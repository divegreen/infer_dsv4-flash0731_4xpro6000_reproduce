#!/usr/bin/env python3
"""CPU mock checks for profile lifecycle; never starts vLLM or contacts 8260.

Uses an owned ephemeral aiohttp server and a tiny local tokenizer. Mock SSE
usage validates protocol/evidence plumbing, not model correctness or GPU traces.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from aiohttp import web
from tokenizers import Tokenizer, models, pre_tokenizers

STAGE = Path(__file__).resolve().parents[2]/'scenarios'
sys.path.insert(0, str(STAGE/'scripts'))
import capacity_audit as audit
import serve_long_context_profile as harness


def raw_plan(port, tokenizer, name='fixture-profile'):
    short = dict(name='short', input_tokens=512, output_tokens=128, weight=1)
    long = dict(name='long', input_tokens=65536, output_tokens=128, weight=1)
    return dict(schema_version=1, experiment_id=name, model='dsv4',
                base_url=f'http://127.0.0.1:{port}', tokenizer_path=str(tokenizer),
                seed=1234, timeout_s=180, max_inflight=16, metrics_interval_s=.1,
                phases=[dict(id='warm-short', role='warmup', mode='closed-loop',
                             requests=1, concurrency=1, seed=100, shape_mix=[short]),
                        dict(id='profile-longshort64', role='profile', mode='trace',
                             seed=101, duration_s=4, shape_mix=[short, long],
                             arrivals=[dict(offset_s=t, shape_name='long' if i == 0 else 'short')
                                       for i, t in enumerate(harness.OFFSETS)])])


def validation_checks(directory, port):
    checks = []
    raw = raw_plan(port, directory)
    plan = harness.validate_profile_plan(raw)
    assert plan == audit.normalize_plan(raw)
    mutations = {'missing warmup': lambda p: p.update(phases=p['phases'][1:]), 'performance role': lambda p: p['phases'][-1].update(role='measure'), 'early step-like shortening': lambda p: p['phases'][-1].update(duration_s=3), 'wrong prompt': lambda p: p['phases'][-1]['shape_mix'][1].update(input_tokens=32768), 'wrong output': lambda p: p['phases'][-1]['shape_mix'][1].update(output_tokens=64), 'wrong offset': lambda p: p['phases'][-1]['arrivals'][1].update(offset_s=0.1), 'missing arrival': lambda p: p['phases'][-1]['arrivals'].pop(), 'EOS length variation': lambda p: p['phases'][-1].update(ignore_eos=False), 'wrong timeout': lambda p: p.update(timeout_s=300)}
    for name, mutate in mutations.items():
        bad = deepcopy(raw)
        mutate(bad)
        try:
            harness.validate_profile_plan(bad)
        except ValueError:
            checks.append('rejected: ' + name)
        else:
            raise AssertionError(name)
    out = (path_at(STAGE, 'results') / raw['experiment_id']).resolve()
    config = dict(port=port, startup_timeout=1200, env={'VLLM_SERVER_DEV_MODE': '1'}, args=['--tensor-parallel-size', '4', '--no-enable-prefix-caching', '--enable-logging-iteration-details', '--profiler-config', json.dumps(harness.profiler_options(out))])
    command = harness.server_command(config, plan, out)
    assert command[command.index('--port') + 2:] == config['args']
    bad = deepcopy(config)
    options = harness.profiler_options(out)
    options['max_iterations'] = 20
    bad['args'][-1] = json.dumps(options)
    try:
        harness.server_command(bad, plan, out)
    except ValueError:
        checks.append('20-step truncation config rejected')
    else:
        raise AssertionError('20-step configuration accepted')
    bad = deepcopy(config)
    bad['args'].remove('--enable-logging-iteration-details')
    try:
        harness.server_command(bad, plan, out)
    except ValueError:
        checks.append('missing iteration evidence rejected')
    else:
        raise AssertionError('missing iteration logging accepted')
    config_path, plan_path = (directory / 'validate-config.json', directory / 'validate-plan.json')
    write_text(config_path, json.dumps(config))
    write_text(plan_path, json.dumps(raw))
    result = subprocess.run([sys.executable, str(Path(harness.__file__)), '--config', str(config_path), '--plan', str(plan_path), '--validate-only'], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    validated = json.loads(result.stdout)
    assert validated['profile_requests'] == 7 and validated['excluded_from_performance'] is True
    assert not out.exists(), 'validate-only must not create experiment directories'
    checks += ['CLI validate-only has no server/filesystem side effect', 'config.args are identical to command args for independent audit']
    return checks


def installed_config_check(directory):
    """Exercise the real wheel's Pydantic config; never initialize CUDA."""
    script = '''
import dataclasses, json, sys
import torch
import vllm
from vllm.config.profiler import ProfilerConfig
assert not torch.cuda.is_initialized()
options = json.loads(sys.argv[1])
parsed = ProfilerConfig(**options)
assert all(getattr(parsed, key) == value for key, value in options.items())
assert parsed.should_profile_frontend is False
rejected_negative = False
try:
    ProfilerConfig(**dict(options, max_iterations=-1))
except ValueError:
    rejected_negative = True
assert rejected_negative
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS', vllm=vllm.__version__, parsed=dataclasses.asdict(parsed),
                      cuda_initialized=torch.cuda.is_initialized(), rejected_negative=True)))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    parsed = subprocess.run([sys.executable, '-c', script,
                             json.dumps(harness.profiler_options(directory))], env=env,
                            text=True, capture_output=True, timeout=60)
    assert parsed.returncode == 0, parsed.stderr
    result = json.loads(parsed.stdout.splitlines()[-1])
    result.update(stdout=parsed.stdout, stderr=parsed.stderr,
                  gpu_visibility='empty; configuration parsing only')
    return result


def startup_failure_checks(directory):
    """No process or HTTP is created: verify launcher failure checkpoints."""
    checks = []
    tokenizer_path = directory / 'tokenizer'
    for case in ('server_exit', 'readiness_timeout'):
        stage = directory / case
        out = path_at(stage, 'results') / case
        out.mkdir(parents=True)
        raw = raw_plan(8231, tokenizer_path, case)
        plan_path, config_path = (stage / 'plan.json', stage / 'config.json')
        write_text(plan_path, json.dumps(raw))
        config = dict(port=8231, startup_timeout=1, env={'VLLM_SERVER_DEV_MODE': '1'}, args=['--tensor-parallel-size', '4', '--no-enable-prefix-caching', '--enable-logging-iteration-details', '--profiler-config', json.dumps(harness.profiler_options(out))])
        write_text(config_path, json.dumps(config))

        class Process:
            pid = 123456
            returncode = 1 if case == 'server_exit' else None
            terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout):
                return self.returncode
        process = Process()
        ticks = iter((i * 0.6 for i in range(20)))
        with patch.object(harness, 'STAGE', stage), patch.object(sys, 'argv', ['serve_long_context_profile.py', '--config', str(config_path), '--plan', str(plan_path)]), patch.dict(os.environ, RESEARCH_RUN_DIR=str(out), CUDA_VISIBLE_DEVICES='0,1,2,3', CUDA_DEVICE_ORDER='PCI_BUS_ID'), patch.object(harness.socket, 'socket'), patch.object(harness.subprocess, 'Popen', return_value=process), patch.object(harness.http.client, 'HTTPConnection') as connection, patch.object(harness.time, 'sleep'), patch.object(harness.time, 'monotonic', side_effect=lambda: next(ticks)):
            connection.return_value.request.side_effect = OSError('mock readiness unavailable')
            try:
                harness.main()
            except (RuntimeError, TimeoutError) as exc:
                assert ('Server exited' if case == 'server_exit' else 'Readiness timeout') in str(exc)
            else:
                raise AssertionError('startup failure was not propagated')
        capacity = load_document(out / 'capacity.json')
        experiment = load_document(out / 'experiment.json')
        assert capacity['status'] == experiment['status'] == 'FAILED'
        assert capacity['fatal_error'] and capacity['finished_utc']
        assert not capacity['runs'] and (not capacity['profiling']['sessions'])
        assert process.terminated == (case == 'readiness_timeout')
        checks.append(case + ': terminal capacity/experiment state and owned child cleanup preserved')
    return checks


async def mock_checks(directory):
    events = []
    state = dict(active=False, fail_start=False, fail_stop=False, fail_completion=False, slow_start=False, completion_count=0)

    async def start(request):
        body = await request.json()
        events.append(dict(kind='start', time=time.time(), body=body))
        state['active'] = True
        if state['slow_start']:
            await asyncio.sleep(0.12)
        return web.Response(status=500 if state['fail_start'] else 200)

    async def stop(request):
        events.append(dict(kind='stop', time=time.time()))
        state['active'] = False
        return web.Response(status=503 if state['fail_stop'] else 200)

    async def complete(request):
        body = await request.json()
        state['completion_count'] += 1
        events.append(dict(kind='completion', time=time.time(), active=state['active'], tokens=len(body['prompt'])))
        if state['fail_completion']:
            return web.Response(status=500)
        if len(body['prompt']) == 65536:
            await asyncio.sleep(3.0)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"index":0,"text":"a"}]}\n\n')
        await asyncio.sleep(0.002)
        output = body['max_tokens']
        terminal = dict(choices=[dict(index=0, text='b', finish_reason='length')], usage=dict(prompt_tokens=len(body['prompt']), completion_tokens=output, total_tokens=len(body['prompt']) + output))
        await response.write(('data: ' + json.dumps(terminal) + '\n\ndata: [DONE]\n\n').encode())
        await response.write_eof()
        return response

    async def metrics(request):
        return web.Response(text='vllm:num_requests_running{model_name="dsv4"} 0\nvllm:num_requests_waiting{model_name="dsv4"} 0\nvllm:kv_cache_usage_perc{model_name="dsv4"} 0.1\n')
    app = web.Application()
    app.router.add_post('/start_profile', start)
    app.router.add_post('/stop_profile', stop)
    app.router.add_post('/v1/completions', complete)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    assert port != 8260
    checks = []
    tokenizer_path = directory / 'tokenizer'
    tokenizer_path.mkdir()
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_path / 'tokenizer.json'))
    try:
        checks += validation_checks(directory, port)
        assert not events, 'validate-only must not issue HTTP'
        for case in ('success', 'start_error', 'stop_error', 'phase_error', 'start_and_stop_error', 'phase_and_stop_error', 'cancelled_phase', 'start_timeout', 'warmup_failure'):
            events.clear()
            state.update(active=False, fail_start=case in ('start_error', 'start_and_stop_error'), fail_stop=case in ('stop_error', 'start_and_stop_error', 'phase_and_stop_error'), fail_completion=case == 'warmup_failure', slow_start=case == 'start_timeout', completion_count=0)
            folder = directory / case
            folder.mkdir()
            plan_path, output = (folder / 'plan.json', folder / 'capacity.json')
            raw = raw_plan(port, tokenizer_path, name=case)
            write_text(plan_path, json.dumps(raw))
            plan = harness.validate_profile_plan(raw)
            report, prepared = harness.prepare_report(plan, plan_path, output)
            original = harness.client.execute_phase

            async def execute(phase, *args, **kwargs):
                if phase['spec']['role'] == 'profile':
                    if case in ('phase_error', 'phase_and_stop_error'):
                        raise RuntimeError('injected phase failure')
                    if case == 'cancelled_phase':
                        raise asyncio.CancelledError('injected cancellation')
                return await original(phase, *args, **kwargs)
            caught = None
            with patch.object(harness.client, 'execute_phase', execute):
                try:
                    await harness.execute_plan(plan, prepared, report, output, control_timeout_s=0.03 if case == 'start_timeout' else 2)
                except BaseException as exc:
                    caught = exc
            saved = load_document(output)
            assert saved['profiling']['excluded_from_performance'] is True
            assert all((r['phase']['role'] in ('profile', 'warmup') for r in saved['runs']))
            assert not state['active']
            controls = saved['profiling']['sessions']
            if case == 'warmup_failure':
                assert not controls and (not any((e['kind'] in ('start', 'stop') for e in events)))
                assert saved['status'] == 'FAILED' and caught is None
                checks.append('warmup failure prevents profile start; failed raw rows retained')
                continue
            controls = controls[0]['controls']
            assert [e['path'] for e in controls] == ['/start_profile', '/stop_profile']
            assert controls[0]['payload']['max_iterations'] == 0
            assert all(('finished_unix' in e for e in controls))
            assert events[-1]['kind'] == 'stop'
            if case == 'success':
                assert caught is None and saved['status'] == 'PASSED'
                assert all((e['status'] == 'PASS' for e in controls))
                run = saved['runs'][-1]
                assert controls[0]['finished_unix'] <= run['start_unix']
                assert run['end_unix'] <= controls[1]['started_unix']
                seen = [e for e in events if e['kind'] == 'completion']
                assert len(seen) == 8 and seen[0]['active'] is False
                assert all((e['active'] is True for e in seen[1:]))
                assert [r['planned_offset_s'] for r in run['requests']] == list(harness.OFFSETS)
                assert sum((r['usage']['prompt_tokens'] for r in run['requests'])) == 68608
                assert sum((r['usage']['completion_tokens'] for r in run['requests'])) == 896
                assert len(Path(run['raw_requests_jsonl']).read_text().splitlines()) == 7
                checker = audit.Auditor(STAGE)
                checker.report(output)
                result = checker.finish()
                assert not result['errors'], result['errors']
                assert result['status'] == 'INCOMPLETE'
                checks += ['unprofiled warmup precedes full seven-request capture', 'start returns before phase origin; stop begins after all terminal rows', 'fixed arrivals, usage totals, JSONL and raw metrics preserved', 'independent audit raw formulas pass; supervisor provenance intentionally absent']
            else:
                assert caught is not None and saved['status'] == 'FAILED'
                assert saved['fatal_error']
                if state['fail_stop']:
                    assert controls[-1]['status'] == 'FAILED'
                    assert saved['profiling']['sessions'][0]['stop_error']
                    if case == 'stop_error':
                        assert saved['runs'][-1]['summary']['successful'] == 7
                    elif case == 'start_and_stop_error':
                        assert '/start_profile: HTTP 500' in str(caught)
                        assert '/start_profile: HTTP 500' in saved['fatal_error']['message']
                    elif case == 'phase_and_stop_error':
                        assert str(caught) == saved['fatal_error']['message'] == 'injected phase failure'
                else:
                    assert controls[-1]['status'] == 'PASS'
                checks.append(case + ': finally stop attempted and both lifecycle/error evidence persisted')
            if case == 'start_timeout':
                await asyncio.sleep(0.13)
        return checks
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = dict(status='RUNNING', scope='CPU lifecycle/HTTP fixture only; no GPU or model service')
    try:
        with tempfile.TemporaryDirectory(prefix='phase-b-profile-fixture-') as temp:
            directory = Path(temp)
            report['installed_profiler_config'] = installed_config_check(directory)
            report['checks'] = asyncio.run(mock_checks(directory))
            report['checks'] += startup_failure_checks(directory)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        report['finished_utc'] = harness.client.utc_now()
        report['harness_sha256'] = harness.client.sha_file(harness.__file__)
        report['fixture_sha256'] = harness.client.sha_file(__file__)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open_output(args.output, 'x') as handle:
                json.dump(report, handle, indent=2)
                handle.write('\n')
        print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
