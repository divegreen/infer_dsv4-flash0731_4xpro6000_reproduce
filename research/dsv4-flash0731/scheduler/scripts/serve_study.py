#!/usr/bin/env python3
"""Stage D owned localhost study. Reuses the frozen capacity request client.

Separate profiled mechanism diagnostics and unprofiled matched performance.
Historical Stage B lifecycle retained as provenance, without editing its files.
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
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import aiohttp

STAGE = Path(__file__).resolve().parents[1]
SCENARIO = STAGE.parent/'scenarios'
sys.path.insert(0, str(SCENARIO/'scripts'))
import study_client as client

STAGE = Path(__file__).resolve().parents[1]
OFFSETS = (0., 0., .25, .75, 1.25, 1.75, 2.25, 2.75)


def profiler_options(out: Path) -> dict:
    """Use this helper when creating the immutable experiment config."""
    return dict(profiler='torch', torch_profiler_dir=str((out/'profiles').resolve()),
                torch_profiler_activities=['CPU', 'CUDA'],
                torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                torch_profiler_with_memory=False, detailed_trace_annotation=False,
                ignore_frontend=True, delay_iterations=0, max_iterations=0,
                warmup_iterations=0, wait_iterations=0)


def validate_study_plan(raw: dict) -> dict:
    plan = client.validate_plan(raw)
    if plan['model'] != 'dsv4' or plan['timeout_s'] != 180:
        raise ValueError('Study requires model=dsv4 and timeout_s=180')
    if not 2 <= len(plan['phases']) <= 16:
        raise ValueError('Require 2..16 bounded phases')
    profiles = [p for p in plan['phases'] if p['role'] == 'profile']
    if profiles:
        if len(profiles) != 1 or plan['phases'][-1] is not profiles[0]:
            raise ValueError('Exactly one final profile phase required')
        if any(p['role'] != 'warmup' for p in plan['phases'][:-1]):
            raise ValueError('Profile must be preceded only by warmup')
        phase = profiles[0]
        if phase['mode'] != 'trace' or phase['duration_s'] != 4:
            raise ValueError('Profile must use four-second dense diagnostic trace')
        items = client.make_items(phase, len(phase['arrivals']))
        actual = [(a['offset_s'], i['input_tokens'], i['requested_output_tokens'])
                  for a, i in zip(phase['arrivals'], items)]
        expected = [(t, 65536 if i < 2 else 512, 128) for i, t in enumerate(OFFSETS)]
        if actual != expected:
            raise ValueError('Profile requires two 64K and six 512-token requests')
    for phase in plan['phases']:
        if not 8 <= phase['max_inflight'] <= 128 or not phase['ignore_eos']:
            raise ValueError('Bounded fixed-output study requires 8..128 max_inflight')
        if any(x['input_tokens'] > 65536 or x['output_tokens'] > 256 for x in phase['shape_mix']):
            raise ValueError('Study bounded to 64K input and 256 output')
        if phase['role'] == 'warmup':
            if phase['mode'] not in ('trace','closed-loop','burst'):
                raise ValueError('Warmup must be finite')
            count = len(phase['arrivals']) if phase['mode'] == 'trace' else phase['requests']
            if count > 32 or (phase['mode']=='trace' and phase['duration_s']>10):
                raise ValueError('Warmup bound exceeded')
    return plan


def profile_control_timeout(config: dict, plan: dict) -> float:
    """Bound only local profiler API export waits; request timeout stays fixed."""
    value = config.get('profile_control_timeout_s', 120.)
    if 'profile_control_timeout_s' in config and not any(p['role'] == 'profile' for p in plan['phases']):
        raise ValueError('profile_control_timeout_s is only valid for Profile runs')
    if type(value) not in (int, float) or not 0 < value <= 600:
        raise ValueError('profile_control_timeout_s must be finite and in (0,600] seconds')
    return float(value)


def quality_command(config: dict, plan: dict, out: Path):
    """Optional fixed natural-EOS sidecar; never part of performance phases."""
    if 'quality_probe' not in config:
        return None
    spec = config['quality_probe']
    expected = dict(rounds=2, max_tokens=128, timeout_s=180)
    if not isinstance(spec, dict) or spec != expected or any(type(spec[k]) is not int for k in expected):
        raise ValueError('quality_probe must be exactly rounds=2, max_tokens=128, timeout_s=180')
    if any(p['role'] == 'profile' for p in plan['phases']):
        raise ValueError('Quality sidecar is only allowed after unprofiled performance')
    return [sys.executable, str(SCENARIO/'scripts/agent_benchmark.py'),
            '--mode', 'quality', '--base-url', plan['base_url'], '--model', plan['model'],
            '--rounds', '2', '--max-tokens', '128', '--timeout', '180',
            '--cache-label', 'unspecified', '--output', str(out/'quality.json')]


def execute_quality(config, plan, out, experiment, checkpoint, *, run_child=None):
    command = quality_command(config, plan, out)
    if command is None:
        return
    if run_child is None:
        run_child = subprocess.run
    row = dict(status='RUNNING', command=command, report=str(out / 'quality.json'), log=str(out / 'quality.log'), started_utc=client.utc_now(), excluded_from_performance=True, quality_scores_gate_performance=False, scope='Eight synthetic natural-EOS probes twice; not general model quality, SWE-bench or a live Agent benchmark.')
    experiment['quality_probe'] = row
    checkpoint()
    try:
        with open_output(out / 'quality.log', 'x') as log:
            completed = run_child(command, stdout=log, stderr=subprocess.STDOUT, timeout=2 * 8 * 180 + 60, check=False)
        row['exit_code'] = completed.returncode
        report = load_document(out / 'quality.json')
        row['report_status'] = report.get('status')
        row['report_sha256'] = client.sha_file(out / 'quality.json')
        if completed.returncode != 0 or report.get('status') != 'COMPLETED' or report.get('mode') != 'quality' or (report.get('rounds') != 2) or (len(report.get('runs', [])) != 16) or (report.get('summary', {}).get('protocol_failures') != 0):
            raise RuntimeError('Independent quality probe protocol/execution failed; completed performance data is preserved')
        row['status'] = 'PASS'
    except BaseException as exc:
        row.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        row['finished_utc'] = client.utc_now()
        checkpoint()


def server_command(config: dict, plan: dict, out: Path) -> list[str]:
    profile_control_timeout(config, plan)
    quality_command(config, plan, out)
    args = config.get('args')
    if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
        raise ValueError('Config args must be a nonempty string list')
    reserved = ('--host', '--port', '--served-model-name')
    if any(a.split('=', 1)[0] in reserved for a in args):
        raise ValueError('Harness owns server address and alias')
    port = config.get('port', 8260)
    if plan['base_url'] != f'http://127.0.0.1:{port}':
        raise ValueError('Plan and owned localhost port differ')
    if '--no-enable-prefix-caching' not in args or '--enable-prefix-caching' in args:
        raise ValueError('Profile comparison requires prefix caching disabled')
    is_profile = any(p['role']=='profile' for p in plan['phases'])
    if is_profile:
        if args.count('--profiler-config') != 1:
            raise ValueError('Profile requires explicit profiler config')
        index = args.index('--profiler-config')
        if json.loads(args[index+1]) != profiler_options(out):
            raise ValueError('Profiler output/options must match unique result directory')
        if '--enable-logging-iteration-details' not in args:
            raise ValueError('Profile requires iteration details')
    elif any(a.startswith('--profiler-config') or a == '--enable-logging-iteration-details' for a in args):
        raise ValueError('Performance runs must not enable profiling or iteration logs')
    if config.get('env', {}).get('VLLM_SERVER_DEV_MODE') != '1':
        raise ValueError('Explicit local development API required for cache reset/profile')
    if args.count('--tensor-parallel-size') != 1 or args[args.index('--tensor-parallel-size')+1] != '4':
        raise ValueError('Stage D initial study is TP4 only')
    if config.get('name') != plan['experiment_id']:
        raise ValueError('Config and plan experiment ID differ')
    prohibited_env = {'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','PYTHONPATH','PYTHONHOME','RESEARCH_RUN_DIR'}
    if prohibited_env.intersection(config.get('env',{})):
        raise ValueError('Config cannot override owned GPU or module environment')
    if not 1 <= config.get('startup_timeout', 1200) <= 1200:
        raise ValueError('startup_timeout must be in 1..1200 seconds')
    return [str(Path(sys.executable).parent/'vllm'), 'serve', client.DEFAULT_TOKENIZER,
            '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', str(port), *args]


def prepare_report(plan: dict, plan_path: Path, output: Path) -> tuple[dict, list]:
    """Prepare all token IDs before HTTP/profile start, preserving client schema."""
    with open_output(output, 'x') as handle:
        handle.write('{}\n')
    report = dict(schema_version=1, status='PREPARING', created_utc=client.utc_now(), plan=plan, plan_path=str(plan_path.resolve()), plan_sha256=client.sha_file(plan_path), client_sha256=client.sha_file(client.__file__), base_helper_sha256=client.sha_file(client.base.__file__), profile_harness_sha256=client.sha_file(__file__), python=sys.version, aiohttp=importlib.metadata.version('aiohttp'), research_slo=client.SLO, transport=dict(connection_policy='fresh_per_request', retries=0), metric_definitions=dict(tpot='(E2E-TTFT)/(actual output tokens-1), not token ITL', scheduled='planned arrival to response; includes client scheduling lag', throughput='Complete cohort horizon plus drain; profile phases excluded from ranking', p95='Descriptive diagnostic or repeated normal workload; not automatic capacity claim', stability='Not assessed; warmup/profile roles excluded from capacity'), profiling=dict(excluded_from_performance=any((p['role'] == 'profile' for p in plan['phases'])), sessions=[], scope='Worker CPU+CUDA and iteration counts; no EngineCore CPU profiler'), runs=[])
    client.save_json(output, report)
    try:
        prepared, tokenizer_file = client.prepare(plan)
        workload = output.parent / (output.stem + '-workload.json')
        with open_output(workload, 'x') as handle:
            json.dump(prepared, handle, separators=(',', ':'))
            handle.write('\n')
        report.update(status='PREPARED', workload_file=str(workload), workload_sha256=client.sha_file(workload), tokenizer_sha256=client.sha_file(tokenizer_file), prepared_requests={p['spec']['id']: len(p['items']) for p in prepared})
        client.save_json(output, report)
        return (report, prepared)
    except BaseException as exc:
        report.update(status='FAILED', fatal_error=dict(type=type(exc).__name__, message=str(exc)), finished_utc=client.utc_now())
        client.save_json(output, report)
        raise


async def control(session, url, path, payload, capture, checkpoint, timeout_s):
    event = dict(path=path, payload=payload, started_unix=time.time(),
                 started_monotonic=time.perf_counter(), status='STARTED')
    capture['controls'].append(event)
    checkpoint()
    try:
        async with session.post(url+path, json=payload, allow_redirects=False,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as response:
            body = await response.text()
            event.update(http_status=response.status, body=body[:4096])
            if response.status != 200:
                raise RuntimeError(f'{path}: HTTP {response.status}: {body[:1000]}')
        event['status'] = 'PASS'
    except BaseException as exc:
        event.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        event.update(finished_unix=time.time(), finished_monotonic=time.perf_counter())
        checkpoint()


async def execute_plan(plan, prepared, report, output, *, control_timeout_s=120.0):
    """Profile begins before execute_phase creates origin; stop is always attempted.

    A failed/timeout start may have started some workers, so it also gets a stop
    attempt. Keep both errors if cleanup fails; never mask the original error.
    """

    def checkpoint():
        client.save_json(output, report)

    async def phase_body(phase):
        name = phase['spec']['id']
        rows_path = output.parent / f'{output.stem}-{name}-requests.jsonl'
        with open_output(rows_path, 'x') as sink:
            result = await client.execute_phase(phase, plan, output.parent / f'{output.stem}-{name}-metrics', sink)
        result['raw_requests_jsonl'] = str(rows_path)
        result['excluded_from_performance'] = phase['spec']['role'] in ('warmup', 'profile')
        report['runs'].append(result)
        checkpoint()
        return result
    report['status'] = 'RUNNING'
    checkpoint()
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    try:
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
            for phase in prepared:
                name = phase['spec']['id']
                report['active_phase'] = name
                checkpoint()
                if phase['spec']['role'] == 'profile':
                    capture = dict(phase_id=name, prefix=name, controls=[])
                    report['profiling']['sessions'].append(capture)
                    error = stop_error = None
                    try:
                        await control(session, plan['base_url'], '/start_profile', dict(profile_prefix=name, delay_iterations=0, max_iterations=0), capture, checkpoint, control_timeout_s)
                        result = await phase_body(phase)
                    except BaseException as exc:
                        error = exc
                    finally:
                        try:
                            await control(session, plan['base_url'], '/stop_profile', None, capture, checkpoint, control_timeout_s)
                        except BaseException as exc:
                            stop_error = exc
                            capture['stop_error'] = dict(type=type(exc).__name__, message=str(exc))
                        checkpoint()
                    if error is not None:
                        raise error
                    if stop_error is not None:
                        raise stop_error
                else:
                    result = await phase_body(phase)
                print(json.dumps(dict(phase=name, role=phase['spec']['role'], summary=result['summary']), sort_keys=True), flush=True)
                if result['summary']['failed'] or result['summary']['censored']:
                    report.update(status='FAILED' if result['summary']['failed'] else 'CENSORED', stop_reason='Stop after failed/censored phase; no retries')
                    break
            else:
                report['status'] = 'PASSED'
            report.pop('active_phase', None)
    except BaseException as exc:
        report.update(status='FAILED', fatal_error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        report['finished_utc'] = client.utc_now()
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    config = load_document(args.config)
    plan = validate_study_plan(load_document(args.plan))
    expected = (path_at(STAGE, 'results') / plan['experiment_id']).resolve()
    if Path(plan['experiment_id']).name != plan['experiment_id']:
        raise ValueError('Experiment ID must be a simple directory name')
    command = server_command(config, plan, expected)
    if args.validate_only:
        print(json.dumps(dict(command=command, result_directory=str(expected), phases=[p['id'] for p in plan['phases']], profile_requests=8 if any((p['role'] == 'profile' for p in plan['phases'])) else 0, excluded_from_performance=any((p['role'] == 'profile' for p in plan['phases']))), indent=2))
        return 0
    out = Path(os.environ['RESEARCH_RUN_DIR']).resolve()
    if out != expected or not out.is_dir():
        raise ValueError('Run under run_logged.py with the matching unique result directory')
    env = os.environ.copy()
    env.update(config.get('env', {}))
    env['PYTHONPATH'] = str(STAGE / 'scripts')
    env['RESEARCH_D_TRACE_DIR'] = str(out) if any((p['role'] == 'profile' for p in plan['phases'])) else ''
    if env.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or env.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('Profile harness must inherit GPU 0,1,2,3 and PCI_BUS_ID from supervisor')
    port = config.get('port', 8260)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    for name, value in [('config.json', config), ('capacity-plan.json', plan), ('server-command.json', dict(command=command, extra_environment=config.get('env', {}), owned_environment={k: env[k] for k in ('PYTHONPATH', 'RESEARCH_D_TRACE_DIR', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}))]:
        with open_output(out / name, 'x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
    start = time.monotonic()
    experiment = dict(status='PREPARING', clients=[], plan=str(args.plan.resolve()), profile=any((p['role'] == 'profile' for p in plan['phases'])), excluded_from_performance=any((p['role'] == 'profile' for p in plan['phases'])))
    server = None
    report = None

    def save():
        client.save_json(out / 'experiment.json', experiment)
    save()
    try:
        report, prepared = prepare_report(plan, args.plan, out / 'capacity.json')
        with open_output(out / 'server.log', 'x') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            experiment.update(status='STARTING', server_pid=server.pid)
            save()
            ready_start = time.monotonic()
            while time.monotonic() - ready_start < config.get('startup_timeout', 1200):
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode}')
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                try:
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    response.read()
                    if response.status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                time.sleep(2)
            else:
                raise TimeoutError('Readiness timeout')
            experiment.update(status='READY', startup_seconds=time.monotonic() - ready_start)
            entry = dict(task='capacity-study', implementation=str(Path(__file__).resolve()), status='RUNNING', excluded_from_performance=experiment['profile'])
            experiment['clients'].append(entry)
            save()
            try:
                asyncio.run(execute_plan(plan, prepared, report, out / 'capacity.json', control_timeout_s=profile_control_timeout(config, plan)))
                code = 0 if report['status'] == 'PASSED' else 1
                entry.update(exit_code=code, status='PASS' if code == 0 else 'FAILED')
                if code:
                    raise RuntimeError('Profile client phase failed; preserve partial evidence')
            except BaseException:
                entry.update(exit_code=1, status='FAILED')
                raise
            finally:
                save()
            execute_quality(config, plan, out, experiment, save)
            experiment['status'] = 'PASS'
    except BaseException as exc:
        experiment.update(status='FAILED', error=repr(exc))
        if report is not None and report['status'] in ('PREPARING', 'PREPARED', 'RUNNING'):
            report.update(status='FAILED', finished_utc=client.utc_now(), fatal_error=dict(type=type(exc).__name__, message=str(exc)))
            client.save_json(out / 'capacity.json', report)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        experiment['elapsed_seconds'] = time.monotonic() - start
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
