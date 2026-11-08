#!/usr/bin/env python3
"""Owned localhost Stage B profile; finite warmup and seven-request microtrace.

Run under run_logged.py. Config/plan are immutable inputs, created by the study
coordinator. Only this new harness controls profiling; the capacity client and
its raw evidence format are reused unchanged. Never use these profiled timings
for the normal performance ranking.
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

import capacity_client as client

STAGE = Path(__file__).resolve().parents[1]
OFFSETS = (0., .25, .75, 1.25, 1.75, 2.25, 2.75)


def profiler_options(out: Path) -> dict:
    """Use this helper when creating the immutable experiment config."""
    return dict(profiler='torch', torch_profiler_dir=str((out/'profiles').resolve()),
                torch_profiler_activities=['CPU', 'CUDA'],
                torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                torch_profiler_with_memory=False, detailed_trace_annotation=False,
                ignore_frontend=True, delay_iterations=0, max_iterations=0,
                warmup_iterations=0, wait_iterations=0)


def validate_profile_plan(raw: dict) -> dict:
    plan = client.validate_plan(raw)
    if plan['model'] != 'dsv4' or plan['timeout_s'] != 180:
        raise ValueError('Profile plan requires model=dsv4 and timeout_s=180')
    phases = plan['phases']
    if not 2 <= len(phases) <= 9:
        raise ValueError('Require 1..8 bounded warmup phases, then one profile phase')
    if any(p['role'] != 'warmup' for p in phases[:-1]) or phases[-1]['role'] != 'profile':
        raise ValueError('Only warmup phases followed by exactly one profile phase are allowed')
    for phase in phases:
        if not 7 <= phase['max_inflight'] <= 32:
            raise ValueError('Profile harness max_inflight must be 7..32')
        if not phase['ignore_eos']:
            raise ValueError('Use fixed output lengths with ignore_eos=true')
    for phase in phases[:-1]:
        if phase['mode'] not in ('trace', 'closed-loop', 'burst'):
            raise ValueError('Warmup must be finite, not Poisson')
        count = len(phase['arrivals']) if phase['mode'] == 'trace' else phase['requests']
        if count > 32 or (phase['mode'] == 'trace' and phase['duration_s'] > 10):
            raise ValueError('Warmup is limited to 32 requests and trace horizon <=10s')
        if any(s['input_tokens'] > 65536 or s['output_tokens'] > 128
               for s in phase['shape_mix']):
            raise ValueError('Warmup must stay within 64K input / 128 output')
    phase = phases[-1]
    if phase['mode'] != 'trace' or phase['duration_s'] != 4:
        raise ValueError('Profile must be a trace with duration_s=4')
    items = client.make_items(phase, len(phase['arrivals']))
    identity = [(a['offset_s'], r['input_tokens'], r['requested_output_tokens'])
                for a, r in zip(phase['arrivals'], items)]
    expected = [(t, 65536 if i == 0 else 512, 128) for i, t in enumerate(OFFSETS)]
    if identity != expected:
        raise ValueError('Profile must contain the fixed 64K + six 512-token microtrace')
    return plan


def server_command(config: dict, plan: dict, out: Path) -> list[str]:
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
    if args.count('--profiler-config') != 1 or any(a.startswith('--profiler-config.') or
                                                 a.startswith('--profiler-config=') for a in args):
        raise ValueError('Provide exactly one explicit --profiler-config JSON value in config.args')
    index = args.index('--profiler-config')
    if index + 1 == len(args):
        raise ValueError('Missing profiler config JSON')
    actual = json.loads(args[index + 1])
    if actual != profiler_options(out):
        raise ValueError('Profiler config must equal profiler_options(result_directory), with no step limit')
    if ('--enable-logging-iteration-details' not in args or '--disable-log-stats' in args
            or '--no-enable-logging-iteration-details' in args):
        raise ValueError('Enable iteration details and retain stats logging')
    if config.get('env', {}).get('VLLM_SERVER_DEV_MODE') != '1':
        raise ValueError('Record VLLM_SERVER_DEV_MODE=1 in config.env')
    if not 1 <= config.get('startup_timeout', 1200) <= 1200:
        raise ValueError('startup_timeout must be in 1..1200 seconds')
    return [str(Path(sys.executable).parent/'vllm'), 'serve', client.DEFAULT_TOKENIZER,
            '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', str(port), *args]


def prepare_report(plan: dict, plan_path: Path, output: Path) -> tuple[dict, list]:
    """Prepare all token IDs before HTTP/profile start, preserving client schema."""
    with open_output(output, 'x') as handle:
        handle.write('{}\n')
    report = dict(schema_version=1, status='PREPARING', created_utc=client.utc_now(), plan=plan, plan_path=str(plan_path.resolve()), plan_sha256=client.sha_file(plan_path), client_sha256=client.sha_file(client.__file__), base_helper_sha256=client.sha_file(client.base.__file__), profile_harness_sha256=client.sha_file(__file__), python=sys.version, aiohttp=importlib.metadata.version('aiohttp'), research_slo=client.SLO, transport=dict(connection_policy='fresh_per_request', retries=0), metric_definitions=dict(tpot='(E2E-TTFT)/(actual output tokens-1), not token ITL', scheduled='planned arrival to response; includes client scheduling lag', throughput='Diagnostic only; profiling excluded from performance ranking', p95='Descriptive only for seven-request profile; no capacity claim', stability='Not assessed; warmup/profile roles excluded from capacity'), profiling=dict(excluded_from_performance=True, sessions=[], scope='Worker CPU+CUDA and iteration counts; no EngineCore CPU profiler'), runs=[])
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
        result['excluded_from_performance'] = True
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
    plan = validate_profile_plan(load_document(args.plan))
    expected = (path_at(STAGE, 'results') / plan['experiment_id']).resolve()
    if Path(plan['experiment_id']).name != plan['experiment_id']:
        raise ValueError('Experiment ID must be a simple directory name')
    command = server_command(config, plan, expected)
    if args.validate_only:
        print(json.dumps(dict(command=command, result_directory=str(expected), phases=[p['id'] for p in plan['phases']], profile_requests=7, excluded_from_performance=True), indent=2))
        return 0
    out = Path(os.environ['RESEARCH_RUN_DIR']).resolve()
    if out != expected or not out.is_dir():
        raise ValueError('Run under run_logged.py with the matching unique result directory')
    env = os.environ.copy()
    env.update(config.get('env', {}))
    if env.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or env.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('Profile harness must inherit GPU 0,1,2,3 and PCI_BUS_ID from supervisor')
    port = config.get('port', 8260)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    for name, value in [('config.json', config), ('capacity-plan.json', plan), ('server-command.json', dict(command=command, extra_environment=config.get('env', {})))]:
        with open_output(out / name, 'x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
    start = time.monotonic()
    experiment = dict(status='PREPARING', clients=[], plan=str(args.plan.resolve()), profile=True, excluded_from_performance=True)
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
            entry = dict(task='capacity-profile', implementation=str(Path(__file__).resolve()), status='RUNNING', excluded_from_performance=True)
            experiment['clients'].append(entry)
            save()
            try:
                asyncio.run(execute_plan(plan, prepared, report, out / 'capacity.json'))
                code = 0 if report['status'] == 'PASSED' else 1
                entry.update(exit_code=code, status='PASS' if code == 0 else 'FAILED')
                if code:
                    raise RuntimeError('Profile client phase failed; preserve partial evidence')
            except BaseException:
                entry.update(exit_code=1, status='FAILED')
                raise
            finally:
                save()
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
