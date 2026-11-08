#!/usr/bin/env python3
"""CPU mock controls for D serial lifecycle; never invokes GPU/HTTP/model."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import zipfile
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parents[1];sys.path.insert(0,str(D/'scripts'))
import run_study_series as q
import check_run_logged as fixtures

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, obj if isinstance(obj, bytes) else (json.dumps(obj) + '\n').encode())

def run():
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append(dict(name=name, status='PASS'))
        except Exception as e:
            checks.append(dict(name=name, status='FAIL', error=repr(e)))

    def reject(fn):
        try:
            fn()
        except (RuntimeError, ValueError, FileExistsError, TimeoutError):
            return
        raise AssertionError('Invalid/uncertain state accepted')
    with tempfile.TemporaryDirectory(prefix='d-series-MOCK-') as temp:
        root = Path(temp)
        phase = root / 'research/scheduler'
        research = phase.parent
        for name in (*q.supervisor.D_RUNTIME, 'run_study_series.py'):
            write(phase / 'scripts' / name, b'# MOCK runtime\n')
        for relative in ('scripts/benchmark_client.py', 'scenarios/scripts/agent_benchmark.py'):
            write(research / relative, b'# MOCK helper\n')
        for relative in q.supervisor.DEPENDENCY_TREES:
            (research / relative).mkdir(parents=True, exist_ok=True)
        for name in ('environment.json', 'model-inventory.json'):
            write(research / name, {})
        for path in q.supervisor.quality_fixture_inputs(D.parent):
            relative = path.relative_to(path_at(D.parent, 'scenarios/workloads/agent'))
            write(path_at(research, 'scenarios/workloads/agent') / relative, path.read_bytes())
        installed, package = fixtures.fake_installed(root)
        q.supervisor.installed_inputs = lambda: (installed, package)
        q.ROOT = phase
        jobs = []
        for i in range(1, 7):
            job = dict(id=f'd-mock-{i}', config=f'd-mock-{i}.json', capacity_plan=f'd-mock-{i}.json', timeout=20, reference='MOCK', hypothesis='CPU ONLY')
            write(path_at(phase, 'configs') / job['config'], dict(name=job['id'], quality_probe=dict(rounds=2, max_tokens=128, timeout_s=180)))
            write(path_at(phase, 'workloads') / job['capacity_plan'], dict(experiment_id=job['id']))
            jobs.append(job)

        def finish(job, command):
            folder = path_at(phase, 'results') / job['id']
            folder.mkdir(parents=True)
            sources = q.resume_sources(job)
            payloads = {k: (installed[k] if k.startswith('installed/') else research / k).read_bytes() for k in sources}
            identity = dict(files={k: dict(path='MOCK/' + k, sha256=sources[k], bytes=len(v)) for k, v in payloads.items()})
            write(folder / 'runtime-identity.json', identity)
            with zipfile.ZipFile(folder / 'source-snapshot.zip', 'w') as z:
                for key, value in payloads.items():
                    z.writestr(key, value)
            launch = dict(experiment_id=job['id'], command=command, source_sha256=sources, source_snapshot_sha256=q.sha((folder / 'source-snapshot.zip').read_bytes()), runtime_identity_sha256=q.sha((folder / 'runtime-identity.json').read_bytes()), environment={'MOCK': 'no_gpu'}, cwd=str(phase))
            write(folder / 'command.json', launch)
            write(folder / 'status.json', dict(launch, supervisor_exit_code=0, exit_code=0, source_identity_unchanged=True, changed_source_members=[], timed_out=False, cleanup_error=None, supervisor_error=None, interrupted_signal=None, end_utc='MOCK'))
            write(folder / 'config.json', load_document(path_at(phase, 'configs') / job['config']))
            write(folder / 'capacity.json', dict(status='PASSED'))
            write(folder / 'quality.json', dict(status='COMPLETED', mode='quality', summary=dict(protocol_failures=0, quality_passed=0)))
            write(folder / 'quality.log', b'MOCK\n')
            write(folder / 'experiment.json', dict(status='PASS', clients=[dict(exit_code=0)], quality_probe=dict(status='PASS', exit_code=0, report_sha256=q.sha((folder / 'quality.json').read_bytes()))))
        calls = []
        idle = []
        stop_after = False

        def gpu(command, **kwargs):
            assert command == ['nvidia-smi', '-i', '0,1,2,3', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
            idle.append(command)
            return '0\n0\n0\n0\n'

        def child(command):
            job = next((j for j in jobs if j['id'] == command[command.index('--name') + 1]))
            calls.append(job['id'])
            finish(job, command[command.index('--') + 1:])
            if stop_after:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0
        q.subprocess.check_output = gpu
        q.subprocess.call = child
        q.time.sleep = lambda _: None

        def execute(selected, name, extra=()):
            plan = path_at(phase, 'workloads') / ('series-' + name + '.json')
            write(plan, dict(jobs=selected))
            state = path_at(phase, 'logs') / ('series-' + name + '.json')
            old = sys.argv
            sys.argv = ['MOCK', '--plan', str(plan), '--state', str(state), *extra]
            try:
                code = q.main()
            finally:
                sys.argv = old
            return (code, load_document(state))

        def normal():
            code, state = execute(jobs[:2], 'normal')
            assert code == 0 and state['status'] == 'COMPLETED' and (calls == ['d-mock-1', 'd-mock-2']) and (len(idle) == 4)
            assert all((row['status'] == 'COMPLETED' and len(row['gpu_idle_checks']) == 2 for row in state['jobs']))
        check('two sequential mock jobs each require two assigned-GPU idle samples and terminal verification', normal)

        def resume():
            before = (len(calls), len(idle))
            code, state = execute(jobs[:2], 'resume')
            assert code == 0 and before == (len(calls), len(idle)) and all((row['status'].startswith('SKIPPED') for row in state['jobs']))
        check('valid exact resume skips without executing or polling GPU', resume)
        first = path_at(phase, 'results') / jobs[0]['id']
        cmd = q.command(jobs[0])
        expected = cmd[cmd.index('--') + 1:]

        def mutate(filename, change):
            path = first / filename
            original = path.read_bytes()
            obj = json.loads(original)
            change(obj)
            write(path, obj)
            try:
                reject(lambda: q.terminal(first, expected, jobs[0]))
            finally:
                write_bytes(path, original)
        cases = [('supervisor exit failure', lambda x: x.update(supervisor_exit_code=1)), ('false boolean exit', lambda x: x.update(supervisor_exit_code=False)), ('cleanup error', lambda x: x.update(cleanup_error='MOCK')), ('source changed', lambda x: x.update(source_identity_unchanged=False)), ('missing end', lambda x: x.pop('end_utc')), ('interrupt', lambda x: x.update(interrupted_signal=15)), ('timeout', lambda x: x.update(timed_out=True))]
        for name, fn in cases:
            check('reject ' + name, lambda fn=fn: mutate('status.json', fn))
        check('reject missing quality protocol success', lambda: mutate('experiment.json', lambda x: x['quality_probe'].update(status='FAILED', exit_code=1)))
        check('reject different actual command', lambda: reject(lambda: q.terminal(first, expected + ['--changed'], jobs[0])))

        def source_change():
            path = phase / 'scripts/study_client.py'
            old = path.read_bytes()
            write_bytes(path, old + b'# drift\n')
            try:
                reject(lambda: q.terminal(first, expected, jobs[0]))
            finally:
                write_bytes(path, old)
        check('reject changed current runtime even if saved archive unchanged', source_change)

        def irrelevant():
            write(phase / 'scripts/new-offline-analysis.py', b'# analysis only\n')
            q.terminal(first, expected, jobs[0])
        check('new offline analysis does not invalidate runtime identity', irrelevant)

        def fixture_change():
            p = path_at(research, 'scenarios/workloads/agent/chunk_tail/extra.py')
            write(p, b'# new fixture\n')
            try:
                reject(lambda: q.terminal(first, expected, jobs[0]))
            finally:
                p.unlink()
        check('reject added fixture set member', fixture_change)

        def stopping():
            nonlocal stop_after
            stop_after = True
            try:
                code, state = execute(jobs[2:4], 'stopping')
            finally:
                stop_after = False
            assert code == 130 and state['status'] == 'STOPPED AFTER CURRENT' and (calls[-1] == 'd-mock-3') and (not path_at(phase, 'results/d-mock-4').exists())
        check('SIGTERM completes current verified child and does not start next', stopping)

        def failed_existing():
            write(path_at(phase, 'results/d-mock-4/command.json'), dict(status='MOCK incomplete'))
            before = len(calls)
            reject(lambda: execute([jobs[3]], 'uncertain'))
            assert len(calls) == before
        check('uncertain existing ID refuses rerun without overwriting', failed_existing)

        def failed_wait():
            path = first / 'status.json'
            old = path.read_bytes()
            data = json.loads(old)
            data['supervisor_exit_code'] = 1
            write(path, data)
            try:
                before = len(calls)
                reject(lambda: execute([jobs[4]], 'failed-wait', ['--wait-for', jobs[0]['id']]))
                assert len(calls) == before
            finally:
                write_bytes(path, old)
        check('wait-for rejects prior failed supervisor despite child exit0', failed_wait)

        def bad_job():
            for changed in (dict(jobs[5], config='../outside.json'), dict(jobs[5], capacity_plan='../outside.json'), dict(jobs[5], timeout=0), dict(jobs[5], timeout=True)):
                reject(lambda: q.command(changed))
        check('input traversal and invalid timeout rejected before launch', bad_job)

        def busy():
            before = len(calls)
            old_gpu = q.subprocess.check_output
            old_clock = q.time.monotonic
            counter = [0]
            q.subprocess.check_output = lambda *a, **k: '999\n0\n0\n0\n'

            def clock():
                counter[0] += 40
                return counter[0]
            q.time.monotonic = clock
            try:
                reject(lambda: execute([jobs[5]], 'busy'))
                assert len(calls) == before
            finally:
                q.subprocess.check_output = old_gpu
                q.time.monotonic = old_clock
        check('busy assigned GPU causes bounded stop without child start', busy)

        def wait_drift():
            target = phase / 'scripts/study_client.py'
            original = target.read_bytes()
            old_gpu = q.subprocess.check_output
            before = len(calls)

            def drifting_gpu(*args, **kwargs):
                write_bytes(target, original + b'# changed during idle wait\n')
                return '0\n0\n0\n0\n'
            q.subprocess.check_output = drifting_gpu
            try:
                reject(lambda: execute([jobs[5]], 'idle-drift'))
                assert len(calls) == before
            finally:
                write_bytes(target, original)
                q.subprocess.check_output = old_gpu
        check('source drift during GPU idle wait rejected before starting child', wait_drift)

        def signal_during_save():
            original = Path.replace
            trigger = [False]
            depth = [0]
            max_depth = [0]
            before = len(calls)

            def replaced(path, target):
                depth[0] += 1
                max_depth[0] = max(max_depth[0], depth[0])
                try:
                    if not trigger[0] and str(target).endswith('series-signal-reentry.json'):
                        trigger[0] = True
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    return original(path, target)
                finally:
                    depth[0] -= 1
            Path.replace = replaced
            try:
                code, state = execute([jobs[5]], 'signal-reentry')
                assert code == 130 and state['stop_after_current'] and (max_depth[0] == 1) and (len(calls) == before)
            finally:
                Path.replace = original
        check('SIGTERM during atomic save never reenters same temporary writer', signal_during_save)
    return dict(status='PASS' if all((c['status'] == 'PASS' for c in checks)) else 'FAIL', checks=checks, scope='Temporary CPU/mock process and nvidia-smi only; no actual GPU HTTP or model', source_sha256={str(p): q.sha(p.read_bytes()) for p in (Path(__file__), D / 'scripts/run_study_series.py', D / 'scripts/run_logged.py')})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use unique output')
    result = run()
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), failures=[c for c in result['checks'] if c['status'] != 'PASS'])))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
