"""Serial, resumable finite experiment plan; each child keeps its own supervisor.

Never repeats an existing failed/uncertain ID. On SIGTERM stop after the current
bounded child, preserving its logs and cleanup. No detached GPU work is created.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
_supervisor_spec = importlib.util.spec_from_file_location('session_series_supervisor', ROOT / 'scripts/run_logged.py')
supervisor = importlib.util.module_from_spec(_supervisor_spec)
_supervisor_spec.loader.exec_module(supervisor)


def resume_sources(job):
    """Exact local runtime inputs required before skipping an existing result.

    run_logged snapshots all scenario scripts/workloads and the shared benchmark
    helper. Check the actual transitive dependencies here, including the entire
    task fixture set; a deleted fixture is a change just as an edited one is.
    """
    captured = supervisor.source_inputs(ROOT)
    paths = [path_at(ROOT, 'configs') / job['config'], path_at(ROOT, 'workloads') / job['session_plan'], ROOT.parent / 'scripts/benchmark_client.py']
    paths += [ROOT / 'scripts' / name for name in ('run_agent_series.py', 'run_logged.py', 'serve_agent_sessions.py', 'agent_session_client.py', 'agent_session_tasks.py', 'capacity_client.py')]
    fixture_root = path_at(ROOT, 'workloads/sessions/tasks')
    if not (fixture_root / 'catalog.json').is_file():
        raise ValueError('Task fixture catalog missing; existing result cannot be verified')
    paths += [p for p in fixture_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    if any((not p.is_file() for p in paths)):
        raise ValueError('Required runtime input missing; existing result cannot be verified')
    required = set(paths)
    # Keep resume checks scoped to execution inputs while reusing the exact
    # snapshot names for project helpers and current environment metadata.
    return {key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in captured.items()
            if path in required or key.startswith(('metadata/', 'project/'))
            or key in ('project_layout.py', 'experiments/catalog.json',
                       'environment.json', 'model-inventory.json')}


def verify_resume_sources(job, saved):
    required = resume_sources(job)
    prefix = str(path_at(ROOT, 'workloads/sessions/tasks').relative_to(ROOT.parent)) + '/'
    saved_fixtures = {name for name in saved if name.startswith(prefix)}
    current_fixtures = {name for name in required if name.startswith(prefix)}
    if saved_fixtures != current_fixtures:
        raise RuntimeError(f'Task fixture set changed for existing ID: {job['id']}')
    changed = [name for name, digest in required.items() if saved.get(name) != digest]
    if changed:
        raise RuntimeError(f'Runtime input changed for existing ID {job['id']}: ' + ', '.join(changed))


def command(job):
    config = path_at(ROOT, 'configs') / job['config']
    if config.parent != path_at(ROOT, 'configs') or not config.is_file():
        raise ValueError('Config must be an existing file in configs')
    name = job['id']
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Invalid experiment ID')
    cmd = [sys.executable, str(ROOT / 'scripts/run_logged.py'), '--name', name, '--timeout', str(job.get('timeout', 6000)), '--reference', job['reference'], '--hypothesis', job['hypothesis'], '--', sys.executable, str(ROOT / 'scripts/serve_agent_sessions.py'), '--config', str(config), '--plan', str(path_at(ROOT, 'workloads') / job['session_plan'])]
    plan = (path_at(ROOT, 'workloads') / job['session_plan']).resolve()
    if not plan.is_relative_to(path_at(ROOT, 'workloads').resolve()) or not plan.is_file():
        raise ValueError('Session plan must exist under workloads')
    if load_document(plan).get('experiment_id') != name:
        raise ValueError('Plan and job experiment IDs differ')
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--wait-for')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = load_document(args.plan)['jobs']
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Repeated experiment ID')
    commands = [command(j) for j in jobs]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.state, 'x') as f:
        f.write('{}\n')
    state = dict(status='IN PROGRESS', pid=os.getpid(), plan=str(args.plan.resolve()), plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(), start_utc=datetime.now(timezone.utc).isoformat(), jobs=[])

    def save():
        temp = args.state.with_suffix('.tmp')
        write_text(temp, json.dumps(state, indent=2) + '\n')
        temp.replace(args.state)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        state['stop_after_current'] = True
        save()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        if args.wait_for:
            source = path_at(ROOT, 'results') / args.wait_for
            if not (source / 'command.json').is_file():
                raise RuntimeError('Wait target has not started')
            deadline = time.monotonic() + 3600
            while not (source / 'status.json').is_file():
                if stopping:
                    state['status'] = 'STOPPED WHILE WAITING'
                    return 130
                if time.monotonic() > deadline:
                    raise TimeoutError('Waiting for preceding experiment')
                time.sleep(5)
            previous = load_document(source / 'status.json')
            if previous.get('exit_code') != 0 or previous.get('timed_out'):
                raise RuntimeError('Preceding experiment failed; inspect before continuing')
        for job, cmd in zip(jobs, commands):
            if stopping:
                break
            folder = path_at(ROOT, 'results') / job['id']
            row = dict(id=job['id'], command=cmd)
            state['jobs'].append(row)
            if folder.exists():
                status = load_document(folder / 'status.json') if (folder / 'status.json').exists() else {}
                if status.get('exit_code') != 0 or status.get('timed_out') or status.get('command') != cmd[cmd.index('--') + 1:]:
                    raise RuntimeError(f'Existing failed, changed or uncertain ID: {job['id']}')
                saved = status.get('source_sha256', {})
                verify_resume_sources(job, saved)
                result = load_document(folder / 'experiment.json')
                if result.get('status') != 'PASS' or any((c.get('exit_code') != 0 for c in result.get('clients', []))):
                    raise RuntimeError(f'Incomplete result for existing ID: {job['id']}')
                row['status'] = 'SKIPPED EXIT/CONFIG/CLIENT VERIFIED; SEE INDEPENDENT DATA AUDIT'
                save()
                continue
            row.update(status='RUNNING', started_utc=datetime.now(timezone.utc).isoformat())
            save()
            deadline = time.monotonic() + 60
            idle_samples = 0
            row['gpu_idle_checks'] = []
            while idle_samples < 2:
                used = [int(x) for x in subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()]
                row['gpu_idle_checks'].append(dict(utc=datetime.now(timezone.utc).isoformat(), memory_mib=used))
                idle_samples = idle_samples + 1 if len(used) == 4 and max(used) <= 512 else 0
                save()
                if time.monotonic() > deadline:
                    raise TimeoutError('GPUs did not become idle; no child started')
                if stopping:
                    row['status'] = 'NOT STARTED: STOP REQUESTED'
                    break
                if idle_samples < 2:
                    time.sleep(2)
            if stopping:
                break
            print('START', job['id'], flush=True)
            rc = subprocess.call(cmd)
            row.update(exit_code=rc, status='COMPLETED' if rc == 0 else 'FAILED', ended_utc=datetime.now(timezone.utc).isoformat())
            save()
            if rc:
                raise RuntimeError(f'{job['id']} failed with exit {rc}')
        state['status'] = 'STOPPED AFTER CURRENT' if stopping else 'COMPLETED'
        return 130 if stopping else 0
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc))
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    raise SystemExit(main())
