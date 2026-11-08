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
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def command(job):
    config = path_at(ROOT, 'configs') / job['config']
    if config.parent != path_at(ROOT, 'configs') or not config.is_file():
        raise ValueError('Config must be an existing file in configs')
    name = job['id']
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Invalid experiment ID')
    cmd = [sys.executable, str(ROOT / 'scripts/run_logged.py'), '--name', name, '--timeout', str(job.get('timeout', 2400)), '--reference', job['reference'], '--hypothesis', job['hypothesis'], '--', sys.executable, str(ROOT / 'scripts/serve_scenarios.py'), '--config', str(config), '--suites', job.get('suites', ''), '--rounds', str(job.get('rounds', 2))]
    if job.get('agent_modes'):
        cmd += ['--agent-modes', job['agent_modes']]
    if job.get('profile'):
        cmd += ['--profile']
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
                required = [str((path_at(ROOT, 'configs') / job['config']).relative_to(ROOT.parent)), str((ROOT / 'scripts/serve_scenarios.py').relative_to(ROOT.parent))]
                required += [str((ROOT / 'scripts' / n).relative_to(ROOT.parent)) for n in (['scenario_client.py'] if job.get('suites') else []) + (['agent_benchmark.py'] if job.get('agent_modes') else [])]
                if any((saved.get(name) != hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() for name in required)):
                    raise RuntimeError(f'Config/client changed for existing ID: {job['id']}')
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
