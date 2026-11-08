#!/usr/bin/env python3
"""Serial bounded D studies; immutable IDs and strict terminal/runtime resume.

Each child owns the shared GPU lock and process-group cleanup. The queue waits
for two idle observations on GPU 0..3, never signals an unrelated GPU owner,
and stops after the current bounded experiment on SIGINT/SIGTERM.
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
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import run_logged as supervisor
ROOT = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()
def load(path):
    return load_document(path)
def utc(): return datetime.now(timezone.utc).isoformat()
def require(ok, message):
    if not ok: raise RuntimeError(message)


def simple(value):
    return isinstance(value,str) and bool(value) and Path(value).name==value and value not in ('.','..')


def command(job):
    require(simple(job.get('id')), 'Invalid experiment ID')
    for key, category in (('config', 'configs'), ('capacity_plan', 'workloads')):
        name = job.get(key)
        require(simple(name), 'Expected plain input filename: ' + key)
        path = path_at(ROOT, category) / name
        require(path.is_file() and path.resolve().parent == path_at(ROOT, category).resolve(), 'Input missing/outside stage: ' + key)
    config = path_at(ROOT, 'configs') / job['config']
    plan = path_at(ROOT, 'workloads') / job['capacity_plan']
    require(load(config).get('name') == job['id'] == load(plan).get('experiment_id'), 'Job/config/plan identities differ')
    timeout = job.get('timeout', 2100)
    require(type(timeout) is int and 0 < timeout <= 7200, 'Bounded timeout must be an integer in 1..7200')
    require(isinstance(job.get('hypothesis'), str) and isinstance(job.get('reference'), str), 'Missing experiment rationale')
    return [sys.executable, str(ROOT / 'scripts/run_logged.py'), '--name', job['id'], '--gpus', '0,1,2,3', '--timeout', str(timeout), '--reference', job['reference'], '--hypothesis', job['hypothesis'], '--', sys.executable, str(ROOT / 'scripts/serve_study.py'), '--config', str(config), '--plan', str(plan)]


def resume_sources(job):
    """Actual runtime closure only; new offline D analyses cannot invalidate it."""
    captured = supervisor.source_inputs(ROOT)
    research = ROOT.parent
    required = {path_at(ROOT, 'configs') / job['config'], path_at(ROOT, 'workloads') / job['capacity_plan']}
    required.update(ROOT / 'scripts' / name for name in (*supervisor.D_RUNTIME, 'run_study_series.py'))
    required.update(research / name for name in ('scripts/benchmark_client.py', 'scenarios/scripts/agent_benchmark.py'))
    required.update(supervisor.quality_fixture_inputs(research))
    inputs = {key: path for key, path in captured.items()
              if path in required or key.startswith(('metadata/', 'project/'))
              or key in ('project_layout.py', 'experiments/catalog.json',
                         'environment.json', 'model-inventory.json')}
    require(all(path.is_file() for path in inputs.values()), 'Runtime dependency missing')
    installed, _ = supervisor.installed_inputs()
    inputs.update(installed)
    return {key: sha(supervisor.stable_read(path)) for key, path in sorted(inputs.items())}


def verify_resume_sources(job,saved):
    current=resume_sources(job)
    prefix = str(path_at(ROOT.parent, 'scenarios/workloads/agent').relative_to(ROOT.parent)) + '/'
    require({k for k in current if k.startswith(prefix)}=={k for k in saved if k.startswith(prefix)},
            'Quality fixture set changed for existing ID: '+job['id'])
    changed=[key for key,value in current.items() if saved.get(key)!=value]
    require(not changed,'Runtime inputs changed for existing ID '+job['id']+': '+', '.join(changed))


def terminal(folder, expected_command=None, job=None):
    """Boundary acceptance only; full request/metric audit remains independent."""
    names=('status.json','command.json','runtime-identity.json','source-snapshot.zip',
           'experiment.json','capacity.json','config.json')
    require(all((folder/name).is_file() for name in names),'Incomplete terminal artifacts: '+folder.name)
    status=load(folder/'status.json');launch=load(folder/'command.json');experiment=load(folder/'experiment.json')
    require(type(status.get('supervisor_exit_code')) is int and type(status.get('exit_code')) is int
            and status.get('supervisor_exit_code')==0 and status.get('exit_code')==0
            and status.get('source_identity_unchanged') is True and status.get('changed_source_members')==[]
            and status.get('timed_out') is False and not any(status.get(k) for k in ('cleanup_error','supervisor_error','interrupted_signal'))
            and status.get('end_utc'),'Failed/uncertain supervisor: '+folder.name)
    require(status.get('experiment_id')==launch.get('experiment_id')==folder.name,'Terminal experiment identity differs')
    for key in ('command','source_sha256','source_snapshot_sha256','runtime_identity_sha256','environment','cwd'):
        require(status.get(key)==launch.get(key),'Launch/terminal mismatch: '+key)
    if expected_command is not None:
        require(launch['command']==expected_command,'Existing command changed: '+folder.name)
    raw=(folder/'runtime-identity.json').read_bytes();identity=json.loads(raw)
    require(sha(raw)==status['runtime_identity_sha256'],'Runtime identity hash mismatch')
    raw=(folder/'source-snapshot.zip').read_bytes()
    require(sha(raw)==status['source_snapshot_sha256'],'Snapshot hash mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist();files=identity['files']
        require(len(names)==len(set(names)) and set(names)==set(files)==set(status['source_sha256']),'Snapshot member inventory differs')
        for key,entry in files.items():
            payload=archive.read(key)
            require(sha(payload)==entry['sha256']==status['source_sha256'][key] and len(payload)==entry['bytes'],'Snapshot bytes differ: '+key)
    require(experiment.get('status')=='PASS' and experiment.get('clients')
            and all(c.get('exit_code')==0 for c in experiment['clients']),'Incomplete study clients')
    require(load(folder/'capacity.json').get('status')=='PASSED','Performance report not complete')
    if 'quality_probe' in load(folder/'config.json'):
        probe=experiment.get('quality_probe',{})
        require(probe.get('status')=='PASS' and probe.get('exit_code')==0,'Quality protocol/execution failed; preserve completed performance separately')
        require((folder/'quality.json').is_file() and (folder/'quality.log').is_file(),'Quality artifacts missing')
        quality=load(folder/'quality.json')
        require(sha((folder/'quality.json').read_bytes())==probe.get('report_sha256')
                and quality.get('status')=='COMPLETED' and quality.get('mode')=='quality'
                and quality.get('summary',{}).get('protocol_failures')==0,'Quality terminal identity/protocol differs')
    if job is not None: verify_resume_sources(job,status['source_sha256'])
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--wait-for')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    jobs = json.loads(raw)['jobs']
    require(isinstance(jobs, list) and 1 <= len(jobs) <= 8, 'Require 1..8 finite registered jobs')
    require(len({j['id'] for j in jobs}) == len(jobs), 'Repeated experiment ID')
    commands = [command(job) for job in jobs]
    if args.wait_for:
        require(simple(args.wait_for), 'Invalid wait target')
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    initial_sources = [resume_sources(job) for job in jobs]
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.state, 'x') as f:
        f.write('{}\n')
    state = dict(status='IN PROGRESS', pid=os.getpid(), plan=str(args.plan.resolve()), plan_sha256=sha(raw), start_utc=utc(), jobs=[])

    def save():
        temporary = args.state.with_suffix(args.state.suffix + '.tmp')
        write_text(temporary, json.dumps(state, indent=2) + '\n')
        temporary.replace(args.state)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        state['stop_after_current'] = True
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, stop)
    save()
    try:
        if args.wait_for:
            source = path_at(ROOT, 'results') / args.wait_for
            require((source / 'command.json').is_file(), 'Wait target has not started')
            deadline = time.monotonic() + 3600
            while not (source / 'status.json').is_file():
                if stopping:
                    state['status'] = 'STOPPED WHILE WAITING'
                    return 130
                if time.monotonic() > deadline:
                    raise TimeoutError('Waiting for preceding experiment')
                time.sleep(5)
            terminal(source)
        for job, cmd, initial in zip(jobs, commands, initial_sources):
            if stopping:
                break
            require(sha(args.plan.read_bytes()) == state['plan_sha256'], 'Series plan changed during execution')
            require(resume_sources(job) == initial, 'Runtime inputs changed while queued: ' + job['id'])
            folder = path_at(ROOT, 'results') / job['id']
            row = dict(id=job['id'], command=cmd)
            state['jobs'].append(row)
            save()
            child = cmd[cmd.index('--') + 1:]
            if folder.exists():
                terminal(folder, child, job)
                row['status'] = 'SKIPPED TERMINAL/COMMAND/RUNTIME VERIFIED; INDEPENDENT DATA AUDIT REQUIRED'
                save()
                continue
            row.update(status='WAITING FOR GPU IDLE', gpu_idle_checks=[])
            save()
            deadline = time.monotonic() + 60
            idle_samples = 0
            while idle_samples < 2:
                used = [int(x) for x in subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()]
                row['gpu_idle_checks'].append(dict(utc=utc(), memory_mib=used))
                idle_samples = idle_samples + 1 if len(used) == 4 and min(used) >= 0 and (max(used) <= 512) else 0
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
            require(sha(args.plan.read_bytes()) == state['plan_sha256'], 'Series plan changed while waiting for GPU idle')
            require(resume_sources(job) == initial, 'Runtime inputs changed while waiting for GPU idle: ' + job['id'])
            row.update(status='RUNNING', started_utc=utc())
            save()
            print('START', job['id'], flush=True)
            rc = subprocess.call(cmd)
            row.update(exit_code=rc, ended_utc=utc(), status='VERIFYING TERMINAL' if rc == 0 else 'FAILED')
            save()
            require(rc == 0, job['id'] + ' failed with exit ' + str(rc))
            terminal(folder, child, job)
            row['status'] = 'COMPLETED'
            save()
        state['status'] = 'STOPPED AFTER CURRENT' if stopping else 'COMPLETED'
        return 130 if stopping else 0
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc))
        raise
    finally:
        state['updated_utc'] = utc()
        save()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

if __name__=='__main__':raise SystemExit(main())
