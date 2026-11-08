#!/usr/bin/env python3
"""Bounded read-only terminal audit watcher for the already running D queue.

No service/GPU commands. It starts only offline evidence auditors once an owned
supervisor's atomic terminal status exists. It never retries an experiment.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def load(path):
    return load_document(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', type=Path, required=True)
    parser.add_argument('--queue-state', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-wait', type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.max_wait <= 14400:
        raise ValueError('max-wait must be 1..14400 seconds')
    if not args.revision.isalnum():
        raise ValueError('Simple audit revision required')
    ids = [j['id'] for j in load(args.series)['jobs']]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated job ID')
    for name in ids:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid job ID')
    source_paths = [Path(__file__), D / 'scripts/audit_study.py', D / 'scripts/audit_quality.py', D.parent / 'scenarios/scripts/capacity_audit.py', args.series]
    source_hashes = {str(p): sha(p) for p in source_paths}
    state = dict(status='IN PROGRESS', started_utc=utc(), experiment_ids=ids, source_sha256=source_hashes, audits=[])
    with open_output(args.state, 'x') as f:
        json.dump(state, f, indent=2)
        f.write('\n')

    def save():
        temp = args.state.with_suffix('.tmp')
        state['updated_utc'] = utc()
        write_text(temp, json.dumps(state, indent=2) + '\n')
        temp.replace(args.state)
    complete = set()
    deadline = time.monotonic() + args.max_wait
    try:
        while time.monotonic() < deadline:
            for name in ids:
                if name in complete:
                    continue
                directory = path_at(D, 'results') / name
                status_path = directory / 'status.json'
                if not status_path.is_file():
                    continue
                if {str(p): sha(p) for p in source_paths} != source_hashes:
                    raise RuntimeError('Audit source/plan changed; inspect before new independent revision')
                status = load(status_path)
                if not status.get('end_utc') or type(status.get('supervisor_exit_code')) is not int:
                    raise RuntimeError('Malformed terminal status ' + name)
                output = path_at(D, 'logs') / ('audit-' + name + '-' + args.revision + '.json')
                log = output.with_suffix('.log')
                if output.exists() or log.exists():
                    raise FileExistsError('Unique audit revision already exists: ' + str(output))
                command = [sys.executable, '-B', str(D / 'scripts/audit_study.py'), '--input', str(directory / 'capacity.json'), '--output', str(output)]
                row = dict(experiment=name, status='AUDITING', started_utc=utc(), command=command, output=str(output), log=str(log), supervisor_exit_code=status['supervisor_exit_code'])
                state['audits'].append(row)
                save()
                with open_output(log, 'x') as stream:
                    run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
                row.update(exit_code=run.returncode, status='FINISHED', finished_utc=utc())
                if output.is_file():
                    audit = load(output)
                    row.update(audit_status=audit.get('status'), runtime_reports=audit.get('runtime_reports'), errors=audit.get('errors'), incomplete=audit.get('incomplete'))
                complete.add(name)
                save()
                print(json.dumps(row, ensure_ascii=False), flush=True)
            queue = load(args.queue_state) if args.queue_state.is_file() else {}
            if len(complete) == len(ids):
                state['status'] = 'COMPLETED'
                save()
                return 0 if all((x.get('audit_status') == 'PASS' for x in state['audits'])) else 1
            if queue.get('status') in ('FAILED', 'STOPPED AFTER CURRENT', 'STOPPED WHILE WAITING', 'COMPLETED'):
                state.update(status='QUEUE TERMINAL', queue_status=queue['status'], unaudited_ids=[x for x in ids if x not in complete])
                save()
                return 1
            time.sleep(10)
        state.update(status='WAIT LIMIT', unaudited_ids=[x for x in ids if x not in complete])
        save()
        return 2
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc))
        save()
        raise

if __name__=='__main__':raise SystemExit(main())
