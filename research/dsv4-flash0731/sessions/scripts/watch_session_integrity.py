#!/usr/bin/env python3
"""Bounded CPU-only terminal integrity/reference observer for explicit IDs.

No model/HTTP/GPU operations. Never retries or overwrites an existing audit.
The production performance analyzer and objective grader are separate observers.
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
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];RESULTS = path_at(ROOT.parent, 'scenarios/results')


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--max-seconds', type=int, default=18000)
    a = p.parse_args()
    if len(set(a.runs)) != len(a.runs) or not all((re.fullmatch('c[0-9]+-[A-Za-z0-9._-]+', n) for n in a.runs)):
        raise ValueError('Explicit unique Stage C IDs required')
    if not re.fullmatch('r[0-9]+', a.revision) or not 1 <= a.max_seconds <= 86400:
        raise ValueError('Bounded observer and simple revision required')
    sources = {k: ROOT / 'scripts' / v for k, v in [('integrity', 'audit_agent_sessions.py'), ('json_reference', 'score_json_reference.py')]}
    hashes = {k: sha(v) for k, v in sources.items()}
    jobs = {}
    for name in a.runs:
        jobs[name] = {}
        for kind, script in sources.items():
            output = path_at(ROOT, 'logs') / (f'audit-{name}-{a.revision}.json' if kind == 'integrity' else f'{name}-json-reference-{a.revision}.json')
            console = path_at(ROOT, 'logs') / f'{name}-{kind}-console-{a.revision}.log'
            if any((x.exists() or x.is_symlink() for x in (output, console))):
                raise FileExistsError('Existing review output for ' + name + '/' + kind)
            cmd = [sys.executable, str(script), '--run', name, '--output', str(output)] if kind == 'integrity' else [sys.executable, str(script), '--result-dir', str(RESULTS / name), '--output', str(output)]
            jobs[name][kind] = dict(command=cmd, output=output, console=console)
    state = dict(status='WAITING', scope='CPU-only terminal integrity/reference observer; no GPU/model/HTTP calls', started_utc=utc(), source_sha256=hashes, runs={name: dict(status='WAITING_FOR_TERMINAL') for name in a.runs})
    a.state.parent.mkdir(parents=True, exist_ok=True)
    with open_output(a.state, 'x') as f:
        json.dump(state, f, indent=2)
        f.write('\n')

    def save():
        tmp = a.state.with_suffix('.json.tmp')
        with open_output(tmp, 'x') as f:
            json.dump(state, f, indent=2)
            f.write('\n')
        tmp.replace(a.state)
    pending = list(a.runs)
    deadline = time.monotonic() + a.max_seconds
    while pending and time.monotonic() < deadline:
        for name in pending.copy():
            path = RESULTS / name / 'status.json'
            if not path.is_file():
                continue
            try:
                terminal = load_document(path)
            except json.JSONDecodeError:
                continue
            if not terminal.get('end_utc'):
                continue
            row = state['runs'][name]
            row.update(status='AUDITING', supervisor_exit_code=terminal.get('exit_code'), observed_terminal_utc=utc())
            save()
            try:
                for kind, job in jobs[name].items():
                    if sha(sources[kind]) != hashes[kind]:
                        raise ValueError('Review source changed since observer launch: ' + kind)
                    with open_output(job['console'], 'x') as log:
                        done = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, timeout=900)
                    result = load_document(job['output']) if job['output'].is_file() else {}
                    row[kind] = dict(exit_code=done.returncode, result_status=result.get('status'), command=job['command'], output=str(job['output']))
                    save()
                row['status'] = 'COMPLETED' if all((row[k]['exit_code'] == 0 for k in sources)) else 'REVIEW_REQUIRED'
            except Exception as exc:
                row.update(status='REVIEW_REQUIRED', error=repr(exc))
            pending.remove(name)
            save()
            print(json.dumps(dict(experiment=name, **row)), flush=True)
        if pending:
            time.sleep(10)
    state.update(status='WAIT_TIMEOUT' if pending else 'REVIEW_REQUIRED' if any((r['status'] == 'REVIEW_REQUIRED' for r in state['runs'].values())) else 'COMPLETED', finished_utc=utc(), pending=pending)
    save()
    print(json.dumps(dict(status=state['status'], pending=pending, state=str(a.state))), flush=True)


if __name__=='__main__':main()
