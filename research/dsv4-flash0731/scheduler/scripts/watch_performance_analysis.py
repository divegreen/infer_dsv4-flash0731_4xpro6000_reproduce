#!/usr/bin/env python3
"""Bounded CPU-only terminal observer for the registered D normal sequence."""

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
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
IDS=('scheduler-tp4-observe-matched-670','scheduler-tp4-fairshare-matched-670',
     'scheduler-tp4-fairshare-matched-671','scheduler-tp4-observe-matched-671')

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(path):
    try:
        return load_document(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-wait-s', type=float, default=7200)
    args = parser.parse_args()
    if not args.revision.isalnum() or not 60 <= args.max_wait_s <= 10800:
        raise ValueError('Bounded revision/deadline required')
    state_path = ROOT / f'logs/normal-analysis-observer-{args.revision}-state.json'
    journal_path = ROOT / f'logs/normal-analysis-observer-{args.revision}.jsonl'
    if state_path.exists() or journal_path.exists():
        raise FileExistsError('Use new observer revision')
    analyzer = ROOT / 'scripts/analyze_performance.py'
    source = sha(analyzer)
    state = dict(status='RUNNING', started_utc=utc(), pid=os.getpid(), python=sys.executable, analyzer_sha256=source, observer_sha256=sha(Path(__file__)), ordered_experiments=IDS, completed=[], active=None)
    start = time.monotonic()

    def save(event):
        state['updated_utc'] = utc()
        temp = state_path.with_suffix('.tmp')
        write_text(temp, json.dumps(state, indent=2) + '\n')
        temp.replace(state_path)
        with open_output(journal_path, 'a') as f:
            f.write(json.dumps(dict(utc=utc(), **event)) + '\n')
    save(dict(event='started', source=source))
    try:
        for index, name in enumerate(IDS):
            directory = path_at(ROOT, 'results') / name
            status_path = directory / 'status.json'
            while True:
                status = read(status_path)
                if status and status.get('end_utc'):
                    break
                queue = read(path_at(ROOT, 'logs/series-matched-performance-r1-state.json')) or {}
                if queue.get('status') in ('FAILED', 'CANCELLED', 'STOPPED', 'COMPLETED'):
                    state.update(status='STOPPED_WITH_UNSTARTED', unstarted=list(IDS[index:]), queue_status=queue['status'])
                    save(dict(event='queue_terminal_before_remaining_results'))
                    print(json.dumps(state))
                    return 0
                if time.monotonic() - start > args.max_wait_s:
                    raise TimeoutError('Bounded terminal-observer wait expired')
                time.sleep(15)
            if sha(analyzer) != source:
                raise RuntimeError('Reviewed analyzer changed; stop instead of mixing revisions')
            prefix = ROOT / f'summary/normal-through-{name.split('-')[0]}-{args.revision}'
            command = [sys.executable, '-B', str(analyzer), '--result-dirs', *[str(path_at(ROOT, 'results') / v) for v in IDS[:index + 1]], '--output-prefix', str(prefix)]
            log = ROOT / f'logs/normal-through-{name.split('-')[0]}-{args.revision}.log'
            if log.exists() or prefix.with_suffix('.json').exists() or prefix.with_suffix('.md').exists():
                raise FileExistsError('Refusing to overwrite analysis evidence')
            state['active'] = dict(experiment=name, command=command, log=str(log), started_utc=utc())
            save(dict(event='analysis_started', **state['active']))
            began = time.monotonic()
            with open_output(log, 'x') as f:
                result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            state['active'].update(exit_code=result.returncode, elapsed_s=time.monotonic() - began, finished_utc=utc(), log_sha256=sha(log), output_prefix=str(prefix))
            save(dict(event='analysis_finished', **state['active']))
            if result.returncode:
                raise RuntimeError('Analysis failed; preserve diagnostics and stop')
            state['completed'].append(state['active'])
            state['active'] = None
            save(dict(event='checkpoint', completed=len(state['completed'])))
        state['status'] = 'COMPLETED'
        save(dict(event='completed'))
        print(json.dumps(state))
        return 0
    except BaseException as exc:
        state.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc))
        save(dict(event='failure', error=state['error']))
        print(json.dumps(state))
        raise
if __name__=='__main__':raise SystemExit(main())
