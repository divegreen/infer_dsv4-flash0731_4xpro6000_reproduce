#!/usr/bin/env python3
"""Bounded offline observer: analyze named experiments only after terminal files.

Never launches inference, queries GPUs, contacts HTTP or edits experiment data.
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

HERE=Path(__file__).resolve().parents[1]
RESULTS = path_at(HERE.parent, 'scenarios/results')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True, type=Path)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--deadline-s', type=int, default=7200)
    p.add_argument('--suffix', required=True)
    args = p.parse_args()
    if len(set(args.ids)) != len(args.ids) or any((Path(x).name != x or x in ('.', '..') for x in args.ids)):
        raise ValueError('Unique simple experiment IDs required')
    if Path(args.suffix).name != args.suffix or args.suffix in ('.', '..'):
        raise ValueError('Simple evidence suffix required')
    if not 1 <= args.deadline_s <= 14400:
        raise ValueError('Bounded observation deadline required')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.state, 'x') as f:
        f.write('{}\n')
    analyzer = HERE / 'scripts/analyze_cache_sessions.py'
    frozen_sha = sha(analyzer)
    state = dict(status='OBSERVING', start_utc=now(), analyzer_sha256=frozen_sha, ids=args.ids, rows=[])

    def save():
        state['updated_utc'] = now()
        tmp = args.state.with_suffix('.tmp')
        write_text(tmp, json.dumps(state, indent=2) + '\n')
        tmp.replace(args.state)
    end = time.monotonic() + args.deadline_s
    save()
    for name in args.ids:
        folder = RESULTS / name
        row = dict(experiment=name, status='WAITING_FOR_TERMINAL')
        state['rows'].append(row)
        save()
        while not (folder / 'status.json').exists():
            if time.monotonic() > end:
                row['status'] = 'DEADLINE_WITHOUT_TERMINAL'
                state['status'] = 'INCOMPLETE'
                save()
                return 2
            time.sleep(10)
        terminal = load_document(folder / 'status.json')
        row.update(supervisor_exit_code=terminal.get('exit_code'), timed_out=terminal.get('timed_out'), terminal_observed_utc=now())
        if sha(analyzer) != frozen_sha:
            raise RuntimeError('Offline analyzer changed after observer start; preserve state and restart with new evidence ID')
        output = path_at(HERE, 'logs') / f'{name}-cache-{args.suffix}.json'
        log = output.with_suffix('.log')
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(analyzer), '--run', str(folder), '--output', str(output)]
        row.update(status='ANALYZING', command=command, output=str(output), log=str(log))
        save()
        with open_output(log, 'x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        row.update(exit_code=result.returncode, status='ANALYSIS_PASS' if result.returncode == 0 else 'ANALYSIS_FAILED', finished_utc=now())
        if result.returncode == 0:
            row.update(output_sha256=sha(output), analysis_status=load_document(output)['status'])
        save()
        print(json.dumps(row), flush=True)
    state['status'] = 'COMPLETED' if all((r['status'] == 'ANALYSIS_PASS' for r in state['rows'])) else 'COMPLETED_WITH_ANALYSIS_FAILURES'
    save()
    return 0 if state['status'] == 'COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
