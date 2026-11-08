#!/usr/bin/env python3
"""Bounded, queue-aware CPU observer for explicitly named terminal experiments.

Never queries a GPU or HTTP, launches inference, or changes runtime/plans.
Cache/resource/runtime analyses write unique evidence paths. Queue failure or
completion without a missing experiment's terminal file stops that wait.
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
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
RESULTS = path_at(HERE.parent, 'scenarios/results')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stopped_queue(queue, name):
    """Return stop evidence, only while the experiment has no terminal file."""
    if queue.get('status') not in ('FAILED', 'COMPLETED', 'COMPLETED_WITH_FAILURES'):
        return None
    row = next((j for j in queue.get('jobs', []) if j.get('id') == name), None)
    return dict(queue_status=queue['status'], experiment=name,
                queue_job_status=None if row is None else row.get('status'),
                reason='Queue terminal without experiment terminal file; no further unbounded wait.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--queue-state', required=True, type=Path)
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--deadline-s', type=int, default=7200)
    args = parser.parse_args()
    if len(set(args.ids)) != len(args.ids) or any((Path(n).name != n or n in ('.', '..') for n in args.ids)):
        raise ValueError('Unique simple experiment IDs required')
    if Path(args.suffix).name != args.suffix or args.suffix in ('.', '..'):
        raise ValueError('Simple unique suffix required')
    if not 1 <= args.deadline_s <= 14400:
        raise ValueError('Observation must be bounded to at most four hours')
    dependencies = [Path(__file__), HERE / 'scripts/analyze_cache_sessions.py', HERE / 'scripts/review_session_runtime.py', HERE / 'scripts/summarize_cache_resources.py', HERE.parent / 'scenarios/scripts/summarize_scenarios.py', HERE.parent / 'long-context/scripts/review_runtime.py']
    frozen = {str(p.resolve()): sha(p) for p in dependencies}
    state = dict(status='OBSERVING', start_utc=now(), ids=args.ids, suffix=args.suffix, deadline_s=args.deadline_s, queue_state=str(args.queue_state.resolve()), offline_source_sha256=frozen, rows=[], scope='CPU local-file observer only; no GPU/HTTP/runtime/plan actions')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.state, 'x') as handle:
        json.dump(state, handle, indent=2)
        handle.write('\n')

    def save():
        state['updated_utc'] = now()
        temp = args.state.with_suffix('.tmp')
        write_text(temp, json.dumps(state, indent=2) + '\n')
        temp.replace(args.state)

    def run_analysis(row, task, script, flags):
        if any((sha(Path(p)) != value for p, value in frozen.items())):
            raise RuntimeError('Offline dependency changed; preserve state and use a new observer ID')
        output = path_at(HERE, 'logs') / f'{row['experiment']}-{task}-{args.suffix}.json'
        log = output.with_suffix('.log')
        if output.exists() or log.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(HERE / 'scripts' / script), *flags, '--output', str(output)]
        op = dict(task=task, status='RUNNING', command=command, output=str(output), log=str(log), start_utc=now())
        row['operations'].append(op)
        save()
        try:
            with open_output(log, 'x') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            op.update(exit_code=completed.returncode, status='PASS' if completed.returncode == 0 else 'FAILED', end_utc=now())
            if completed.returncode == 0:
                op.update(output_sha256=sha(output), analysis_status=load_document(output)['status'])
        except Exception as exc:
            op.update(status='FAILED', error=repr(exc), end_utc=now())
        save()
        print(json.dumps(op), flush=True)
        return output if op['status'] == 'PASS' else None
    deadline = time.monotonic() + args.deadline_s
    print(json.dumps(dict(status='OBSERVING', state=str(args.state), ids=args.ids)), flush=True)
    try:
        for name in args.ids:
            folder = RESULTS / name
            row = dict(experiment=name, status='WAITING_FOR_TERMINAL', operations=[])
            state['rows'].append(row)
            save()
            while not (folder / 'status.json').exists():
                if args.queue_state.exists():
                    queue = load_document(args.queue_state)
                    stop = stopped_queue(queue, name)
                    if stop:
                        row.update(status='QUEUE_STOPPED_WITHOUT_TERMINAL', queue_stop=stop)
                        state.update(status='BLOCKED_QUEUE_TERMINAL', end_utc=now(), unresolved_ids=args.ids[args.ids.index(name):])
                        save()
                        print(json.dumps(state), flush=True)
                        return 2
                if time.monotonic() >= deadline:
                    row['status'] = 'DEADLINE_WITHOUT_TERMINAL'
                    state.update(status='INCOMPLETE_DEADLINE', end_utc=now())
                    save()
                    return 2
                time.sleep(10)
            terminal = load_document(folder / 'status.json')
            row.update(status='ANALYZING', observed_terminal_utc=now(), supervisor_exit_code=terminal.get('exit_code'), timed_out=terminal.get('timed_out'))
            save()
            cache = run_analysis(row, 'cache', 'analyze_cache_sessions.py', ['--run', str(folder)])
            healthy = type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0 and (terminal.get('timed_out') is False) and (not terminal.get('supervisor_error')) and bool(terminal.get('end_utc'))
            if healthy and cache:
                run_analysis(row, 'runtime', 'review_session_runtime.py', ['--runs', name])
                run_analysis(row, 'resources', 'summarize_cache_resources.py', ['--inputs', str(cache)])
            else:
                row['healthy_only_analyses_skipped'] = True
            row.update(status='ANALYSIS_PASS' if healthy and len(row['operations']) == 3 and all((o['status'] == 'PASS' for o in row['operations'])) else 'ANALYSIS_REVIEW_REQUIRED', completed_utc=now())
            save()
        state.update(status='COMPLETED' if all((r['status'] == 'ANALYSIS_PASS' for r in state['rows'])) else 'COMPLETED_WITH_REVIEW_REQUIRED', end_utc=now())
        save()
        return 0 if state['status'] == 'COMPLETED' else 1
    except BaseException as exc:
        state.update(status='OBSERVER_FAILED', error=repr(exc), end_utc=now())
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
