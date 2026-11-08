#!/usr/bin/env python3
"""CPU-only terminal evidence queue; never starts inference or contacts HTTP.

Only explicit existing experiment IDs are followed. Terminal raw trees are
hashed before and after analysis and copy-only grading. Every output is unique.
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

ROOT=Path(__file__).resolve().parents[1]
RESULTS = path_at(ROOT.parent, 'scenarios/results')
DEFAULT=('session-tp4-cache-screen','session-tp4-noapc-screen','session-tp2pp2-agent-screen','session-dspark3-agent-screen','session-dp4ep-affinity-screen')


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def raw_manifest(directory):
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Unexpected symlink in raw tree')
        if path.is_file():result[str(path.relative_to(directory))]=dict(sha256=sha(path),bytes=path.stat().st_size)
        elif not path.is_dir():raise ValueError('Unexpected special raw file')
    return result


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(path, 'x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def inspect(name, revision):
    directory = RESULTS / name
    status = load_document(directory / 'status.json')
    if not status.get('end_utc') or status.get('exit_code') is None:
        raise ValueError('Nonterminal supervisor')
    prefix = path_at(ROOT, 'summary') / f'{name}-terminal-{revision}'
    grade = path_at(ROOT, 'logs') / f'{name}-objective-regrade-{revision}.json'
    integrity = path_at(ROOT, 'logs') / f'{name}-raw-integrity-{revision}.json'
    command_log = path_at(ROOT, 'logs') / f'{name}-offline-review-{revision}.json'
    if any((path.exists() for path in (prefix.with_suffix('.json'), prefix.with_suffix('.md'), grade, integrity, command_log))):
        raise ValueError('Existing deliverable; do not rerun or overwrite: ' + name)
    before = raw_manifest(directory)
    commands = [[sys.executable, str(ROOT / 'scripts/analyze_sessions.py'), '--result-dirs', str(directory), '--output-prefix', str(prefix)], [sys.executable, str(ROOT / 'scripts/regrade_sessions.py'), '--result-dir', str(directory), '--output', str(grade)]]
    operations = []
    for command in commands:
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        operations.append(dict(command=command, exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, wall_s=time.perf_counter() - started))
        if completed.returncode:
            break
    after = raw_manifest(directory)
    unchanged = before == after
    save_new(integrity, dict(status='PASS' if unchanged else 'FAIL', utc=now(), experiment=name, original_raw_files=len(before), before_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(), after_sha256=hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(), files=before, scope='Whole terminal raw result tree read-only check around offline analysis and copy-only grading.'))
    success = unchanged and len(operations) == 2 and all((op['exit_code'] == 0 for op in operations))
    result = dict(status='PASS' if success else 'FAIL', utc=now(), experiment=name, supervisor_exit_code=status['exit_code'], operations=operations, raw_integrity=str(integrity), summary=str(prefix.with_suffix('.json')), regrade=str(grade))
    if success:
        summary = load_document(prefix.with_suffix('.json'))
        regraded = load_document(grade)
        result['task_classifications'] = regraded['sessions']
        phases = []
        for stage in summary['stages']:
            stats = stage['summary']
            gpu = stage['gpu']
            gauges = stage['gauges']
            phases.append(dict(id=stage['phase']['id'], role=stage['phase']['role'], kind=stage['phase']['kind'], sessions=stats['sessions'], protocol_completed=stats['protocol_completed'], successful_tasks=stats['successful_tasks'], completed_sessions_s=stats['completed_sessions_s'], successful_tasks_s=stats['successful_tasks_s'], visible_p95_s=stats['user_visible_latency_s']['p95'], cache_fraction=stats['cache_fraction'], model_calls=stats['model_calls'], tool_calls=stats['tool_calls'], terminal_reasons=stats['terminal_reasons'], sampled_gpu_status=gpu['status'], total_mean_board_w=gpu['energy_j'] / gpu['window_s'] if gpu.get('energy_j') is not None else None, kv_max=gauges.get('vllm:kv_cache_usage_perc', {}).get('max'), preemptions=stage['metrics'].get('totals', {}).get('vllm:num_preemptions_total')))
        result['phases'] = phases
    save_new(command_log, result)
    return {key: result[key] for key in result if key not in ('operations', 'task_classifications')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=DEFAULT)
    parser.add_argument('--revision', default='r1')
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    if args.journal.exists():
        raise ValueError('Use a new journal path')
    if any((Path(name).name != name or not name.startswith('c') for name in args.runs)):
        raise ValueError('Explicit simple Stage C IDs required')
    if Path(args.revision).name != args.revision:
        raise ValueError('Simple revision ID required')
    pending = list(args.runs)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.journal, 'x') as journal:
        while pending:
            ready = []
            for name in pending:
                path = RESULTS / name / 'status.json'
                if path.exists():
                    value = load_document(path)
                    if value.get('end_utc') and value.get('exit_code') is not None:
                        ready.append(name)
            for name in ready:
                try:
                    result = inspect(name, args.revision)
                except Exception as exc:
                    result = dict(status='FAIL', experiment=name, utc=now(), error=repr(exc))
                journal.write(json.dumps(result, ensure_ascii=False) + '\n')
                journal.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                pending.remove(name)
            if not pending or not args.follow:
                break
            time.sleep(30)
    print(json.dumps(dict(status='WATCH_COMPLETE' if not pending else 'PENDING', pending=pending, journal=str(args.journal))), flush=True)


if __name__=='__main__':main()
