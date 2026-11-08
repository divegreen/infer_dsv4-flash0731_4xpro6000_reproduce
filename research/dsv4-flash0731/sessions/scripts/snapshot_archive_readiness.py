#!/usr/bin/env python3
"""Record Stage C archive prerequisites without packaging or touching a GPU.

This inventory is deliberately not an archive approval: running experiments,
observers and report revisions require a fresh final check by the coordinator.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PHASE = Path(__file__).resolve().parents[1]
RESEARCH = PHASE.parent
SCENARIO = RESEARCH / 'scenarios'


def snapshot():
    ledger = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw

    def obj(path):
        return json.loads(read(path))
    packager = PHASE / 'scripts/package_sessions.py'
    constants = {}
    for node in ast.parse(read(packager)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('SHARED', 'CROSS_STAGE_REQUIRED', 'OFFLINE_REFERENCE_REQUIRED'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    dependencies = []
    for key, base in [('SHARED', SCENARIO / 'scripts'), ('CROSS_STAGE_REQUIRED', RESEARCH), ('OFFLINE_REFERENCE_REQUIRED', SCENARIO)]:
        for name in constants[key]:
            path = base / name
            read(path)
            dependencies.append(dict(group=key, path=str(path), **ledger[str(path)]))
    plans = {p.stem: p for p in path_at(SCENARIO, 'workloads/sessions').glob('c*.json') if re.fullmatch('c\\d+[A-Za-z0-9._-]*', p.stem)}
    actual = {p.name: p for p in path_at(SCENARIO, 'results').iterdir() if re.fullmatch('c\\d+[A-Za-z0-9._-]*', p.name)}
    runs = []
    for name in sorted(set(plans) | set(actual)):
        row = dict(id=name, planned=name in plans, result_directory_exists=name in actual)
        if name in plans:
            plan = obj(plans[name])
            row['planned_roles'] = sorted({p.get('role', 'measure') for p in plan['phases']})
        folder = path_at(SCENARIO, 'results') / name
        command_path = folder / 'command.json'
        if command_path.is_file():
            command = obj(command_path)
            row.update(git_commit=command.get('git_commit'), command_id=command.get('experiment_id'))
        status_path = folder / 'status.json'
        if status_path.is_file():
            status = obj(status_path)
            terminal = bool(status.get('end_utc')) and (type(status.get('exit_code')) is int or status.get('timed_out') is True or bool(status.get('supervisor_error')))
            row.update(terminal=terminal, exit_code=status.get('exit_code'), timed_out=status.get('timed_out'), supervisor_error=status.get('supervisor_error'), end_utc=status.get('end_utc'), state='TERMINAL_OUTCOME_RETAIN' if terminal else 'NONTERMINAL')
        else:
            row.update(terminal=False, state='STARTED_OR_PARTIAL' if name in actual else 'NOT_STARTED')
        runs.append(row)
    states = []
    for name in ('series-open-screen-r1-state.json', 'independent-open-audit-watch-r1.json', 'cache-observer-open-screen-r1-state.json'):
        path = path_at(PHASE, 'logs') / name
        if path.exists():
            value = obj(path)
            states.append(dict(path=str(path), status=value.get('status'), pid=value.get('pid'), dynamic_snapshot=True))
    for name in ('package-c-selftest-r4.json', 'independent-regrade-applicability-r1.json'):
        value = obj(path_at(PHASE, 'logs') / name)
        if value.get('status') != 'PASS':
            raise ValueError('Prerequisite CPU check is not PASS: ' + name)
    read(Path(__file__))
    return dict(status='INVENTORY_ONLY_NOT_ARCHIVE_APPROVAL', created_utc=datetime.now(timezone.utc).isoformat(), scope='Read-only CPU metadata and dependency inventory; no package, GPU, HTTP or task execution.', experiments=runs, dependencies=dependencies, dynamic_observers=states, not_terminal=[r['id'] for r in runs if not r['terminal']], terminal_failed_ids=[r['id'] for r in runs if r['terminal'] and (r.get('exit_code') != 0 or r.get('timed_out') or r.get('supervisor_error'))], archive_created=False, evidence_ledger=ledger, limitations=['Snapshot is stale as soon as a live experiment or observer changes; refresh before final packaging.', 'No full raw-tree verification is repeated; terminal audits and package verification retain that responsibility.', 'All experiment IDs, including failed pilots and later confirmation runs, require explicit final selection.', 'Final report/Notebook/figures must be frozen and all writers stopped before taking the shared GPU lock.', 'Only manually identified local offline/runtime dependencies are inventoried; model, tokenizer and installed libraries remain external.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = snapshot()
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'not_terminal', 'terminal_failed_ids', 'archive_created')}))


if __name__ == '__main__':
    main()
