#!/usr/bin/env python3
"""Repeat terminal objective grades on copies; never write original workspaces.

This checks repeatability of the same objective grader, not an independently
designed quality rubric. Non-natural/budget/protocol failures are not converted
to semantic failures by grading a nonexistent final answer.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
TASK_SOURCE=ROOT.parent/'scenarios/scripts/agent_session_tasks.py'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(path):
    result={}
    for member in sorted(path.rglob('*')):
        if member.is_symlink():raise ValueError('Reject symlink in task workspace')
        if member.is_file():result[str(member.relative_to(path))]=sha(member)
        elif not member.is_dir():raise ValueError('Reject special task workspace file')
    return result


def failure_class(row,repeated):
    if row['terminal_reason']!='natural_stop':return row['terminal_reason']
    if repeated['success']:return 'OBJECTIVE_TASK_SUCCESS'
    details=repeated.get('details') or {}
    if row['task_id'] in ('reservation_book','invoice_adjustments'):
        return 'CODE_FUNCTIONAL_FAILURE' if details.get('executed') else 'CODE_TEST_EXECUTION_FAILURE'
    if details.get('error'):return 'FINAL_FORMAT_OR_SCHEMA_FAILURE'
    checks=details.get('checks',{})
    if any(value.get('value_correct') and value.get('evidence_supported') is False for value in checks.values()):return 'ANSWER_OR_CITATION_FAILURE'
    return 'ANSWER_VALUE_OR_METHOD_FAILURE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = args.result_dir.resolve()
    output = args.output.resolve()
    if output.exists() or d in output.parents:
        raise ValueError('Use a new output outside immutable original results')
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)
    status = read(d / 'status.json')
    report = read(d / 'sessions.json')
    command = read(d / 'command.json')
    if status.get('exit_code') is None or not status.get('end_utc'):
        raise ValueError('Experiment is not terminal')
    has_agent = any((phase['phase']['kind'] == 'agent' for phase in report['phases']))
    expected = command['source_sha256'].get('scenarios/scripts/agent_session_tasks.py')
    if has_agent:
        if not expected or sha(TASK_SOURCE) != expected or report['helper_sha256'].get('agent_session_tasks') != expected:
            raise ValueError('Current task grader differs from executed source; reproduce matching source before grading')
        spec = importlib.util.spec_from_file_location('frozen_objective_task_grader', TASK_SOURCE)
        tasks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tasks)
    elif any((row['kind'] != 'replay' or row.get('grade') is not None or row.get('task_success') is not None for phase in report['phases'] for row in phase['sessions'])):
        raise ValueError('Pure replay run must not claim an objective task grade')
    checked = []
    source_hashes = {}
    source_workspace_hashes = {}
    with tempfile.TemporaryDirectory(prefix='phasec-regrade-') as temporary:
        for phase in report['phases']:
            if phase['phase']['kind'] != 'agent':
                continue
            raw = d / Path(phase['raw_dir']).name
            for row in phase['sessions']:
                entry = dict(phase_id=phase['phase']['id'], session_index=row['index'], task_id=row['task_id'], terminal_reason=row['terminal_reason'], recorded_task_success=row.get('task_success'))
                if row['terminal_reason'] != 'natural_stop':
                    entry.update(regraded=False, classification=row['terminal_reason'], reason='No natural final task result; preserve budget/transport/runtime outcome separately.')
                    checked.append(entry)
                    continue
                if not row['protocol_ok'] or not isinstance(row.get('grade'), dict) or type(row['grade'].get('success')) is not bool:
                    raise ValueError('Natural terminal session lacks a valid original grade')
                original = raw / f'session-{row['index']:04d}'
                before = tree_hashes(original)
                source_workspace_hashes[str(original)] = before
                for relative, digest in before.items():
                    source_hashes[str(original / relative)] = digest
                target = Path(temporary) / (phase['phase']['id'] + '-' + str(row['index']))
                shutil.copytree(original, target)
                state = deepcopy(row['final_state'])
                state['workspace'] = str(target)
                tasks.context(state)
                repeated = tasks.grade(state, row['final_text'])
                if repeated['success'] is not row['grade']['success'] or repeated['success'] is not row['task_success']:
                    raise ValueError('Objective grade does not repeat: ' + row['task_id'])
                if tree_hashes(original) != before:
                    raise ValueError('Original task workspace changed')
                entry.update(regraded=True, repeat_matches=True, classification=failure_class(row, repeated), repeated_grade=repeated, original_workspace_sha256=hashlib.sha256(tasks.compact(before).encode()).hexdigest(), copied_state_workspace='temporary owned copy; removed after grading')
                checked.append(entry)
    for path, info in inputs.items():
        if sha(Path(path)) != info['sha256']:
            raise ValueError('Terminal input changed')
    for path, digest in source_hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError('Original workspace changed after grade')
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), experiment=d.name, experiment_execution_status=report['status'], supervisor_exit_code=status['exit_code'], sessions=checked, regraded=sum((e.get('regraded', False) for e in checked)), successful_tasks=sum((e.get('repeated_grade', {}).get('success') is True for e in checked)), input_ledger=inputs, original_workspace_hashes=source_workspace_hashes, grader_sha256=sha(TASK_SOURCE) if has_agent else None, grading_applicability='AGENT_TASKS' if has_agent else 'NOT_APPLICABLE_REPLAY_ONLY', script_sha256=sha(Path(__file__)), scope='Same frozen objective grader rerun on copies; original result trees read-only. Not an independent grading-rubric validity proof or task quality benchmark score.' if has_agent else 'Pure fixed replay has no objective task scoring. No task grader imported or executed; regraded=0 is not task success.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', sessions=len(checked), regraded=result['regraded'], successful_tasks=result['successful_tasks'], output=str(output)), indent=2))


if __name__=='__main__':main()
