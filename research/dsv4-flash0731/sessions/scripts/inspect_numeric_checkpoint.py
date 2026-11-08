#!/usr/bin/env python3
"""Read a completed interface-pilot session while its enclosing run may be live."""

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
import zipfile

ROOT = path_at(Path(__file__).resolve().parents[2], 'scenarios/results')


def sha(data):return hashlib.sha256(data).hexdigest()


def read(path):
    raw=path.read_bytes()
    return json.loads(raw),dict(path=str(path.resolve()),sha256=sha(raw),bytes=len(raw))


def observations(s):
    operations=[]
    for t in s['tools']:
        args=json.loads(t['call']['function']['arguments'])
        filters=[dict(column=f['column'],op=f['op'],value=f['value'],python_type=type(f['value']).__name__)
                 for f in args.get('filters',[]) if f['column']=='input_tokens']
        if filters:
            operations.append(dict(name=t['call']['function']['name'],filters=filters,
                                   valid=t['valid'],action_ok=t['action_ok'],result=t['result']))
    errors=[t['result'] for t in s['tools'] if not t['action_ok']]
    return dict(task_id=s['task_id'],session_index=s['index'],terminal_reason=s['terminal_reason'],
                protocol_ok=s['protocol_ok'],task_success=s['task_success'],grade=s['grade'],
                model_calls=len(s['requests']),tool_calls=len(s['tools']),
                model_seeds=sorted({r['payload']['seed'] for r in s['requests']}),
                numeric_filter_operations=operations,tool_errors=errors,
                type_mismatch_errors=sum(t.get('message')=='Filter value type differs from column' for t in errors))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    run = ROOT / 'session-tp4-cache-screen'
    phase = run / 'sessions-pilot-numeric-tool-schema'
    raw = (phase / 'sessions.jsonl').read_bytes()
    rows = [json.loads(x) for x in raw.splitlines() if x]
    assert len(rows) == 1 and rows[0]['end_offset_s'] is not None
    current = observations(rows[0])
    previous, old_identity = read(ROOT / 'session-tp4-agent-pilot2/sessions.json')
    old = observations(next((s for ph in previous['phases'] for s in ph['sessions'] if s['task_id'] == 'service_latency')))
    report, report_identity = read(run / 'sessions.json')
    command, command_identity = read(run / 'command.json')
    with zipfile.ZipFile(run / 'source-snapshot.zip') as archive:
        task_member = 'scenarios/scripts/agent_session_tasks.py'
        assert sha(archive.read(task_member)) == command['source_sha256'][task_member]
    assert current['model_seeds'] == old['model_seeds'] == [202611608]
    assert current['numeric_filter_operations'] and all((type(f['value']) is int for op in current['numeric_filter_operations'] for f in op['filters']))
    assert current['type_mismatch_errors'] == 0 and old['type_mismatch_errors'] == 10
    assert raw == (phase / 'sessions.jsonl').read_bytes(), 'Closed pilot evidence changed while reading'
    result = dict(status='CHECKPOINT_INTERFACE_FIX_VERIFIED_TASK_GRADE_FALSE_RUN_NOT_CERTIFIED', created_utc=datetime.now(timezone.utc).isoformat(), git_commit=command['git_commit'], runtime_task_sha256=command['source_sha256'][task_member], observed_report_status=report['status'], observed_active_phase=report.get('active_phase'), observed_completed_phase_ids=[p['phase']['id'] for p in report['phases']], current=current, previous=old, input_identities=[old_identity, report_identity, command_identity, dict(path=str((phase / 'sessions.jsonl').resolve()), sha256=sha(raw), bytes=len(raw))], limitations=['Only the completed pilot session is examined; surrounding experiment may still be RUNNING.', 'Same task/seed and fixed budgets, but schema changed and generated histories/workload differ.', 'Numeric filtering now works; strict task success remains false due to final output JSON format.', 'Model raw DSML was not captured, and parsed arguments do not reveal pre-parser original token spellings.', 'No throughput/capacity/quality ranking or task score override follows from this checkpoint.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], current_model_calls=current['model_calls'], old_model_calls=old['model_calls'], current_numeric_operations=len(current['numeric_filter_operations']), old_type_errors=old['type_mismatch_errors'], new_type_errors=current['type_mismatch_errors'], task_success=current['task_success'], output=str(args.output)), indent=2))


if __name__=='__main__':main()
