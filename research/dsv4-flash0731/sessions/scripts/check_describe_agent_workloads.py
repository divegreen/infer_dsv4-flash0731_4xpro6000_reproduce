#!/usr/bin/env python3
"""Synthetic CPU controls for the independent raw-JSONL workload descriptor."""

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

SOURCE=Path(__file__).with_name('describe_agent_workloads.py')
spec=importlib.util.spec_from_file_location('workload_description',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Unique control output required')
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, status='PASS' if condition else 'FAIL'))

    def reject(name, call):
        try:
            call()
        except ValueError:
            check(name, True)
        else:
            check(name, False)

    def operation(name, a, b, c, d):
        return dict(name=name, ready_offset_s=a, worker_start_offset_s=b, worker_end_offset_s=c, end_offset_s=d, queue_s=b - a, worker_s=c - b)
    req = dict(ok=True, usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120, prompt_tokens_details=dict(cached_tokens=50)), cached_tokens=50, dispatch_offset_s=1.0, end_offset_s=7.0, turn=0, tool_calls=[])
    row = dict(index=0, kind='agent', task_id='fixture', profile=None, admitted=True, planned_offset_s=0.0, start_offset_s=1.0, end_offset_s=11.0, elapsed_s=10.0, scheduled_latency_s=11.0, user_visible_latency_s=8.0, task_success=True, protocol_ok=True, terminal_reason='natural_stop', requests=[req], tools=[dict(wall_s=0.4)], grade=dict(wall_s=0.1), host_operations=[operation('create_session', 7, 7.1, 7.3, 7.4), operation('execute_tool', 7.4, 7.5, 8, 8.1), operation('grade', 8.1, 8.2, 8.4, 8.5)])
    t = m.timing_projection(row)
    check('Nonoverlapping components close active time', t['complete'] and abs(sum((t[k] for k in m.COMPONENTS)) + t['residual_wall_s'] - 10) < 1e-09)
    check('Worker wall differs from nested tool wall', abs(t['tool_worker_wall_s'] - 0.5) < 1e-09 and t['tool_internal_wall_s'] == 0.4)
    check('Queue/resume distinguished from worker', abs(t['host_queue_wall_s'] - 0.3) < 1e-09 and abs(t['host_resume_wall_s'] - 0.3) < 1e-09)
    check('Arrival lag separate from active session', t['dispatch_lag_s'] == 1 and t['arrival_to_end_s'] == 11 and (t['active_session_wall_s'] == 10))
    bad = deepcopy(row)
    bad['requests'][0]['end_offset_s'] = 8
    reject('Within-session overlap rejected', lambda: m.timing_projection(bad))
    bad = deepcopy(row)
    del bad['host_operations'][1]['worker_start_offset_s']
    t2 = m.timing_projection(bad)
    check('Missing time null, never synthetic zero', not t2['complete'] and t2['tool_worker_wall_s'] is None and (t2['host_queue_wall_s'] is None) and (t2['residual_wall_s'] is None))
    bad = deepcopy(row)
    del bad['host_operations']
    t2 = m.timing_projection(bad)
    check('Absent host evidence not interpreted as no host work', not t2['complete'] and t2['tool_worker_wall_s'] is None)
    failed = deepcopy(row)
    failed.update(index=1, task_success=False, protocol_ok=False, terminal_reason='request_error')
    failed['requests'][0].update(ok=False, usage=dict(prompt_tokens=200, completion_tokens=40, total_tokens=240), cached_tokens=None)
    a, b = (m.session_projection(row), m.session_projection(failed))
    group = m.group_summary([a, b], 'agent', 12)
    cost = group['all_attempt_cost']
    check('Failed known usage included in complete all-attempt costs', cost['complete'] and cost['known_prompt_tokens'] == 300 and (cost['known_output_tokens'] == 60) and (cost['total_tokens_per_successful_task'] == 360))
    check('Failed usage included in lengths', group['request_lengths']['prompt_tokens']['n'] == 2 and group['failed_request_lengths']['prompt_tokens']['p50'] == 200)
    check('Known zero cache distinguished from missing', group['request_lengths']['cached_tokens']['n'] == 1 and group['request_lengths']['cached_tokens']['missing'] == 1 and (not cost['cache_complete']))
    check('Concurrent session sums not cohort wall', group['cohort_wall_s'] == 12 and group['timing']['summed_session_components_s']['active_session_wall_s'] == 20)
    failed['requests'][0]['usage'] = None
    b = m.session_projection(failed)
    group = m.group_summary([a, b], 'agent', 12)
    cost = group['all_attempt_cost']
    check('Missing failed usage creates explicit lower bound', not cost['complete'] and cost['unknown_usage_calls'] == 1 and (cost['total_tokens_per_successful_task'] == 120))
    check('Incomplete Session excluded from complete token-total distribution', group['session_token_totals_complete_only']['prompt_tokens']['n'] == 1 and group['session_token_totals_complete_only']['prompt_tokens']['missing'] == 1)
    a['task_success'] = False
    group = m.group_summary([a, b], 'agent', 12)
    check('Zero successes do not imply zero token cost', group['all_attempt_cost']['total_tokens_per_successful_task'] is None)
    replay = deepcopy(row)
    replay.update(kind='replay', task_id=None, profile='fixture', task_success=None, planned_offset_s=0.0, start_offset_s=0.0, end_offset_s=5.0, elapsed_s=5.0, scheduled_latency_s=5.0, grade=None, host_operations=[], tools=[dict(synthetic_wait=True, planned_s=1.0, start_offset_s=3.0, end_offset_s=4.0, wall_s=1.0)])
    replay['requests'][0].update(dispatch_offset_s=0.0, end_offset_s=3.0)
    p = m.session_projection(replay)
    group = m.group_summary([p], 'replay', 5)
    check('Replay waits not counted as actual tools', group['actual_tool_calls'] == 0 and group['synthetic_waits'] == 1 and (group['successful_tasks'] is None))
    check('Replay wait separate from worker wall', p['timing']['synthetic_wait_wall_s'] == 1 and p['timing']['tool_worker_wall_s'] == 0 and (p['timing']['residual_wall_s'] == 1))
    censored = deepcopy(row)
    censored.update(admitted=False, start_offset_s=None, requests=[], tools=[], host_operations=[])
    t = m.timing_projection(censored)
    check('Censored no fictitious active duration', not t['complete'] and t['active_session_wall_s'] is None)
    p = m.session_projection(censored)
    check('Zero calls have known zero cached total, not unknown', p['cache_usage_complete'] and p['known_cached_tokens'] == 0)
    g = m.group_summary([p], 'agent', 12)
    check('Zero-call cohort has complete zero known token totals', g['all_attempt_cost']['complete'] and g['all_attempt_cost']['known_cached_tokens'] == 0 and (g['session_token_totals_complete_only']['cached_tokens']['n'] == 1))
    bad = deepcopy(req)
    bad['usage']['prompt_tokens'] = True
    check('Boolean rejected as token integer', not m.usage_projection(bad)['usage_complete'])
    d = m.distribution([0, 10, None])
    check('Explicit linear quantiles and missing samples', d['n'] == 2 and d['missing'] == 1 and (d['p50'] == 5) and (d['p95'] == 9.5) and (d['p99'] == 9.9))
    report = dict(status='PASS' if all((x['status'] == 'PASS' for x in checks)) else 'FAIL', utc=datetime.now(timezone.utc).isoformat(), checks=checks, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='SYNTHETIC_CPU_CONTROLS_NOT_MODEL_EVIDENCE; validates arithmetic and missing-data boundaries, not independent proof of raw experiment correctness.')
    with open_output(args.output, 'x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], checks=len(checks), failed=[x for x in checks if x['status'] != 'PASS'], output=str(args.output)), indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__=='__main__':main()
