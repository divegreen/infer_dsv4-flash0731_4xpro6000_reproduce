#!/usr/bin/env python3
"""Offline independent step-policy review; never imports the runtime/vLLM.

Reads one terminal D diagnostic. CPU scheduling positions are not GPU completion.
This checks request/token/decision evidence, not a full HTTP audit or capacity.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def review(result, phase):
    inputs = {}
    checks = Counter()
    errors = []

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = dict(sha256=digest(raw), bytes=len(raw))
        return raw

    def check(condition, name, detail=None):
        checks[name] += 1
        if not condition:
            errors.append(dict(check=name, detail=detail))

    status = json.loads(read(result/'status.json'))
    report = json.loads(read(result/'capacity.json'))
    identity = json.loads(read(result/'runtime-identity.json'))
    config = json.loads(read(result/'config.json'))
    check(status['supervisor_exit_code'] == status['exit_code'] == 0
          and status['source_identity_unchanged'] and not status['timed_out'], 'terminal_success')
    check(report['status'] == 'PASSED', 'client_terminal')
    logs = list(result.glob('scheduler-steps-*.jsonl'))
    ids = list(result.glob('scheduler-identity-*.json'))
    check(len(logs) == len(ids) == 1, 'one_engine_trace_and_identity')
    if len(logs) != 1 or len(ids) != 1:
        raise ValueError('One TP4 engine trace and identity required')
    runtime = json.loads(read(ids[0]))
    trace = [json.loads(row) for row in read(logs[0]).splitlines()]
    check(trace[0] == runtime, 'identity_trace_equal')
    check(runtime['config'] == dict(adaptive=False, input_budget=4096, max_active=64,
        max_running=64, original_cap=2048, prefill_interval=8, token_budget=4096), 'matched_configuration')
    key = 'scheduler/scripts/research_scheduler.py'
    check(runtime['module_sha256'] == identity['files'][key]['sha256'], 'executed_external_source_bound')
    key = 'installed/vllm/v1/core/sched/async_scheduler.py'
    check(runtime['parent_sha256'] == identity['files'][key]['sha256'], 'executed_async_parent_bound')
    check(not any(r['type'] == 'schedule_error' for r in trace), 'no_schedule_error')
    rows = [json.loads(x) for x in read(result/f'capacity-{phase}-requests.jsonl').splitlines()]
    steps = [r for r in trace if r['type'] == 'step']
    server_ids = {key for s in steps for key in s['before']['requests']}
    mapped = {}
    for r in rows:
        # Frozen InputProcessor appends a random eight-hex suffix after the
        # single-prompt '-0'; preserve full exact ID after this anchored join.
        pattern = re.compile(r'cmpl-' + re.escape(r['wire_request_id']) + r'-0-[0-9a-f]{8}')
        found = [key for key in server_ids if pattern.fullmatch(key)]
        check(len(found) == 1, 'client_scheduler_id_unique', r['request_id'])
        if len(found) == 1:
            mapped[found[0]] = r
            check(r['server_response_ids'] == ['cmpl-'+r['wire_request_id']], 'sse_id_binding')
    check(len(mapped) == len(rows) == 8, 'all_eight_requests_bound')
    selected = [s for s in steps if set(s['before']['requests']) & mapped.keys()]
    counters = Counter()
    events = {}
    scheduled_total, seen_time, first_service, prompt_done = Counter(), {}, {}, {}
    first_seen, last_service = {}, {}
    max_len = int(config['args'][config['args'].index('--max-model-len')+1])

    def event(name, s):
        counters[name] += 1
        events.setdefault(name, []).append(s['scheduler_step'])

    def pending(r, current):
        # max_tokens is fixed at128 in this diagnostic; other runs unsupported.
        if (r['num_output_placeholders'] > 0 and r['num_computed_tokens'] + 2
                - r['num_output_placeholders'] >= r['num_prompt_tokens'] + 128):
            return 0
        if current + 1 < r['next_decode_eligible_step']:
            return 0
        return max(0, min(r['num_tokens_with_spec'] + r['num_output_placeholders']
                         - r['num_computed_tokens'], max_len-r['num_computed_tokens']-1))

    def prefill(r):
        return r['is_prefill_chunk'] or r['num_computed_tokens'] < r['num_prompt_tokens']

    for s in steps:
        b, a, o = s['before'], s['after'], s['output']
        req, actual = b['requests'], o['num_scheduled_tokens']
        step = b['current_step']+1
        first_seen = {k:v for k,v in first_seen.items() if k in req}
        last_service = {k:v for k,v in last_service.items() if k in req}
        for k in req:
            first_seen.setdefault(k, b['current_step'])
        if s['policy'] == 'fairshare-v1':
            decision = s['decision']
            running_prefills = [k for k in b['running'] if prefill(req[k]) and pending(req[k],b['current_step'])>0]
            decode = sum(pending(req[k],b['current_step']) for k in b['running'] if not prefill(req[k]))
            eligible = [k for k in b['waiting'] if k not in b['deferred_waiting']
                        and req[k]['status'] in ('WAITING','PREEMPTED')]
            shorts = [k for k in eligible if req[k]['num_prompt_tokens'] <= 1024]
            aged = [k for k in req if step-last_service.get(k,first_seen[k]) >= 64]
            reserve = min(512, sum(max(0,req[k]['num_tokens']-req[k]['num_computed_tokens']) for k in shorts))
            if not shorts and set(aged) & set(eligible):
                reserve = 256
            cap = min(2048,max(256,(4096-decode-reserve)//max(1,len(running_prefills))))
            check((decision['dynamic_cap'], decision['decode_demand'],decision['waiting_reserve'],decision['running_prefills'])
                  == (cap,decode,reserve,len(running_prefills)), 'fair_cap_recomputed', step)
            check(sorted(decision['aged_promotions']) == sorted(aged), 'fair_age_recomputed', step)
            check(sorted(decision['running_order']) == sorted(b['running'])
                  and sorted(decision['waiting_order']) == sorted(b['waiting']), 'fair_queue_id_sets_retained',step)
        else:
            check(s['decision'] is None, 'observe_no_policy_decision',step)
        for k,n in actual.items():
            if n>0:
                last_service[k] = a['current_step']
        if not (set(req) & mapped.keys()):
            continue
        check(s['end_monotonic_ns'] >= s['begin_monotonic_ns'], 'monotonic_step_clock',step)
        check(sum(actual.values()) == o['total_num_scheduled_tokens'] <= 4096, 'token_budget',step)
        check(len(a['running']) <= 64 and len(a['running']) == len(set(a['running'])), 'running_slot_identity',step)
        check(set(req) == set(a['requests']), 'request_registry_preserved_in_step',step)
        check(not o['preempted_request_ids'], 'diagnostic_no_preemptions',step)
        for k,n in actual.items():
            check(k in mapped, 'scheduled_id_in_diagnostic',k)
            check(n>0, 'positive_scheduled_tokens',step)
            before,after = req[k], a['requests'][k]
            check(after['num_computed_tokens']-before['num_computed_tokens']==n,
                  'computed_delta_equals_scheduled',dict(step=step,request=k))
            check(after['num_in_flight_tokens']-before['num_in_flight_tokens']==n,
                  'inflight_delta_equals_scheduled',dict(step=step,request=k))
            expected_placeholder = before['num_output_placeholders'] + (not after['is_prefill_chunk'])
            check(after['num_output_placeholders']==expected_placeholder,
                  'parent_async_placeholder_delta',dict(step=step,request=k))
            scheduled_total[k] += n
            first_service.setdefault(k,s['begin_monotonic_ns'])
            if (before['num_computed_tokens']<before['num_prompt_tokens']
                    and after['num_computed_tokens']>=before['num_prompt_tokens']):
                prompt_done.setdefault(k,s['begin_monotonic_ns'])
        for k in req:
            seen_time.setdefault(k,s['begin_monotonic_ns'])
        longs = [k for k in b['running'] if req[k]['num_prompt_tokens']==65536 and req[k]['is_prefill_chunk']]
        short_wait = [k for k in b['waiting'] if req[k]['num_prompt_tokens']==512]
        has_decode = any(not req[k]['is_prefill_chunk'] for k in b['running'])
        distance = b['num_executed_steps']+1-b['last_prefill_step']
        long_compute = any(n>0 and req[k]['num_prompt_tokens']==65536 and
                           req[k]['num_computed_tokens']<65536 for k,n in actual.items())
        for name,predicate in dict(
            two_running_long_prefills=len(longs)==2,
            two_running_long_prefills_short_wait=len(longs)==2 and bool(short_wait),
            longs_exhaust4096_short_wait=len(longs)==2 and bool(short_wait) and sum(actual.get(k,0) for k in longs)==4096,
            no_decode_while_long_prefill=bool(longs) and not has_decode,
            capacity_bound_before=b['prefill_capacity_bound'],
            capacity_escape_with_long_prefill=(b['prefill_capacity_bound'] and has_decode
                and distance<8 and long_compute and not s['throttle_prefills']),
            cadence_deferral_condition=(not b['prefill_capacity_bound'] and has_decode
                and distance<8 and not s['throttle_prefills']),
            capacity_rise=not b['prefill_capacity_bound'] and a['prefill_capacity_bound'],
            capacity_drop=b['prefill_capacity_bound'] and not a['prefill_capacity_bound'],
            short_admission=any(req[k]['num_prompt_tokens']==512 for k in o['admitted_request_ids'])
        ).items():
            if predicate:
                event(name,s)
    by_request=[]
    for k,r in sorted(mapped.items()):
        expected = r['input_tokens'] + r['usage']['completion_tokens'] - 1
        check(scheduled_total[k] == expected, 'request_token_conservation',k)
        check(r['usage']['completion_tokens']==r['requested_output_tokens']==128 and r['ok']
              and r['done_received'] and not r['timed_out'], 'client_output_complete',k)
        by_request.append(dict(server_id=k,client_request_id=r['request_id'],input_tokens=r['input_tokens'],
            scheduled_tokens=scheduled_total[k],expected_scheduled_tokens=expected,
            scheduler_seen_to_first_service_s=(first_service[k]-seen_time[k])/1e9,
            scheduler_seen_to_prompt_scheduled_s=(prompt_done[k]-seen_time[k])/1e9,
            profiled_client_ttft_s=r['ttft_s'],profiled_client_tpot_s=r['tpot_s']))
    for path,info in inputs.items():
        check(digest(Path(path).read_bytes()) == info['sha256'], 'input_unchanged_after_review',path)
    return dict(schema_version=1,status='PASS' if not errors else 'FAIL',experiment_id=result.name,
        phase=phase,policy=runtime['policy'],checks=dict(checks),errors=errors,
        profile_steps=len(selected),mechanism_counts=dict(counters),mechanism_steps=events,
        requests=by_request,input_identity=inputs,analysis_source_sha256=digest(Path(__file__).read_bytes()),
        limits=['Profile, not normal performance or capacity evidence.',
                'Scheduler timestamps and after-counts are optimistic CPU scheduling positions, not GPU completion.',
                'Capacity-bound alone does not establish cadence escape: decode must exist and cadence distance be below8.',
                'This is an independent narrow diagnostic policy check, not a complete HTTP/source-package audit.',
                'No actual arrival_time is logged, so tie-break order among identical age/category peers is not independently reconstructed.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--phase', default='profile_interference64')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    outcome = review(a.result, a.phase)
    with open_output(a.output, 'x') as sink:
        json.dump(outcome, sink, indent=2)
        sink.write('\n')
    print(json.dumps({k: outcome[k] for k in ('status', 'experiment_id', 'profile_steps', 'mechanism_counts', 'errors')}, indent=2))
    return 0 if outcome['status'] == 'PASS' else 1


if __name__=='__main__':
    sys.exit(main())
