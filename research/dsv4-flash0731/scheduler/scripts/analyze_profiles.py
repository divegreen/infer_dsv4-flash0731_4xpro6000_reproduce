#!/usr/bin/env python3
"""CPU-only Stage D profile and exact request-ID scheduler evidence.

Reuse frozen Stage B control/trace parsers with their one-long inference OFF.
Never import or execute captured runtime code, vLLM, CUDA, or an HTTP client.
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
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
B_PATH = ROOT.parent/'long-context/scripts/analyze_long_context_profiles.py'
SPEC = importlib.util.spec_from_file_location('stage_d_frozen_b_profile', B_PATH)
base = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(base)
metrics, need = base.metrics, base.require
AUDIT_PATH = ROOT/'scripts/audit_study.py'
AUDIT_SPEC = importlib.util.spec_from_file_location('profile_path_auditor', AUDIT_PATH)
path_audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(path_audit)
OFFSETS = (0., 0., .25, .75, 1.25, 1.75, 2.25, 2.75)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def number(value):
    return type(value) is int and value >= 0


def bind_requests(rows, server_ids):
    """Source-defined single-prompt n=1 mapping; no substring/fuzzy joins.

    Completion API prepends cmpl-, adds -0 for its single prompt, then
    InputProcessor appends eight lower-case hex characters by default.
    """
    mapping = {}; used = set()
    for row in rows:
        wire = row.get('wire_request_id')
        need(isinstance(wire, str) and wire and wire not in used, 'Missing/duplicate client wire ID')
        used.add(wire)
        response = 'cmpl-'+wire
        need(row.get('server_response_ids') == [response], 'Actual SSE response ID does not match wire ID')
        pattern = re.compile(re.escape(response+'-0-')+r'[0-9a-f]{8}\Z')
        matches = [key for key in server_ids if pattern.fullmatch(key)]
        need(len(matches) == 1, 'Expected exactly one internal ID for '+wire+', found '+str(len(matches)))
        internal = matches[0]
        need(internal not in mapping, 'One internal request bound to multiple clients')
        mapping[internal] = dict(index=row['index'], wire_request_id=wire, response_id=response,
            internal_request_id=internal, input_tokens=row['input_tokens'],
            output_tokens=row['usage']['completion_tokens'],
            request_class='long' if row['input_tokens'] == 65536 else 'short',
            mapping_basis='Single token-array prompt n=1; cmpl-wire + -0 + -8hex; exact anchored unique join')
    return mapping


def normalize_launch_plan(raw):
    """Match documented D defaults, without importing runtime/captured code.

    D deliberately requires explicit top-level settings and shape lists. Fail
    closed on unknown implicit settings rather than executing a ZIP member.
    """
    plan = json.loads(json.dumps(raw))
    need(all(key in plan for key in ('base_url','model','tokenizer_path','seed',
         'timeout_s','max_inflight','metrics_interval_s')), 'D launch plan must be explicit')
    need(plan['base_url'] == 'http://127.0.0.1:8260', 'Unexpected D launch base URL')
    for index, phase in enumerate(plan['phases']):
        need(isinstance(phase.get('shape_mix'), list), 'D launch needs explicit shape lists')
        for key, value in (('seed',plan['seed']+index),('ignore_eos',True),
                           ('role','measure'),('max_inflight',plan['max_inflight'])):
            phase.setdefault(key,value)
        if phase['mode'] in ('burst','closed-loop'):
            phase.setdefault('concurrency',phase['requests'] if phase['mode']=='burst' else 1)
    return plan


def snapshot_identity(directory, records, ledger):
    """Bind bytes in the launch ZIP to runtime identity without executing them."""
    command = records['command.json']; status = records['status.json']
    identity_path = directory/'runtime-identity.json'
    raw = metrics.raw_bytes(identity_path, ledger); identity = json.loads(raw)
    need(command.get('runtime_identity_sha256') == sha(raw), 'Runtime identity hash differs from command')
    need(status.get('source_identity_unchanged') is True and not status.get('changed_source_members')
         and status.get('supervisor_exit_code') == 0 and not status.get('cleanup_error')
         and status.get('exit_code') == 0 and status.get('timed_out') is False
         and not status.get('interrupted') and not status.get('interrupted_signal') and not status.get('supervisor_error'), 'Runtime source identity/cleanup incomplete')
    archive_raw = metrics.raw_bytes(directory/'source-snapshot.zip', ledger)
    need(command.get('source_snapshot_sha256') == sha(archive_raw), 'Launch archive hash differs from command')
    files = identity['files']; source_hashes = command['source_sha256']
    need(set(source_hashes) == set(files) and status.get('source_sha256') == source_hashes, 'Runtime/command/status source inventories differ')
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) and set(names) == set(files), 'Source ZIP member inventory differs')
        for name, entry in files.items():
            data = archive.read(name)
            need(sha(data) == entry['sha256'] == source_hashes[name] and len(data) == entry['bytes'], 'Source ZIP member changed: '+name)
        for filename, target in [('config.json', 'configs'), ('capacity-plan.json', 'workloads')]:
            option = '--config' if target == 'configs' else '--plan'
            argv = command.get('command', [])
            if option in argv:
                key = path_audit.base.snapshot_source_key(argv[argv.index(option)+1], ROOT, target)
            else:
                keys = ['experiments/'+ROOT.name+'/'+target+'/'+directory.name+'.json',
                        ROOT.name+'/'+target+'/'+directory.name+'.json']
                key = next((name for name in keys if name in files), None)
            need(key in files, 'Launch source missing '+str(key))
            launch = json.loads(archive.read(key))
            if key.startswith('experiments/'):
                context = command.get('path_context', {})
                need(status.get('path_context') == context, 'Launch/terminal path context differs')
                required = ('project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py',
                            'metadata/environment.json', 'metadata/model-inventory.json')
                need(all(name in files for name in required), 'Portable runtime dependency closure incomplete')
                launch = path_audit.base.expand_captured_document(launch, context)
            if filename == 'capacity-plan.json': launch = normalize_launch_plan(launch)
            need(launch == records[filename], 'Runtime saved input differs from launch '+key)
        report = records['capacity.json']
        for filename, field in [('study_client.py', 'client_sha256'), ('serve_study.py', 'profile_harness_sha256')]:
            key = ROOT.name+'/scripts/'+filename
            need(key in files and report.get(field) == files[key]['sha256'], 'Executed '+filename+' SHA not bound')
        required = ('v1/engine/input_processor.py', 'v1/engine/async_llm.py',
                    'entrypoints/openai/completion/serving.py', 'v1/core/sched/async_scheduler.py',
                    'v1/core/sched/scheduler.py')
        for name in required:
            need('installed/vllm/'+name in files, 'Request/scheduler semantics source absent: '+name)
    return identity


def scheduler_analysis(records, client, run, controls, source_identity=None):
    need(records and records[0].get('type') == 'scheduler_identity', 'First scheduler row must be identity')
    identities = [r for r in records if r.get('type') == 'scheduler_identity']
    need(len(identities) == 1, 'Expected one TP4 engine identity')
    identity = identities[0]
    policy = identity.get('policy')
    need(identity.get('schema_version') == 1 and identity.get('step_trace_enabled') is True
         and identity.get('async_scheduler') is True and policy in ('observe-v1', 'fairshare-v1'), 'Unverified scheduler identity')
    expected_class = 'research_scheduler.'+('ObserveResearchScheduler' if policy == 'observe-v1' else 'FairShareResearchScheduler')
    need(identity.get('class_name') == expected_class, 'Scheduler class/policy mismatch')
    if source_identity is not None:
        files = source_identity['files']
        module = files[ROOT.name+'/scripts/research_scheduler.py']
        parent = files['installed/vllm/v1/core/sched/async_scheduler.py']
        need(identity.get('module_sha256') == module['sha256'] and identity.get('parent_sha256') == parent['sha256'], 'Actual scheduler self-report SHA differs from captured source')
        need(identity.get('module_file') == module['path'] and identity.get('parent_file') == parent['path'], 'Actual scheduler import path differs from launch source')
    need(identity.get('config', {}).get('input_budget') == 4096 and identity['config'].get('token_budget') == 4096, 'Unexpected scheduler budgets')
    steps = []; server_ids = set(); previous_step = 0; previous_end = None
    for record in records[1:]:
        need(record.get('type') == 'step', 'Unexpected/error scheduler row: '+str(record.get('type')))
        need(record.get('schema_version') == 1 and record.get('policy') == policy, 'Scheduler row schema/policy changed')
        serial = record.get('scheduler_step')
        need(number(serial) and serial == previous_step+1, 'Missing/duplicate/nonmonotonic scheduler step')
        previous_step = serial
        for clock in ('monotonic_ns', 'unix_ns'):
            begin, end = record.get('begin_'+clock), record.get('end_'+clock)
            need(number(begin) and number(end) and begin <= end, 'Invalid scheduler '+clock)
        need(previous_end is None or record['begin_monotonic_ns'] >= previous_end, 'Overlapping scheduler call intervals')
        previous_end = record['end_monotonic_ns']
        for slot in ('before', 'after'):
            snapshot = record.get(slot)
            need(isinstance(snapshot, dict) and isinstance(snapshot.get('requests'), dict), 'Missing scheduler snapshot')
            server_ids.update(snapshot['requests'])
            for key in ('running', 'waiting', 'kv_holding_waiting', 'deferred_waiting'):
                values = snapshot.get(key)
                need(isinstance(values, list) and len(values) == len(set(values)), 'Invalid snapshot queue '+key)
                need(set(values) <= set(snapshot['requests']), 'Queue ID missing request snapshot')
        output = record.get('output'); need(isinstance(output, dict), 'Missing scheduler output')
        allocation = output.get('num_scheduled_tokens'); need(isinstance(allocation, dict), 'Missing allocation map')
        need(all(number(v) and v > 0 for v in allocation.values()), 'Invalid/nonpositive scheduled tokens')
        need(number(output.get('total_num_scheduled_tokens')) and sum(allocation.values()) == output['total_num_scheduled_tokens'] <= 4096, 'Scheduled tokens exceed budget or total mismatch')
        server_ids.update(allocation)
        for key in ('admitted_request_ids', 'preempted_request_ids', 'finished_request_ids'):
            need(isinstance(output.get(key), list) and len(output[key])==len(set(output[key])), 'Missing/duplicate scheduler output IDs '+key)
            server_ids.update(output[key])
        steps.append(record)
    binding = bind_requests(client, server_ids)
    selected_ids = set(binding)
    left, right = controls['controls'][0]['started_unix'], controls['controls'][1]['finished_unix']
    totals = {key: dict(context_tokens=0, generation_tokens=0, allocations=[], observed_steps=[], observed_unallocated_steps=[],
                       finished_observed=False, short_queued_long_only_steps=[]) for key in binding}
    timeline = []; observed = set(); frontiers = {}
    for record in steps:
        before = record['before']; after = record['after']; output = record['output']
        allocation = output['num_scheduled_tokens']; known = selected_ids.intersection(before['requests'])
        targeted = selected_ids.intersection(allocation)
        observed.update(known)
        for key in selected_ids.intersection(output['finished_request_ids']): totals[key]['finished_observed'] = True
        if not known and not targeted:
            continue
        need(record['begin_unix_ns']/1e9 >= left-.1 and record['end_unix_ns']/1e9 <= right+.1,
             'Profile request observed outside capture interval')
        need(not set(allocation)-selected_ids, 'Other-phase requests allocated concurrently with profile requests')
        need(not output['preempted_request_ids'], 'Profile scheduler recorded preemption')
        per_request = []
        for key in known:
            state = before['requests'][key]
            need(state.get('num_prompt_tokens') == binding[key]['input_tokens'], 'Scheduler/client prompt length mismatch')
            need(state.get('num_preemptions') == 0, 'Request snapshot has preemptions')
            computed = state.get('num_computed_tokens')
            need(number(computed) and computed == frontiers.get(key, 0),
                 'Request computed frontier is not continuous from preceding scheduler state: '+key)
            need(key in after['requests'], 'Request disappeared inside nonpreempting schedule call')
            after_computed = after['requests'][key].get('num_computed_tokens')
            need(number(after_computed) and after_computed == computed+allocation.get(key, 0),
                 'Request after-before computed delta differs from scheduled allocation: '+key)
            frontiers[key] = after_computed
            totals[key]['observed_steps'].append(record['scheduler_step'])
            if key not in allocation and (key in before['running'] or key in before['waiting']):
                totals[key]['observed_unallocated_steps'].append(record['scheduler_step'])
        for key in targeted:
            need(key in before['requests'], 'Allocated request absent from before snapshot')
            state = before['requests'][key]; count = allocation[key]
            computed, prompt = state['num_computed_tokens'], state['num_prompt_tokens']
            need(number(computed) and number(prompt), 'Invalid computed/prompt counters')
            context = min(count, max(0, prompt-computed)); generation = count-context
            totals[key]['context_tokens'] += context; totals[key]['generation_tokens'] += generation
            per_request.append(dict(request_id=key, request_class=binding[key]['request_class'], scheduled_tokens=count,
                context_tokens=context, generation_tokens=generation, before_computed=computed,
                after_computed=after['requests'].get(key, {}).get('num_computed_tokens'),
                before_output_placeholders=state['num_output_placeholders'], before_is_prefill_chunk=state['is_prefill_chunk']))
            totals[key]['allocations'].append(dict(step=record['scheduler_step'], unix_s=record['begin_unix_ns']/1e9,
                monotonic_ns=record['begin_monotonic_ns'], context_tokens=context, generation_tokens=generation))
        long_context = any(x['request_class']=='long' and x['context_tokens'] for x in per_request)
        short_context = any(x['request_class']=='short' and x['context_tokens'] for x in per_request)
        short_generation = any(x['request_class']=='short' and x['generation_tokens'] for x in per_request)
        if long_context and not any(x['request_class']=='short' for x in per_request):
            for key in known:
                if binding[key]['request_class']=='short' and key in before['running']+before['waiting']:
                    totals[key]['short_queued_long_only_steps'].append(record['scheduler_step'])
        timeline.append(dict(step=record['scheduler_step'], begin_unix_s=record['begin_unix_ns']/1e9,
            end_unix_s=record['end_unix_ns']/1e9, host_schedule_call_wall_ms=(record['end_monotonic_ns']-record['begin_monotonic_ns'])/1e6,
            allocations=per_request, total_scheduled_tokens=output['total_num_scheduled_tokens'],
            short_queued_ids=sorted(k for k in known if binding[k]['request_class']=='short' and k in before['running']+before['waiting']),
            long_context=long_context, short_context=short_context, short_generation=short_generation,
            long_and_short_context=long_context and short_context, long_context_short_generation=long_context and short_generation,
            before_running=before['running'], before_waiting=before['waiting'],
            prefill_capacity_bound_before=before['prefill_capacity_bound'], prefill_capacity_bound_after=after['prefill_capacity_bound'],
            last_prefill_step_before=before['last_prefill_step'], executed_steps_before=before['num_executed_steps'],
            local_prefill_interval=before['local_prefill_interval'], throttle_prefills=record['throttle_prefills'], decision=record['decision']))
    need(observed == selected_ids, 'Not every client request observed in scheduler state')
    for key, value in totals.items():
        expected = binding[key]
        need(value['context_tokens'] == expected['input_tokens'], 'Per-request scheduled context conservation failed: '+key)
        need(value['generation_tokens'] == expected['output_tokens']-1, 'Per-request scheduled generation conservation failed: '+key)
        need(value['allocations'], 'Request never allocated')
        allocations = value['allocations']; first, last = allocations[0], allocations[-1]
        gaps = [(b['monotonic_ns']-a['monotonic_ns'])/1e9 for a,b in zip(allocations, allocations[1:])]
        row = next(row for row in client if row['index']==expected['index'])
        origin = run['start_unix']
        value.update(first_allocation_step=first['step'], last_allocation_step=last['step'],
                     first_observed_step=value['observed_steps'][0],
                     observed_to_first_allocation_steps=first['step']-value['observed_steps'][0],
                     max_between_allocations_wall_s=max(gaps) if gaps else 0,
                     allocated_calls=len(allocations),
                     client_dispatch_to_first_schedule_s=first['unix_s']-(origin+row['dispatch_offset_s']),
                     client_ttft_s=row['ttft_s'], client_scheduled_ttft_s=row['scheduled_ttft_s'],
                     client_tpot_s=row['tpot_s'], client_scheduled_e2e_s=row['scheduled_e2e_s'])
    active = [step for step in timeline if step['total_scheduled_tokens']]
    return dict(status='PASS', identity=identity, total_logged_steps=len(steps), profile_observed_steps=len(timeline),
        profile_active_steps=len(active), binding=binding, requests=totals, timeline=timeline,
        request_position_continuity='PASS: cold position starts at 0; every before equals preceding after; each after-before equals allocated input tokens, including zero-allocation calls',
        context_tokens_sum=sum(r['context_tokens'] for r in totals.values()),
        generation_tokens_sum=sum(r['generation_tokens'] for r in totals.values()),
        long_short_context_steps=sum(r['long_and_short_context'] for r in active),
        long_context_short_generation_steps=sum(r['long_context_short_generation'] for r in active),
        capacity_bound_steps=sum(r['prefill_capacity_bound_before'] for r in timeline),
        attribution='Exact anchored unique client/response/internal request-ID join. Allocation input positions are async scheduler decisions, not GPU completion. Queued/observed-but-omitted does not itself prove eligibility or intentional policy deferral.',
        finished_scope='finished_request_ids may be consumed on a later scheduler call; complete client DONE/usage separately establishes completion.')



TIMEOUT_HARNESS_PAIR = ('b53d15c6c117a5d768513410025e0a3bf82c4a06e9e62222b5a3dbb876aa89b0',
                        '40f54805c5294f4a6a4153306244a8e1cbf458eb4ec4d4da1041e6d8217ff974')


def comparison_conditions(config, identity):
    value = json.loads(json.dumps(config)); value.pop('name')
    timeout = value.pop('profile_control_timeout_s', 120.)
    args = value['args']
    need(args.count('--scheduler-cls') == 1 and args.count('--profiler-config') == 1,
         'Comparison needs explicit scheduler and profiler options')
    idx = args.index('--scheduler-cls'); policy_class = args[idx+1]; del args[idx:idx+2]
    idx = args.index('--profiler-config'); profiler = json.loads(args[idx+1])
    profiler.pop('torch_profiler_dir'); args[idx+1] = json.dumps(profiler, sort_keys=True)
    sources = {key:entry['sha256'] for key,entry in identity['files'].items()
               if '/scripts/' in key or key.startswith(('scripts/', 'installed/', 'metadata/', 'project/'))
               or key in ('environment.json', 'model-inventory.json', 'project_layout.py', 'experiments/catalog.json')}
    return dict(config_sha256=metrics.hashed(value), profile_control_timeout_s=timeout,
                policy_class=policy_class, runtime_sources=sources)


def matched_profile_conditions(experiments, ledger):
    if len(experiments) < 2:
        return dict(status='SINGLE_OBSERVATION', declared_differences=[])
    need(len({e['same_workload_identity']['sha256'] for e in experiments}) == 1, 'Different exact prompt/arrival profile identity')
    need(len({e['comparison_conditions']['config_sha256'] for e in experiments}) == 1, 'Non-treatment Profile configuration changed')
    allowed = []
    anchor = experiments[0]['comparison_conditions']
    harness_key = ROOT.name + '/scripts/serve_study.py'
    for experiment in experiments[1:]:
        current = experiment['comparison_conditions']
        keys = set(anchor['runtime_sources']) | set(current['runtime_sources'])
        different = [key for key in keys if anchor['runtime_sources'].get(key) != current['runtime_sources'].get(key)]
        timeout_changed = anchor['profile_control_timeout_s'] != current['profile_control_timeout_s']
        if not different and (not timeout_changed):
            continue
        decision = metrics.read_json(ROOT / 'decisions/profile-export-retry-r1.json', ledger)
        if not different:
            # A new reproduction runs both policies with the same captured
            # harness. The original retry protocol still declares 120 -> 600
            # seconds for profile export, but no historical source diff exists.
            before, after = experiments[0], experiment
            if before['comparison_conditions']['profile_control_timeout_s'] == 600:
                before, after = after, before
            before_conditions = before['comparison_conditions']
            after_conditions = after['comparison_conditions']
            need(decision['controlled_change'] == {'profile_control_timeout_s': {'before': 120, 'after': 600}}
                 and before['experiment'] == 'scheduler-tp4-observe-dense-profile'
                 and after['experiment'] == decision['retry']
                 and before_conditions.get('policy_class') == 'research_scheduler.ObserveResearchScheduler'
                 and after_conditions.get('policy_class') == 'research_scheduler.FairShareResearchScheduler'
                 and before_conditions['profile_control_timeout_s'] == 120
                 and after_conditions['profile_control_timeout_s'] == 600,
                 'Same-source export deadlines do not match the registered Observe/retry pair')
            need(bool(anchor['runtime_sources'].get(harness_key)),
                 'Same-source comparison lacks a captured profile harness')
            allowed.append(dict(experiment=after['experiment'], field='profile_control_timeout_s',
                before=120, after=600, harness_sha256=anchor['runtime_sources'][harness_key],
                runtime_source_identity='identical', decision='decisions/profile-export-retry-r1.json',
                scope='Registered start/stop profile export deadline only; identical runtime source, '
                      'measured request timeout/arrival/config identity unchanged; original failed run remains excluded'))
            continue
        need(different == [harness_key] and (anchor['runtime_sources'].get(harness_key), current['runtime_sources'].get(harness_key)) == TIMEOUT_HARNESS_PAIR, 'Undeclared runtime source revision in Profile comparison')
        need(experiment['experiment'] == decision['retry'] and decision['harness_sha256'] == TIMEOUT_HARNESS_PAIR[1] and (decision['controlled_change'] == {'profile_control_timeout_s': {'before': 120, 'after': 600}}) and (anchor['profile_control_timeout_s'] == 120) and (current['profile_control_timeout_s'] == 600), 'Timeout/source difference does not match frozen retry decision')
        metrics.raw_bytes(path_at(ROOT, 'logs/profile-timeout-harness-diff-r1.txt'), ledger)
        allowed.append(dict(experiment=experiment['experiment'], field='profile_control_timeout_s', before=120, after=600, previous_harness_sha256=TIMEOUT_HARNESS_PAIR[0], harness_sha256=TIMEOUT_HARNESS_PAIR[1], decision='decisions/profile-export-retry-r1.json', source_diff='logs/profile-timeout-harness-diff-r1.txt', scope='Start/stop profile HTTP control deadline only; measured request timeout/arrival unchanged; original failed run remains excluded'))
    return dict(status='PASS', declared_differences=allowed, basis='Exact workload; all config excluding name/policy/output directory; identical runtime sources with registered export deadlines, or the exact historical reviewed harness revision')


def analyze_directory(directory, ledger):
    directory = directory.resolve()
    names = ('command.json','status.json','runtime-identity.json','experiment.json','capacity.json','config.json','capacity-plan.json','server-command.json')
    records = {name: metrics.read_json(directory/name, ledger) for name in names}
    identity = snapshot_identity(directory, records, ledger)
    report, config = records['capacity.json'], records['config.json']
    need(report['plan'] == records['capacity-plan.json'], 'Report/saved plan mismatch')
    need(report['plan']['experiment_id'] == config['name'] == directory.name, 'Experiment input identities differ')
    need('--no-enable-prefix-caching' in config['args'] and '--enable-prefix-caching' not in config['args'], 'Require APC off')
    need(base.option(config,'--tensor-parallel-size') == '4' and base.option(config,'--speculative-config') is None, 'Require TP4 no speculative')
    need(base.option(config,'--max-num-batched-tokens') == '4096' and base.option(config,'--max-num-seqs') == '64', 'Unexpected resource budgets')
    server = records['server-command.json']
    env = server.get('owned_environment',{})
    need(env.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3' and env.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'Server GPU scope changed')
    need('VLLM_DISABLE_REQUEST_ID_RANDOMIZATION' not in server.get('extra_environment',{})
         and records['command.json']['environment'].get('VLLM_DISABLE_REQUEST_ID_RANDOMIZATION') not in ('1','true','True'), 'Unsupported randomized-ID override')
    options = json.loads(base.option(config,'--profiler-config','{}'))
    need(options.get('max_iterations') == 0 and options.get('delay_iterations') == 0
         and options.get('detailed_trace_annotation') is False, 'Require complete non-detailed capture')
    run, rows, coverage = base.control_coverage(report, records['status.json'], records['experiment.json'])
    raw_rows_path = metrics.capacity.resolve_raw(run['raw_requests_jsonl'],directory)
    raw_rows = [json.loads(line) for line in metrics.raw_bytes(raw_rows_path,ledger).decode().splitlines() if line.strip()]
    need(sorted(raw_rows,key=lambda r:r['index']) == sorted(run['requests'],key=lambda r:r['index']), 'Client JSONL/report mismatch')
    rows = sorted(rows,key=lambda r:r['index'])
    need(len(rows)==8 and [r['index'] for r in rows]==list(range(8)), 'Eight unique profile requests required')
    need([(r['planned_offset_s'],r['input_tokens'],r['requested_output_tokens']) for r in rows]
         == [(t,65536 if i<2 else 512,128) for i,t in enumerate(OFFSETS)], 'Dense profile identity changed')
    need(all(r['usage']['completion_tokens']==128 for r in rows), 'Profile fixed output count changed')
    workload_path = metrics.capacity.resolve_raw(report['workload_file'],directory)
    workload_raw = metrics.raw_bytes(workload_path,ledger)
    need(sha(workload_raw)==report['workload_sha256'], 'Prepared workload SHA mismatch')
    workload = json.loads(workload_raw)
    prepared = [r for r in workload if r['spec']['id']==run['phase']['id']]
    need(len(prepared)==1, 'Profile prepared workload absent/duplicate')
    workload_identity = metrics.arrival_identity(report,run['phase'],rows)
    for endpoint in ('metrics_before','metrics_after'): metrics.read_prom(run[endpoint],directory,ledger)
    counters = metrics.capacity.metric_evidence(run,directory,ledger)
    need(counters.get('status')=='PASS' and counters.get('measurement_basis')=='explicit_before_after'
         and counters['totals'].get('vllm:num_preemptions_total')==0, 'Profile zero preemptions not verified')
    paths = sorted(set((directory/'profiles').glob('*.pt.trace.json.gz'))|set((directory/'profiles').glob('*.pt.trace.json')))
    manifest = base.legacy.validate_manifest(paths,(run['phase']['id'],),(0,1,2,3))
    need(manifest['complete'], 'Incomplete rank trace manifest: '+str(manifest['errors']))
    traces = [base.trace_analysis(path,run,coverage,False,ledger) for path in paths]
    for trace in traces:
        for key in ('inferred_long_short_context_steps','inferred_long_context_short_generation_steps','decode_only_between_first_last_inferred_long_context_steps'):
            trace[key]=None
        for step in trace['timeline']:
            for key in ('inferred_long_context','inferred_long_plus_short_context','inferred_long_context_plus_short_generation'): step[key]=None
        need(trace['context_tokens_sum']==134144 and trace['generation_tokens_sum']==1016, 'Worker annotation token conservation failed')
    need(len({t['active_iteration_sequence_sha256'] for t in traces})==1, 'Active worker sequence differs across ranks')
    log_paths = sorted(directory.glob('scheduler-steps-*.jsonl'))
    need(len(log_paths)==1,'Expected exactly one TP4 engine scheduler step file')
    log_rows=[json.loads(line) for line in metrics.raw_bytes(log_paths[0],ledger).decode().splitlines() if line.strip()]
    scheduler=scheduler_analysis(log_rows,rows,run,coverage,identity)
    self_paths=sorted(directory.glob('scheduler-identity-*.json'));need(len(self_paths)==1,'Missing/duplicate actual scheduler identity')
    need(metrics.read_json(self_paths[0],ledger)==scheduler['identity'],'Scheduler JSONL/self-report identities differ')
    need(base.option(config,'--scheduler-cls')==scheduler['identity']['class_name'],'Requested and actual scheduler classes differ')
    need(all(t['active_iteration_count']==scheduler['profile_active_steps'] for t in traces), 'Worker/scheduler active step count differs')
    return dict(experiment=directory.name,status='PASS',same_workload_identity=workload_identity,
        comparison_conditions=comparison_conditions(config,identity),
        profile_request_count=8,profile_control_coverage=coverage,preemption_counter_evidence=counters,
        actual_scheduler=scheduler['identity'],scheduler=scheduler,traces=traces,manifest=manifest,
        attribution='Request-level allocation attribution from scheduler records only. No GPU kernel-to-request or timestamp-based CPU/GPU ownership inference.')


def analyze(directories):
    ledger={};experiments=[];errors=[]
    # Every imported frozen Python helper is bound by bytes, without running snapshots.
    sources=[Path(__file__),B_PATH,Path(base.legacy.__file__),Path(metrics.__file__),Path(metrics.capacity.__file__),Path(metrics.historical.__file__),AUDIT_PATH,path_audit.SOURCE]
    for path in sources:metrics.raw_bytes(path,ledger)
    for directory in directories:
        try:experiments.append(analyze_directory(directory,ledger))
        except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,zipfile.BadZipFile) as exc:
            errors.append(dict(directory=str(directory),type=type(exc).__name__,error=str(exc)))
    ids={e['same_workload_identity']['sha256'] for e in experiments}
    comparison='SINGLE_OBSERVATION' if len(experiments)==1 else 'PASS' if len(ids)==1 and experiments else 'NOT VERIFIED'
    if len(ids)>1:errors.append(dict(error='Different prompt/arrival/workload identities; no matched comparison'))
    conditions = {'status':'NOT VERIFIED','declared_differences':[]}
    try: conditions = matched_profile_conditions(experiments,ledger)
    except (ValueError,KeyError,OSError,TypeError) as exc:errors.append(dict(error='Profile condition matching: '+str(exc)))
    for path,entry in ledger.items():
        data=Path(path).read_bytes()
        if sha(data)!=entry['sha256'] or len(data)!=entry['bytes']:errors.append(dict(path=path,error='Input changed during analysis'))
    return dict(schema_version=1,status='PASS' if experiments and not errors else 'NOT VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,python_executable=sys.executable,
        same_workload_comparison=comparison,matched_conditions=conditions,experiments=experiments,errors=errors,evidence_ledger=ledger,
        limitations=['Profile and scheduler tracing alter timing and are excluded from normal performance rankings.',
        'One eight-request diagnostic cannot certify capacity or a population P95/P99.',
        'Async scheduler after positions are optimistic, not actual GPU completion or KV release.',
        'Kernel duration sums overlap across streams and ranks; categories are names, not complete module ownership.',
        'Request-ID attribution concerns scheduled token allocations; CUDA graph/stream kernels have no proven request-level ownership.',
        'All old one-long/six-short aggregate-label long/short inferences are disabled.',
        'Control coverage and token accounting do not establish model numerical correctness or broad task quality.'])


def markdown(report):
    lines=['# 阶段 D 请求级调度与独立 Profile','',f"状态：{report['status']}；匹配负载：{report['same_workload_comparison']}。",'',
        '带Profiler的时间仅诊断，不进入正常性能或容量排名。长短归属来自唯一请求ID映射，不来自aggregate标签。','',
        '| 实验 | 实际policy | scheduler active | worker active/rank | 长短同时Prefill步 | 长Prefill+短Decode步 | capacity_bound观测步 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for e in report['experiments']:
        s=e['scheduler'];lines.append(f"| {e['experiment']} | {s['identity']['policy']} | {s['profile_active_steps']} | {e['traces'][0]['active_iteration_count']} | {s['long_short_context_steps']} | {s['long_context_short_generation_steps']} | {s['capacity_bound_steps']} |")
    lines+=['','## 每请求分配与客户端观测','','| 实验 / 请求index | 类型 | Context / generation输入 | 首次观察到首次分配步差 | 最大两次分配间隔s | 客户端scheduled TTFT s | TPOT ms |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for e in report['experiments']:
        s=e['scheduler']
        for key,binding in sorted(s['binding'].items(),key=lambda kv:kv[1]['index']):
            r=s['requests'][key];lines.append(f"| {e['experiment']} / {binding['index']} | {binding['request_class']} | {r['context_tokens']}/{r['generation_tokens']} | {r['observed_to_first_allocation_steps']} | {r['max_between_allocations_wall_s']:.6f} | {r['client_scheduled_ttft_s']:.6f} | {r['client_tpot_s']*1000:.3f} |")
    if report.get('matched_conditions',{}).get('declared_differences'):
        lines+=['','已声明的配对差异：']+['- '+json.dumps(x,ensure_ascii=False) for x in report['matched_conditions']['declared_differences']]
    lines+=['','首次观察是scheduler快照首次出现，不是精确API admission时刻；未获分配不等于policy故意延后，可能是cadence/异步eligibility/KV等。完整队列、分配、decision与请求定位在JSON。','',
        '## 每rank Kernel描述','','| 实验 / rank | Kernel数 | sum ms | union ms | span ms |','| --- | ---: | ---: | ---: | ---: |']
    for e in report['experiments']:
        for t in e['traces']:lines.append(f"| {e['experiment']} / {t['identity']['rank']} | {t['cuda_kernel_events']} | {t['kernel_duration_sum_ms']:.3f} | {t['kernel_union_ms']:.3f} | {t['kernel_span_ms']:.3f} |")
    lines+=['','局限：']+['- '+v for v in report['limitations']]
    if report['errors']:lines+=['','错误：']+['- '+json.dumps(v,ensure_ascii=False) for v in report['errors']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(s) for s in ('.json', '.md')]
    need(not any((p.exists() for p in outputs)), 'Use unique output prefix')
    need(all((not path_at(ROOT, 'results').resolve() in p.resolve().parents for p in outputs)), 'Analysis outputs cannot be raw result files')
    report = analyze(args.result_dirs)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, body in zip(outputs, (json.dumps(report, indent=2, ensure_ascii=False) + '\n', markdown(report))):
        with open_output(path, 'x') as f:
            f.write(body)
    print(json.dumps(dict(status=report['status'], experiments=len(report['experiments']), errors=report['errors'], output_prefix=str(args.output_prefix))))
    return 0 if report['status'] == 'PASS' else 1


if __name__=='__main__':raise SystemExit(main())
