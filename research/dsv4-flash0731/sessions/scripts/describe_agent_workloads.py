#!/usr/bin/env python3
"""Describe terminal Session workloads from authoritative raw JSONL, offline.

An existing PASS audit supplies phase metadata and expected raw hashes. No
runtime helper is imported; no model, HTTP, tools, or objective grader runs.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ('model_request_wall_s', 'tool_worker_wall_s', 'create_session_worker_wall_s',
              'grading_worker_wall_s', 'other_host_worker_wall_s', 'host_queue_wall_s',
              'host_resume_wall_s', 'synthetic_wait_wall_s')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distribution(values, total=None):
    raw = list(values)
    values = sorted(v for v in raw if finite(v))
    def quantile(q):
        if not values:
            return None
        at = (len(values) - 1) * q
        low = int(at)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (at - low)
    return dict(n=len(values), missing=(len(raw) if total is None else total)-len(values),
                min=min(values) if values else None, p50=quantile(.5), p95=quantile(.95),
                p99=quantile(.99), max=max(values) if values else None,
                mean=sum(values)/len(values) if values else None)


def read_json(path, ledger, expected=None):
    raw = path.read_bytes()
    info = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    need(expected is None or info == expected, 'Changed audited input: ' + str(path))
    ledger[str(path.resolve())] = info
    return json.loads(raw)


def usage_projection(request):
    usage = request.get('usage')
    known = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    known = known and usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens']
    result = dict(ok=request.get('ok') is True, usage_complete=bool(known),
                  prompt_tokens=usage['prompt_tokens'] if known else None,
                  output_tokens=usage['completion_tokens'] if known else None,
                  cached_tokens=None, turn=request.get('turn'))
    if known:
        details = usage.get('prompt_tokens_details')
        cached = details.get('cached_tokens') if isinstance(details, dict) else None
        if type(cached) is int and 0 <= cached <= usage['prompt_tokens']:
            result['cached_tokens'] = cached
        if request.get('cached_tokens') is not None:
            need(type(request['cached_tokens']) is int and request['cached_tokens'] == result['cached_tokens'],
                 'Recorded cached_tokens differs from authoritative usage')
    return result


def timing_projection(row):
    result = dict(admitted=row.get('admitted') is True, complete=False, reasons=[],
                  active_session_wall_s=None, arrival_to_end_s=None, dispatch_lag_s=None,
                  user_visible_latency_s=row.get('user_visible_latency_s'),
                  residual_wall_s=None, tool_internal_wall_s=None, grading_internal_wall_s=None,
                  synthetic_wait_planned_s=None, host_operations=len(row.get('host_operations', [])))
    result.update({name: None for name in COMPONENTS})
    if not result['admitted']:
        result['reasons'].append('Not admitted/censored; no active Session wall is inferred.')
        return result
    start, end, planned = (row.get(k) for k in ('start_offset_s', 'end_offset_s', 'planned_offset_s'))
    if not all(finite(v) for v in (start, end, planned)) or not planned <= start <= end:
        result['reasons'].append('Missing/invalid session time bounds.')
        return result
    active = end - start
    result.update(active_session_wall_s=active, arrival_to_end_s=end-planned, dispatch_lag_s=start-planned)
    if finite(row.get('elapsed_s')):
        need(abs(row['elapsed_s']-active) <= .003, 'Recorded active Session wall differs')
    if finite(row.get('scheduled_latency_s')):
        need(abs(row['scheduled_latency_s']-(end-planned)) <= .003, 'Arrival-to-end wall differs')
    values = dict.fromkeys(COMPONENTS, 0.)
    intervals = []
    missing = []
    missing_components = set()
    def add(name, begin, finish):
        if not finite(begin) or not finite(finish) or finish < begin:
            missing.append(name + ': missing/invalid interval')
            missing_components.add(name)
            return
        need(begin >= start-.003 and finish <= end+.003, 'Component outside Session bounds')
        values[name] += finish-begin
        intervals.append((begin, finish, name))
    for request in row.get('requests', []):
        add('model_request_wall_s', request.get('dispatch_offset_s'), request.get('end_offset_s'))
    if 'host_operations' not in row:
        missing.append('host_operations field absent')
        missing_components.update(k for k in COMPONENTS if k not in ('model_request_wall_s','synthetic_wait_wall_s'))
    for op in row.get('host_operations', []):
        ready, worker_start, worker_end, returned = (op.get(k) for k in
            ('ready_offset_s', 'worker_start_offset_s', 'worker_end_offset_s', 'end_offset_s'))
        add('host_queue_wall_s', ready, worker_start)
        worker_name = dict(execute_tool='tool_worker_wall_s', create_session='create_session_worker_wall_s',
                           grade='grading_worker_wall_s').get(op.get('name'), 'other_host_worker_wall_s')
        add(worker_name, worker_start, worker_end)
        add('host_resume_wall_s', worker_end, returned)
        if all(finite(v) for v in (ready, worker_start)) and finite(op.get('queue_s')):
            need(abs(op['queue_s']-(worker_start-ready)) < 1e-6, 'Host queue mismatch')
        if all(finite(v) for v in (worker_start, worker_end)) and finite(op.get('worker_s')):
            need(abs(op['worker_s']-(worker_end-worker_start)) < 1e-6, 'Host worker wall mismatch')
    synthetic = [t for t in row.get('tools', []) if t.get('synthetic_wait') is True]
    actual = [t for t in row.get('tools', []) if t.get('synthetic_wait') is not True]
    need(not synthetic or row['kind']=='replay', 'Agent tools cannot be synthetic replay waits')
    need(not actual or row['kind']=='agent', 'Replay cannot claim actual tool work')
    for tool in synthetic:
        add('synthetic_wait_wall_s', tool.get('start_offset_s'), tool.get('end_offset_s'))
    internal = [t.get('wall_s') for t in actual]
    if all(finite(x) and x>=0 for x in internal):
        result['tool_internal_wall_s'] = sum(internal)
    waits = [t.get('planned_s') for t in synthetic]
    if all(finite(x) and x>=0 for x in waits):
        result['synthetic_wait_planned_s'] = sum(waits)
    grade = row.get('grade')
    if isinstance(grade, dict) and finite(grade.get('wall_s')):
        result['grading_internal_wall_s'] = grade['wall_s']
    elif row['kind']=='replay':
        result['grading_internal_wall_s'] = 0.
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        need(left[1] <= right[0]+1e-6, 'Overlapping within-Session timing components')
    result.update({key:None if key in missing_components else value for key,value in values.items()})
    result['reasons'] = missing
    result['complete'] = not missing
    if not missing:
        residual = active-sum(values.values())
        need(residual >= -1e-6, 'Negative unexplained residual')
        result['residual_wall_s'] = max(0., residual)
    return result


def session_projection(row):
    calls = [usage_projection(r) for r in row['requests']]
    known = [r for r in calls if r['usage_complete']]
    cached = [r['cached_tokens'] for r in calls if r['cached_tokens'] is not None]
    return dict(index=row['index'], kind=row['kind'], task=row.get('task_id') or row.get('profile'),
                task_success=row.get('task_success'), protocol_ok=row.get('protocol_ok') is True,
                terminal_reason=row['terminal_reason'], calls=calls,
                actual_tool_calls=sum(t.get('synthetic_wait') is not True for t in row['tools']),
                tool_worker_operations=sum(op.get('name')=='execute_tool' for op in row.get('host_operations', [])),
                synthetic_waits=sum(t.get('synthetic_wait') is True for t in row['tools']),
                requested_tool_calls=sum(len(r.get('tool_calls', [])) for r in row['requests']),
                known_prompt_tokens=sum(r['prompt_tokens'] for r in known),
                known_output_tokens=sum(r['output_tokens'] for r in known),
                known_cached_tokens=sum(cached) if cached or not calls else None,
                token_usage_complete=len(known)==len(calls), cache_usage_complete=len(cached)==len(calls),
                timing=timing_projection(row))


def group_summary(rows, kind, cohort_wall):
    calls = [r for s in rows for r in s['calls']]
    known = [r for r in calls if r['usage_complete']]
    failed = [r for r in calls if not r['ok']]
    cached = [r['cached_tokens'] for r in calls if r['cached_tokens'] is not None]
    prompt = sum(r['prompt_tokens'] for r in known)
    output = sum(r['output_tokens'] for r in known)
    wins = sum(s['task_success'] is True for s in rows) if kind=='agent' else None
    admitted = [s['timing'] for s in rows if s['timing']['admitted']]
    complete_times = [t for t in admitted if t['complete']]
    timing_names = ('active_session_wall_s', 'arrival_to_end_s', 'dispatch_lag_s', 'user_visible_latency_s',
                    *COMPONENTS, 'residual_wall_s', 'tool_internal_wall_s', 'grading_internal_wall_s',
                    'synthetic_wait_planned_s')
    def cost(value):
        return value/wins if wins else None
    result = dict(sessions=len(rows), successful_tasks=wins,
                  protocol_completed=sum(s['protocol_ok'] for s in rows),
                  terminal_reasons=dict(Counter(s['terminal_reason'] for s in rows)),
                  model_calls=len(calls), successful_model_calls=sum(r['ok'] for r in calls),
                  actual_tool_calls=sum(s['actual_tool_calls'] for s in rows),
                  tool_worker_operations=sum(s['tool_worker_operations'] for s in rows),
                  requested_tool_calls=sum(s['requested_tool_calls'] for s in rows),
                  synthetic_waits=sum(s['synthetic_waits'] for s in rows),
                  calls_per_session=distribution([len(s['calls']) for s in rows]),
                  actual_tools_per_session=distribution([s['actual_tool_calls'] for s in rows]),
                  request_lengths={k:distribution([r[k] for r in calls]) for k in
                      ('prompt_tokens','output_tokens','cached_tokens')},
                  successful_request_lengths={k:distribution([r[k] for r in calls if r['ok']]) for k in
                      ('prompt_tokens','output_tokens','cached_tokens')},
                  failed_request_lengths={k:distribution([r[k] for r in failed]) for k in
                      ('prompt_tokens','output_tokens','cached_tokens')},
                  session_token_totals_complete_only={},
                  all_attempt_cost=dict(known_prompt_tokens=prompt,known_output_tokens=output,
                      known_cached_tokens=sum(cached) if cached or not calls else None,known_usage_calls=len(known),
                      unknown_usage_calls=len(calls)-len(known),complete=len(known)==len(calls),
                      known_cache_calls=len(cached),cache_complete=len(cached)==len(calls),
                      failed_calls=len(failed),failed_known_usage_calls=sum(r['usage_complete'] for r in failed),
                      prompt_tokens_per_successful_task=cost(prompt),output_tokens_per_successful_task=cost(output),
                      total_tokens_per_successful_task=cost(prompt+output),
                      model_calls_per_successful_task=cost(len(calls)),
                      actual_tools_per_successful_task=cost(sum(s['actual_tool_calls'] for s in rows))),
                  cohort_wall_s=cohort_wall,
                  timing=dict(admitted_sessions=len(admitted),complete_decompositions=len(complete_times),
                      incomplete_session_indices=[s['index'] for s in rows if s['timing']['admitted'] and not s['timing']['complete']],
                      per_session={k:distribution([t.get(k) for t in admitted]) for k in timing_names},
                      summed_session_components_s={k:sum(t[k] for t in complete_times) if complete_times else None
                          for k in (*COMPONENTS,'residual_wall_s','active_session_wall_s')},
                      sum_scope='Sums include only complete admitted Session decompositions; concurrent sessions overlap. Not cohort wall, CPU time, GPU kernel time or additive latency quantiles.'))
    for k in ('prompt_tokens','output_tokens','cached_tokens'):
        complete_flag='cache_usage_complete' if k=='cached_tokens' else 'token_usage_complete'
        selected=[s['known_'+k] for s in rows if s[complete_flag]]
        result['session_token_totals_complete_only'][k]=distribution(selected,total=len(rows))
    return result


def describe(summaries, result_root):
    ledger = {}; stages = []; experiments = []; seen = set(); raw_metadata = {}
    for source in summaries:
        audit = read_json(source.resolve(),ledger)
        need(audit.get('status')=='PASS' and not audit.get('errors'),'Require an existing evidence-analysis PASS')
        audited = audit['input_ledger']
        for experiment in audit['experiments']:
            need(experiment['id'] not in seen,'Overlapping experiment inputs; select nonoverlapping summaries')
            seen.add(experiment['id']); experiments.append(experiment)
            name=experiment['id'];need(Path(name).name==name,'Invalid experiment name')
            directory=(result_root/name).resolve()
            def metadata_file(filename):
                matches=[v for k,v in audited.items() if k.endswith('/'+name+'/'+filename)]
                need(len(matches)==1,'Audited raw metadata not uniquely identified: '+filename)
                return read_json(directory/filename,ledger,matches[0])
            status=metadata_file('status.json')
            need(status.get('end_utc') and type(status.get('exit_code')) is int,'Nonterminal/invalid raw supervisor')
            owned=metadata_file('experiment.json')
            report=metadata_file('sessions.json')
            saved_plan=metadata_file('session-plan.json')
            need(report['plan']==saved_plan,'Raw report differs from original saved plan')
            terminal=(status['exit_code']==0 and status.get('timed_out') is False
                and not status.get('supervisor_error') and owned.get('status')=='PASS'
                and report.get('status')=='COMPLETED' and not report.get('active_phase'))
            need(experiment['terminal_success'] is terminal,'Summary terminal flag contradicts raw execution')
            need(experiment['supervisor_exit_code']==status['exit_code'] and experiment['execution_status']==report['status'],
                 'Summary exit/report state contradicts raw execution')
            raw_phases={run['phase']['id']:dict(phase=run['phase'],status=run['status'],total_wall_s=run['total_wall_s']) for run in report['phases']}
            need(len(raw_phases)==len(report['phases']),'Duplicate raw phase ID')
            plan_phases={phase['id']:phase for phase in saved_plan['phases']}
            need(all(key in plan_phases and value['phase']==plan_phases[key] for key,value in raw_phases.items()),
                 'Executed phase metadata differs from original plan')
            raw_metadata[name]=dict(terminal_success=terminal,phases=raw_phases)
            del report, owned, saved_plan
        for stage in audit['stages']:
            name=stage['experiment'];phase=stage['phase'];identifier=phase['id']
            need(Path(identifier).name==identifier,'Invalid phase ID')
            meta=raw_metadata[name];original=meta['phases'].get(identifier)
            need(original is not None and phase==original['phase'],'Summary phase metadata contradicts original execution')
            need(stage['terminal_success'] is meta['terminal_success'],'Phase terminal flag contradicts raw execution')
            included=meta['terminal_success'] and original['status']=='COMPLETED' and phase['role'] in ('measure','screen','confirm')
            need(stage['included_as_measurement'] is included,'Summary formal inclusion contradicts raw phase role/terminal status')
            need(stage['total_wall_s']==original['total_wall_s'],'Summary cohort wall contradicts raw phase')
            relative=name+'/sessions-'+identifier+'/sessions.jsonl'
            matches=[v for k,v in audited.items() if k.endswith('/'+relative)]
            need(len(matches)==1,'Raw Session JSONL absent from audit ledger: '+relative)
            path=result_root/relative;before=sha(path);need(before==matches[0]['sha256'] and path.stat().st_size==matches[0]['bytes'],'Raw JSONL differs from audited input')
            projected=[];digest=hashlib.sha256();size=0
            with path.open('rb') as stream:
                for line in stream:
                    digest.update(line);size+=len(line)
                    if line.strip():projected.append(session_projection(json.loads(line)))
            info=dict(sha256=digest.hexdigest(),bytes=size);need(info==matches[0],'Raw JSONL changed during parsing')
            ledger[str(path.resolve())]=info
            need(sorted(s['index'] for s in projected)==list(range(len(projected))),'Session index coverage mismatch')
            full=group_summary(projected,phase['kind'],stage['total_wall_s']);old=stage['summary']
            for key in ('sessions','successful_tasks','protocol_completed','model_calls','successful_model_calls'):
                need(full[key]==old[key], 'Independent raw count disagrees with audit: '+key)
            need(full['actual_tool_calls']+full['synthetic_waits']==old['tool_calls'],'Tool/wait count disagrees with audit')
            costs=old['all_calls_known_usage'];actual=full['all_attempt_cost']
            need((actual['known_prompt_tokens'],actual['known_output_tokens'],actual['known_usage_calls'],actual['complete']) ==
                 (costs['prompt_tokens'],costs['output_tokens'],costs['known_calls'],costs['complete']),'Known all-attempt token costs disagree')
            stages.append(dict(experiment=name,phase=phase,included_as_measurement=included,
                exclusion_reason=stage['exclusion_reason'],workload_sha256=stage['arrival_workload_sha256'],
                authoritative_raw=str(path.resolve()),summary=full,
                by_task={task:group_summary([s for s in projected if s['task']==task],phase['kind'],stage['total_wall_s']) for task in sorted({s['task'] for s in projected})},
                sessions=sorted(projected,key=lambda s:s['index'])))
    for path,expected in ledger.items():
        need(sha(Path(path))==expected['sha256'] and Path(path).stat().st_size==expected['bytes'],'Input changed during description: '+path)
    return dict(schema_version=1,status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,script_sha256=sha(Path(__file__)),experiments=experiments,stages=stages,input_ledger=ledger,
        quantiles='Linear interpolation at (n-1)*q; n/missing explicitly shown; small n is descriptive.',
        timing_definition=dict(model_request_wall_s='Client dispatch to stream return including network/server queue/prefill/decode; not GPU execution time.',
            tool_worker_wall_s='execute_tool worker start to worker end; includes wall waits/subprocess/IO; not measured CPU time.',
            host_queue_wall_s='Host operation ready to worker start.',host_resume_wall_s='Worker end to awaiting coroutine continuation.',
            create_session_worker_wall_s='Owned task workspace setup worker wall.',grading_worker_wall_s='Objective grader worker wall after user-visible final response.',
            residual_wall_s='Active Session end-start minus disjoint recorded intervals; includes client serialization/state/event bookkeeping and scheduling, not attributed to an invented subsystem.',
            synthetic_wait_wall_s='Replay sleep intervals, separately labeled; no real tool CPU work.',
            tool_internal_wall_s='Nested execute_tool-reported wall, descriptive only; already included in tool worker interval.',
            grading_internal_wall_s='Nested grader-reported wall, descriptive only; already included in grading worker interval.'),
        limitations=['Existing audit PASS is required; this descriptor does not repeat source/SSE/objective-grading verification.',
            'Formal inclusion is re-derived from hashed raw supervisor/report/experiment terminal state and original phase role; summary labels, plan identity and cohort wall must agree.',
            'Every raw Session JSONL input is SHA-checked against audit and again after processing; originals are never written.',
            'Normal, warmup, pilot, Profile and failed execution roles remain separate; no pooled normal score includes probes.',
            'Usage distributions include valid known failed-call usage; missing usage remains missing, complete-only Session totals exclude incomplete attempts.',
            'Success costs distribute all attempts over strictly successful tasks. Zero successes produce null; incomplete known token costs are lower bounds.',
            'Request cached usage is not necessarily unique cached storage or bytes saved.',
            'Concurrent Session component sums overlap in wall time, and component quantiles do not add to total quantiles.',
            'Recorded Session end precedes final state/JSONL persistence; final persistence and monitor/drain tails remain in cohort wall and cannot be attributed from within-Session residuals.',
            'No Session latency SLO or new capacity certification; actual dynamic Agent histories need not match.'])


def markdown(report):
    lines=['# 实际 Agent 工作量与 Session 墙钟分解','',
        '状态：PASS（离线一致性）。原始 Session JSONL 为权威输入；每项已对照既有审计 SHA 并在读后复核。',
        'P50/P95/P99 为实际请求usage分布，n为已知样本；包括有效的失败调用usage。成本分摊全部尝试，缺usage为下界。',
        '组件是客户端/worker墙钟，不是GPU kernel或CPU计时；并发Session的时长加总不能替代cohort wall。合成回放等待单列。','']
    def f(v):return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
    def quant(d):return '/'.join(f(d[k]) for k in ('p50','p95','p99'))+' (n='+str(d['n'])+')'
    for normal,title in ((True,'正式 Agent'),(False,'非正式 Agent：预热、pilot/Profile或失败运行')):
        lines.extend(['## '+title,'','| phase / task | N / success | calls / tools | prompt P50/P95/P99 (n) | output P50/P95/P99 (n) | cached P50/P95/P99 (n) | total tokens / success | complete usage |','|---|---|---|---|---|---|---|---|'])
        for stage in report['stages']:
            if stage['phase']['kind']!='agent' or stage['included_as_measurement'] is not normal:continue
            for task,g in [('ALL',stage['summary']),*stage['by_task'].items()]:
                lens=g['request_lengths'];cost=g['all_attempt_cost']
                lines.append('| '+' | '.join([stage['experiment']+'/'+stage['phase']['id']+' / '+task,str(g['sessions'])+' / '+str(g['successful_tasks']),str(g['model_calls'])+' / '+str(g['actual_tool_calls']),quant(lens['prompt_tokens']),quant(lens['output_tokens']),quant(lens['cached_tokens']),f(cost['total_tokens_per_successful_task']),str(cost['complete'])])+' |')
    lines.extend(['','## 逐阶段墙钟分解','','以下为完整admitted Session组件的**重叠时长加总**，不是阶段总时间；完整cohort wall独立列示。task细分与每Session分位数见JSON。内部tool/grader计时已被worker区间包含，不再次相加。','','| phase | role / kind | complete / admitted | cohort wall | summed active | model | tool worker | setup | grading | host queue | host resume | synthetic wait | residual |','|---|---|---|---|---|---|---|---|---|---|---|---|---|'])
    for stage in report['stages']:
        g=stage['summary'];t=g['timing'];v=t['summed_session_components_s']
        lines.append('| '+' | '.join([stage['experiment']+'/'+stage['phase']['id'],stage['phase']['role']+' / '+stage['phase']['kind'],str(t['complete_decompositions'])+' / '+str(t['admitted_sessions']),f(g['cohort_wall_s']),*[f(v[k]) for k in ('active_session_wall_s','model_request_wall_s','tool_worker_wall_s','create_session_worker_wall_s','grading_worker_wall_s','host_queue_wall_s','host_resume_wall_s','synthetic_wait_wall_s','residual_wall_s')]])+' |')
    lines.extend(['','## 限制','',*[ '- '+s for s in report['limitations']]])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, nargs='+', required=True)
    parser.add_argument('--result-root', type=Path, default=path_at(ROOT.parent, 'scenarios/results'))
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output_prefix.with_suffix(s).resolve() for s in ('.json', '.md')]
    need(all((not p.exists() and args.result_root.resolve() not in p.parents for p in paths)), 'Use unique outputs outside raw results')
    report = describe(args.summaries, args.result_root.resolve())
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(paths[0], 'x') as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write('\n')
    with open_output(paths[1], 'x') as out:
        out.write(markdown(report))
    print(json.dumps(dict(status=report['status'], experiments=len(report['experiments']), phases=len(report['stages']), sessions=sum((s['summary']['sessions'] for s in report['stages'])), normal_agent_attempts=sum((s['summary']['sessions'] for s in report['stages'] if s['included_as_measurement'] and s['phase']['kind'] == 'agent')), input_files=len(report['input_ledger']), outputs=[str(p) for p in paths]), indent=2))


if __name__=='__main__':main()
