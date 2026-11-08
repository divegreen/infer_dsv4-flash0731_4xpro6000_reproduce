#!/usr/bin/env python3
"""Raw-derived cumulative Session evidence, with treatments and roles separate.

Reuses the reviewed raw JSONL descriptor without modifying it. This script never
runs a model, HTTP request, task tool, objective grader, or capacity certification.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = Path(__file__).with_name('describe_agent_workloads.py')
spec = importlib.util.spec_from_file_location('raw_workload_descriptor', DESCRIPTOR)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def checked_bytes(path, ledger, expected=None):
    raw = path.read_bytes()
    info = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    need(expected is None or info == expected, 'Audited input changed: ' + str(path))
    key = str(path.resolve())
    need(key not in ledger or ledger[key] == info, 'Input changed between reads: ' + str(path))
    ledger[key] = info
    return raw


def turn_distributions(rows):
    groups = defaultdict(list)
    for session_number, row in enumerate(rows):
        for call in row['calls']:
            turn = call.get('turn')
            key = turn if type(turn) is int and turn >= 0 else None
            groups[key].append((session_number, call))
    output = []
    for turn in sorted(groups, key=lambda v: (v is None, v if v is not None else 0)):
        pairs = groups[turn]
        calls = [call for _, call in pairs]
        known = [call for call in calls if call['usage_complete']]
        output.append(dict(raw_turn_index=turn,
            missing_or_invalid_turn=turn is None,
            assigned_sessions=len(rows), sessions_reaching_turn=len({i for i, _ in pairs}),
            model_calls=len(calls), failed_model_calls=sum(not call['ok'] for call in calls),
            usage_known_calls=len(known), usage_missing_calls=len(calls)-len(known),
            request_lengths={name:d.distribution([call[name] for call in calls])
                             for name in ('prompt_tokens','output_tokens','cached_tokens')}))
    return output


def quality_summary(rows, kind):
    if kind == 'replay':
        return dict(applicability='NOT_APPLICABLE_REPLAY_ONLY', assigned_attempts=len(rows),
                    strict_successes=None, reference_covered_attempts=0,
                    reference_eligible_attempts=None, reference_successes=None)
    references = [row['_reference'] for row in rows if row.get('_reference') is not None]
    eligible = [r for r in references if r['reference_eligible']]
    return dict(applicability='STRICT_PRIMARY_WITH_SEPARATE_OPTIONAL_JSON_REFERENCE',
        assigned_attempts=len(rows), strict_successes=sum(row['task_success'] is True for row in rows),
        reference_covered_attempts=len(references), reference_missing_attempts=len(rows)-len(references),
        reference_eligible_attempts=len(eligible),
        reference_successes=(sum(r['reference_passed'] is True for r in eligible) if references else None),
        reference_citation_gap_attempts=sum(any(v.get('evidence_supported') is False
            for v in (r.get('reference_grade') or {}).get('checks',{}).values()) for r in eligible),
        reference_extraction_errors=dict(Counter((r.get('reference_error') or {}).get('message')
            for r in eligible if (r.get('reference_error') or {}).get('message'))),
        primary_recheck_classifications=dict(Counter(row.get('_primary_classification','OBJECTIVE_RECHECK_NOT_SELECTED') for row in rows)))


def aggregate_group(rows, kind, total_wall):
    summary = d.group_summary(rows, kind, total_wall)
    summary['summed_phase_cohort_wall_s'] = summary.pop('cohort_wall_s')
    summary['wall_scope'] = 'Sum of unique source phase cohort walls, each including drain/grading; not continuous elapsed time or attributed task service time.'
    summary['turn_distributions'] = turn_distributions(rows)
    summary['quality'] = quality_summary(rows, kind)
    return summary


def cumulative(description, catalogs, references, regrades):
    """Pure aggregation, supplied with independently hash-checked input objects."""
    grouped = defaultdict(list)
    inventories = []
    seen_phases = set()
    for experiment in description['experiments']:
        identifier = experiment['id']
        selected = [s for s in description['stages'] if s['experiment'] == identifier]
        inventory = dict(experiment=identifier, terminal_success=experiment['terminal_success'],
            supervisor_exit_code=experiment['supervisor_exit_code'], phases=len(selected),
            phase_ids=[s['phase']['id'] for s in selected],
            roles=dict(Counter(s['phase']['role'] for s in selected)),
            all_sessions=sum(s['summary']['sessions'] for s in selected),
            formal_sessions=sum(s['summary']['sessions'] for s in selected if s['included_as_measurement']),
            nonformal_sessions=sum(s['summary']['sessions'] for s in selected if not s['included_as_measurement']))
        inventories.append(inventory)
    for stage in description['stages']:
        identity = (stage['experiment'], stage['phase']['id'])
        need(identity not in seen_phases, 'Duplicate source phase')
        seen_phases.add(identity)
        treatment = {key:value for key,value in stage['phase'].items() if key not in ('id','seed')}
        key = canonical(dict(experiment=stage['experiment'], treatment=treatment,
                             included_as_measurement=stage['included_as_measurement']))
        grouped[key].append(stage)
    cohorts = []
    used_references = defaultdict(set)
    for key, stages in sorted(grouped.items()):
        treatment = json.loads(key)
        identifier = treatment['experiment']
        kind = treatment['treatment']['kind']
        rows = []
        for stage in stages:
            for raw in stage['sessions']:
                row = deepcopy(raw)
                row['_source'] = dict(experiment=identifier, phase=stage['phase']['id'], session_index=raw['index'])
                # Descriptor timing incompleteness indexes remain unique in a cumulative cohort.
                row['index'] = len(rows)
                if kind == 'agent':
                    need(row['task'] in catalogs[identifier], 'Task absent from launch catalog')
                    row['_task_kind'] = catalogs[identifier][row['task']]
                    refkey = (stage['phase']['id'], raw['index'])
                    reference = references.get(identifier, {}).get(refkey)
                    if identifier in references:
                        need(reference is not None, 'Reference omits a recorded Agent attempt')
                    if reference is not None:
                        need(reference['task_id'] == row['task'] and reference['kind'] == row['_task_kind'],
                             'Reference task identity differs from launch catalog/raw')
                        need(reference['original_primary_success'] is row['task_success'], 'Reference primary score differs from raw')
                        need(reference['terminal_reason'] == row['terminal_reason'], 'Reference terminal reason differs from raw')
                        need(type(reference['reference_eligible']) is bool, 'Reference eligibility must be boolean')
                        need((type(reference['reference_passed']) is bool) if reference['reference_eligible']
                             else reference['reference_passed'] is None,
                             'Eligible reference requires a boolean score; ineligible requires null')
                        need(reference['reference_eligible'] or reference['reference_passed'] is None,
                             'Ineligible reference cannot report a pass/fail score')
                        used_references[identifier].add(refkey)
                    row['_reference'] = reference
                    objective = regrades.get(identifier, {}).get(refkey)
                    if objective is not None:
                        need(objective['recorded_task_success'] is row['task_success'], 'Objective recheck score differs from raw')
                        row['_primary_classification'] = objective['classification']
                rows.append(row)
        total_wall = sum(stage['summary']['cohort_wall_s'] for stage in stages)
        summary = aggregate_group(rows, kind, total_wall)
        by_task = {task:aggregate_group([row for row in rows if row['task']==task], kind, total_wall)
                   for task in sorted({row['task'] for row in rows})}
        by_type = ({taskkind:aggregate_group([row for row in rows if row['_task_kind']==taskkind], kind, total_wall)
                    for taskkind in sorted({row['_task_kind'] for row in rows})} if kind=='agent' else {})
        cohorts.append(dict(cohort_sha256=hashlib.sha256(key).hexdigest(), **treatment,
            source_phases=[stage['phase']['id'] for stage in stages],
            source_phase_concurrency=[dict(phase=stage['phase']['id'],evidence=stage.get('concurrency')) for stage in stages],
            seeds=[stage['phase'].get('seed') for stage in stages],
            source_sessions=[row['_source'] for row in rows],
            summary=summary, by_task=by_task, by_task_type=by_type))
    for identifier, refmap in references.items():
        need(used_references[identifier] == set(refmap), 'Reference includes extra/unmatched Agent attempts')
    formal = [cohort for cohort in cohorts if cohort['included_as_measurement']]
    formal_agents = [cohort for cohort in formal if cohort['treatment']['kind']=='agent']
    inventory = dict(experiments=len(inventories), phases=len(seen_phases),
        all_sessions=sum(e['all_sessions'] for e in inventories),
        formal_sessions=sum(c['summary']['sessions'] for c in formal),
        formal_agent_attempts=sum(c['summary']['sessions'] for c in formal_agents),
        formal_agent_strict_successes=sum(c['summary']['successful_tasks'] for c in formal_agents),
        formal_agent_model_calls=sum(c['summary']['model_calls'] for c in formal_agents),
        note='Inventory counts across heterogeneous conditions, never a pooled performance/quality ranking or capacity point.')
    return dict(inventory=inventory, experiments=inventories, cohorts=cohorts)



def bind_score_inputs(data, expected, experiment, is_reference, command, catalog_entries):
    ledger = data.get('input_ledger')
    need(isinstance(ledger, dict), 'Score lacks an input ledger')
    required = ['status.json', 'sessions.json', 'command.json']
    if is_reference:
        required.append('source-snapshot.zip')
    for filename in required:
        suffix = '/' + experiment + '/' + filename
        audited = [info for path,info in expected.items() if path.endswith(suffix)]
        scored = [info for path,info in ledger.items() if path.endswith(suffix)]
        need(len(audited)==1 and len(scored)==1 and audited[0]==scored[0],
             'Score provenance differs from audited raw: '+experiment+'/'+filename)
    sources = command['source_sha256']
    if is_reference:
        fixtures = data.get('fixture_source_sha256')
        need(isinstance(fixtures,dict), 'Reference fixture identity is absent')
        prefix = 'scenarios/workloads/sessions/tasks/'
        required_fixtures = {prefix+'catalog.json'}
        tasks = {task['id']:task for task in catalog_entries}
        for row in data['sessions']:
            if row.get('reference_eligible') is True:
                task = tasks.get(row['task_id'])
                need(task is not None and task['kind'] in ('data','research'), 'Unexpected reference-eligible task')
                required_fixtures.add(prefix+task['id']+'/private/answers.json')
                required_fixtures.update(prefix+task['id']+'/public/'+name for name in task['readable'])
        need(required_fixtures <= fixtures.keys(), 'Reference required fixture identity is incomplete')
        need(all(key in sources and digest==sources[key] for key,digest in fixtures.items()),
             'Reference fixture source differs from executed source')
    else:
        need(data.get('grader_sha256')==sources.get('scenarios/scripts/agent_session_tasks.py')
             and isinstance(data.get('grader_sha256'),str), 'Objective regrader identity differs from executed task helper')


def interval_peak(intervals):
    events=[]
    for start,end in intervals:
        if start == end:
            continue
        events.extend(((start,1),(end,-1)))
    active=peak=0
    for _,delta in sorted(events):
        active+=delta;need(active>=0,'Invalid interval sweep');peak=max(peak,active)
    need(active==0,'Unclosed interval sweep')
    return peak


def concurrency_projection(run):
    rows=run['sessions']
    def project(selected,start_field):
        intervals=[];missing=0
        for row in selected:
            start,end=row.get(start_field),row.get('end_offset_s')
            if not all(d.finite(value)for value in (start,end))or end<start:
                missing+=1
            else:
                intervals.append((start,end))
        lower=interval_peak(intervals)
        return dict(peak=lower if not missing else None,known_interval_peak=lower,
                    complete=not missing,intervals=len(intervals),missing_intervals=missing)
    admitted=[row for row in rows if row.get('admitted')is True]
    return dict(client_reported_peak_active_sessions=run.get('peak_active_sessions'),
        client_reported_peak_is_measured=run['phase']['arrival']=='poisson',
        client_reported_scope='Open: live asyncio Session tasks including wrapper/persistence tails. Closed: runtime preset min(concurrency,N), not independent measurement.',
        admitted_active_session_intervals=project(admitted,'start_offset_s'),
        open_outstanding_session_intervals=(project(rows,'planned_offset_s')if run['phase']['arrival']=='poisson'else None),
        interval_scope='Half-open intervals; ends precede starts at equal time. Active is admitted start to recorded end; open outstanding is planned arrival to recorded end, including admission wait. Final persistence tails excluded.')


def build(summaries, result_root, references_root, reference_map=None):
    # This re-reads authoritative raw JSONL and original terminal metadata;
    # summary inclusion flags and aggregate token numbers are not trusted alone.
    description = d.describe(summaries, result_root)
    ledger = dict(description['input_ledger'])
    expected = {}
    explicit = {}
    if reference_map is not None:
        mapping = json.loads(checked_bytes(reference_map.resolve(), ledger))
        need(mapping.get('schema_version')==1 and isinstance(mapping.get('experiments'),dict), 'Invalid explicit reference map')
        explicit = mapping['experiments']
    for path in summaries:
        audit = json.loads(checked_bytes(path.resolve(), ledger))
        for name, info in audit['input_ledger'].items():
            need(name not in expected or expected[name] == info, 'Conflicting audit hashes')
            expected[name] = info
    def original(experiment, filename):
        matches = [info for path,info in expected.items() if path.endswith('/'+experiment+'/'+filename)]
        need(len(matches)==1, 'Audited original not unique: '+experiment+'/'+filename)
        return checked_bytes(result_root/experiment/filename, ledger, matches[0])
    catalogs = {}; references = {}; regrades = {}; reference_inventory = []
    for experiment in description['experiments']:
        identifier = experiment['id']
        selected = [s for s in description['stages'] if s['experiment']==identifier]
        report = json.loads(original(identifier, 'sessions.json'))
        need({s['phase']['id'] for s in selected} == {run['phase']['id'] for run in report['phases']},
             'Summary omits or adds an executed phase')
        raw_phases = {run['phase']['id']:run for run in report['phases']}
        for stage in selected:
            stage['concurrency'] = concurrency_projection(raw_phases[stage['phase']['id']])
        if not any(stage['phase']['kind']=='agent' for stage in selected):
            continue
        command = json.loads(original(identifier, 'command.json'))
        snapshot = original(identifier, 'source-snapshot.zip')
        catalog_member = 'scenarios/workloads/sessions/tasks/catalog.json'
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            catalog_raw = archive.read(catalog_member)
        need(hashlib.sha256(catalog_raw).hexdigest()==command['source_sha256'][catalog_member],
             'Launch task catalog SHA mismatch')
        entries = json.loads(catalog_raw)['tasks']
        catalogs[identifier] = {task['id']:task['kind'] for task in entries}
        need(len(catalogs[identifier])==len(entries), 'Duplicate launch task ID')
        for suffix, destination, keyfields in [('-json-reference-r1.json', references, ('phase','session_index')),
                                               ('-objective-regrade-r1.json', regrades, ('phase_id','session_index'))]:
            field = 'json_reference' if destination is references else 'objective_regrade'
            configured = explicit.get(identifier,{})
            need(isinstance(configured,dict) and set(configured)<= {'json_reference','objective_regrade'}, 'Invalid reference map entry')
            if field in configured:
                need(isinstance(configured[field],str) and configured[field], 'Explicit reference path is empty')
                path = (reference_map.resolve().parent/configured[field]).resolve()
                need(path.is_file(), 'Explicit reference file is absent: '+str(path))
            else:
                path = references_root/(identifier+suffix)
            available = path.is_file()
            reference_inventory.append(dict(experiment=identifier,path=str(path.resolve()),available=available,
                                            selection='explicit_map' if field in configured else 'default'))
            if not available:
                continue
            data = json.loads(checked_bytes(path, ledger))
            need(data.get('status')=='PASS' and data.get('experiment')==identifier, 'Reference/regrade not verified for this experiment')
            bind_score_inputs(data, expected, identifier, destination is references, command, entries)
            if destination is references:
                need(data.get('primary_unchanged') is True and data.get('applied_to_runtime') is False
                     and data.get('feedback_to_model') is False, 'Reference must remain separate from runtime/primary score')
            values = {(row[keyfields[0]], row[keyfields[1]]):row for row in data['sessions']}
            need(len(values)==len(data['sessions']), 'Duplicate reference/regrade attempt')
            destination[identifier] = values
    result = cumulative(description, catalogs, references, regrades)
    for source in (Path(__file__), DESCRIPTOR):
        checked_bytes(source, ledger)
    for name, info in ledger.items():
        need(d.sha(Path(name))==info['sha256'] and Path(name).stat().st_size==info['bytes'], 'Input changed during cumulative analysis: '+name)
    need(all(Path(item['path']).is_file() is item['available'] for item in reference_inventory), 'Reference availability changed during analysis')
    return dict(schema_version=1,status='PASS',utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        scope='Cumulative snapshot of explicitly selected terminal experiments; not automatically Stage C final.',
        **result, reference_inventory=reference_inventory, input_ledger=ledger,
        descriptor_sha256=d.sha(DESCRIPTOR), script_sha256=d.sha(Path(__file__)), inputs_unchanged=True,
        limitations=['Raw-derived via the independently reviewed descriptor; original source/SSE/task grading is not rerun here.',
          'Coalescing is only within one experiment and identical complete phase treatment excluding ID/seed. Role, cache, routing, load, budget and formal inclusion remain separate.',
          'All assigned attempts contribute to strict success denominators and cost numerators, including failed calls with known usage. Zero successes give null costs; incomplete usage is a known lower bound.',
          'All-session inventory counts are not a global throughput or accuracy estimate; no cross-architecture pooled winner is computed.',
          'Turn indices retain raw numbering. Each turn reports reaching sessions and all assigned attempts; later-turn lengths are conditional on reaching that turn.',
          'Missing or invalid turn metadata is retained in an explicit bucket, never silently discarded.',
          'Session time sums overlap under concurrency; nested tool/grader time must not be added twice. Model request wall is not GPU time.',
          'Optional JSON references/regrades are separate and may be absent. Their required original input hashes bind to the same audited raw; fixture/grader identities bind to executed source. Top-level score totals are ignored.',
          'Reference eligible attempts require a boolean result; missing score is rejected, not treated as failure. Historical short filenames require an explicit hashed reference map.',
          'Actual Session concurrency is re-derived from original timestamp intervals per source phase; the closed runtime reported peak is a preset limit, not independent measurement. These are not GPU max-num-seqs.',
          'Six repeated authored tasks are not production workloads or SWE-bench; fixed replay has no task score.',
          'No Session latency SLO or all-started capacity certification is performed.'])


def markdown(report):
    lines=['# Session累计工作量证据快照','',
        '明确选择的终态实验；不是自动认定的阶段最终报告。正式/非正式、角色、负载、缓存/路由和实验保持分组。',
        '跨实验仅汇总清单计数，不合成全局吞吐或质量排名。逐turn分布的样本条件是实际到达该轮，另列全部分配尝试。','',
        '清单：'+json.dumps(report['inventory'],ensure_ascii=False),'',
        '| experiment | terminal | phases | all/formal/nonformal sessions |','|---|---|---:|---|']
    for e in report['experiments']:
        lines.append(f"| {e['experiment']} | {e['terminal_success']} | {e['phases']} | {e['all_sessions']}/{e['formal_sessions']}/{e['nonformal_sessions']} |")
    def fmt(v):
        return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
    for c in report['cohorts']:
        role=c['treatment']['role'];kind=c['treatment']['kind']
        lines.extend(['', '## '+c['experiment']+' / '+role+' / '+kind,
            '', '正式：'+str(c['included_as_measurement'])+'；source phases：'+', '.join(c['source_phases']),
            '完整处理条件：`'+json.dumps(c['treatment'],sort_keys=True,ensure_ascii=False)+'`',
            '逐phase实际Session并发证据：`'+json.dumps(c['source_phase_concurrency'],ensure_ascii=False)+'`','',
            '| task | attempts/strict success | calls/tools | known total tokens/success | usage complete | reference passed/eligible/covered |',
            '|---|---|---|---|---|---|'])
        for task,g in [('ALL',c['summary']),*c['by_task'].items()]:
            cost=g['all_attempt_cost'];q=g['quality']
            lines.append('| '+' | '.join([task,fmt(g['sessions'])+'/'+fmt(g['successful_tasks']),
                fmt(g['model_calls'])+'/'+fmt(g['actual_tool_calls']),fmt(cost['total_tokens_per_successful_task']),str(cost['complete']),
                fmt(q['reference_successes'])+'/'+fmt(q['reference_eligible_attempts'])+'/'+fmt(q['reference_covered_attempts'])])+' |')
        lines.extend(['','| raw turn | reaching/assigned sessions | calls/failed | prompt P50/P95/P99 | output P50/P95/P99 | known/missing usage |',
                      '|---|---|---|---|---|---|'])
        for turn in c['summary']['turn_distributions']:
            lengths=turn['request_lengths']
            quant=lambda name:'/'.join(fmt(lengths[name][key]) for key in ('p50','p95','p99'))
            lines.append('| '+' | '.join([fmt(turn['raw_turn_index']),str(turn['sessions_reaching_turn'])+'/'+str(turn['assigned_sessions']),
                str(turn['model_calls'])+'/'+str(turn['failed_model_calls']),quant('prompt_tokens'),quant('output_tokens'),
                str(turn['usage_known_calls'])+'/'+str(turn['usage_missing_calls'])])+' |')
    lines.extend(['','## 限制','',*['- '+item for item in report['limitations']]])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, nargs='+', required=True)
    parser.add_argument('--result-root', type=Path, default=path_at(ROOT.parent, 'scenarios/results'))
    parser.add_argument('--references-root', type=Path, default=path_at(ROOT, 'logs'))
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--reference-map', type=Path)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(ext).resolve() for ext in ('.json', '.md')]
    need(all((not path.exists() and args.result_root.resolve() not in path.parents for path in outputs)), 'Use unique outputs outside raw results')
    result = build(args.summaries, args.result_root.resolve(), args.references_root.resolve(), args.reference_map)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(outputs[0], 'x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    with open_output(outputs[1], 'x') as f:
        f.write(markdown(result))
    print(json.dumps(dict(status=result['status'], **result['inventory'], cohorts=len(result['cohorts']), input_files=len(result['input_ledger']), outputs=[str(p) for p in outputs]), indent=2))


if __name__=='__main__':main()
