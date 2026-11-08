#!/usr/bin/env python3
"""Synthetic CPU controls; these fixtures are not model measurements."""

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

SOURCE=Path(__file__).with_name('summarize_session_workloads.py')
spec=importlib.util.spec_from_file_location('cumulative_controls',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def reject(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    def row(index, success=True, known=True, turn=0):
        usage = dict(prompt_tokens=100 if success else 200, completion_tokens=20 if success else 40, total_tokens=120 if success else 240, prompt_tokens_details=dict(cached_tokens=0)) if known else None
        raw = dict(index=index, kind='agent', task_id='task1', profile=None, task_success=success, protocol_ok=True, terminal_reason='natural_stop', admitted=True, start_offset_s=0.0, end_offset_s=4.0, planned_offset_s=0.0, elapsed_s=4.0, scheduled_latency_s=4.0, user_visible_latency_s=2.0, requests=[dict(ok=success, usage=usage, cached_tokens=0 if known else None, turn=turn, dispatch_offset_s=0.0, end_offset_s=2.0, tool_calls=[])], tools=[], host_operations=[], grade=None)
        return m.d.session_projection(raw)

    def stage(identifier, seed, role='screen', included=True, rate=0.2, rows=None, experiment='cMOCK'):
        rows = rows if rows is not None else [row(0), row(1, False)]
        phase = dict(id=identifier, seed=seed, role=role, kind='agent', arrival='poisson', rate=rate, duration_s=120, cache_scope='session', dp_routing='none')
        return dict(experiment=experiment, phase=phase, included_as_measurement=included, exclusion_reason=None, sessions=rows, summary=m.d.group_summary(rows, 'agent', 10.0))
    e = dict(id='cMOCK', terminal_success=True, supervisor_exit_code=0)
    base = dict(experiments=[e], stages=[stage('one', 1), stage('two', 2), stage('probe', 3, 'pilot', False, rows=[row(0)])])
    cat = {'cMOCK': {'task1': 'coding'}}

    def build(value=None, refs=None):
        return m.cumulative(value or deepcopy(base), cat, refs or {}, {})
    result = build()
    formal = [c for c in result['cohorts'] if c['included_as_measurement']]
    check('two seeds same treatment coalesce; pilot remains separate', len(result['cohorts']) == 2 and len(formal) == 1 and (formal[0]['seeds'] == [1, 2]))
    check('formal inventory excludes pilot attempt and success', result['inventory']['formal_agent_attempts'] == 4 and result['inventory']['formal_agent_strict_successes'] == 2 and (result['inventory']['all_sessions'] == 5))
    cost = formal[0]['summary']['all_attempt_cost']
    check('all known failed-call costs included, not successful-only numerator', cost['known_prompt_tokens'] == 600 and cost['known_output_tokens'] == 120 and (cost['total_tokens_per_successful_task'] == 360))
    check('all source attempts retain exact phase/index identities', len({(r['phase'], r['session_index']) for r in formal[0]['source_sessions']}) == 4)
    check('missing reference is absent coverage, not an inferred zero score', formal[0]['summary']['quality']['reference_missing_attempts'] == 4 and formal[0]['summary']['quality']['reference_successes'] is None)
    check('summed source phase walls are explicit, not continuous runtime', formal[0]['summary']['summed_phase_cohort_wall_s'] == 20 and 'cohort_wall_s' not in formal[0]['summary'])
    check('concurrent active wall not replaced by phase wall', formal[0]['summary']['timing']['summed_session_components_s']['active_session_wall_s'] == 16)
    for key, value in [('role', 'confirm'), ('rate', 0.3), ('cache_scope', 'shared'), ('dp_routing', 'sticky4')]:
        b = deepcopy(base)
        b['stages'][1]['phase'][key] = value
        check('separate treatment ' + key, len(build(b)['cohorts']) == 3)
    b = deepcopy(base)
    b['stages'].append(deepcopy(b['stages'][0]))
    reject('duplicate phase input rejected', lambda: build(b))
    b = deepcopy(base)
    b['experiments'].append(dict(id='cOTHER', terminal_success=True, supervisor_exit_code=0))
    b['stages'].append(stage('one', 1, experiment='cOTHER'))
    other = m.cumulative(b, {**cat, 'cOTHER': {'task1': 'coding'}}, {}, {})
    check('different experiments never pooled', len(other['cohorts']) == 3)
    b = deepcopy(base)
    b['stages'][0] = stage('one', 1, rows=[row(0), row(1, False, False)])
    result = build(b)
    g = next((c for c in result['cohorts'] if c['included_as_measurement']))['summary']
    check('missing failed usage produces lower-bound cost, not missing attempt', not g['all_attempt_cost']['complete'] and g['all_attempt_cost']['unknown_usage_calls'] == 1 and (g['sessions'] == 4) and (g['all_attempt_cost']['total_tokens_per_successful_task'] == 240))
    rows = [row(0, False), row(1, False)]
    g = m.aggregate_group(rows, 'agent', 10)
    check('zero task successes produce null cost', g['all_attempt_cost']['total_tokens_per_successful_task'] is None)
    r = [row(0), row(1, False), row(2, False, False)]
    r[0]['calls'].append({**r[0]['calls'][0], 'turn': 1, 'output_tokens': 30, 'prompt_tokens': 300})
    r[2]['calls'][0]['turn'] = None
    turns = m.turn_distributions(r)
    check('later turn has reaching-session denominator, not all-attempt length sample', turns[1]['raw_turn_index'] == 1 and turns[1]['sessions_reaching_turn'] == 1 and (turns[1]['assigned_sessions'] == 3) and (turns[1]['request_lengths']['output_tokens']['n'] == 1))
    check('missing turn retained explicit bucket', turns[-1]['missing_or_invalid_turn'] and turns[-1]['model_calls'] == 1 and (turns[-1]['usage_missing_calls'] == 1))
    check('all call records conserved across turn buckets', sum((t['model_calls'] for t in turns)) == 4)
    check('failed known usage contributes to turn quantiles', turns[0]['request_lengths']['output_tokens']['p50'] == 30 and turns[0]['failed_model_calls'] == 1)
    reference = {}
    for s in base['stages']:
        for r in s['sessions']:
            reference[s['phase']['id'], r['index']] = dict(task_id='task1', kind='coding', original_primary_success=r['task_success'], terminal_reason='natural_stop', reference_eligible=False, reference_passed=None)
    res = build(refs={'cMOCK': reference})
    g = next((c for c in res['cohorts'] if c['included_as_measurement']))['summary']
    check('reference joins exact attempts without promoting pilot', g['quality']['reference_covered_attempts'] == 4 and g['successful_tasks'] == 2)
    bad = deepcopy(reference)
    del bad['one', 0]
    reject('partial reference file cannot silently shrink denominator', lambda: build(refs={'cMOCK': bad}))
    bad = deepcopy(reference)
    bad['extra', 0] = deepcopy(bad['one', 0])
    reject('unmatched extra reference rejected', lambda: build(refs={'cMOCK': bad}))
    bad = deepcopy(reference)
    bad['one', 0]['original_primary_success'] = False
    reject('reference cannot replace raw strict grade', lambda: build(refs={'cMOCK': bad}))
    bad = deepcopy(reference)
    bad['one', 0]['kind'] = 'data'
    reject('reference task type bound to launch catalog', lambda: build(refs={'cMOCK': bad}))
    bad = deepcopy(reference)
    bad['one', 0]['reference_passed'] = True
    reject('ineligible reference cannot imply score', lambda: build(refs={'cMOCK': bad}))
    replay = deepcopy(base)
    replay['stages'] = replay['stages'][:1]
    st = replay['stages'][0]
    st['phase']['kind'] = 'replay'
    for r in st['sessions']:
        r.update(kind='replay', task_success=None)
    res = m.cumulative(replay, {}, {}, {})
    g = res['cohorts'][0]['summary']
    check('replay has no strict task score or reference quality', g['successful_tasks'] is None and g['quality']['applicability'] == 'NOT_APPLICABLE_REPLAY_ONLY')
    bad = deepcopy(reference)
    bad['one', 0].update(reference_eligible=True, reference_passed=None)
    reject('eligible missing score cannot be counted as an actual failure', lambda: build(refs={'cMOCK': bad}))
    experiment = 'cIDENTITY'
    prefix = 'scenarios/workloads/sessions/tasks/'
    catalog = [dict(id='data_task', kind='data', readable=['README.md', 'table.csv'])]
    fixtures = {prefix + 'catalog.json': 'a' * 64, prefix + 'data_task/private/answers.json': 'b' * 64, prefix + 'data_task/public/README.md': 'c' * 64, prefix + 'data_task/public/table.csv': 'd' * 64}
    cmd = dict(source_sha256={**fixtures, 'scenarios/scripts/agent_session_tasks.py': 'e' * 64})
    expected = {'/mock/results/' + experiment + '/' + name: dict(sha256=str(i) * 64, bytes=i + 10) for i, name in enumerate(['status.json', 'sessions.json', 'command.json', 'source-snapshot.zip'], 1)}
    score = dict(input_ledger=deepcopy(expected), fixture_source_sha256=fixtures, sessions=[dict(task_id='data_task', reference_eligible=True)], grader_sha256='e' * 64)
    m.bind_score_inputs(score, expected, experiment, True, cmd, catalog)
    check('complete reference original input and fixture closure accepted', True)
    m.bind_score_inputs(score, expected, experiment, False, cmd, catalog)
    check('complete objective regrade raw and helper closure accepted', True)
    for name in ['status.json', 'sessions.json', 'command.json', 'source-snapshot.zip']:
        key = '/mock/results/' + experiment + '/' + name
        bad = deepcopy(score)
        bad['input_ledger'][key]['sha256'] = '0' * 64
        reject('wrong required scored raw hash rejected ' + name, lambda bad=bad: m.bind_score_inputs(bad, expected, experiment, True, cmd, catalog))
        bad = deepcopy(score)
        del bad['input_ledger'][key]
        reject('absent scored raw identity rejected ' + name, lambda bad=bad: m.bind_score_inputs(bad, expected, experiment, True, cmd, catalog))
    bad = deepcopy(score)
    bad['fixture_source_sha256'][prefix + 'data_task/private/answers.json'] = '0' * 64
    reject('wrong reference answer source rejected', lambda: m.bind_score_inputs(bad, expected, experiment, True, cmd, catalog))
    bad = deepcopy(score)
    del bad['fixture_source_sha256'][prefix + 'data_task/private/answers.json']
    reject('reference required fixture set cannot shrink', lambda: m.bind_score_inputs(bad, expected, experiment, True, cmd, catalog))
    bad = deepcopy(score)
    bad['grader_sha256'] = '0' * 64
    reject('wrong objective grader identity rejected', lambda: m.bind_score_inputs(bad, expected, experiment, False, cmd, catalog))
    intervals = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (2.0, 2.0)]
    check('half-open equal endpoint ordering avoids artificial concurrency spike', m.interval_peak(intervals) == 2)
    run = dict(phase=dict(arrival='poisson'), peak_active_sessions=4, sessions=[dict(admitted=True, start_offset_s=a, end_offset_s=b, planned_offset_s=c) for a, b, c in [(0.0, 2.0, 0.0), (1.0, 3.0, 0.5), (2.0, 4.0, 1.5)]])
    concurrency = m.concurrency_projection(run)
    check('open outstanding includes waiting before admission', concurrency['admitted_active_session_intervals']['peak'] == 2 and concurrency['open_outstanding_session_intervals']['peak'] == 3)
    check('client task peak remains separately labeled', concurrency['client_reported_peak_active_sessions'] == 4 and concurrency['client_reported_peak_is_measured'])
    run['phase']['arrival'] = 'closed'
    concurrency = m.concurrency_projection(run)
    check('closed preset peak not claimed as independent measurement', not concurrency['client_reported_peak_is_measured'] and concurrency['open_outstanding_session_intervals'] is None)
    run['sessions'][0]['end_offset_s'] = None
    concurrency = m.concurrency_projection(run)
    check('missing timestamp yields explicit partial concurrency not zero', concurrency['admitted_active_session_intervals']['peak'] is None and (not concurrency['admitted_active_session_intervals']['complete']))
    out = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), checks=checks, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='SYNTHETIC_CPU_CONTROLS_NOT_MODEL_EVIDENCE; pure aggregation only. File/raw metadata validation receives separate real and independent review.')
    with open_output(args.output, 'x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), output=str(args.output))))


if __name__=='__main__':main()
