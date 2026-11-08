#!/usr/bin/env python3
"""Actual final Session summary/descriptor/cumulative review; no model execution."""

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

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks, ledger = ([], {})

    def load(path):
        raw = path.read_bytes()
        ledger[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    def check(name, value):
        if not value:
            raise ValueError(name)
        checks.append(name)

    def equal(a, b):
        return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    def distribution(values):
        xs = sorted((x for x in values if type(x) in (int, float) and math.isfinite(x)))

        def quantile(q):
            if not xs:
                return None
            pos = (len(xs) - 1) * q
            lo, hi = (math.floor(pos), math.ceil(pos))
            return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
        return dict(n=len(xs), missing=len(values) - len(xs), min=min(xs) if xs else None, max=max(xs) if xs else None, p50=quantile(0.5), p95=quantile(0.95), p99=quantile(0.99), mean=sum(xs) / len(xs) if xs else None)
    main_report = load(path_at(ROOT, 'summary/final-r1.json'))
    descriptor = load(path_at(ROOT, 'summary/workloads-final-r1.json'))
    cumulative = load(path_at(ROOT, 'summary/cumulative-final-r1.json'))
    confirmation = load(path_at(ROOT, 'logs/session-confirmations-final-r1.json'))
    prior = [load(path_at(ROOT, 'summary') / name) for name in ('cumulative-c03-c12-r2.json', 'cumulative-pilots-c01-c02-r2.json')]
    source = ROOT / 'scripts/summarize_session_workloads.py'
    check('frozen accepted cumulative implementation', hashlib.sha256(source.read_bytes()).hexdigest() == cumulative['script_sha256'] == '234eaeff442809c264a200374d172fa5666c5298ac3011a72e72e5ab211ab531')
    check('all final artifacts completed', all((obj['status'] == 'PASS' for obj in (main_report, descriptor, cumulative, confirmation))))
    ids = {e['id'] for e in main_report['experiments']}
    check('exact all-ID membership', len(ids) == 16 and ids == {e['id'] for e in descriptor['experiments']} == {e['experiment'] for e in cumulative['experiments']})
    main_stages = {(s['experiment'], s['phase']['id']): s for s in main_report['stages']}
    stages = {(s['experiment'], s['phase']['id']): s for s in descriptor['stages']}
    check('all 82 executed phases represented once', len(main_stages) == len(main_report['stages']) == 82 and len(stages) == len(descriptor['stages']) == 82 and (main_stages.keys() == stages.keys()))
    for key, stage in stages.items():
        check('raw-derived role and health agree ' + str(key), stage['phase'] == main_stages[key]['phase'] and stage['included_as_measurement'] is main_stages[key]['included_as_measurement'] and (len(stage['sessions']) == main_stages[key]['summary']['sessions']))
    expected = dict(experiments=16, phases=82, all_sessions=sum((len(s['sessions']) for s in stages.values())), formal_sessions=sum((len(s['sessions']) for s in stages.values() if s['included_as_measurement'])), formal_agent_attempts=sum((len(s['sessions']) for s in stages.values() if s['included_as_measurement'] and s['phase']['kind'] == 'agent')), formal_agent_strict_successes=sum((row['task_success'] is True for s in stages.values() if s['included_as_measurement'] and s['phase']['kind'] == 'agent' for row in s['sessions'])), formal_agent_model_calls=sum((len(row['calls']) for s in stages.values() if s['included_as_measurement'] and s['phase']['kind'] == 'agent' for row in s['sessions'])))
    check('final inventory independently counted', all((cumulative['inventory'][k] == v for k, v in expected.items())) and expected == dict(experiments=16, phases=82, all_sessions=2564, formal_sessions=2114, formal_agent_attempts=908, formal_agent_strict_successes=307, formal_agent_model_calls=4386))
    old_cohorts = {c['cohort_sha256']: c for report in prior for c in report['cohorts']}
    finals = {c['cohort_sha256']: c for c in cumulative['cohorts']}
    check('61 unique final cohorts', len(finals) == len(cumulative['cohorts']) == 61 and old_cohorts.keys() <= finals.keys())
    for digest, cohort in old_cohorts.items():
        check('previous independent numerical evidence unchanged ' + digest, finals[digest] == cohort)
    for selection in cumulative['reference_inventory']:
        path = Path(selection['path'])
        check('reference availability accurate ' + str(path), path.is_file() is selection['available'])
        if not selection['available']:
            continue
        score = load(path)
        check('score actual consumed bytes ' + str(path), ledger[str(path)] == cumulative['input_ledger'][str(path)])
        name = selection['experiment']
        is_reference = 'fixture_source_sha256' in score
        for filename in ('status.json', 'sessions.json', 'command.json') + (('source-snapshot.zip',) if is_reference else ()):
            suffix = '/' + name + '/' + filename
            expected_infos = [info for p, info in main_report['input_ledger'].items() if p.endswith(suffix)]
            scored_infos = [info for p, info in score['input_ledger'].items() if p.endswith(suffix)]
            check('score/main raw identity ' + name + '/' + filename, len(expected_infos) == 1 and scored_infos == expected_infos)
    catalog_path = path_at(ROOT.parent, 'scenarios/workloads/sessions/tasks/catalog.json')
    catalog = {row['id']: row['kind'] for row in load(catalog_path)['tasks']}
    new_groups = new_turns = 0
    all_sources = []
    for cohort in cumulative['cohorts']:
        key = cohort['cohort_sha256']
        name = cohort['experiment']
        source_stages = [stages[name, phase] for phase in cohort['source_phases']]
        source_rows = [row for stage in source_stages for row in stage['sessions']]
        identity_rows = [(name, stage['phase']['id'], row['index']) for stage in source_stages for row in stage['sessions']]
        all_sources.extend(identity_rows)
        check('cohort exact source identities ' + key, identity_rows == [(row['experiment'], row['phase'], row['session_index']) for row in cohort['source_sessions']])
        check('complete treatment separation ' + key, all(({k: v for k, v in stage['phase'].items() if k not in ('id', 'seed')} == cohort['treatment'] and stage['included_as_measurement'] is cohort['included_as_measurement'] for stage in source_stages)))
        if key in old_cohorts:
            continue
        kind = cohort['treatment']['kind']
        wall = sum((stage['summary']['cohort_wall_s'] for stage in source_stages))
        groups = [('ALL', source_rows, cohort['summary'])]
        groups.extend(((task, [r for r in source_rows if r['task'] == task], value) for task, value in cohort['by_task'].items()))
        groups.extend(((task_kind, [r for r in source_rows if catalog[r['task']] == task_kind], value) for task_kind, value in cohort['by_task_type'].items()))
        for group_name, rows, out in groups:
            label = name + '/' + key + '/' + group_name
            calls = [call for row in rows for call in row['calls']]
            known = [call for call in calls if call['usage_complete']]
            wins = sum((row['task_success'] is True for row in rows)) if kind == 'agent' else None
            cost = out['all_attempt_cost']
            check('new counts and all attempts ' + label, out['sessions'] == len(rows) and out['successful_tasks'] == wins and (out['model_calls'] == len(calls)) and (cost['failed_calls'] == sum((not c['ok'] for c in calls))) and (out['actual_tool_calls'] == sum((row['actual_tool_calls'] for row in rows))))
            check('summed cohort denominator ' + label, equal(out['summed_phase_cohort_wall_s'], wall))
            prompt, output = (sum((c['prompt_tokens'] for c in known)), sum((c['output_tokens'] for c in known)))
            check('new all-attempt cost ' + label, cost['known_prompt_tokens'] == prompt and cost['known_output_tokens'] == output and (cost['known_usage_calls'] == len(known)) and (cost['unknown_usage_calls'] == len(calls) - len(known)) and (cost['complete'] is (len(calls) == len(known))))
            check('new zero/null success cost ' + label, equal(cost['total_tokens_per_successful_task'], (prompt + output) / wins if wins else None) and equal(cost['model_calls_per_successful_task'], len(calls) / wins if wins else None))
            turn_groups = {}
            for index, row in enumerate(rows):
                for call in row['calls']:
                    turn = call.get('turn')
                    turn = turn if type(turn) is int and turn >= 0 else None
                    turn_groups.setdefault(turn, []).append((index, call))
            check('new turn coverage ' + label, {t['raw_turn_index'] for t in out['turn_distributions']} == turn_groups.keys() and sum((t['model_calls'] for t in out['turn_distributions'])) == len(calls))
            for turn in out['turn_distributions']:
                pairs = turn_groups[turn['raw_turn_index']]
                tcalls = [call for _, call in pairs]
                check('new conditional turn denominator ' + label + '/' + str(turn['raw_turn_index']), turn['assigned_sessions'] == len(rows) and turn['sessions_reaching_turn'] == len({i for i, _ in pairs}) and (turn['model_calls'] == len(tcalls)) and (turn['failed_model_calls'] == sum((not c['ok'] for c in tcalls))))
                for field in ('prompt_tokens', 'output_tokens', 'cached_tokens'):
                    stats = distribution([call[field] for call in tcalls])
                    measured = turn['request_lengths'][field]
                    check('new independent turn quantiles ' + label + '/' + str(turn['raw_turn_index']) + '/' + field, stats.keys() == measured.keys() and all((equal(stats[k], measured[k]) for k in stats)))
                new_turns += 1
            new_groups += 1
    expected_sources = {(name, phase, row['index']) for (name, phase), stage in stages.items() for row in stage['sessions']}
    check('every raw-projected attempt exactly once', len(all_sources) == len(set(all_sources)) == 2564 and set(all_sources) == expected_sources)
    failed_formal = [s for key, s in stages.items() if key[0] == 'session-dspark3-agent-confirm' and s['phase']['role'] == 'confirm']
    check('failed executed cohort retained alongside accepted set', len(failed_formal) == 3 and sum((len(s['sessions']) for s in failed_formal)) == 371 and all((not s['included_as_measurement'] for s in failed_formal)) and (sum((r['task_success'] is True for s in failed_formal for r in s['sessions'])) == 125))
    failed_rows = [r for s in failed_formal for r in s['sessions'] if r['terminal_reason'] == 'request_error']
    check('failed request usage preserved with ungraded primary', len(failed_rows) == 1 and failed_rows[0]['task_success'] is None and (failed_rows[0]['calls'][-1]['usage_complete'] is True) and (failed_rows[0]['calls'][-1]['output_tokens'] == 4096))
    check('full-scan zero confirmation not replaced', confirmation['started_confirmation_windows'] == 16 and confirmation['confirmed_protocol_points'] == 0 and (len(confirmation['unresolved_started_identity']) == 3))
    for name in ('final-r1.json', 'workloads-final-r1.json'):
        path = path_at(ROOT, 'summary') / name
        if str(path) in cumulative['input_ledger']:
            check('consumed final input hash ' + name, ledger[str(path)] == cumulative['input_ledger'][str(path)])
    check('all consumed input bytes unchanged', all((hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'] and Path(path).stat().st_size == info['bytes'] for path, info in ledger.items())))
    result = dict(status='PASS_INDEPENDENT_FINAL_NUMERICAL_REVIEW', utc=datetime.now(timezone.utc).isoformat(), checks=checks, inventory=expected, reused_prior_cohorts=len(old_cohorts), new_numerically_checked_groups=new_groups, new_numerically_checked_turns=new_turns, failed_c15_formal_role_attempts=371, evidence_ledger=ledger, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitations=['Old cohorts are byte-structure equal to separately reviewed r2 values; old 5625 arithmetic and 1081 provenance/concurrency checks reused.', 'New c13-c16 task/turn arithmetic independently recomputed from audited raw projections, not by importing cumulative implementation.', 'Accepted formal set excludes failed deployments. Executed c15 formal-role 371 attempts remain separately visible, not discarded.', 'Counts across heterogeneous conditions are inventory, not pooled task accuracy/throughput or capacity.', 'Full raw/SSE/task execution audits are reused; this does not certify model representativeness or production SLA.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), inventory=expected, prior_cohorts=len(old_cohorts), new_groups=new_groups, new_turns=new_turns)))


if __name__ == '__main__':
    main()
