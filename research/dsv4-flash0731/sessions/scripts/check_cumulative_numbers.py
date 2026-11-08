#!/usr/bin/env python3
"""Independent cumulative arithmetic using previously audited raw projections.

No cumulative implementation import or GPU/task calls. C12 replay inventory
uses its independent audit; c03-c11 task/turn arithmetic uses frozen descriptors.
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
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cumulative', type=Path, required=True)
    p.add_argument('--pilots', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    checks = []
    ledger = {}

    def load(path):
        raw = path.read_bytes()
        ledger[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def need(label, value):
        if not value:
            raise ValueError(label)
        checks.append(label)

    def close(x, y):
        return x is y if x is None or y is None else math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)

    def stats(values):
        xs = sorted((v for v in values if type(v) in (int, float) and math.isfinite(v)))

        def q(at):
            if not xs:
                return None
            x = (len(xs) - 1) * at
            i = math.floor(x)
            j = math.ceil(x)
            return xs[i] + (xs[j] - xs[i]) * (x - i)
        return dict(n=len(xs), missing=len(values) - len(xs), min=min(xs) if xs else None, max=max(xs) if xs else None, p50=q(0.5), p95=q(0.95), p99=q(0.99), mean=sum(xs) / len(xs) if xs else None)
    cumulative = load(a.cumulative)
    pilots = load(a.pilots)
    need('actual explicit snapshots PASS', cumulative['status'] == pilots['status'] == 'PASS')
    descs = [load(path_at(ROOT, 'summary') / name) for name in ('screen-c03-c07-workloads-r1.json', 'open-screen-c08-c11-workloads-r1.json')]
    stages = {(s['experiment'], s['phase']['id']): s for d in descs for s in d['stages']}
    need('prior projections have unique phase identities', len(stages) == sum((len(d['stages']) for d in descs)))
    audited = load(path_at(ROOT, 'logs/audit-session-tp4-replay-confirm-r1.json'))['experiments'][0]
    need('exact cumulative inventory', cumulative['inventory']['all_sessions'] == 1351 and cumulative['inventory']['formal_sessions'] == 1314 and (cumulative['inventory']['formal_agent_attempts'] == 392) and (cumulative['inventory']['formal_agent_strict_successes'] == 133) and (cumulative['inventory']['formal_agent_model_calls'] == 1892))
    expected_source = {(exp, phase, r['index']) for (exp, phase), s in stages.items() for r in s['sessions']}
    actual_source = []
    groups_checked = 0
    task_groups = 0
    turns_checked = 0
    for cohort in cumulative['cohorts']:
        name = cohort['experiment']
        source_ids = [(name, phase) for phase in cohort['source_phases']]
        if name == 'session-tp4-replay-confirm':
            phases = {f['id']: f for f in audited['phases']}
            need('c12 replay no task quality ' + cohort['cohort_sha256'], cohort['treatment']['kind'] == 'replay' and cohort['summary']['quality']['strict_successes'] is None and (cohort['summary']['quality']['applicability'] == 'NOT_APPLICABLE_REPLAY_ONLY') and (cohort['summary']['sessions'] == sum((phases[p]['sessions'] for p in cohort['source_phases']))))
            continue
        need('source phases present ' + cohort['cohort_sha256'], all((k in stages for k in source_ids)))
        source = [stages[k] for k in source_ids]
        rows = [r for s in source for r in s['sessions']]
        need('complete treatment and role preserved ' + cohort['cohort_sha256'], all(({k: v for k, v in s['phase'].items() if k not in ('id', 'seed')} == cohort['treatment'] and s['included_as_measurement'] is cohort['included_as_measurement'] for s in source)))
        identities = [(name, s['phase']['id'], r['index']) for s in source for r in s['sessions']]
        actual_source += identities
        need('exact source attempt mapping ' + cohort['cohort_sha256'], identities == [(r['experiment'], r['phase'], r['session_index']) for r in cohort['source_sessions']])
        wall = sum((s['summary']['cohort_wall_s'] for s in source))
        subsets = [('ALL', rows, cohort['summary'])] + [(task, [r for r in rows if r['task'] == task], out) for task, out in cohort['by_task'].items()]
        for label, selected, out in subsets:
            context = cohort['cohort_sha256'] + '/' + label
            calls = [call for row in selected for call in row['calls']]
            known = [c for c in calls if c['usage_complete']]
            wins = sum((r['task_success'] is True for r in selected))
            is_agent = cohort['treatment']['kind'] == 'agent'
            cost = out['all_attempt_cost']
            need('attempts and calls ' + context, out['sessions'] == len(selected) and out['model_calls'] == len(calls) and (out['successful_tasks'] == (wins if is_agent else None)) and (out['actual_tool_calls'] == sum((r['actual_tool_calls'] for r in selected))))
            need('cohort wall is source wall sum ' + context, close(out['summed_phase_cohort_wall_s'], wall))
            pin = sum((c['prompt_tokens'] for c in known))
            pout = sum((c['output_tokens'] for c in known))
            need('all known attempt costs ' + context, cost['known_prompt_tokens'] == pin and cost['known_output_tokens'] == pout and (cost['known_usage_calls'] == len(known)) and (cost['unknown_usage_calls'] == len(calls) - len(known)) and (cost['complete'] is (len(known) == len(calls))) and (cost['failed_calls'] == sum((not c['ok'] for c in calls))))
            need('zero/null or all-attempt per-success cost ' + context, close(cost['total_tokens_per_successful_task'], (pin + pout) / wins if is_agent and wins else None) and close(cost['model_calls_per_successful_task'], len(calls) / wins if is_agent and wins else None))
            turn_map = {}
            for index, row in enumerate(selected):
                for call in row['calls']:
                    t = call.get('turn')
                    t = t if type(t) is int and t >= 0 else None
                    turn_map.setdefault(t, []).append((index, call))
            need('every call in exactly one turn bucket ' + context, sum((t['model_calls'] for t in out['turn_distributions'])) == len(calls) and {t['raw_turn_index'] for t in out['turn_distributions']} == set(turn_map))
            for turn in out['turn_distributions']:
                pairs = turn_map[turn['raw_turn_index']]
                tc = [c for i, c in pairs]
                need('turn denominator ' + context + '/' + str(turn['raw_turn_index']), turn['assigned_sessions'] == len(selected) and turn['sessions_reaching_turn'] == len({i for i, c in pairs}) and (turn['model_calls'] == len(tc)) and (turn['failed_model_calls'] == sum((not c['ok'] for c in tc))) and (turn['usage_known_calls'] == sum((c['usage_complete'] for c in tc))))
                for field in ['prompt_tokens', 'output_tokens', 'cached_tokens']:
                    expected = stats([c[field] for c in tc])
                    observed = turn['request_lengths'][field]
                    need('independent turn quantile ' + context + '/' + str(turn['raw_turn_index']) + '/' + field, expected.keys() == observed.keys() and all((close(expected[k], observed[k]) for k in expected)))
                turns_checked += 1
            if is_agent:
                q = out['quality']
                need('reference separate from strict ' + context, q['assigned_attempts'] == len(selected) and q['strict_successes'] == wins and (q['reference_covered_attempts'] + q['reference_missing_attempts'] == len(selected)))
            task_groups += 1
        groups_checked += 1
    need('no source attempt duplicate or omission', len(actual_source) == len(set(actual_source)) and set(actual_source) == expected_source)
    inv = pilots['inventory']
    need('real pilots stay nonformal', inv['all_sessions'] == 21 and inv['formal_sessions'] == 0 and (inv['formal_agent_attempts'] == 0) and all((not c['included_as_measurement'] for c in pilots['cohorts'])))
    failed = [c for c in pilots['cohorts'] if c['experiment'] == 'session-tp4-apc-pilot' and c['treatment']['kind'] == 'agent']
    need('failed pilot unknown usage and zero-success null retained', len(failed) == 1 and failed[0]['summary']['successful_tasks'] == 0 and (not failed[0]['summary']['all_attempt_cost']['complete']) and (failed[0]['summary']['all_attempt_cost']['unknown_usage_calls'] > 0) and (failed[0]['summary']['all_attempt_cost']['total_tokens_per_successful_task'] is None))
    need('input artifacts unchanged', all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == v for p, v in ledger.items())))
    result = dict(status='PASS_NUMERICAL_REVIEW', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, checked_projected_cohorts=groups_checked, checked_task_subgroups=task_groups, checked_turn_subgroups=turns_checked, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), evidence_ledger=ledger, scope='Independent primitive counts/cost/quantiles from previously audited raw projections; no cumulative implementation imported.', limitations=['C12 fixed replay uses independent terminal inventory only; all c03-c11 aggregate/task/turn arithmetic is recomputed.', 'Previous full raw/SSE/workspace/metadata audits are reused, not repeated.', 'Source-reference binding and malformed-reference rejection receive separate CPU review; numerical PASS does not resolve those checks.', 'Inventory across heterogeneous experiments is not pooled task accuracy, throughput ranking or capacity.'])
    with open_output(a.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), cohorts=groups_checked, turns=turns_checked)))


if __name__=='__main__':main()
