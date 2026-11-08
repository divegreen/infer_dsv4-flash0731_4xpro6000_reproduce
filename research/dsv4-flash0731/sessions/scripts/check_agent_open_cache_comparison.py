#!/usr/bin/env python3
"""Review c10/c11 Agent comparison numerics without assuming equal token work."""

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
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    checks = []
    ledger = {}
    descriptive = []

    def read(path):
        raw = path.read_bytes()
        ledger[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return raw

    def load(path):
        return json.loads(read(path))

    def need(label, value):
        if not value:
            raise ValueError(label)
        checks.append(label)

    def close(a, b):
        return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    x = load(path_at(ROOT, 'logs/c10-c11-cache-matched-open-r1.json'))
    need('actual paired Agent artifact', x['ids'] == ['session-tp4-agent-open', 'session-dspark3-agent-open'] and x['status'] == 'PASS_MATCHED_OPEN_CACHE_COMPARISON' and (len(x['matched_groups']) == 3))
    need('frozen calculation reused', hashlib.sha256(read(Path(x['calculation_record']))).hexdigest() == x['calculation_sha256'] == '8a2854bba1db6160a967de8b31943c1904dc83bcf38428fe867d39038a6d4f43')
    for path, digest in x['inputs'].items():
        need('exact input ' + path, hashlib.sha256(read(Path(path))).hexdigest() == digest)
    datasets = {}
    for name in x['ids']:
        cache_path = path_at(ROOT, 'logs') / f'{name}-cache-open-screen-r1.json'
        cache = load(cache_path)
        resource = load(path_at(ROOT, 'logs') / f'{name}-resources-open-screen-r1.json')
        summary = load(path_at(ROOT, 'summary') / f'{name}-terminal-r1.json')
        need('resource source ' + name, resource['inputs'][str(cache_path.resolve())] == ledger[str(cache_path.resolve())])
        datasets[name] = ({r['id']: r for r in cache['phases']}, {r['phase']: r for r in resource['rows']}, {r['phase']['id']: r for r in summary['stages']})
    for group, rate in zip(x['matched_groups'], [0.15, 0.3, 0.6]):
        need('only planned Agent matching ' + str(rate), group['kind'] == 'agent' and group['seed'] == 202611630 and (group['comparison_scope'] == 'Matched planned task/arrival/budget cohort; actual work and task grades can differ'))
        need('candidate ordering ' + str(rate), [p['experiment'] for p in group['points']] == x['ids'])
        pair = []
        for point in group['points']:
            name, label = (point['experiment'], point['phase'])
            ca, re, su = datasets[name]
            c, r, s = (ca[label], re[label], su[label])
            need('embedded resource identity ' + name + '/' + label, point['resource_row'] == r and group['workload_sha256'] == c['workload_sha256'] == r['workload_sha256'])
            need('excluded warmup and correct open role ' + name + '/' + label, s['included_as_measurement'] is True and s['phase']['role'] == 'screen' and (s['phase']['arrival'] == 'poisson') and (s['phase']['rate'] == point['nominal_session_rate'] == rate) and (s['phase']['duration_s'] == point['horizon_s'] == 120) and (s['stability']['classification'] == 'SCREEN_ONLY'))
            need('full horizon plus drain ' + name + '/' + label, close(point['wall_s'], c['wall_s']) and close(point['wall_s'], s['total_wall_s']) and close(point['drain_s'], max(0, point['wall_s'] - 120)))
            need('all attempts denominator and strict grade ' + name + '/' + label, r['sessions'] == s['summary']['sessions'] and r['successful_tasks'] == s['summary']['successful_tasks'] and close(r['session_rate'], r['sessions'] / point['wall_s']) and close(r['successful_task_rate'], r['successful_tasks'] / point['wall_s']) and close(s['summary']['task_success_rate'], r['successful_tasks'] / r['sessions']))
            need('actual output and prompt cost ' + name + '/' + label, r['prompt_tokens'] == c['known_prompt_tokens'] and r['output_tokens'] == c['known_output_tokens'] == s['summary']['output_tokens'] and (r['logical_computed_tokens'] == r['prompt_tokens'] - r['cached_tokens']) and close(r['output_token_rate'], r['output_tokens'] / point['wall_s']))
            need('scheduled latency is not user-visible latency ' + name + '/' + label, r['session_latency_s'] == c['session_latency_s'] and all((close(r['session_latency_s'][k], v) for k, v in s['summary']['session_latency_s'].items())))
            need('whole four-GPU sampled energy ' + name + '/' + label, r['resource']['complete_four_gpu_window'] is True and {g['gpu'] for g in r['resource']['devices']} == {0, 1, 2, 3} and close(r['resource']['energy_j'], c['gpu']['energy_j']) and close(r['resource']['mean_board_power_sum_w'], r['resource']['energy_j'] / c['gpu']['window_s']))
            for field in ['queue', 'prefill', 'decode', 'inference']:
                timing = r['server_timings'][field]
                need('counter interval mean ' + name + '/' + label + '/' + field, close(timing['mean_s'], timing['sum_s'] / timing['count']))
            pair.append(r)
            descriptive.append(dict(experiment=name, rate=rate, sessions=r['sessions'], strict_successes=r['successful_tasks'], model_calls=r['requests'], prompt_tokens=r['prompt_tokens'], output_tokens=r['output_tokens'], wall_s=point['wall_s'], scheduled_p95_s=r['session_latency_s']['p95'], visible_p95_s=s['summary']['user_visible_latency_s']['p95']))
        left, right = pair
        pairs = {'cohort_output_token_rate': (right['output_token_rate'], left['output_token_rate']), 'cohort_session_rate': (right['session_rate'], left['session_rate']), 'session_p95': (right['session_latency_s']['p95'], left['session_latency_s']['p95']), 'ttft_text_p95': (right['ttft_text_s']['p95'], left['ttft_text_s']['p95']), 'tpot_p95': (right['tpot_output_s']['p95'], left['tpot_output_s']['p95']), 'server_prefill_mean': (right['server_timings']['prefill']['mean_s'], left['server_timings']['prefill']['mean_s']), 'server_decode_mean': (right['server_timings']['decode']['mean_s'], left['server_timings']['decode']['mean_s']), 'board_energy': (right['resource']['energy_j'], left['resource']['energy_j'])}
        need('all ratios present ' + str(rate), pairs.keys() == group['candidate_over_reference'].keys())
        for field, (num, den) in pairs.items():
            need('candidate/reference ' + field + '/' + str(rate), close(group['candidate_over_reference'][field], num / den))
    need('all bytes unchanged', all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == v for p, v in ledger.items())))
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), evidence_ledger=ledger, actual_work=descriptive, scope='Independent CPU identity/numerical review of c10/c11 comparison; prior full raw/runtime audits reused.', limitations=['Matched tasks/arrivals/budgets do not establish equal dynamic token work, calls, output text or quality equivalence.', 'Scheduled session latency includes queue, tools and grading; user-visible latency is reported separately.', 'Cohort rates include horizon plus drain and all attempted tasks, not successful-only cost.', 'The comparison is single-seed screen evidence, not stable capacity or a pure kernel/communication attribution.', 'Measurement-phase JIT and after-phase cleanup warnings remain recorded by the separate runtime audit.'])
    with open_output(a.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks))))


if __name__=='__main__':main()
