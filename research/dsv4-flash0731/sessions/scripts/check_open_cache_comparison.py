#!/usr/bin/env python3
"""Independent numerical review of the saved c08/c09 open cache comparison.

Reuses already independently audited terminal summaries; no model/GPU calls,
no re-execution of the calculation script and no raw result modifications.
"""

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
    ledger, checks = ({}, [])

    def raw(path):
        data = path.read_bytes()
        ledger[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def load(path):
        return json.loads(raw(path))

    def check(label, condition):
        if not condition:
            raise ValueError(label)
        checks.append(label)

    def equal(a, b):
        return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    path = path_at(ROOT, 'logs/c08-c09-cache-matched-open-r1.json')
    comparison = load(path)
    check('explicit scope and candidates', comparison['status'] == 'PASS_MATCHED_OPEN_CACHE_COMPARISON' and comparison['ids'] == ['session-tp4-replay-open', 'session-dspark3-replay-open'])
    check('calculation source bytes', hashlib.sha256(raw(Path(comparison['calculation_record']))).hexdigest() == comparison['calculation_sha256'])
    for name, expected in comparison['inputs'].items():
        check('comparison input hash ' + name, hashlib.sha256(raw(Path(name))).hexdigest() == expected)
    check('exact three screen groups', len(comparison['matched_groups']) == 3 and len({g['workload_sha256'] for g in comparison['matched_groups']}) == 3)
    sources = {}
    for name, revision in zip(comparison['ids'], ['r2', 'r1']):
        cache_path = path_at(ROOT, 'logs') / f'{name}-cache-open-screen-r1.json'
        cache = load(cache_path)
        resource = load(path_at(ROOT, 'logs') / f'{name}-resources-open-screen-r1.json')
        summary = load(path_at(ROOT, 'summary') / f'{name}-terminal-{revision}.json')
        check('resource/cache input identity ' + name, resource['inputs'][str(cache_path.resolve())] == ledger[str(cache_path.resolve())])
        check('healthy independently audited summary ' + name, summary['status'] == 'PASS' and summary['experiments'][0]['terminal_success'] is True)
        sources[name] = ({r['id']: r for r in cache['phases']}, {r['phase']: r for r in resource['rows']}, {r['phase']['id']: r for r in summary['stages']})
    denominators = []
    for group, expected_rate in zip(comparison['matched_groups'], [0.15, 0.3, 0.6]):
        check('exact seed and replay ' + str(expected_rate), group['kind'] == 'replay' and group['seed'] == 202611630)
        check('candidate order ' + str(expected_rate), [p['experiment'] for p in group['points']] == comparison['ids'])
        for point in group['points']:
            name, phase = (point['experiment'], point['phase'])
            cache, resources, summaries = sources[name]
            c, r, s = (cache[phase], resources[phase], summaries[phase])
            check('embedded row identity ' + name + '/' + phase, point['resource_row'] == r and r['workload_sha256'] == c['workload_sha256'] == group['workload_sha256'])
            check('formal open identity ' + name + '/' + phase, s['included_as_measurement'] is True and s['phase']['role'] == 'screen' and (s['phase']['arrival'] == 'poisson') and (point['nominal_session_rate'] == s['phase']['rate'] == expected_rate) and (point['horizon_s'] == s['phase']['duration_s'] == 120) and (s['stability']['classification'] == 'SCREEN_ONLY'))
            check('horizon plus drain denominator ' + name + '/' + phase, equal(point['wall_s'], c['wall_s']) and equal(point['wall_s'], s['total_wall_s']) and equal(point['drain_s'], max(0, point['wall_s'] - 120)))
            check('count per cohort wall ' + name + '/' + phase, equal(r['session_rate'], r['sessions'] / point['wall_s']) and equal(r['output_token_rate'], r['output_tokens'] / point['wall_s']) and (r['sessions'] == c['sessions'] == s['summary']['sessions']))
            check('cache token arithmetic ' + name + '/' + phase, r['cached_tokens'] == c['known_cached_tokens'] and r['prompt_tokens'] == c['known_prompt_tokens'] and (r['logical_computed_tokens'] == r['prompt_tokens'] - r['cached_tokens']) and (r['output_tokens'] == c['known_output_tokens']))
            check('four-GPU integrated energy ' + name + '/' + phase, r['resource']['complete_four_gpu_window'] is True and {g['gpu'] for g in r['resource']['devices']} == {0, 1, 2, 3} and equal(r['resource']['energy_j'], c['gpu']['energy_j']) and (abs(c['gpu']['window_s'] - point['wall_s']) < 1e-06) and equal(r['resource']['mean_board_power_sum_w'], r['resource']['energy_j'] / c['gpu']['window_s']))
            for field in ('queue', 'prefill', 'decode', 'inference'):
                timing = r['server_timings'][field]
                check('counter timing mean ' + name + '/' + phase + '/' + field, equal(timing['mean_s'], timing['sum_s'] / timing['count']))
            denominators.append(dict(experiment=name, phase=phase, nominal_rate=expected_rate, measured_sessions=r['sessions'], cohort_wall_s=point['wall_s'], cohort_session_rate=r['session_rate'], drain_s=point['drain_s']))
        a, b = [p['resource_row'] for p in group['points']]
        values = {'cohort_output_token_rate': (b['output_token_rate'], a['output_token_rate']), 'cohort_session_rate': (b['session_rate'], a['session_rate']), 'session_p95': (b['session_latency_s']['p95'], a['session_latency_s']['p95']), 'ttft_text_p95': (b['ttft_text_s']['p95'], a['ttft_text_s']['p95']), 'tpot_p95': (b['tpot_output_s']['p95'], a['tpot_output_s']['p95']), 'server_prefill_mean': (b['server_timings']['prefill']['mean_s'], a['server_timings']['prefill']['mean_s']), 'server_decode_mean': (b['server_timings']['decode']['mean_s'], a['server_timings']['decode']['mean_s']), 'board_energy': (b['resource']['energy_j'], a['resource']['energy_j'])}
        check('ratio fields complete ' + str(expected_rate), values.keys() == group['candidate_over_reference'].keys())
        for field, (numerator, denominator) in values.items():
            check('independent candidate/reference ' + field + '/' + str(expected_rate), equal(group['candidate_over_reference'][field], numerator / denominator))
        check('not task scoring ' + str(expected_rate), a['successful_tasks'] is b['successful_tasks'] is None)
    check('all input bytes unchanged', all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == v for p, v in ledger.items())))
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), denominator_checks=denominators, evidence_ledger=ledger, scope='Independent CPU numerical and identity review of one c08/c09 artifact, not a generic validator or capacity audit.', limitations=['Ratios use candidate/reference; greater throughput is favorable, lower latency/energy is favorable.', 'Cohort throughput denominator includes 120-second horizon and drain; it is neither nominal offered rate nor an in-window departure rate.', 'Lower-rate similar cohort throughput can coexist with faster response because arrival-limited idle time remains in the denominator.', 'Prefill/decode counter intervals overlap across requests; they do not establish exclusive GPU kernel or PCIe communication time.', 'Reuse prior full raw-integrity and cache/runtime audits; do not repeat every large raw request or GPU sample.', 'Power means use the GPU Unix-time sampling window; cohort rates use the client monotonic duration. Their sub-microsecond rounding difference is checked separately.', 'Single-seed screen comparisons neither certify stable capacity nor provide repeated performance confidence intervals.'])
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks))))


if __name__ == '__main__':main()
