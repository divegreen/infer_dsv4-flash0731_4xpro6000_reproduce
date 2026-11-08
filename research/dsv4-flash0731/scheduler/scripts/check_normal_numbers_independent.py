#!/usr/bin/env python3
"""Independently recompute selected normal D statistics from capacity.json only.

No project analysis imports, no repeated JSONL/Prom/GPU trace audit. Earlier
terminal auditors bind those raw files. Resource sampling is outside this check.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def quantile(values,p):
    values=sorted(values)
    if not values:return None
    n=(len(values)-1)*p;lo=int(n);hi=min(lo+1,len(values)-1)
    return values[lo]*(hi-n)+values[hi]*(n-lo) if hi!=lo else values[lo]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use unique output')
    summary = load_document(args.summary)
    checks = []
    errors = []
    observations = []
    inputs = {str(args.summary.resolve()): digest(args.summary)}

    def same(actual, expected, where):
        checks.append(where)
        ok = math.isclose(actual, expected, rel_tol=1e-08, abs_tol=1e-08) if type(expected) is float else actual == expected
        if not ok:
            errors.append(dict(where=where, actual=actual, expected=expected))
    saved_performance = {}
    for entry in summary['runs']:
        source = next((Path(s['source']) for s in summary['stages'] if s['experiment'] == entry['experiment']))
        performance = load_document(source)
        inputs[str(source)] = digest(source)
        saved_performance[entry['experiment']] = performance
        stages = {s['phase_id']: s for s in summary['stages'] if s['experiment'] == entry['experiment']}
        same(set(stages), {r['phase']['id'] for r in performance['runs']}, entry['experiment'] + '.exact_phase_set')
        official = []
        for run in performance['runs']:
            ident = run['phase']['id']
            stage = stages[ident]
            role = run['phase']['role']
            rows = run['requests']
            same(stage['included_as_measurement'], role not in ('warmup', 'profile'), ident + '.role_exclusion')
            if role in ('warmup', 'profile'):
                same(stage['ranking_eligible'], False, ident + '.no_warmup_ranking')
                continue
            official.extend(rows)
            wall = run['summary']['total_wall_s']
            window = run['phase']['duration_s']
            same(stage['total_wall_s'], wall, ident + '.cohort_wall')
            same(stage['drain_s'], max(0.0, wall - window), ident + '.drain')
            same(wall >= max((r['end_offset_s'] for r in rows)), True, ident + '.wall_covers_all_completions')
            good = [r for r in rows if r['ok']]
            same(stage['stats']['requests'], len(rows), ident + '.requests')
            same(stage['stats']['successful'], len(good), ident + '.successes')
            output = sum((r['usage']['completion_tokens'] for r in good))
            same(stage['successful_output_tokens'], output, ident + '.actual_tokens')
            same(stage['throughput']['cohort_output_tokens_s'], output / wall, ident + '.token_throughput')
            same(stage['throughput']['cohort_requests_s'], len(good) / wall, ident + '.request_throughput')
            eligible_goodput = 0
            for r in good:
                lim = 1.0 if r['input_tokens'] <= 1024 else 3.0 if r['input_tokens'] <= 4096 else 8.0 if r['input_tokens'] <= 16384 else None
                if lim is not None:
                    eligible_goodput += r['ttft_s'] + r['dispatch_lag_s'] <= lim and (r['e2e_s'] - r['ttft_s']) / (r['usage']['completion_tokens'] - 1) <= 0.05 and (r['e2e_s'] + r['dispatch_lag_s'] <= lim + 0.05 * (r['usage']['completion_tokens'] - 1))
            same(stage['throughput']['cohort_slo_goodput_requests_s'], eligible_goodput / wall, ident + '.overall_goodput')
            energy = stage['gpu'].get('energy_j')
            if energy is not None:
                same(stage['gpu']['successful_output_tokens_per_j'], output / energy, ident + '.resource_token_per_j')
                same(stage['gpu']['slo_output_tokens_per_j'], stage['slo_output_tokens'] / energy, ident + '.resource_slo_token_per_j')
                same(stage['gpu']['window_s'], run['start_unix'] + wall - run['start_unix'], ident + '.resource_monotonic_cohort_window')
                same(energy, sum((d['measurements']['power_w']['integral'] for d in stage['gpu']['devices'])), ident + '.four_board_energy_sum')
            recomputed = {}
            for shape in stage['by_shape']:
                cohort = [r for r in rows if r['shape_name'] == shape]
                success = [r for r in cohort if r['ok']]
                stats = stage['by_shape'][shape]
                ttft = [r['ttft_s'] + r['dispatch_lag_s'] for r in success]
                e2e = [r['e2e_s'] + r['dispatch_lag_s'] for r in success]
                tpot = [(r['e2e_s'] - r['ttft_s']) / (r['usage']['completion_tokens'] - 1) for r in success]
                same(stats['requests'], len(cohort), ident + '.' + shape + '.count')
                for metric, values in [('scheduled_ttft_s', ttft), ('scheduled_e2e_s', e2e), ('tpot_s', tpot)]:
                    for name, p in [('p50', 0.5), ('p95', 0.95), ('p99', 0.99)]:
                        same(stats['metrics'][metric][name], quantile(values, p), ident + '.' + shape + '.' + metric + '.' + name)
                passed = 0
                eligible = 0
                for r in cohort:
                    limit = 1.0 if r['input_tokens'] <= 1024 else 3.0 if r['input_tokens'] <= 4096 else 8.0 if r['input_tokens'] <= 16384 else None
                    if limit is None:
                        continue
                    eligible += 1
                    if r['ok']:
                        out = r['usage']['completion_tokens']
                        post = (r['e2e_s'] - r['ttft_s']) / (out - 1)
                        passed += r['ttft_s'] + r['dispatch_lag_s'] <= limit and post <= 0.05 and (r['e2e_s'] + r['dispatch_lag_s'] <= limit + 0.05 * (out - 1))
                same(stats['slo_eligible_requests'], eligible, ident + '.' + shape + '.slo_eligible')
                same(stats['slo_passes'], passed, ident + '.' + shape + '.slo_passes')
                same(stats['cohort_slo_goodput_requests_s'], passed / wall, ident + '.' + shape + '.goodput')
                if shape == 'p512_o128':
                    recomputed = dict(short_requests=len(cohort), short_slo_passes=passed, short_goodput=passed / wall, short_ttft_p95=quantile(ttft, 0.95), short_tpot_p95=quantile(tpot, 0.95))
            if run['phase']['mode'] == 'poisson':
                bins = []
                for i in range(3):
                    left = window * 0.25 + i * window * 0.25
                    right = left + window * 0.25
                    arrivals = sum((left <= r['planned_offset_s'] < right for r in rows))
                    completions = sum((r['ok'] and left <= r['end_offset_s'] < right for r in rows))
                    mean = sum((max(0.0, min(r['end_offset_s'], right) - max(r['dispatch_offset_s'], left)) for r in rows if r['admitted'])) / (right - left)
                    saved = stage['stability']['analysis_bins'][i]
                    for k, v in dict(arrivals=arrivals, successful_completions=completions, client_mean_outstanding=mean).items():
                        same(saved[k], v, ident + '.bin' + str(i) + '.' + k)
                    waits = [sum((v for k, v in m.get('values', {}).items() if k.split('{')[0] == 'vllm:num_requests_waiting' and isinstance(v, (float, int)) and (not isinstance(v, bool)) and math.isfinite(v))) for m in run['metric_samples'] if m.get('http_status') == 200 and (not m.get('error')) and (left <= m['offset_s'] < right) and any((k.split('{')[0] == 'vllm:num_requests_waiting' for k in m.get('values', {})))]
                    waiting = sum(waits) / len(waits) if waits else None
                    same(saved['server_mean_waiting'], waiting, ident + '.bin' + str(i) + '.server_waiting')
                    same(saved['server_sample_count'], len(waits), ident + '.bin' + str(i) + '.server_samples')
                    bins.append((arrivals, completions, mean, waiting, len(waits)))
                arrivals = sum((x[0] for x in bins))
                completed = sum((x[1] for x in bins))
                tolerance = max(2.0, 0.05 * arrivals)
                growth = bins[-1][2] - bins[0][2]
                same(stage['stability']['completion_deficit_requests'], arrivals - completed, ident + '.deficit')
                same(stage['stability']['backlog_tolerance_requests'], tolerance, ident + '.tolerance')
                same(stage['stability']['client_mean_outstanding_increase'], growth, ident + '.client_growth')
                server_growth = bins[-1][3] - bins[0][3] if all((x[3] is not None for x in bins)) else None
                coverage = min(1.0, sum((x[4] for x in bins)) * performance['plan']['metrics_interval_s'] / (window * 0.75))
                lag = quantile([r['dispatch_lag_s'] for r in rows if r['admitted']], 0.95)
                enough = all((r['ok'] and (not r['censored']) and (not r['timed_out']) for r in rows)) and arrivals >= 20 and (coverage >= 0.8) and (server_growth is not None) and (lag is not None) and (lag <= 0.1)
                growing = arrivals - completed > tolerance and growth > tolerance and (bins[0][2] < bins[1][2] < bins[2][2])
                stable = arrivals - completed <= tolerance and growth <= tolerance and (server_growth is not None) and (server_growth <= tolerance)
                verdict = 'INCONCLUSIVE' if not enough else 'GROWING_BACKLOG' if growing else 'STABLE_IN_WINDOW' if stable else 'INCONCLUSIVE'
                same(stage['stability']['server_sample_coverage_fraction'], coverage, ident + '.server_coverage')
                same(stage['stability']['server_mean_waiting_increase'], server_growth, ident + '.server_growth')
                same(stage['stability']['classification'], verdict if window >= 180 else 'SCREEN_ONLY', ident + '.fixed_finite_capacity_classification')
            observations.append(dict(experiment=entry['experiment'], phase=ident, requests=len(rows), wall_s=wall, token_throughput=output / wall, **recomputed))
        quality_path = source.parent / 'quality.json'
        if quality_path.exists():
            q = load_document(quality_path)
            inputs[str(quality_path)] = digest(quality_path)
            same(entry['quality']['requests'], len(q['runs']), entry['experiment'] + '.quality_count')
            same(entry['quality']['quality_passed'], sum((r['quality']['passed'] for r in q['runs'])), entry['experiment'] + '.quality_score')
            same(entry['quality']['excluded_from_performance'], True, entry['experiment'] + '.quality_excluded')
            same(entry['quality']['quality_scores_gate_performance'], False, entry['experiment'] + '.score_nongating')
            same(datetime.fromisoformat(q['started_at']) >= datetime.fromisoformat(performance['finished_utc']), True, entry['experiment'] + '.quality_after_performance')
        same(sum((s['offered_requests'] for s in stages.values() if s['included_as_measurement'])), len(official), entry['experiment'] + '.only_formal_performance_count')
    paired = []
    for pair in summary['paired_comparisons']:
        if len(pair['members']) == 1:
            same(pair['status'], 'SINGLE_OBSERVATION', 'no_claim_from_one_policy')
            continue
        same(pair['status'], 'PASS', 'matched_pair_status')
        a = next((s for s in summary['stages'] if s['experiment'] == pair['reference'] and s['comparison_id'] == pair['comparison_id']))
        b = next((s for s in summary['stages'] if s['experiment'] == pair['candidate'] and s['comparison_id'] == pair['comparison_id']))

        def selected_requests(stage):
            run = next((r for r in saved_performance[stage['experiment']]['runs'] if r['phase']['id'] == stage['phase_id']))
            return [{k: r[k] for k in ('index', 'planned_offset_s', 'input_tokens', 'requested_output_tokens', 'prompt_sha256', 'seed', 'ignore_eos')} for r in run['requests']]
        same(selected_requests(a), selected_requests(b), pair['comparison_id'] + '.actual_prompt_arrival_identity')
        same(a['matched_conditions'], b['matched_conditions'], pair['comparison_id'] + '.non_treatment_runtime')
        same(a['seed'], b['seed'], pair['comparison_id'] + '.seed')

        def scalar(stage, key):
            if key == 'wall_s':
                return stage['total_wall_s']
            if key == 'drain_s':
                return stage['drain_s']
            if key.startswith('cohort_'):
                return stage['throughput'][key]
            if key == 'gpu_energy_j':
                return stage['gpu'].get('energy_j')
            if key == 'gpu_output_tokens_per_j':
                return stage['gpu'].get('successful_output_tokens_per_j')
            if key == 'preemptions':
                return stage['preemptions']
            if key == 'client_peak_outstanding':
                return stage['client_peak_dispatched_outstanding']
            if key.startswith('p'):
                shape, field, *rest = key.split('.')
                if shape in stage['by_shape']:
                    item = stage['by_shape'][shape]
                    return item['metrics'][field][rest[0]] if rest else item[field]
            metric, stat = key.rsplit('.', 1)
            return stage['raw_gauges']['aggregate'][metric][stat]
        for key, delta in pair['deltas'].items():
            av, bv = (scalar(a, key), scalar(b, key))
            same(delta['observe'], av, pair['comparison_id'] + '.' + key + '.observe')
            same(delta['fairshare'], bv, pair['comparison_id'] + '.' + key + '.fairshare')
            same(delta['difference'], bv - av if av is not None and bv is not None else None, pair['comparison_id'] + '.' + key + '.difference')
            expected = (bv - av) / av * 100 if av is not None and bv is not None and (av != 0) else None
            same(delta['percent_change'], expected, pair['comparison_id'] + '.' + key + '.percent_change')
        paired.append(dict(comparison_id=pair['comparison_id'], reference=pair['reference'], candidate=pair['candidate'], delta_count=len(pair['deltas']), short_slo=[a['short512']['slo_passes'], b['short512']['slo_passes']], short_count=[a['short512']['requests'], b['short512']['requests']], classifications=[a['stability']['classification'], b['stability']['classification']]))
    report = dict(status='PASS' if not errors else 'FAIL', checks=len(checks), errors=errors, observations=observations, paired=paired, scope='Independent formulas from saved capacity rows; existing raw/Prom/source audit reused, no independent GPU integration or new capacity certification', input_sha256=inputs, reviewer_sha256=digest(Path(__file__)))
    with open_output(args.output, 'x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], checks=report['checks'], errors=errors, observations=observations)))
    return 0 if not errors else 1

if __name__=='__main__':raise SystemExit(main())
