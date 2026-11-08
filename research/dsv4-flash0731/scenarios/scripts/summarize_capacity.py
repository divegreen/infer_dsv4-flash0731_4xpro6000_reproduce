#!/usr/bin/env python3
"""CPU-only, independently recomputed Stage A capacity evidence.

Reads terminal capacity reports; never contacts a server or changes raw results.
Finite-window stability rules are versioned below and in capacity-methods.md.
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
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('historical_capacity_metrics', ROOT/'scripts/summarize_scenarios.py')
historical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(historical)
RULES = {
    'version': 'finite-capacity-v1',
    'min_confirmation_duration_s': 180.,
    'min_distinct_confirmation_seeds': 2,
    'analysis_start_fraction': .25,
    'analysis_bins': 3,
    'min_analysis_arrivals': 20,
    'max_dispatch_lag_p95_s': .1,
    'min_server_sample_coverage_fraction': .8,
    'backlog_tolerance': 'max(2 requests, 0.05 * arrivals in the final 75% of the window)',
    'stable': 'No request failure/timeout/censor; all rows complete; adequate server samples; completion deficit and first-to-last client/server mean queue increases <= tolerance.',
    'growing': 'Completion deficit > tolerance AND client outstanding means increase in both steps and by > tolerance overall.',
    'inconclusive': 'Anything between these rules or insufficient measurement evidence.',
    'slo': 'Per applicable input tier: P95 normalized TTFT<=1, P95 normalized E2E<=1, P95 TPOT<=50ms; no P99 or joint-rate gate.',
    'qualification': 'At least two long windows with different seeds; every long window at that architecture/workload/rate must pass stability. SLO capacity additionally requires every such window to pass P95 SLO.',
    'not_claimed': 'No finite run establishes a production SLA or an exact mathematical maximum.',
}
METRICS = ('ttft_s', 'scheduled_ttft_s', 'tpot_s', 'e2e_s', 'scheduled_e2e_s',
           'dispatch_lag_s', 'ttft_ratio', 'e2e_ratio')


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, ledger):
    raw = path.read_bytes()
    ledger[str(path.resolve())] = {'sha256': digest(raw), 'bytes': len(raw)}
    return json.loads(raw)


def dist(values):
    values = [float(v) for v in values if finite(v)]
    return {'n': len(values), 'p50': historical.percentile(values, .50),
            'p95': historical.percentile(values, .95), 'p99': historical.percentile(values, .99),
            'max': max(values) if values else None}


def ratio(a, b):
    return a/b if b and b > 0 else None


def tier(tokens):
    for upper, name, limit in ((1024, 'le1k', 1.), (4096, 'le4k', 3.), (16384, 'le16k', 8.)):
        if tokens <= upper:
            return name, limit
    return 'gt16k', None


def recompute_row(original, mode):
    row = dict(original)
    require(type(row.get('input_tokens')) is int and row['input_tokens'] > 0, 'Invalid input tokens')
    require(finite(row.get('planned_offset_s')) and row['planned_offset_s'] >= 0, 'Invalid planned time')
    require(finite(row.get('end_offset_s')) and row['end_offset_s'] >= 0, 'Missing request terminal time')
    if row.get('admitted'):
        require(finite(row.get('dispatch_offset_s')) and row['dispatch_offset_s'] >= 0, 'Invalid dispatch time')
        require(row['end_offset_s'] >= row['dispatch_offset_s'], 'Negative request elapsed time')
        expected_lag = max(0., row['dispatch_offset_s'] - row['planned_offset_s'])
        require(math.isclose(row.get('dispatch_lag_s', -1), expected_lag, abs_tol=1e-7), 'Dispatch lag mismatch')
    if row.get('ok'):
        require(row.get('admitted') and not row.get('censored') and not row.get('timed_out'), 'False success state')
        require(row.get('http_status') == 200 and row.get('done_received') and not row.get('error'), 'False protocol success')
        usage = row.get('usage') or {}
        p, o = usage.get('prompt_tokens'), usage.get('completion_tokens')
        require(type(p) is int and p == row['input_tokens'] and type(o) is int and o > 0, 'Invalid success usage')
        require(usage.get('total_tokens', p+o) == p+o, 'Total token usage mismatch')
        require(o <= row['requested_output_tokens'], 'Actual output exceeds budget')
        if row.get('ignore_eos', True):
            require(o == row['requested_output_tokens'], 'Controlled output length mismatch')
            require(row.get('finish_reason') == 'length', 'Controlled output did not finish by length')
        else:
            require(row.get('finish_reason') in ('stop', 'length'), 'Invalid finish reason')
        require(bool(row.get('full_text')), 'Successful request has no text')
        require(digest(row['full_text'].encode()) == row.get('text_sha256'), 'Output text hash mismatch')
        require(finite(row.get('ttft_s')) and finite(row.get('e2e_s')) and 0 <= row['ttft_s'] <= row['e2e_s'], 'Invalid successful latency')
        row['tpot_s'] = (row['e2e_s']-row['ttft_s'])/(o-1) if o > 1 else None
        for key in ('ttft_s', 'e2e_s'):
            row['scheduled_'+key] = row['dispatch_lag_s']+row[key]
    name, limit = tier(row['input_tokens'])
    row['input_tier'], row['slo_applicable'] = name, limit is not None
    row['ttft_ratio'] = row['e2e_ratio'] = None
    row['slo_pass'] = False if limit is not None else None
    row['legacy_slo_pass'] = False
    if row.get('ok'):
        o = row['usage']['completion_tokens']
        basis = 'scheduled_' if mode in ('poisson', 'burst') else ''
        ttft, e2e = row[basis+'ttft_s'], row[basis+'e2e_s']
        token_pass = o == 1 or row['tpot_s'] <= .05
        row['legacy_slo_pass'] = ttft <= 1 and e2e <= 10 and token_pass
        if limit is not None:
            row['ttft_ratio'] = ttft/limit
            row['e2e_ratio'] = e2e/(limit+.05*(o-1))
            row['slo_pass'] = row['ttft_ratio'] <= 1 and row['e2e_ratio'] <= 1 and token_pass
    return row


def group(rows):
    ok = [r for r in rows if r.get('ok')]
    eligible = [r for r in rows if r['slo_applicable']]
    applicable_ok = [r for r in ok if r['slo_applicable']]
    metrics = {k: dist([r.get(k) for r in ok]) for k in METRICS}
    p95 = None
    if applicable_ok:
        ttft, e2e = dist([r['ttft_ratio'] for r in applicable_ok])['p95'], dist([r['e2e_ratio'] for r in applicable_ok])['p95']
        tpot = dist([r['tpot_s'] for r in applicable_ok])['p95']
        p95 = ttft <= 1 and e2e <= 1 and (tpot is None or tpot <= .05)
    passed = sum(r['slo_pass'] is True for r in rows)
    return {'requests': len(rows), 'successful': len(ok), 'failed': sum(r.get('admitted') and not r.get('ok') for r in rows),
            'censored': sum(bool(r.get('censored')) for r in rows), 'timeouts': sum(bool(r.get('timed_out')) for r in rows),
            'slo_eligible_requests': len(eligible), 'slo_passes': passed, 'slo_joint_pass_rate': ratio(passed, len(eligible)),
            'legacy_slo_passes': sum(r['legacy_slo_pass'] for r in rows),
            'legacy_slo_joint_pass_rate': ratio(sum(r['legacy_slo_pass'] for r in rows), len(rows)),
            'overall_ratio_p95_slo_pass': p95, 'metrics': metrics}


def workload_key(phase):
    shapes = phase['shape_mix']
    total = sum(s.get('weight', 1) for s in shapes)
    normalized = sorted((s['name'], s['input_tokens'], s['output_tokens'], s.get('weight', 1)/total) for s in shapes)
    descriptor = {'shape_mix': normalized, 'ignore_eos': phase.get('ignore_eos', True)}
    return digest(json.dumps(descriptor, sort_keys=True).encode()), descriptor


def completion_context(report, supervisor, experiment):
    """Qualification needs the whole owned run to be terminal and successful.

    A client can finish before the supervisor cleans up. Earlier phase data from
    a failed run remains descriptive but must not silently certify capacity.
    """
    reasons = []
    if report.get('status') != 'PASSED':
        reasons.append('Capacity client is not PASSED.')
    if report.get('active_phase'):
        reasons.append('Capacity client has an unresolved active_phase.')
    planned = [p.get('id') for p in report.get('plan', {}).get('phases', [])]
    completed = [r.get('phase', {}).get('id') for r in report.get('runs', [])]
    if not planned or completed != planned:
        reasons.append('Completed phase sequence does not equal the complete plan.')
    if not (supervisor and supervisor.get('exit_code') == 0 and supervisor.get('timed_out') is False
            and not supervisor.get('supervisor_error') and supervisor.get('end_utc')):
        reasons.append('Supervisor is missing or not terminal exit0 without timeout/error.')
    if not (experiment and experiment.get('status') == 'PASS' and experiment.get('elapsed_seconds') is not None):
        reasons.append('Experiment is missing or not terminal PASS.')
    return {'eligible_for_capacity_qualification': not reasons, 'reasons': reasons,
            'client_status': report.get('status'), 'supervisor_exit_code': (supervisor or {}).get('exit_code'),
            'experiment_status': (experiment or {}).get('status'),
            'planned_phase_ids': planned, 'completed_phase_ids': completed}


def unresolved_attempts(report, architecture, deployment_key, source, context):
    """Only begun attempts block: never turn future unstarted plans into failures."""
    if context['eligible_for_capacity_qualification']:
        return []
    begun = {r.get('phase', {}).get('id') for r in report.get('runs', [])}
    if report.get('active_phase'):
        begun.add(report['active_phase'])
    known = {p.get('id'): p for p in report.get('plan', {}).get('phases', [])}
    for run in report.get('runs', []):
        known.setdefault(run.get('phase', {}).get('id'), run.get('phase', {}))
    blockers = []
    for name in sorted(begun):
        phase = known.get(name, {})
        if (phase.get('mode') != 'poisson' or phase.get('role', 'measure') in ('warmup', 'profile')
                or phase.get('duration_s', 0) < RULES['min_confirmation_duration_s']):
            continue
        key, _ = workload_key(phase)
        blockers.append({'source': str(source), 'phase_id': name, 'architecture': architecture,
                         'deployment_key': deployment_key, 'workload_key': key, 'rate': phase['rate'],
                         'reasons': context['reasons']})
    return blockers


def waiting_samples(samples):
    result = []
    for sample in samples:
        if sample.get('error') or sample.get('http_status') != 200:
            continue
        values = sample.get('values', {})
        waiting = [v for key, v in values.items() if key.split('{')[0] == 'vllm:num_requests_waiting' and finite(v)]
        if waiting and finite(sample.get('offset_s')):
            result.append((sample['offset_s'], sum(waiting)))
    return result


def classify(rows, phase, window, samples, metrics_interval):
    if phase['mode'] != 'poisson':
        return {'classification': 'NOT_OPEN_ARRIVAL', 'trend_verdict': None, 'reasons': ['Burst/closed-loop is not a sustained offered-rate qualification.']}
    left0 = window*RULES['analysis_start_fraction']
    width = (window-left0)/RULES['analysis_bins']
    waits = waiting_samples(samples)
    bins = []
    for index in range(RULES['analysis_bins']):
        left, right = left0+index*width, left0+(index+1)*width
        ws = [v for t, v in waits if left <= t < right]
        outstanding_integral = sum(max(0., min(r['end_offset_s'], right)-max(r['dispatch_offset_s'], left))
                                   for r in rows if r.get('admitted'))
        bins.append({'start_s': left, 'end_s': right,
                     'arrivals': sum(left <= r['planned_offset_s'] < right for r in rows),
                     'successful_completions': sum(r.get('ok') and left <= r['end_offset_s'] < right for r in rows),
                     'client_mean_outstanding': outstanding_integral/width,
                     'client_outstanding_at_end': sum(r.get('admitted') and r['dispatch_offset_s'] <= right < r['end_offset_s'] for r in rows),
                     'server_mean_waiting': sum(ws)/len(ws) if ws else None,
                     'server_sample_count': len(ws)})
    arrivals = sum(b['arrivals'] for b in bins)
    completed = sum(b['successful_completions'] for b in bins)
    tolerance = max(2., .05*arrivals)
    client = [b['client_mean_outstanding'] for b in bins]
    server = [b['server_mean_waiting'] for b in bins]
    coverage = min(1., sum(b['server_sample_count'] for b in bins)*metrics_interval/(window-left0))
    client_growth = client[-1]-client[0]
    server_growth = server[-1]-server[0] if all(v is not None for v in server) else None
    deficit = arrivals-completed
    dispatch = dist([r.get('dispatch_lag_s') for r in rows if r.get('admitted')])['p95']
    reasons = []
    if any(not r.get('ok') or r.get('censored') or r.get('timed_out') for r in rows):
        reasons.append('Request failure, timeout, censor, or incomplete terminal result.')
    if arrivals < RULES['min_analysis_arrivals']:
        reasons.append('Insufficient arrivals in analysis interval.')
    if coverage < RULES['min_server_sample_coverage_fraction'] or server_growth is None:
        reasons.append('Insufficient server waiting samples.')
    if dispatch is None or dispatch > RULES['max_dispatch_lag_p95_s']:
        reasons.append('Client dispatch P95 exceeds 100ms or is unavailable.')
    growing = deficit > tolerance and client_growth > tolerance and client[1] > client[0] and client[2] > client[1]
    if reasons:
        verdict = 'INCONCLUSIVE'
    elif growing:
        verdict = 'GROWING_BACKLOG'
        reasons.append('Arrival-completion deficit and sustained client outstanding growth exceed frozen tolerance.')
    elif deficit <= tolerance and client_growth <= tolerance and server_growth <= tolerance:
        verdict = 'STABLE_IN_WINDOW'
    else:
        verdict = 'INCONCLUSIVE'
        reasons.append('Queue/throughput evidence lies between stable and sustained-growth rules.')
    classification = verdict if window >= RULES['min_confirmation_duration_s'] else 'SCREEN_ONLY'
    if classification == 'SCREEN_ONLY':
        reasons.append('Duration below 180s: cannot qualify finite-window stable capacity.')
    return {'classification': classification, 'trend_verdict': verdict, 'reasons': reasons,
            'analysis_bins': bins, 'analysis_arrivals': arrivals, 'analysis_completions': completed,
            'completion_deficit_requests': deficit, 'backlog_tolerance_requests': tolerance,
            'client_mean_outstanding_increase': client_growth, 'server_mean_waiting_increase': server_growth,
            'server_sample_coverage_fraction': coverage, 'dispatch_lag_p95_s': dispatch}


def resolve_raw(value, run_dir):
    path = Path(value)
    if path.exists():
        return path
    candidate = run_dir/path.name
    if candidate.exists():
        return candidate
    # Capacity metric directories are children of the run directory.
    return run_dir/path.parent.name/path.name


def metric_evidence(run, directory, ledger):
    before, after = run.get('metrics_before'), run.get('metrics_after')
    source = 'explicit_before_after'
    if not before or not after:
        saved = [s for s in run.get('metric_samples', []) if s.get('raw_file') and not s.get('error')]
        if len(saved) < 2:
            return {'status': 'MISSING', 'reason': 'No complete metrics endpoints.'}
        before, after, source = saved[0], saved[-1], 'sampled_interval_only'
    def load_metric(item):
        name = item if isinstance(item, str) else item.get('raw_file') or item.get('path')
        require(bool(name), 'Missing metric raw file')
        path = resolve_raw(name, directory)
        raw = path.read_bytes()
        ledger[str(path.resolve())] = {'sha256': digest(raw), 'bytes': len(raw)}
        if isinstance(item, dict) and item.get('sha256'):
            require(item['sha256'] == digest(raw), 'Metrics snapshot hash mismatch')
        return historical.parse_metrics(raw.decode())
    try:
        delta = historical.metrics_delta(load_metric(before), load_metric(after))
        delta['measurement_basis'] = source
        delta['caveat'] = None if source == 'explicit_before_after' else 'Counter delta covers sampled interval, which can omit first/last requests.'
        return delta
    except (ValueError, OSError, KeyError) as exc:
        return {'status': 'INCOMPLETE', 'error': str(exc), 'measurement_basis': source}


def analyze_phase(report, run, path, architecture, devices, ledger):
    phase = run['phase']
    rows = [recompute_row(r, phase['mode']) for r in run['requests']]
    expected = report.get('prepared_requests', {}).get(phase['id'])
    require(expected is not None and len(rows) == expected, 'Request count differs from prepared plan')
    require(len({r['request_id'] for r in rows}) == len(rows), 'Duplicate request identity')
    wall = run['summary']['total_wall_s']
    require(finite(wall) and wall > 0, 'Invalid measurement wall')
    window = phase['duration_s'] if phase['mode'] == 'poisson' else wall
    require(window > 0 and wall+1e-5 >= window, 'Poisson fixed window did not elapse')
    require(all(r['end_offset_s'] <= wall+.05 for r in rows), 'Request completion after measured wall')
    group_result = group(rows)
    by_tier = {name: group([r for r in rows if r['input_tier'] == name]) for name in sorted({r['input_tier'] for r in rows})}
    by_shape = {name: group([r for r in rows if r['shape_name'] == name]) for name in sorted({r['shape_name'] for r in rows})}
    checks = [g['overall_ratio_p95_slo_pass'] for name, g in by_tier.items() if name != 'gt16k']
    group_result['p95_slo_pass'] = all(v is True for v in checks) if checks else None
    ok = [r for r in rows if r['ok']]
    within = [r for r in ok if r['end_offset_s'] <= window]
    good = [r for r in ok if r['slo_pass'] is True]
    good_within = [r for r in within if r['slo_pass'] is True]
    output = lambda seq: sum(r['usage']['completion_tokens'] for r in seq)
    throughput = {'cohort_requests_s': len(ok)/wall, 'cohort_output_tokens_s': output(ok)/wall,
                  'completed_in_window_requests_s': len(within)/window,
                  'completion_credited_in_window_output_tokens_s': output(within)/window,
                  'cohort_slo_goodput_requests_s': len(good)/wall,
                  'cohort_slo_goodput_output_tokens_s': output(good)/wall,
                  'completed_in_window_slo_goodput_requests_s': len(good_within)/window,
                  'offered_window_slo_goodput_requests_s': len(good)/window}
    key, shape_descriptor = workload_key(phase)
    start = run['start_unix']
    gpu = historical.integrate_gpu(devices, start, start+wall, [0, 1, 2, 3], 2.5) if devices else {'status': 'MISSING', 'energy_j': None}
    if gpu.get('energy_j'):
        gpu['successful_output_tokens_per_j'] = output(ok)/gpu['energy_j']
        gpu['slo_output_tokens_per_j'] = output(good)/gpu['energy_j']
    stability = classify(rows, phase, window, run.get('metric_samples', []), report['plan'].get('metrics_interval_s', 1.))
    measured = phase.get('role', 'measure') not in ('warmup', 'profile')
    return {'experiment': path.parent.name, 'architecture': architecture, 'source': str(path.resolve()),
            'phase_id': phase['id'], 'role': phase.get('role', 'measure'), 'mode': phase['mode'],
            'seed': phase['seed'], 'rate': phase.get('rate'), 'workload_key': key, 'workload': shape_descriptor,
            'included_as_measurement': measured, 'start_unix': start, 'total_wall_s': wall,
            'measurement_window_s': window, 'drain_s': max(0., wall-window),
            'offered_requests': len(rows), 'successful_output_tokens': output(ok), 'slo_output_tokens': output(good),
            'in_flight_at_window_end': sum(r.get('admitted') and r['dispatch_offset_s'] <= window < r['end_offset_s'] for r in rows),
            'stats': group_result, 'by_input_tier': by_tier, 'by_shape': by_shape, 'throughput': throughput,
            'stability': stability, 'gpu': gpu, 'prometheus': metric_evidence(run, path.parent, ledger)}


def aggregate(stages, blockers=None):
    blockers = blockers or []
    buckets = defaultdict(list)
    for s in stages:
        if s['included_as_measurement']:
            buckets[s['architecture'], s.get('deployment_key', ''), s['workload_key']].append(s)
    result = []
    for (architecture, deployment_key, key), rows in sorted(buckets.items()):
        rates = defaultdict(list)
        for s in rows:
            if s['mode'] == 'poisson':
                rates[s['rate']].append(s)
        candidates = []
        for rate, observed in sorted(rates.items()):
            long = [s for s in observed if s['measurement_window_s'] >= RULES['min_confirmation_duration_s']]
            seeds = {s['seed'] for s in long}
            related_blockers = [b for b in blockers if b['architecture'] == architecture
                                and b.get('deployment_key', '') == deployment_key
                                and b['workload_key'] == key and b['rate'] == rate]
            confirmed = (len(seeds) >= RULES['min_distinct_confirmation_seeds'] and not related_blockers
                         and all(s.get('eligible_for_capacity_qualification', True)
                                 and s['stability']['classification'] == 'STABLE_IN_WINDOW' for s in long))
            slo_confirmed = confirmed and all(s['stats']['p95_slo_pass'] is True for s in long)
            duration = sum(s['total_wall_s'] for s in long)
            candidate = {'offered_rate_s': rate, 'observed_phase_ids': [s['experiment']+'/'+s['phase_id'] for s in observed],
                         'confirmation_phase_ids': [s['experiment']+'/'+s['phase_id'] for s in long],
                         'distinct_confirmation_seeds': len(seeds), 'finite_stable_confirmed': confirmed,
                         'finite_p95_slo_confirmed': slo_confirmed,
                         'unresolved_attempt_blockers': related_blockers,
                         'aggregate_confirmation_cohort_requests_s': ratio(sum(s['stats']['successful'] for s in long), duration),
                         'aggregate_confirmation_cohort_output_tokens_s': ratio(sum(s['successful_output_tokens'] for s in long), duration),
                         'aggregate_confirmation_cohort_slo_goodput_requests_s': ratio(sum(s['stats']['slo_passes'] for s in long), duration),
                         'aggregate_confirmation_cohort_slo_goodput_output_tokens_s': ratio(sum(s['slo_output_tokens'] for s in long), duration),
                         'all_observed_p95_slo_pass': all(s['stats']['p95_slo_pass'] is True for s in observed),
                         'classification_counts': {c: sum(s['stability']['classification'] == c for s in observed) for c in sorted({s['stability']['classification'] for s in observed})}}
            candidates.append(candidate)
        stable = [c for c in candidates if c['finite_stable_confirmed']]
        slo = [c for c in candidates if c['finite_p95_slo_confirmed']]
        peak = max(rows, key=lambda s: s['throughput']['cohort_output_tokens_s'])
        peak_record = {name: peak[name] for name in ('experiment', 'phase_id', 'mode', 'rate')}
        peak_record.update(peak['throughput'])
        peak_record['p95_slo_pass'] = peak['stats']['p95_slo_pass']
        peak_record['stability'] = peak['stability']['classification']
        result.append({'architecture': architecture, 'deployment_key': deployment_key,
                       'workload_key': key, 'workload': rows[0]['workload'],
                       'rates': candidates,
                       'highest_finite_confirmed_slo_rate': max(slo, key=lambda c: c['offered_rate_s']) if slo else None,
                       'highest_finite_confirmed_stable_rate': max(stable, key=lambda c: c['offered_rate_s']) if stable else None,
                       'maximum_finite_confirmed_stable_output_throughput': max(stable, key=lambda c: c['aggregate_confirmation_cohort_output_tokens_s']) if stable else None,
                       'maximum_finite_stable_slo_goodput': max(stable, key=lambda c: c['aggregate_confirmation_cohort_slo_goodput_requests_s']) if stable else None,
                       'observed_peak_without_slo_constraint': peak_record,
                       'interpretation': 'Highest tested finite-confirmed point, not an exact maximum; screens and inconclusive points remain visible.'})
    return result


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(report):
    lines = ['# 阶段 A 容量复算', '', f"状态：{report['status']}。规则 `{RULES['version']}`；P95按输入档位判定，P99单列，稳定性独立于SLO。", '',
             '吞吐主列为 cohort 成功输出 tokens / (arrival window + drain)；completed-in-window 是完成归属窗口，不能当逐 token 生成时间窗。', '',
             '| 架构 | 工作负载 | 最高确认 SLO rate | 最高确认稳定 rate | 已确认最大稳定 tok/s | 观测 peak tok/s |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for item in report['capacities']:
        slo, stable, fastest = item['highest_finite_confirmed_slo_rate'], item['highest_finite_confirmed_stable_rate'], item['maximum_finite_confirmed_stable_output_throughput']
        name = ', '.join(s[0] for s in item['workload']['shape_mix'])
        lines.append(f"| {item['architecture']} | {name} | {number(slo['offered_rate_s'] if slo else None)} | {number(stable['offered_rate_s'] if stable else None)} | {number(fastest['aggregate_confirmation_cohort_output_tokens_s'] if fastest else None, 1)} | {number(item['observed_peak_without_slo_constraint']['cohort_output_tokens_s'], 1)} |")
    lines += ['', '未确认不是容量为零。60 秒粗扫及重复/证据不足的点只形成范围线索；阶段内稳定也不代表生产 SLA。完整分桶P50/P95/P99、GPU、服务端counter与原始来源见同名JSON。', '',
              '| 实验 / phase | rate | window / drain s | 成功 / 全部 | P95 SLO | 联合满足率 | cohort tok/s | good req/s | 稳定性 |',
              '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |']
    for s in report['stages']:
        if not s['included_as_measurement']:
            continue
        lines.append(f"| {s['experiment']} / {s['phase_id']} | {number(s['rate'])} | {number(s['measurement_window_s'], 0)} / {number(s['drain_s'], 1)} | {s['stats']['successful']} / {s['offered_requests']} | {s['stats']['p95_slo_pass']} | {number(s['stats']['slo_joint_pass_rate'])} | {number(s['throughput']['cohort_output_tokens_s'], 1)} | {number(s['throughput']['cohort_slo_goodput_requests_s'])} | {s['stability']['classification']} |")
    if report['excluded']:
        lines += ['', '未纳入文件/phase：'] + [f"- {x['source']}: {x['reason']}" for x in report['excluded']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--files', type=Path, nargs='*')
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    files = args.files if args.files is not None else sorted(path_at(args.root, 'results').glob('*/capacity.json'))
    ledger, stages, excluded, fatal, blockers, qualifications = ({}, [], [], [], [], [])
    for path in files:
        try:
            report = read(path, ledger)
            require(report.get('schema_version') == 1 and isinstance(report.get('plan'), dict), 'Not a capacity schema-1 report')
            config_path = path.parent / 'config.json'
            config = read(config_path, ledger) if config_path.exists() else {}
            architecture = config.get('name', report['plan'].get('architecture', path.parent.name))
            deployment_key = digest(json.dumps(config or {'architecture': architecture}, sort_keys=True, separators=(',', ':')).encode())
            supervisor_path, experiment_path = (path.parent / 'status.json', path.parent / 'experiment.json')
            supervisor = read(supervisor_path, ledger) if supervisor_path.exists() else None
            experiment = read(experiment_path, ledger) if experiment_path.exists() else None
            context = completion_context(report, supervisor, experiment)
            qualifications.append({'source': str(path), 'architecture': architecture, 'deployment_key': deployment_key, **context})
            blockers.extend(unresolved_attempts(report, architecture, deployment_key, path, context))
            if report.get('status') not in ('PASSED', 'FAILED', 'CENSORED'):
                excluded.append({'source': str(path), 'reason': 'Not terminal; begun long attempts retained as qualification blockers: ' + str(report.get('status'))})
                continue
            devices, warnings = historical.load_gpu(path.parent / 'gpu.csv', ZoneInfo('UTC'), ledger)
            for run in report.get('runs', []):
                try:
                    stage = analyze_phase(report, run, path, architecture, devices, ledger)
                    stage['gpu_warnings'] = warnings
                    stage['deployment_key'] = deployment_key
                    stage['eligible_for_capacity_qualification'] = context['eligible_for_capacity_qualification']
                    stages.append(stage)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    item = {'source': str(path) + '/' + str(run.get('phase', {}).get('id')), 'reason': str(exc)}
                    excluded.append(item)
                    fatal.append(item)
                    broken_context = {'eligible_for_capacity_qualification': False, 'reasons': ['Independent phase validation failed: ' + str(exc)]}
                    blockers.extend(unresolved_attempts({'plan': {'phases': [run.get('phase', {})]}, 'runs': [run]}, architecture, deployment_key, path, broken_context))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            item = {'source': str(path), 'reason': str(exc)}
            excluded.append(item)
            fatal.append(item)
    result = {'schema_version': 1, 'status': 'FAIL' if fatal else 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'rules': RULES, 'summarizer_sha256': digest(Path(__file__).read_bytes()), 'stages': stages, 'capacities': aggregate(stages, blockers), 'excluded': excluded, 'qualification_contexts': qualifications, 'unresolved_attempt_blockers': blockers, 'evidence_ledger': ledger}
    prefix = args.output_prefix or path_at(args.root, 'summary/capacity-summary')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_text(prefix.with_suffix('.json'), json.dumps(result, indent=2, sort_keys=True) + '\n')
    write_text(prefix.with_suffix('.md'), markdown(result))
    print(json.dumps({'status': result['status'], 'files': len(files), 'stages': len(stages), 'excluded': len(excluded), 'output_prefix': str(prefix)}))
    return 1 if fatal else 0


if __name__ == '__main__':
    raise SystemExit(main())
