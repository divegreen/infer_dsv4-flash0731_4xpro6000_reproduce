#!/usr/bin/env python3
"""Independent finite-window classifier/aggregation fixtures; no GPU or HTTP."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

path = Path(__file__).with_name('summarize_capacity.py')
spec = importlib.util.spec_from_file_location('capacity_summary_under_test', path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def rows(arrival_rate=1., duration=180., service=1., growth=0.):
    result = []
    index = 0
    while (index+.5)/arrival_rate < duration:
        planned = (index+.5)/arrival_rate
        latency = service+growth*planned
        result.append({'input_tokens': 512, 'planned_offset_s': planned, 'dispatch_offset_s': planned+.001,
                       'end_offset_s': planned+.001+latency, 'dispatch_lag_s': .001,
                       'admitted': True, 'ok': True, 'timed_out': False, 'censored': False})
        index += 1
    return result


def samples(duration=180, waiting_growth=0.):
    return [{'offset_s': i+.01, 'http_status': 200,
             'values': {'vllm:num_requests_waiting{engine="0"}': max(0., waiting_growth*i)}}
            for i in range(duration)]


def classify(data, duration=180., telemetry=None):
    return summary.classify(data, {'mode': 'poisson'}, duration,
                            samples(int(duration)) if telemetry is None else telemetry, 1.)


tests = []


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    tests.append(label)


steady = classify(rows())
check('constant-latency open arrivals stable with nonzero final inflight', steady['classification'] == 'STABLE_IN_WINDOW')
long_latency = classify(rows(service=30.))
check('constant 30s response latency independent of SLO and natural drain', long_latency['classification'] == 'STABLE_IN_WINDOW')
overload = classify(rows(growth=.8), telemetry=samples(waiting_growth=.4))
check('sustained arrival-completion deficit and queue growth detected', overload['classification'] == 'GROWING_BACKLOG')
check('60s screen never confirmed stable', classify(rows(duration=60.), 60.)['classification'] == 'SCREEN_ONLY')
check('server evidence missing means inconclusive', classify(rows(), telemetry=[])['classification'] == 'INCONCLUSIVE')
check('sparse traffic cannot establish a tail-window conclusion', classify(rows(arrival_rate=.05))['classification'] == 'INCONCLUSIVE')
broken = rows()
broken[-1]['ok'] = False
broken[-1]['timed_out'] = True
check('timeout cannot confirm stable', classify(broken)['classification'] == 'INCONCLUSIVE')
lagged = rows()
for row in lagged:
    row['dispatch_lag_s'] = .2
check('client scheduling delay blocks server capacity certification', classify(lagged)['classification'] == 'INCONCLUSIVE')
check('closed-loop does not qualify offered-rate stability', summary.classify(rows(), {'mode': 'closed-loop'}, 180., samples(), 1.)['classification'] == 'NOT_OPEN_ARRIVAL')
check('linear p99 interpolation and n', summary.dist([0, 10, 20, 30]) == {'n': 4, 'p50': 15., 'p95': 28.499999999999996, 'p99': 29.699999999999996, 'max': 30.})


def stage(seed, slo=True, stability='STABLE_IN_WINDOW', duration=180.):
    return {'architecture': 'a', 'workload_key': 'w', 'workload': {'shape_mix': []}, 'included_as_measurement': True,
            'mode': 'poisson', 'rate': 1., 'seed': seed, 'measurement_window_s': duration,
            'total_wall_s': duration+1., 'experiment': 'exp', 'phase_id': 'p'+str(seed),
            'successful_output_tokens': 2000, 'slo_output_tokens': 1000,
            'stats': {'p95_slo_pass': slo, 'successful': 20, 'slo_passes': 10},
            'stability': {'classification': stability}, 'throughput': {'cohort_output_tokens_s': 2000/(duration+1)}}


check('one long phase does not establish repeated capacity', summary.aggregate([stage(1)])[0]['highest_finite_confirmed_stable_rate'] is None)
check('same seed twice does not establish repeated capacity', summary.aggregate([stage(1), stage(1)])[0]['highest_finite_confirmed_stable_rate'] is None)
check('two distinct seeds establish finite confirmation', summary.aggregate([stage(1), stage(2)])[0]['highest_finite_confirmed_stable_rate'] is not None)
out = summary.aggregate([stage(1, slo=False), stage(2, slo=False)])[0]
check('stable throughput does not require SLO pass', out['highest_finite_confirmed_stable_rate'] is not None and out['highest_finite_confirmed_slo_rate'] is None)
check('failed third confirmation cannot be cherry-picked away', summary.aggregate([stage(1), stage(2), stage(3, stability='GROWING_BACKLOG')])[0]['highest_finite_confirmed_stable_rate'] is None)
check('two screens never establish repeated stable capacity', summary.aggregate([stage(1, duration=60), stage(2, duration=60)])[0]['highest_finite_confirmed_stable_rate'] is None)

confirmation = {'id': 'confirm3', 'mode': 'poisson', 'rate': 1., 'duration_s': 180.,
                'shape_mix': [{'name': 'short', 'input_tokens': 512, 'output_tokens': 4, 'weight': 1}]}
future = dict(confirmation, id='not-started')
supervisor = {'exit_code': 0, 'timed_out': False, 'end_utc': '2026-01-01T00:00:00Z', 'supervisor_error': None}
experiment = {'status': 'PASS', 'elapsed_seconds': 1.}
finished_report = {'status': 'PASSED', 'plan': {'phases': [confirmation]}, 'runs': [{'phase': confirmation}]}
context = summary.completion_context(finished_report, supervisor, experiment)
check('complete plan and terminal supervisor are necessary evidence', context['eligible_for_capacity_qualification'])
check('client PASSED before supervisor cleanup cannot qualify', not summary.completion_context(finished_report, None, experiment)['eligible_for_capacity_qualification'])
check('supervisor timeout cannot qualify a PASSED client', not summary.completion_context(finished_report, dict(supervisor, timed_out=True), experiment)['eligible_for_capacity_qualification'])
check('incomplete phase plan cannot qualify a PASSED client', not summary.completion_context(dict(finished_report, runs=[]), supervisor, experiment)['eligible_for_capacity_qualification'])
for terminal in ('RUNNING', 'FAILED'):
    pending = {'status': terminal, 'active_phase': 'confirm3', 'plan': {'phases': [confirmation, future]}, 'runs': []}
    context = summary.completion_context(pending, supervisor, experiment)
    blockers = summary.unresolved_attempts(pending, 'a', 'deployment', 'unfinished.json', context)
    check(terminal+' active confirmation retained but future plan not treated as failed', len(blockers) == 1 and blockers[0]['phase_id'] == 'confirm3')
    observed = [dict(stage(i), deployment_key='deployment', workload_key=blockers[0]['workload_key']) for i in (1, 2)]
    check('two successes plus '+terminal+' confirmation cannot qualify', summary.aggregate(observed, blockers)[0]['highest_finite_confirmed_stable_rate'] is None)
separate_configs = summary.aggregate([dict(stage(1), deployment_key='old-config'), dict(stage(2), deployment_key='changed-config')])
check('same architecture name with changed deployment config does not combine', len(separate_configs) == 2 and all(x['highest_finite_confirmed_stable_rate'] is None for x in separate_configs))
qualified = [stage(i, slo=True) for i in (1, 2)]
more_good_requests = [dict(stage(i, slo=False), rate=2., stats={'p95_slo_pass': False, 'successful': 60, 'slo_passes': 30}) for i in (3, 4)]
different_optima = summary.aggregate(qualified+more_good_requests)[0]
best_goodput = different_optima['maximum_finite_stable_slo_goodput']
check('maximum stable individual Goodput can occur at P95-unqualified rate', best_goodput['offered_rate_s'] == 2. and best_goodput['finite_p95_slo_confirmed'] is False)
check('maximum stable individual Goodput differs from highest P95-qualified capacity', different_optima['highest_finite_confirmed_slo_rate']['offered_rate_s'] == 1. and best_goodput['aggregate_confirmation_cohort_slo_goodput_requests_s'] > different_optima['highest_finite_confirmed_slo_rate']['aggregate_confirmation_cohort_slo_goodput_requests_s'])


with tempfile.TemporaryDirectory(prefix='capacity-summary-fixture-') as temporary:
    directory = Path(temporary)
    phase = {'id': 'fixture', 'mode': 'poisson', 'rate': 1.5, 'duration_s': 2.0, 'seed': 1, 'shape_mix': [{'name': 'short', 'input_tokens': 512, 'output_tokens': 4, 'weight': 1}]}
    request_rows = []
    for index in range(3):
        offset = index * 0.5
        ok = index != 2
        row = {'request_id': f'fixture/{index}', 'input_tokens': 512, 'shape_name': 'short', 'requested_output_tokens': 4, 'planned_offset_s': offset, 'dispatch_offset_s': offset + 0.001, 'dispatch_lag_s': 0.001, 'end_offset_s': offset + 0.101, 'admitted': True, 'ok': ok, 'timed_out': False, 'censored': False, 'ignore_eos': True, 'http_status': 200 if ok else 500, 'done_received': ok, 'error': None if ok else {'type': 'HTTPError'}, 'ttft_s': 0.02 if ok else None, 'e2e_s': 0.1 if ok else None, 'full_text': 'abc' if ok else '', 'text_sha256': hashlib.sha256(('abc' if ok else '').encode()).hexdigest(), 'finish_reason': 'length' if ok else None, 'usage': {'prompt_tokens': 512, 'completion_tokens': 4, 'total_tokens': 516} if ok else None}
        request_rows.append(row)
    before, after = (directory / 'before.prom', directory / 'after.prom')
    write_text(before, 'vllm:request_success_total{model_name="fixture"} 0\nvllm:num_requests_waiting{model_name="fixture"} 0\n')
    write_text(after, 'vllm:request_success_total{model_name="fixture"} 2\nvllm:num_requests_waiting{model_name="fixture"} 0\n')
    start = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    with open_output(directory / 'gpu.csv', 'w') as handle:
        for sec in range(3):
            stamp = datetime.fromtimestamp(start + sec, timezone.utc).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
            for gpu in range(4):
                handle.write(f'{stamp}, {gpu}, fixture-{gpu}, 1000, 50, 100, 40\n')
    report = {'schema_version': 1, 'status': 'FAILED', 'prepared_requests': {'fixture': 3}, 'plan': {'architecture': 'fixture', 'metrics_interval_s': 1.0}, 'runs': [{'phase': phase, 'requests': request_rows, 'summary': {'total_wall_s': 2.0}, 'start_unix': start, 'metric_samples': samples(2), 'metrics_before': {'raw_file': str(before)}, 'metrics_after': {'raw_file': str(after)}}]}
    source = directory / 'capacity.json'
    write_text(source, json.dumps(report))
    process = subprocess.run([sys.executable, str(path), '--files', str(source), '--output-prefix', str(directory / 'derived')], text=True, capture_output=True)
    check('terminal-failure raw evidence still recomputes through full CLI', process.returncode == 0)
    output = load_document(directory / 'derived.json')
    s = output['stages'][0]
    check('independent recompute retains failures in joint SLO denominator', s['stats']['slo_joint_pass_rate'] == 2 / 3 and s['stats']['failed'] == 1)
    check('cohort throughput counts only successful actual outputs', s['throughput']['cohort_output_tokens_s'] == 4.0)
    check('explicit metrics endpoints used and counter delta validated', s['prometheus']['totals']['vllm:request_success_total'] == 2.0 and s['prometheus']['measurement_basis'] == 'explicit_before_after')
    check('four-device sampled energy integral matches 800 joules', abs(s['gpu']['energy_j'] - 800.0) < 1e-08)
    check('failed short fixture never qualifies stable capacity', output['capacities'][0]['highest_finite_confirmed_stable_rate'] is None)

print(json.dumps({'status': 'PASS', 'checks': len(tests), 'tests': tests}, indent=2))
