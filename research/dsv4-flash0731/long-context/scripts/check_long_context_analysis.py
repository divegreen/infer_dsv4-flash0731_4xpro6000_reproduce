#!/usr/bin/env python3
"""CPU-only end-to-end fixtures for Stage B offline analysis; no model/server."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name('analyze_long_context.py')
SPEC = importlib.util.spec_from_file_location('phase_b_analysis_test', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
checks = []


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    checks.append(label)


def fixture(directory, architecture='tp4', changed_prompt=False, priority=False):
    directory.mkdir()
    horizon, wall = (10.0, 12.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    definitions = [(16384, 0.0, 0.0, 2.0, 6.0, True, True), (32768, 0.1, 0.1, 1.0, 5.0, True, True), (65536, 0.2, 0.2, 4.0, 11.8, True, True), (512, 1.0, 1.0, 0.1, 0.2, True, True), (512, 2.0, 2.0, 0.2, 0.3, True, True), (512, 6.0, 6.0, 0.2, 0.3, True, True), (512, 7.0, 7.9, 0.2, 0.3, True, True), (512, 9.0, 9.0, None, None, False, True), (512, 9.5, None, None, None, False, False)]
    rows = []
    for index, (tokens, planned, dispatched, ttft, e2e, ok, admitted) in enumerate(definitions):
        row = {'index': index, 'request_id': f'fixture/{index}', 'phase_id': 'fixture', 'shape_name': f'p{tokens}_o4', 'input_tokens': tokens, 'requested_output_tokens': 4, 'planned_offset_s': planned, 'end_offset_s': dispatched + (e2e if ok else 0.1) if admitted else planned, 'admitted': admitted, 'ok': ok, 'timed_out': admitted and (not ok), 'censored': not admitted, 'ignore_eos': True, 'http_status': 200 if ok else None, 'done_received': ok, 'error': None if ok else {'type': 'TimeoutError' if admitted else 'ProtectedStop'}, 'ttft_s': ttft, 'e2e_s': e2e, 'full_text': 'abc' if ok else '', 'text_sha256': analysis.capacity.digest(('abc' if ok else '').encode()), 'prompt_sha256': analysis.hashed((tokens, index, changed_prompt)), 'usage': {'prompt_tokens': tokens, 'completion_tokens': 4, 'total_tokens': tokens + 4} if ok else None, 'finish_reason': 'length' if ok else None, 'priority': -1 if priority and tokens == 512 else 0}
        if admitted:
            row.update(dispatch_offset_s=dispatched, dispatch_lag_s=max(0.0, dispatched - planned))
        rows.append(row)
    phase = {'id': 'fixture', 'mode': 'trace', 'role': 'measure', 'seed': 1, 'duration_s': horizon, 'shape_mix': [{'name': f'p{x}_o4', 'input_tokens': x, 'output_tokens': 4, 'weight': 1} for x in (512, 16384, 32768, 65536)], 'arrivals': [{'offset_s': r['planned_offset_s'], 'shape_name': r['shape_name']} for r in rows]}
    metric_samples = []

    def save_metrics(name, offset, kv0, kv1, preemptions):
        body = f'vllm:kv_cache_usage_perc{{engine="0"}} {kv0}\nvllm:kv_cache_usage_perc{{engine="1"}} {kv1}\nvllm:num_requests_waiting{{engine="0"}} 2\nvllm:num_requests_waiting{{engine="1"}} 3\nvllm:num_requests_running{{engine="0"}} 1\nvllm:num_requests_running{{engine="1"}} 2\nvllm:num_preemptions_total{{engine="0"}} {preemptions}\nvllm:request_success_total{{engine="0"}} {preemptions}\n'
        path = directory / name
        write_text(path, body)
        embedded = {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1]) for line in body.splitlines() if any((x in line for x in analysis.GAUGES))}
        return {'http_status': 200, 'offset_s': offset, 'raw_file': str(path), 'sha256': analysis.capacity.digest(body.encode()), 'values': embedded}
    for i, (kv0, kv1) in enumerate(((0.2, 0.4), (0.4, 0.5), (0.6, 0.6), (0.8, 0.7))):
        metric_samples.append(save_metrics(f'{i}.prom', i * 4.0, kv0, kv1, 3 + i))
    before = save_metrics('before.prom', 0.0, 0.0, 0.0, 3)
    after = save_metrics('after.prom', 12.0, 0.0, 0.0, 7)
    jsonl = directory / 'requests.jsonl'
    write_text(jsonl, ''.join((json.dumps(r) + '\n' for r in reversed(rows))))
    report = {'schema_version': 1, 'status': 'CENSORED', 'tokenizer_sha256': 'fixture-tokenizer', 'prepared_requests': {'fixture': len(rows)}, 'plan': {'phases': [phase], 'metrics_interval_s': 4.0}, 'runs': [{'phase': phase, 'requests': rows, 'summary': {'total_wall_s': wall}, 'start_unix': start, 'end_unix': start + wall, 'metric_samples': metric_samples, 'metrics_before': before, 'metrics_after': after, 'raw_requests_jsonl': str(jsonl)}]}
    write_text(directory / 'capacity.json', json.dumps(report))
    write_text(directory / 'config.json', json.dumps({'name': architecture}))
    write_text(directory / 'status.json', json.dumps({'exit_code': 1, 'timed_out': False, 'end_utc': '2026-01-01T00:00:12Z'}))
    write_text(directory / 'experiment.json', json.dumps({'status': 'FAIL', 'elapsed_seconds': wall}))
    with open_output(directory / 'gpu.csv', 'w') as handle:
        for sec in range(13):
            stamp = datetime.fromtimestamp(start + sec, timezone.utc).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
            for gpu in range(4):
                handle.write(f'{stamp}, {gpu}, fixture-{gpu}, 1000, 50, 100, 40\n')
    return report


with tempfile.TemporaryDirectory(prefix='phase-b-analysis-') as tmp:
    root = Path(tmp)
    a, b = (root / 'b01', root / 'b02')
    source = fixture(a)
    fixture(b, architecture='tp2pp2', priority=True)
    before = {str(p): analysis.capacity.digest(p.read_bytes()) for directory in (a, b) for p in directory.iterdir()}
    result = analysis.analyze([a, b])
    check('terminal failed/censored experiments retained as descriptive evidence', result['status'] == 'PASS' and len(result['stages']) == 2)
    s = result['stages'][0]
    check('fixed trace horizon and drain are independent of cohort wall', s['measurement_window_s'] == 10 and s['drain_s'] == 2 and (s['total_wall_s'] == 12))
    check('trace never certifies sustained capacity', s['stability']['classification'] == 'NOT_SUSTAINED_ARRIVAL' and (not s['eligible_for_capacity_qualification']))
    check('delayed dispatch makes short scheduled TTFT fail although HTTP TTFT passes', s['short512']['slo_passes'] == 3 and s['short512']['successful'] == 4)
    check('failed and undispatched short requests stay in SLO denominator', s['short512']['requests'] == 6 and s['short512']['slo_joint_pass_rate'] == 0.5)
    check('short goodput uses whole phase cohort wall', s['short512']['cohort_slo_goodput_requests_s'] == 0.25)
    check('completion-window throughput uses fixed trace horizon', abs(s['throughput']['completed_in_window_requests_s'] - 0.6) < 1e-10)
    check('cohort throughput includes successful output only', abs(s['throughput']['cohort_output_tokens_s'] - 28 / 12) < 1e-10)
    check('P50/P95/P99 scheduled TTFT TPOT E2E retained by shape', all((q in s['by_shape']['p512_o4']['metrics'][key] for q in ('p50', 'p95', 'p99') for key in ('scheduled_ttft_s', 'tpot_s', 'scheduled_e2e_s'))))
    check('raw KV aggregates busiest pool without summing fractions', abs(s['raw_gauges']['aggregate'][analysis.GAUGES[0]]['sample_mean'] - 0.575) < 1e-10 and s['raw_gauges']['aggregate'][analysis.GAUGES[0]]['max'] == 0.8)
    check('raw wait and running counts sum across engines', s['raw_gauges']['aggregate'][analysis.GAUGES[1]]['max'] == 5 and s['raw_gauges']['aggregate'][analysis.GAUGES[2]]['sample_mean'] == 3)
    check('preemption count comes from exact endpoints', s['preemptions'] == 4)
    check('GPU power integration uses all four GPUs and full cohort', s['gpu']['energy_j'] == 4800)
    check('GPU memory and device busy retain observed/time-weighted evidence', all((d['measurements']['memory_mib']['observed_max'] == 1000 and d['measurements']['utilization_pct']['time_weighted_mean'] == 50 for d in s['gpu']['devices'])))
    check('request event sweep derives peak outstanding', s['client_peak_dispatched_outstanding'] == 4)
    groups = s['associations']['groups']
    check('pre-first-text association never looks at future long arrivals', groups['p16384_pre_first_text']['with_existing_overlap']['requests'] == 1)
    check('first-text boundary enters post group and end boundary excludes request', groups['p16384_post_first_text']['with_existing_overlap']['requests'] == 1)
    check('64K pre/post intervals keep independent shape associations', groups['p65536_pre_first_text']['with_existing_overlap']['requests'] == 2 and groups['p65536_post_first_text']['with_existing_overlap']['requests'] == 3)
    check('undispatched short requests are explicitly outside dispatch associations', s['associations']['undispatched_short_requests'] == 1)
    check('same planned arrivals and prompts match across deployment and priority treatments', result['arrival_identity_checks'][0]['status'] == 'PASS')
    check('every raw periodic metric and JSONL is hashed', all((str(p.resolve()) in result['evidence_ledger'] for p in (a / '0.prom', a / '3.prom', a / 'requests.jsonl', a / 'gpu.csv', a / 'capacity.json'))))
    check('analysis does not mutate original input files', all((analysis.capacity.digest(Path(p).read_bytes()) == h for p, h in before.items())))
    check('association caveat explicitly denies kernel causality', 'causal' in s['associations']['interpretation'] and 'prefill' in s['associations']['interpretation'])
    output = root / 'derived'
    proc = subprocess.run([sys.executable, str(SCRIPT), '--result-dirs', str(a), str(b), '--output-prefix', str(output)], capture_output=True, text=True)
    check('full CLI emits JSON and Markdown', proc.returncode == 0 and output.with_suffix('.json').exists() and output.with_suffix('.md').exists())
    proc2 = subprocess.run([sys.executable, str(SCRIPT), '--result-dirs', str(a), '--output-prefix', str(output)], capture_output=True, text=True)
    check('CLI refuses to overwrite an existing analysis', proc2.returncode != 0)
    c = root / 'b03'
    fixture(c, changed_prompt=True)
    mismatch = analysis.analyze([a, c])
    check('same comparison ID with changed prompt is a hard mismatch', mismatch['status'] == 'FAIL' and mismatch['arrival_identity_checks'][0]['status'] == 'MISMATCH')
    raw = a / '0.prom'
    write_text(raw, raw.read_text().replace('0.2', '0.3'))
    corrupt = analysis.analyze([a])
    check('raw metric hash mismatch fails validation', corrupt['status'] == 'FAIL' and any(('hash mismatch' in e.get('error', '') for e in corrupt['errors'])))
    d = root / 'b04'
    fixture(d)
    jsonl = d / 'requests.jsonl'
    write_text(jsonl, jsonl.read_text().replace('fixture/0', 'altered/0'))
    corrupt = analysis.analyze([d])
    check('JSONL/report mismatch fails validation', corrupt['status'] == 'FAIL' and any(('JSONL differs' in e.get('error', '') for e in corrupt['errors'])))
    e = root / 'b05'
    report = fixture(e)
    report['runs'][0]['phase']['arrivals'][0]['offset_s'] = 0.01
    write_text(e / 'capacity.json', json.dumps(report))
    corrupt = analysis.analyze([e])
    check('trace plan/observed scheduled offset mismatch fails validation', corrupt['status'] == 'FAIL' and any(('planned arrival' in e.get('error', '') for e in corrupt['errors'])))
    f = root / 'b06'
    fixture(f)
    endpoint = f / 'after.prom'
    write_text(endpoint, endpoint.read_text().replace(' 7\n', ' 8\n'))
    corrupt = analysis.analyze([f])
    check('counter endpoint hash mismatch fails instead of merely omitting delta', corrupt['status'] == 'FAIL' and any(('hash mismatch' in e.get('error', '') for e in corrupt['errors'])))

print(json.dumps({'status': 'PASS', 'checks': len(checks), 'tests': checks,
                  'analyzer_sha256': analysis.capacity.digest(SCRIPT.read_bytes()),
                  'fixture_sha256': analysis.capacity.digest(Path(__file__).read_bytes())}, indent=2))
