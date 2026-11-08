"""CPU fixtures for scenario aggregation; no model/client/GPU/network imports."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("summarize_scenarios.py")
spec = importlib.util.spec_from_file_location("scenario_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def request(ttft=.2, dispatch=0., prompt_hash="fixture", seed=100, index=0, prefix="1/case/measure", start=240.):
    text = "fixture output"
    return {"request_id": f"{prefix}/{index}", "prompt_sha256": prompt_hash, "input_tokens": 2,
        "requested_output_tokens": 3, "seed": seed, "start_unix": start + index * .01,
        "ok": True, "error": None, "http_status": 200, "done_received": True,
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "finish_reason": "length", "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ttft_s": ttft, "e2e_s": ttft + .2, "tpot_s": .1, "elapsed_s": ttft + .201,
        "dispatch_lag_s": dispatch, "scheduled_ttft_s": ttft + dispatch,
        "scheduled_e2e_s": ttft + .2 + dispatch, "planned_offset_s": 0.}


def fixture(root, suite='decode', rounds=2):
    directory = path_at(root, 'results') / 'fixture'
    directory.mkdir(parents=True)
    source = b'# source provenance fixture\n'
    source_key = 'scenarios/scripts/scenario_client.py'
    source_hash = hashlib.sha256(source).hexdigest()
    command = {'environment': {'CUDA_VISIBLE_DEVICES': '0,1,2,3'}, 'git_commit': 'synthetic', 'source_sha256': {source_key: source_hash}}

    def put(name, data):
        write_text(directory / name, json.dumps(data))
    put('command.json', command)
    put('config.json', {'name': 'fixture', 'args': ['--no-enable-prefix-caching']})
    put('experiment.json', {'status': 'PASS', 'profile': suite == 'profile'})
    put('status.json', {'exit_code': 0, 'timed_out': False, 'supervisor_error': None})
    with zipfile.ZipFile(directory / 'source-snapshot.zip', 'w') as z:
        z.writestr(source_key, source)
    token_ids = [101, 102]
    prompt_hash = hashlib.sha256(json.dumps(token_ids, separators=(',', ':')).encode()).hexdigest()
    case = {'name': 'case', 'p': 2, 'o': 3, 'c': 1 if suite == 'cache' else 2, 'n': 8 if suite == 'cache' else 2}
    report = {'schema_version': 2, 'suite': suite, 'status': 'PASSED', 'rounds': rounds, 'client_sha256': source_hash, 'tokenizer_sha256': 'synthetic-tokenizer', 'protocol': {'test_fixture': True}, 'cases': [case], 'prompts': {prompt_hash: {'token_ids': token_ids, 'input_tokens': 2, 'sha256': prompt_hash}}, 'banks': {}, 'runs': []}
    phases = ['cold', 'exact_replay', 'fork_first', 'warm_serial', 'warm_parallel'] if suite == 'cache' else ['warmup', 'measure']
    max_end = 0
    for r in range(1, rounds + 1):
        run = {'round': r, 'case': 'case', 'status': 'PASSED', 'stages': [], 'reset': {'status': 200, 'body': '{"success":true}'}}
        for j, phase in enumerate(phases):
            n = 1 if phase in ('cold', 'exact_replay', 'fork_first') else case['n']
            start = 100.0 + r * 300 + j * 40
            wall = 10.0 if r == 1 else 30.0
            rows = [request(prompt_hash=prompt_hash, seed=100 + i, index=i, prefix=f'{r}/case/{phase}', start=start) for i in range(n)]
            bank = [{'prompt_sha256': prompt_hash, 'output_tokens': 3, 'seed': 100 + i} for i in range(n)]
            report['banks'][f'{r}/case/{phase}'] = bank
            raw = {'requests': n, 'successful': n, 'failed': 0, 'wall_s': wall, 'output_tokens': n * 3, 'output_tokens_per_s': n * 3 / wall}
            for k, value in (('ttft_s', 0.2), ('e2e_s', 0.4), ('tpot_s', 0.1)):
                raw[k] = {'p50': value, 'p95': value, 'max': value}
            stage = {'phase': phase, 'start_unix': start, 'end_unix': start + wall, 'requests': rows, 'summary': raw, 'offered_rate': None, 'concurrency_limit': 8 if phase == 'warm_parallel' else 1 if phase in ('cold', 'exact_replay', 'fork_first') else case['c'], 'planned_arrival_span_s': 0.0, 'metrics_before': f'metrics-{r}-{phase}-before.txt', 'metrics_after': f'metrics-{r}-{phase}-after.txt'}
            for when, multiplier in (('before', 0), ('after', 1)):
                values = {'prompt_tokens_total': 2 * n, 'generation_tokens_total': 3 * n, 'request_success_total': n, 'prefix_cache_queries_total': 0, 'prefix_cache_hits_total': 0, 'num_preemptions_total': 0}
                write_text(directory / stage['metrics_' + when], '\n'.join((f'vllm:{name}{{engine="0",model_name="fixture"}} {value * multiplier}' for name, value in values.items())) + '\n')
            run['stages'].append(stage)
            max_end = max(max_end, start + wall)
        report['runs'].append(run)
    put(f'scenario-{suite}.json', report)
    with open_output(directory / 'gpu.csv', 'w') as f:
        for timestamp in range(390, int(max_end) + 2):
            stamp = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y/%m/%d %H:%M:%S.%f')
            for gpu in range(4):
                f.write(f'{stamp}, {gpu}, GPU-fixture-{gpu}, 1000, 50, 100, 40\n')
    return (directory, report)


class Tests(unittest.TestCase):

    def test_weighted_throughput_and_pooled_quantiles(self):
        rows = [request(x) for x in (1.0, 2.0, 10.0, 20.0)]
        actual = summary.request_summary(rows, 40.0, False)
        self.assertAlmostEqual(actual['output_tokens_per_s'], 0.3)
        self.assertAlmostEqual(actual['latencies']['ttft_s']['p50'], 6.0)
        self.assertAlmostEqual(actual['latencies']['ttft_s']['p95'], 18.5)
        self.assertIsNone(actual['exploratory_slo'])

    def test_poisson_only_scheduled_goodput(self):
        rows = [request(0.2, 0.1), request(0.2, 2.0)]
        closed = summary.request_summary(rows, 10.0, False)
        self.assertIsNone(closed['scheduled_latencies'])
        self.assertIsNone(closed['exploratory_slo'])
        opened = summary.request_summary(rows, 10.0, True)
        self.assertEqual(opened['exploratory_slo']['passed'], 0)
        for row in rows:
            row['tpot_s'] = 0.02
        opened = summary.request_summary(rows, 10.0, True)
        self.assertEqual(opened['exploratory_slo']['passed'], 1)
        self.assertAlmostEqual(opened['exploratory_slo']['goodput_requests_per_s'], 0.1)

    def test_prometheus_labels_counters_histogram_means(self):
        a = summary.parse_metrics('vllm:prefix_cache_hits_total{engine="0",model_name="x"} 10\nvllm:request_decode_time_seconds_sum{engine="0"} 3\nvllm:request_decode_time_seconds_count{engine="0"} 2\n')
        b = summary.parse_metrics('vllm:prefix_cache_hits_total{model_name="x",engine="0"} 30\nvllm:request_decode_time_seconds_sum{engine="0"} 9\nvllm:request_decode_time_seconds_count{engine="0"} 4\n')
        value = summary.metrics_delta(a, b)
        self.assertEqual(value['totals']['vllm:prefix_cache_hits_total'], 20)
        self.assertEqual(value['derived']['histogram_means']['vllm:request_decode_time_seconds'], 3)
        self.assertNotIn('p95', value['derived']['histogram_means'])

    def test_counter_reset_and_missing_series_not_zero(self):
        a = summary.parse_metrics('vllm:num_preemptions_total{engine="0"} 5\nvllm:prefix_cache_hits_total 10\n')
        b = summary.parse_metrics('vllm:num_preemptions_total{engine="0"} 1\n')
        value = summary.metrics_delta(a, b)
        self.assertEqual(value['status'], 'INCOMPLETE')
        self.assertIsNone(value['totals']['vllm:num_preemptions_total'])
        self.assertIsNone(value['totals']['vllm:prefix_cache_hits_total'])

    def test_gpu_clipping_and_linear_integral(self):
        device = [{'t': float(t), 'uuid': 'fixture', 'memory_mib': 1000.0, 'utilization_pct': 50.0, 'power_w': 100.0 + 20 * t, 'temperature_c': 40.0} for t in range(3)]
        actual = summary.integrate_gpu({g: device for g in range(4)}, 0.5, 1.5, [0, 1, 2, 3], 3.0)
        self.assertAlmostEqual(actual['energy_j'], 480.0)
        self.assertAlmostEqual(actual['devices'][0]['measurements']['power_w']['time_weighted_mean'], 120.0)
        self.assertEqual(actual['devices'][0]['samples_inside'], 1)
        second = summary.integrate_gpu({g: device for g in range(4)}, 1.5, 2.0, [0, 1, 2, 3], 3.0)
        merged = summary.combine_gpu([actual, second])
        self.assertAlmostEqual(merged['energy_j'], 750.0)

    def test_missing_gpu_and_gaps_suppress_total_energy(self):
        device = [{'t': float(t), 'uuid': 'fixture', 'memory_mib': 1000.0, 'utilization_pct': 50.0, 'power_w': 100.0, 'temperature_c': 40.0} for t in (0, 10)]
        missing = summary.integrate_gpu({0: device}, 1.0, 2.0, [0, 1], 20.0)
        self.assertIsNone(missing['energy_j'])
        gap = summary.integrate_gpu({0: device}, 1.0, 2.0, [0], 3.0)
        self.assertIsNone(gap['energy_j'])
        self.assertEqual(gap['covered_energy_j'], 0.0)

    def test_end_to_end_weighted_rows_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['audit']['status'], 'PASS')
            self.assertEqual(len(data['rounds']), 2)
            self.assertEqual(len(data['audit']['excluded_stages']), 2)
            self.assertAlmostEqual(data['combined'][0]['summary']['output_tokens_per_s'], 0.3)
            self.assertTrue(data['combined'][0]['final_execution_verified'])
            summary.write_outputs(path_at(root, 'summary'), data)
            self.assertEqual({p.name for p in path_at(root, 'summary').iterdir()}, {'summary.json', 'summary.md', 'rounds.csv', 'combined.csv'})

    def test_cache_warm_phases_are_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 'cache', 1)
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['audit']['status'], 'PASS', data['audit']['errors'])
            self.assertEqual({r['phase'] for r in data['rounds']}, summary.CACHE_PHASES)
            self.assertEqual(sum((r['summary']['requests'] for r in data['rounds'])), 19)

    def test_running_report_excluded_but_finished_suite_provisional(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, report = fixture(root)
            (directory / 'status.json').unlink()
            write_text(directory / 'experiment.json', '{"status":"READY","profile":false}')
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(len(data['rounds']), 2)
            self.assertFalse(data['combined'][0]['final_execution_verified'])
            report['status'] = 'RUNNING'
            write_text(directory / 'scenario-decode.json', json.dumps(report))
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['rounds'], [])

    def test_failed_and_profile_reports_never_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, report = fixture(root, 'profile', 1)
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['rounds'], [])
            report['status'] = 'FAILED'
            report['suite'] = 'decode'
            report['runs'][0]['status'] = 'FAILED'
            failed = report['runs'][0]['stages'][1]['requests'][0]
            failed.update(ok=False, error={'type': 'FixtureFailure'})
            (directory / 'scenario-profile.json').unlink()
            write_text(directory / 'experiment.json', '{"status":"FAILED","profile":false}')
            write_text(directory / 'scenario-decode.json', json.dumps(report))
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['rounds'], [])
            self.assertTrue(any((x['failures'] for x in data['audit']['excluded_stages'])))

    def test_tampered_text_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, report = fixture(root)
            report['runs'][0]['stages'][1]['requests'][0]['text'] = 'changed'
            write_text(directory / 'scenario-decode.json', json.dumps(report))
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['audit']['status'], 'FAILED')
            self.assertTrue(any(('Text/hash mismatch' in x['error'] for x in data['audit']['errors'])))
            self.assertEqual(data['rounds'], [])
            self.assertEqual(data['combined'], [])

    def test_cache_phase_order_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, report = fixture(root, 'cache', 1)
            report['runs'][0]['stages'].reverse()
            write_text(directory / 'scenario-cache.json', json.dumps(report))
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertEqual(data['audit']['status'], 'FAILED')
            self.assertEqual(data['combined'], [])
            self.assertTrue(any(('reordered phases' in x['error'] for x in data['audit']['errors'])))

    def test_request_end_and_dispatch_must_fit_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, report = fixture(root, rounds=1)
            run = report['runs'][0]
            stage = deepcopy(run['stages'][1])
            for row in stage['requests']:
                row.update(ttft_s=100.0, e2e_s=100.2, elapsed_s=100.201, scheduled_ttft_s=100.0, scheduled_e2e_s=100.2)
            with self.assertRaisesRegex(ValueError, 'finishes after stage'):
                summary.validate_stage(report, run, stage, set())
            stage = deepcopy(run['stages'][1])
            stage['requests'][0]['start_unix'] += 1
            with self.assertRaisesRegex(ValueError, 'Actual dispatch'):
                summary.validate_stage(report, run, stage, set())

    def test_different_tokenizer_not_falsely_fair_singletons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, report = fixture(root)
            other = directory.with_name('other')
            shutil.copytree(directory, other)
            report['tokenizer_sha256'] = 'another-tokenizer'
            write_text(other / 'scenario-decode.json', json.dumps(report))
            data = summary.collect(path_at(root, 'results'), set(), 'Etc/UTC', 3.0)
            self.assertTrue(data['audit']['cross_configuration_inputs'])
            self.assertTrue(all((x['status'] == 'UNCOMPARABLE_COHORTS' and x['matched'] is False for x in data['audit']['cross_configuration_inputs'])))

    def test_profile_only_ast_exception_is_narrow(self):
        original = 'def workloads(suite):\n    if suite == "decode":\n        return [1]\n    if suite == "profile":\n        return [2]\n    raise ValueError(suite)\n'
        profile_changed = original.replace('return [2]', 'return [2, 3]')
        decode_changed = original.replace('return [1]', 'return [4]')
        self.assertEqual(summary.non_profile_ast_hash(original), summary.non_profile_ast_hash(profile_changed))
        self.assertNotEqual(summary.non_profile_ast_hash(original), summary.non_profile_ast_hash(decode_changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    log = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if result.wasSuccessful() else 'FAILED', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'summary_script_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), 'test_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'output': stream.getvalue(), 'scope': 'CPU synthetic fixtures; independent code review and actual saved-data audit remain separate evidence.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.output, json.dumps(log, indent=2) + '\n')
    print(stream.getvalue(), end='')
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
