#!/usr/bin/env python3
"""Synthetic CPU-only rank traces; validate Stage B profile analysis semantics."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name('analyze_long_context_profiles.py')
SPEC = importlib.util.spec_from_file_location('phase_b_profiles_fixture', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def fixture(directory, cap=0, labels=None):
    directory.mkdir()
    (directory / 'profiles').mkdir()
    phase_id = 'profile_micro'
    rows = []
    for index, offset in enumerate((0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75)):
        inputs = 65536 if index == 0 else 512
        first = 3.2 if index == 0 else 3.4
        rows.append(dict(index=index, request_id=f'{phase_id}/{index}', input_tokens=inputs, requested_output_tokens=128, shape_name=f'p{inputs}', planned_offset_s=offset, dispatch_offset_s=offset, dispatch_lag_s=0.0, end_offset_s=7.5, admitted=True, ok=True, timed_out=False, censored=False, http_status=200, done_received=True, error=None, ignore_eos=True, ttft_s=first - offset, e2e_s=7.5 - offset, usage=dict(prompt_tokens=inputs, completion_tokens=128, total_tokens=inputs + 128), finish_reason='length', full_text='answer', text_sha256=analysis.metrics.capacity.digest(b'answer'), prompt_sha256=analysis.metrics.hashed((inputs, index))))
    warm = dict(id='warm', role='warmup')
    phase = dict(id=phase_id, role='profile', mode='trace', duration_s=4.0, seed=1, arrivals=[dict(offset_s=r['planned_offset_s'], shape_name=r['shape_name']) for r in rows], shape_mix=[dict(name=f'p{x}', input_tokens=x, output_tokens=128, weight=1) for x in (512, 65536)])
    controls = [dict(path='/start_profile', payload=dict(profile_prefix=phase_id, delay_iterations=0, max_iterations=0), status='PASS', http_status=200, started_unix=999.0, finished_unix=999.5, started_monotonic=99.0, finished_monotonic=99.5), dict(path='/stop_profile', payload=None, status='PASS', http_status=200, started_unix=1009.0, finished_unix=1010.0, started_monotonic=109.0, finished_monotonic=110.0)]
    report = dict(schema_version=1, status='PASSED', tokenizer_sha256='fixture-tokenizer', plan=dict(phases=[warm, phase]), prepared_requests={phase_id: 7}, runs=[dict(phase=warm, excluded_from_performance=True), dict(phase=phase, excluded_from_performance=True, start_unix=1000.0, end_unix=1008.0, requests=rows)], profiling=dict(excluded_from_performance=True, sessions=[dict(phase_id=phase_id, prefix=phase_id, controls=controls)]))
    for endpoint in ('before', 'after'):
        path = directory / (endpoint + '.prom')
        write_text(path, 'vllm:num_preemptions_total{engine="0"} 0\n')
        report['runs'][-1]['metrics_' + endpoint] = dict(raw_file=str(path), sha256=analysis.metrics.capacity.digest(path.read_bytes()), http_status=200)
    config = dict(name=f'fixture-cap{cap}', args=['--tensor-parallel-size', '4', '--no-enable-prefix-caching', '--max-num-batched-tokens', '4096', '--long-prefill-token-threshold', str(cap), '--profiler-config', json.dumps(dict(max_iterations=0, delay_iterations=0, detailed_trace_annotation=False))])
    sources = {'status.json': dict(end_utc='fixture', exit_code=0, timed_out=False), 'experiment.json': dict(status='PASS', profile=True, excluded_from_performance=True), 'capacity.json': report, 'config.json': config}
    for name, value in sources.items():
        write_text(directory / name, json.dumps(value))
    if labels is None:
        labels = [(1, 4096, 0, 0)] * 16 + [(6, 3072, 1, 1)] + [(0, 0, 7, 7)] * 126 + [(0, 0, 6, 6)] if cap == 0 else [(1, 2048, 0, 0)] * 30 + [(2, 2560, 0, 0), (1, 2048, 1, 1), (5, 2560, 2, 2)] + [(0, 0, 7, 7)] * 125 + [(0, 0, 6, 6), (0, 0, 5, 5)]
    for rank in range(4):
        events = []
        time_us = 10000.0
        for index, (cn, ct, gn, gt) in enumerate(labels):
            name = f'execute_context_{cn}({ct})_generation_{gn}({gt})'
            scope = dict(cat='user_annotation', ph='X', name=name, ts=time_us, dur=1000.0, args={'External id': index}, pid=rank + 100, tid=rank + 100)
            events += [scope, dict(scope)]
            events += [dict(scope, cat='gpu_user_annotation')] * 4
            events.append(dict(cat='cpu_op', ph='X', name='aten::fixture', ts=time_us, dur=10.0))
            for kernel, duration in (('ncclDevKernel_AllReduce', 100.0), ('marlin_moe_kernel', 200.0), ('cutlass::SparseGemm', 150.0), ('sparse_mla_decode', 50.0)):
                events.append(dict(cat='kernel', ph='X', name=kernel, ts=time_us, dur=duration, args={'device': rank}))
            time_us += (200000.0 if cap == 0 else 100000.0) if cn else 5000.0
        trace = dict(distributedInfo=dict(rank=rank, world_size=4), baseTimeNanoseconds=1000000000000, displayTimeUnit='ms', traceEvents=events)
        path = directory / 'profiles' / f'{phase_id}_dp0_pp0_tp{rank}_dcp0_ep{rank}_rank{rank}.1.pt.trace.json.gz'
        write_bytes(path, gzip.compress(json.dumps(trace).encode()))
    return report


def alter_trace(directory, function):
    path = next((directory / 'profiles').glob('*tp0*trace.json.gz'))
    trace = json.loads(gzip.decompress(path.read_bytes()))
    function(trace)
    write_bytes(path, gzip.compress(json.dumps(trace).encode()))


with tempfile.TemporaryDirectory(prefix='phase-b-profile-analysis-') as temporary:
    root = Path(temporary)
    baseline, capped = (root / 'b06-baseline', root / 'b07-capped')
    fixture(baseline)
    fixture(capped, cap=2048)
    result = analysis.analyze([baseline, capped])
    check('matched complete four-rank profile fixtures pass', result['status'] == 'PASS' and result['same_workload_comparison'] == 'PASS')
    a, b = result['experiments']
    ta, tb = (a['traces'][0], b['traces'][0])
    check('CPU scopes deduplicated; repeated GPU scopes excluded', ta['iteration_count'] == 144)
    check('actual max/median context token counts come from labels', ta['context_tokens']['max'] == 4096 and tb['context_tokens']['max'] == 2560 and (tb['context_tokens']['p50'] == 2048))
    check('context totals independently reconcile all seven prompt lengths', a['prompt_token_accounting_matches_all_ranks'] and b['prompt_token_accounting_matches_all_ranks'] and (ta['context_tokens_sum'] == 68608))
    check('active iteration sequences match all four ranks', a['active_iteration_sequence_matches_all_ranks'] and b['active_iteration_sequence_matches_all_ranks'])
    check('generic mixed step alone does not prove long-short interleaving', ta['mixed_iteration_count'] == 1 and ta['inferred_long_short_context_steps'] == 0 and (ta['inferred_long_context_short_generation_steps'] == 0))
    check('workload bound identifies observed long plus short context', tb['inferred_long_short_context_steps'] == 1)
    check('workload bound identifies observed long context plus short decode', tb['inferred_long_context_short_generation_steps'] == 1)
    check('long-short attribution requires explicit zero-preemption evidence', b['preemptions'] == 0 and b['attribution_assumptions_verified'])
    check('pure decode total is separate from inserted decode during context progression', ta['decode_only_iteration_count'] == 127 and tb['decode_only_iteration_count'] == 127 and (tb['decode_only_between_first_last_inferred_long_context_steps'] == 0))
    check('single-request long context distribution excludes aggregate short batches', tb['single_request_long_context_tokens']['max'] == 2048)
    check('start/stop controls cover every complete long and short request', len(b['profile_control_coverage']['requests']) == 7 and all((r['enclosed'] for r in b['profile_control_coverage']['requests'])))
    categories = {c['category']: c for c in ta['categories']}
    check('legacy category priority keeps SparseGemm out of attention', categories['gemm_unspecified']['calls'] == 144 and categories['attention_explicit']['calls'] == 144)
    check('kernel categories retain calls and duration sums', categories['communication']['summed_ms'] == 14.4 and ta['cuda_kernel_events'] == 576)
    check('overlap-aware kernel union differs from summed durations', ta['kernel_union_ms'] < ta['kernel_duration_sum_ms'])
    check('trace clock microseconds plus base nanoseconds despite display ms', abs(ta['timeline'][0]['phase_offset_s'] - 0.01) < 1e-09)
    check('input trace and report hashes registered', all((str(p.resolve()) in result['evidence_ledger'] for p in [baseline / 'capacity.json', *list((baseline / 'profiles').glob('*.gz'))])))
    check('profile limitations reject E2E percentage interpretation', any(('not E2E' in x for x in result['limitations'])))
    unchanged = root / 'b08-cap-with-unchanged-execution'
    labels = [(1, 4096, 0, 0)] * 16 + [(6, 3072, 1, 1)] + [(0, 0, 7, 7)] * 126 + [(0, 0, 6, 6)]
    fixture(unchanged, cap=2048, labels=labels)
    actual = analysis.analyze([unchanged])
    check('configured cap never substitutes for observed 4096-token steps', actual['status'] == 'PASS' and actual['experiments'][0]['traces'][0]['context_tokens']['max'] == 4096)
    cadence = root / 'cadence-insertion'
    labels = [(1, 2048, 0, 0)] * 30 + [(2, 2560, 0, 0), (0, 0, 1, 1), (1, 2048, 1, 1), (5, 2560, 2, 2)] + [(0, 0, 7, 7)] * 125 + [(0, 0, 5, 5)] * 2
    fixture(cadence, cap=2048, labels=labels)
    cadence_result = analysis.analyze([cadence])['experiments'][0]['traces'][0]
    check('pure decode inserted between verified long context steps counted directly', cadence_result['decode_only_between_first_last_inferred_long_context_steps'] == 1 and cadence_result['decode_only_iteration_count'] == 128)
    output = root / 'derived'
    process = subprocess.run([sys.executable, str(SCRIPT), '--result-dirs', str(baseline), str(capped), '--output-prefix', str(output)], capture_output=True, text=True)
    check('full CLI creates JSON and Markdown', process.returncode == 0 and output.with_suffix('.json').exists() and output.with_suffix('.md').exists())
    repeat = subprocess.run([sys.executable, str(SCRIPT), '--result-dirs', str(baseline), '--output-prefix', str(output)], capture_output=True, text=True)
    check('existing analysis outputs are not overwritten', repeat.returncode != 0)
    missing = root / 'missing-rank'
    fixture(missing)
    next((missing / 'profiles').glob('*tp3*trace.json.gz')).unlink()
    check('missing rank invalidates capture manifest', analysis.analyze([missing])['status'] == 'NOT VERIFIED')
    no_cpu = root / 'missing-cpu'
    fixture(no_cpu)
    alter_trace(no_cpu, lambda t: t.update(traceEvents=[e for e in t['traceEvents'] if e.get('cat') != 'cpu_op']))
    check('CUDA alone is insufficient without rank CPU operations', analysis.analyze([no_cpu])['status'] == 'NOT VERIFIED')
    conflict = root / 'conflicting-annotation'
    fixture(conflict)
    alter_trace(conflict, lambda t: t['traceEvents'][1].update(dur=999.0))
    check('conflicting duplicate External id rejected', analysis.analyze([conflict])['status'] == 'NOT VERIFIED')
    invalid = root / 'invalid-kernel'
    fixture(invalid)
    alter_trace(invalid, lambda t: next((e for e in t['traceEvents'] if e.get('cat') == 'kernel')).update(ph='B'))
    check('non-X CUDA kernel rejected', analysis.analyze([invalid])['status'] == 'NOT VERIFIED')
    uncovered = root / 'uncovered-request'
    report = fixture(uncovered)
    report['profiling']['sessions'][0]['controls'][1]['started_unix'] = 1007.0
    write_text(uncovered / 'capacity.json', json.dumps(report))
    check('controls must enclose entire request phase', analysis.analyze([uncovered])['status'] == 'NOT VERIFIED')
    mismatch = root / 'prompt-mismatch'
    report = fixture(mismatch)
    report['runs'][-1]['requests'][0]['prompt_sha256'] = 'different'
    write_text(mismatch / 'capacity.json', json.dumps(report))
    check('different prompt identity prevents matched comparison', analysis.analyze([baseline, mismatch])['status'] == 'NOT VERIFIED')
    preempted = root / 'preempted'
    report = fixture(preempted, cap=2048)
    path = preempted / 'after.prom'
    write_text(path, 'vllm:num_preemptions_total{engine="0"} 1\n')
    report['runs'][-1]['metrics_after']['sha256'] = analysis.metrics.capacity.digest(path.read_bytes())
    write_text(preempted / 'capacity.json', json.dumps(report))
    observed = analysis.analyze([preempted])['experiments'][0]
    check('preemptions retain generic observed mixing but disable long-short inference', not observed['attribution_assumptions_verified'] and observed['traces'][0]['mixed_iteration_count'] == 2 and (observed['traces'][0]['inferred_long_short_context_steps'] is None))

print(json.dumps(dict(status='PASS', checks=len(checks), tests=checks,
                     analyzer_sha256=analysis.metrics.capacity.digest(SCRIPT.read_bytes()),
                     fixture_sha256=analysis.metrics.capacity.digest(Path(__file__).read_bytes())), indent=2))
