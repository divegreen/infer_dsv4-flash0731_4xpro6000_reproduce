#!/usr/bin/env python3
"""Offline Stage B CPU/CUDA profile evidence, never a performance ranking.

Iteration labels describe actual scheduled request/token counts. They contain
no request IDs: long/short attribution below is explicitly workload-constrained.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'scenarios'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load('phase_b_legacy_profile', OLD/'scripts/analyze_profiles.py')
metrics = load('phase_b_request_evidence', ROOT/'scripts/analyze_long_context.py')
require = metrics.capacity.require
finite = metrics.capacity.finite
EXECUTE = re.compile(r'^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$')


def option(config, name, default=None):
    args = config.get('args', [])
    hits = [i for i, item in enumerate(args) if item == name or item.startswith(name+'=')]
    require(len(hits) <= 1, 'Duplicate configuration option: '+name)
    if not hits:
        return default
    index = hits[0]
    if args[index].startswith(name+'='):
        return args[index].split('=', 1)[1]
    require(index+1 < len(args), 'Missing configuration value: '+name)
    return args[index+1]


def control_coverage(report, status, experiment):
    require(status.get('end_utc') and status.get('exit_code') == 0
            and status.get('timed_out') is False and not status.get('supervisor_error'), 'Supervisor is not terminal exit0')
    require(experiment.get('status') == 'PASS' and experiment.get('profile') is True
            and experiment.get('excluded_from_performance') is True, 'Not a successful excluded profile experiment')
    require(report.get('schema_version') == 1 and report.get('status') == 'PASSED'
            and not report.get('active_phase'), 'Capacity-schema profile report is not terminal PASSED')
    phases = report['plan']['phases']
    require(len(phases) >= 2 and phases[-1].get('role') == 'profile'
            and all(p.get('role') == 'warmup' for p in phases[:-1]), 'Profile plan must contain only warmup then one profile')
    require([r['phase'] for r in report['runs']] == phases, 'Profile runs differ from planned phases')
    require(all(r.get('excluded_from_performance') is True for r in report['runs']), 'Profile run lacks exclusion flag')
    run = report['runs'][-1]
    phase = run['phase']
    require(phase['mode'] == 'trace', 'Expected fixed profile trace')
    profiling = report['profiling']
    require(profiling.get('excluded_from_performance') is True, 'Missing profile exclusion flag')
    sessions = profiling['sessions']
    require(len(sessions) == 1, 'Expected exactly one profile session')
    capture = sessions[0]
    require(capture.get('phase_id') == phase['id'] == capture.get('prefix') and not capture.get('stop_error'), 'Profile capture identity/stop failure')
    controls = capture['controls']
    require(len(controls) == 2 and [x.get('path') for x in controls] == ['/start_profile', '/stop_profile'], 'Incomplete profile controls')
    require(all(x.get('status') == 'PASS' and x.get('http_status') == 200 and not x.get('error') for x in controls), 'Profile controls not HTTP200/PASS')
    start, stop = controls
    require(start.get('payload') == dict(profile_prefix=phase['id'], delay_iterations=0, max_iterations=0)
            and stop.get('payload') is None, 'Profile controls unexpectedly cap iterations or use other payload')
    require(all(finite(x.get(k)) for x in controls for k in ('started_unix', 'finished_unix', 'started_monotonic', 'finished_monotonic')), 'Invalid control timestamps')
    require(all(x['started_unix'] <= x['finished_unix'] and x['started_monotonic'] <= x['finished_monotonic'] for x in controls)
            and start['finished_monotonic'] <= stop['started_monotonic'], 'Invalid profile control order')
    require(start['finished_unix'] <= run['start_unix'] <= run['end_unix'] <= stop['started_unix'], 'Controls do not enclose entire phase')
    rows = [metrics.capacity.recompute_row(row, 'burst') for row in run['requests']]
    require(len(rows) == report['prepared_requests'][phase['id']] and all(r.get('ok') for r in rows), 'Incomplete/failed profile requests')
    require(len({r['index'] for r in rows}) == len(rows), 'Duplicate profile request index')
    arrivals = phase['arrivals']
    require(len(arrivals) == len(rows), 'Profile trace arrival count mismatch')
    coverage = []
    for row in sorted(rows, key=lambda r: r['index']):
        arrival = arrivals[row['index']]
        require(arrival['offset_s'] == row['planned_offset_s'] and arrival['shape_name'] == row['shape_name'], 'Profile request differs from fixed arrivals')
        dispatched, finished = run['start_unix']+row['dispatch_offset_s'], run['start_unix']+row['end_offset_s']
        require(start['finished_unix'] <= dispatched <= finished <= stop['started_unix'], 'Profile control misses a request interval')
        coverage.append(dict(index=row['index'], input_tokens=row['input_tokens'],
                             dispatch_unix=dispatched, first_text_unix=dispatched+row['ttft_s'],
                             end_unix=finished, enclosed=True))
    return run, rows, dict(status='PASS', phase_id=phase['id'], controls=controls, requests=coverage,
                          interpretation='Control acknowledgements enclose each complete client request. This does not by itself prove every CUDA kernel has a request ID attribution.')


def trace_analysis(path, run, controls, workload_bound, ledger):
    raw = metrics.raw_bytes(path, ledger)
    trace = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
    require(isinstance(trace, dict) and isinstance(trace.get('traceEvents'), list), 'Invalid trace root/events')
    identity = legacy.trace_identity(path)
    require(identity is not None, 'Unrecognized rank trace filename')
    distributed = trace.get('distributedInfo', {})
    require(distributed.get('rank') == identity['rank'] and distributed.get('world_size') == 4, 'Trace distributed rank/world-size mismatch')
    require(identity['tp'] == identity['rank'] and identity['dp'] == identity['pp'] == identity['dcp'] == 0, 'Expected single TP4 engine trace')
    base_ns = trace.get('baseTimeNanoseconds')
    require(finite(base_ns), 'Trace requires baseTimeNanoseconds for host/request alignment')
    events = trace['traceEvents']
    kernel_names = defaultdict(lambda: dict(calls=0, duration_us=0.))
    categories = defaultdict(lambda: dict(calls=0, duration_us=0.))
    intervals = []
    cpu_ops = 0
    external = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get('cat') == 'cpu_op' and event.get('ph') == 'X':
            require(finite(event.get('ts')) and finite(event.get('dur')) and event['dur'] >= 0, 'Invalid CPU operation event')
            cpu_ops += 1
        if event.get('cat') == 'user_annotation' and str(event.get('name', '')).startswith('execute_context'):
            require(event.get('ph') == 'X' and finite(event.get('ts'))
                    and finite(event.get('dur')) and event['dur'] >= 0, 'Invalid CPU iteration scope')
            require(EXECUTE.fullmatch(event['name']), 'Unrecognized actual iteration label')
            eid = (event.get('args') or {}).get('External id')
            if eid is not None:
                signature = (event['name'], event['ts'], event['dur'])
                require(eid not in external or external[eid] == signature, 'Conflicting duplicate CPU External id')
                external[eid] = signature
        if event.get('cat') != 'kernel':
            continue
        require(event.get('ph') == 'X' and finite(event.get('ts')) and finite(event.get('dur'))
                and event['dur'] > 0, 'Invalid CUDA kernel event')
        device = (event.get('args') or {}).get('device')
        require(device == identity['rank'], 'CUDA kernel device differs from rank')
        name = str(event.get('name', '<unnamed>'))
        for target in (kernel_names[name], categories[legacy.category(name)]):
            target['calls'] += 1
            target['duration_us'] += event['dur']
        intervals.append((event['ts'], event['ts']+event['dur']))
    require(cpu_ops > 0 and intervals, 'Each rank must contain CPU operation and CUDA kernel events')
    scopes = legacy.cpu_iterations(events)
    require(scopes, 'Missing CPU execute iteration annotations')
    timeline = []
    for index, item in enumerate(scopes):
        context_n, context_t, generation_n, generation_t = map(int, EXECUTE.fullmatch(item['name']).groups())
        require((context_n == 0) == (context_t == 0) and (generation_n == 0) == (generation_t == 0), 'Inconsistent iteration request/token counts')
        begin = base_ns/1e9+item['ts']/1e6
        end = begin+item['duration_ms']/1000
        long_present = workload_bound and context_t > 512*context_n
        timeline.append(dict(index=index, name=item['name'], external_id=item['external_id'],
                             context_requests=context_n, context_tokens=context_t,
                             generation_requests=generation_n, generation_tokens=generation_t,
                             host_duration_ms=item['duration_ms'], start_unix=begin, end_unix=end,
                             phase_offset_s=begin-run['start_unix'],
                             mixed_context_generation=context_n > 0 and generation_n > 0,
                             multiple_context_requests=context_n > 1,
                             inferred_long_context=long_present,
                             inferred_long_plus_short_context=long_present and context_n > 1,
                             inferred_long_context_plus_short_generation=long_present and generation_n > 0))
    # Capture boundaries use the whole POST control operation, allowing clocks
    # to differ by at most 100ms. Client requests have the stricter ack boundary.
    left, right = controls['controls'][0]['started_unix'], controls['controls'][1]['finished_unix']
    observed = [t['start_unix'] for t in timeline]+[t['end_unix'] for t in timeline]
    observed += [base_ns/1e9+min(a for a, _ in intervals)/1e6,
                 base_ns/1e9+max(b for _, b in intervals)/1e6]
    require(min(observed) >= left-.1 and max(observed) <= right+.1, 'Trace event times exceed profile control capture bounds')
    active = [t for t in timeline if t['context_requests'] or t['generation_requests']]
    context = [t for t in timeline if t['context_requests']]
    generation_only = [t for t in timeline if not t['context_requests'] and t['generation_requests']]
    long_context = [t for t in timeline if t['inferred_long_context']]
    def between_context(bounds):
        return sum(bounds[0]['index'] < t['index'] < bounds[-1]['index'] for t in generation_only) if bounds else 0
    single_long = [t['context_tokens'] for t in context if t['context_requests'] == 1 and t['inferred_long_context']]
    counts = Counter(t['name'] for t in timeline)
    total = sum(v['duration_us'] for v in categories.values())
    return dict(path=str(path.resolve()), sha256=metrics.capacity.digest(raw), bytes=len(raw), identity=identity,
                trace_clock_basis='Unix seconds = baseTimeNanoseconds/1e9 + Chrome ts/1e6; dur is microseconds regardless of displayTimeUnit.',
                cpu_operation_events=cpu_ops, cuda_kernel_events=len(intervals),
                iteration_count=len(timeline), active_iteration_count=len(active),
                context_iteration_count=len(context), context_tokens=metrics.capacity.dist([t['context_tokens'] for t in context]),
                context_tokens_sum=sum(t['context_tokens'] for t in context),
                generation_tokens_sum=sum(t['generation_tokens'] for t in active),
                single_request_long_context_tokens=metrics.capacity.dist(single_long),
                mixed_iteration_count=sum(t['mixed_context_generation'] for t in timeline),
                context_only_iteration_count=sum(t['context_requests'] > 0 and t['generation_requests'] == 0 for t in timeline),
                decode_only_iteration_count=len(generation_only),
                decode_only_between_first_last_context_steps=between_context(context),
                decode_only_between_first_last_inferred_long_context_steps=between_context(long_context),
                multi_context_iteration_count=sum(t['multiple_context_requests'] for t in timeline),
                inferred_long_short_context_steps=sum(t['inferred_long_plus_short_context'] for t in timeline),
                inferred_long_context_short_generation_steps=sum(t['inferred_long_context_plus_short_generation'] for t in timeline),
                iteration_label_distribution=dict(sorted(counts.items())), timeline=timeline,
                active_iteration_sequence_sha256=metrics.hashed([t['name'] for t in active]),
                kernel_duration_sum_ms=total/1000, kernel_union_ms=legacy.union_duration(intervals)/1000,
                kernel_span_ms=(max(b for _, b in intervals)-min(a for a, _ in intervals))/1000,
                categories=[dict(category=name, calls=value['calls'], summed_ms=value['duration_us']/1000,
                                 fraction_of_kernel_sum=value['duration_us']/total)
                            for name, value in sorted(categories.items(), key=lambda x: -x[1]['duration_us'])],
                top_kernels=[dict(name=name, calls=value['calls'], category=legacy.category(name), summed_ms=value['duration_us']/1000)
                             for name, value in sorted(kernel_names.items(), key=lambda x: -x[1]['duration_us'])[:40]])


def analyze_directory(directory, ledger):
    directory = directory.resolve()
    records = {name: metrics.read_json(directory/name, ledger) for name in ('status.json', 'experiment.json', 'capacity.json', 'config.json')}
    report, config = records['capacity.json'], records['config.json']
    run, rows, coverage = control_coverage(report, records['status.json'], records['experiment.json'])
    raw_jsonl = run.get('raw_requests_jsonl')
    if raw_jsonl:
        data = metrics.raw_bytes(metrics.capacity.resolve_raw(raw_jsonl, directory), ledger)
        streamed = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        require(sorted(streamed, key=lambda r: r['index']) == sorted(run['requests'], key=lambda r: r['index']), 'Profile JSONL/report mismatch')
    shape_counts = Counter((r['input_tokens'], r['requested_output_tokens']) for r in rows)
    fixed = shape_counts == {(65536, 128): 1, (512, 128): 6}
    args = config.get('args', [])
    require('--no-enable-prefix-caching' in args, 'Profile analysis expects APC off')
    require(option(config, '--tensor-parallel-size') == '4', 'Profile analysis expects TP4')
    require(option(config, '--speculative-config') is None, 'Speculative path outside profile attribution scope')
    options = json.loads(option(config, '--profiler-config', '{}'))
    require(options.get('max_iterations') == 0 and options.get('delay_iterations') == 0
            and options.get('detailed_trace_annotation') is False, 'Profile is capped/delayed or uses unsupported detailed annotations')
    for endpoint in ('metrics_before', 'metrics_after'):
        if run.get(endpoint) and not (isinstance(run[endpoint], dict) and run[endpoint].get('error')):
            metrics.read_prom(run[endpoint], directory, ledger)
    counters = metrics.capacity.metric_evidence(run, directory, ledger)
    preemptions = counters.get('totals', {}).get('vllm:num_preemptions_total')
    zero_preemption = preemptions == 0 and counters.get('status') == 'PASS' and counters.get('measurement_basis') == 'explicit_before_after'
    paths = sorted(set((directory/'profiles').glob('*.pt.trace.json.gz')) | set((directory/'profiles').glob('*.pt.trace.json')))
    manifest = legacy.validate_manifest(paths, (run['phase']['id'],), (0, 1, 2, 3))
    require(manifest['complete'], 'Incomplete rank trace manifest: '+str(manifest['errors']))
    traces = [trace_analysis(path, run, coverage, fixed and zero_preemption, ledger) for path in paths]
    prompt_tokens = sum(r['input_tokens'] for r in rows)
    matches = all(t['context_tokens_sum'] == prompt_tokens for t in traces)
    attribution_verified = fixed and zero_preemption and matches
    if not attribution_verified:
        for trace in traces:
            trace['inferred_long_short_context_steps'] = None
            trace['inferred_long_context_short_generation_steps'] = None
            trace['decode_only_between_first_last_inferred_long_context_steps'] = None
            trace['single_request_long_context_tokens'] = metrics.capacity.dist([])
            for step in trace['timeline']:
                for field in ('inferred_long_context', 'inferred_long_plus_short_context', 'inferred_long_context_plus_short_generation'):
                    step[field] = None
    same_sequence = len({t['active_iteration_sequence_sha256'] for t in traces}) == 1
    warnings = []
    if not fixed:
        warnings.append('Workload is not exactly one 64K + six 512 requests; long/short bound inference disabled.')
    if not matches:
        warnings.append('CPU context token sum differs from requested prompt total; do not assume every prefill token was captured exactly once.')
    if not zero_preemption:
        warnings.append('Explicit profile counter endpoints do not establish zero preemptions; long/short count-bound attribution is not verified.')
    if not same_sequence:
        warnings.append('Active iteration labels differ across ranks; capture completeness/rank behavior requires inspection.')
    return dict(experiment=directory.name, config_name=config.get('name'), profile_phase=run['phase']['id'],
                profile_request_count=len(rows), input_token_total=prompt_tokens,
                same_workload_identity=metrics.arrival_identity(report, run['phase'], rows),
                fixed_one_long_six_short=fixed, profile_control_coverage=coverage,
                attribution_assumptions_verified=attribution_verified, preemptions=preemptions,
                preemption_counter_evidence=counters,
                long_prefill_token_threshold=int(option(config, '--long-prefill-token-threshold', 0)),
                max_num_batched_tokens=int(option(config, '--max-num-batched-tokens', 4096)),
                prompt_token_accounting_matches_all_ranks=matches,
                active_iteration_sequence_matches_all_ranks=same_sequence,
                manifest=manifest, traces=traces, warnings=warnings,
                attribution='With exactly one 64K and six <=512-token prompts, explicit zero preemptions, APC off and fully reconciled prompt-token totals, a step with context_tokens > 512 * context_requests must include the long context. If it also has >1 context request, a short context shares that step; if it has generation requests, those belong to other (short) requests. Aggregate labels provide this bound, not request-ID attribution. Last small long chunks can remain ambiguous. No kernel duration is assigned to an individual request.')


def analyze(directories):
    ledger, experiments, errors = {}, [], []
    for script in (Path(__file__), ROOT/'scripts/analyze_long_context.py', OLD/'scripts/analyze_profiles.py',
                   OLD/'scripts/summarize_capacity.py', OLD/'scripts/summarize_scenarios.py'):
        metrics.raw_bytes(script, ledger)
    source_root = ROOT.parents[2]/'.venv-vllm/lib/python3.12/site-packages/vllm'
    semantics = []
    for relative in ('v1/utils.py', 'v1/worker/gpu_worker.py'):
        path = source_root/relative
        if path.exists():
            metrics.raw_bytes(path, ledger)
            semantics.append(str(path))
    for directory in directories:
        try:
            experiments.append(analyze_directory(directory, ledger))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(dict(directory=str(directory), error=str(exc)))
    identities = {x['same_workload_identity']['sha256'] for x in experiments}
    comparison = 'SINGLE_OBSERVATION' if len(experiments) == 1 else 'PASS' if len(identities) == 1 else 'MISMATCH'
    if comparison == 'MISMATCH':
        errors.append(dict(error='Profile workloads/arrivals/prompts differ; no matched threshold comparison'))
    for path, entry in ledger.items():
        if metrics.capacity.digest(Path(path).read_bytes()) != entry['sha256']:
            errors.append(dict(path=path, error='Input changed during analysis'))
    return dict(schema_version=1, status='PASS' if experiments and not errors else 'NOT VERIFIED',
                scope='Independent CPU/CUDA trace evidence; profile results excluded from normal performance rankings',
                same_workload_comparison=comparison, experiments=experiments, errors=errors,
                installed_annotation_semantics_sources=semantics, evidence_ledger=ledger,
                limitations=['CPU labels report actual scheduler counts, not request IDs or GPU completion boundaries.',
                             'NCCL and other kernel durations overlap across streams/ranks; summed category fractions are not E2E percentages.',
                             'Profile instrumentation changes execution timing and can change scheduling, so confirm benefit using unprofiled experiments.',
                             'Context token totals and identical active rank sequences are cross-checks, not independent model-correctness proof.',
                             'Presence proves observed co-scheduling; absence of bound-qualified steps does not exclude ambiguous small long chunks.',
                             'CPU timestamp alignment uses trace baseTimeNanoseconds; no CPU-to-CUDA kernel ownership is inferred.'])


def markdown(report):
    num = metrics.number
    lines = ['# 阶段 B：独立 Profile 与实际调度证据', '', f"状态：{report['status']}；跨实验负载身份：{report['same_workload_comparison']}。", '',
             '以下只分析独立 Profile，不把带 profiler 的延迟/吞吐并入正常性能表。CPU execute 标签来自实际调度请求数与 token 数，不能仅从 CLI threshold 推断分块或交织成功。', '',
             '| 实验 / rank | cap 配置 | context token P50 / max | context步 | mixed步 | 纯decode总步 / 长context之间 | 多context步 | 推断长+短context步 | 推断长context+短generation步 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for exp in report['experiments']:
        for trace in exp['traces']:
            lines.append(f"| {exp['experiment']} / {trace['identity']['rank']} | {exp['long_prefill_token_threshold']} | {num(trace['context_tokens']['p50'], 0)} / {num(trace['context_tokens']['max'], 0)} | {trace['context_iteration_count']} | {trace['mixed_iteration_count']} | {trace['decode_only_iteration_count']} / {num(trace['decode_only_between_first_last_inferred_long_context_steps'], 0)} | {trace['multi_context_iteration_count']} | {num(trace['inferred_long_short_context_steps'], 0)} | {num(trace['inferred_long_context_short_generation_steps'], 0)} |")
    lines += ['', 'mixed 表示同一步同时含 context 和 generation；多context本身也可能只是多个短请求。只有固定的一个64K加六个512请求、APC off、明确零抢占且prompt token总量核对一致时，context tokens > 512×context请求数才可推断该步含长请求；再结合请求数/generation数得到上表两个受限推断。最后较小的长chunk可能无法辨认；前提不足时推断列为空。完整逐步时间线和标签分布在JSON中。', '',
              '## 捕获与按 rank 完整性', '',
              '| 实验 | 控制覆盖请求 | prompt token核对 | 各rank active序列相同 | CPU / CUDA 事件数（每rank） |',
              '| --- | ---: | --- | --- | --- |']
    for exp in report['experiments']:
        events = '; '.join(f"r{t['identity']['rank']}: {t['cpu_operation_events']} / {t['cuda_kernel_events']}" for t in exp['traces'])
        lines.append(f"| {exp['experiment']} | {len(exp['profile_control_coverage']['requests'])} | {exp['prompt_token_accounting_matches_all_ranks']} | {exp['active_iteration_sequence_matches_all_ranks']} | {events} |")
        for warning in exp['warnings']:
            lines.append('\n- '+exp['experiment']+': '+warning)
    lines += ['', '## CUDA kernel 描述统计', '',
              '累计时长允许 stream/rank 重叠；比例分母仅为该 rank 的 kernel duration sum，不是端到端耗时比例，也不表示PCIe饱和度。名称分类沿用历史分析器，仅为启发式归类。', '',
              '| 实验 / rank | 类别 | 调用数 | 累计 ms | 占kernel sum % |',
              '| --- | --- | ---: | ---: | ---: |']
    for exp in report['experiments']:
        for trace in exp['traces']:
            for category in trace['categories']:
                lines.append(f"| {exp['experiment']} / {trace['identity']['rank']} | {category['category']} | {category['calls']} | {num(category['summed_ms'])} | {num(category['fraction_of_kernel_sum']*100, 2)} |")
    lines += ['', '局限：']+['- '+text for text in report['limitations']]
    for error in report['errors']:
        lines.append('- '+json.dumps(error, ensure_ascii=False))
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(suffix) for suffix in ('.json', '.md')]
    require(not any((p.exists() for p in outputs)), 'Use unique output prefix; previous analyses are immutable')
    result = analyze(args.result_dirs)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, body in zip(outputs, (json.dumps(result, indent=2, ensure_ascii=False) + '\n', markdown(result))):
        with open_output(path, 'x') as output:
            output.write(body)
    print(json.dumps(dict(status=result['status'], experiments=len(result['experiments']), errors=result['errors'], output_prefix=str(args.output_prefix))))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
