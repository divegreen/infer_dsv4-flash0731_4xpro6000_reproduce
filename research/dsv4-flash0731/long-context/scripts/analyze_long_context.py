#!/usr/bin/env python3
"""Read-only Stage B analysis of capacity-schema result directories.

Reuses frozen Stage A request validation, percentiles, SLOs, Prometheus deltas
and GPU integration. Trace mode uses scheduled latency and its fixed horizon;
finite traces never certify sustained online service capacity.
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
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'scenarios'
SPEC = importlib.util.spec_from_file_location('phase_b_capacity_metrics', OLD/'scripts/summarize_capacity.py')
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)
historical = capacity.historical
GAUGES = ('vllm:kv_cache_usage_perc', 'vllm:num_requests_waiting', 'vllm:num_requests_running')
LATENCIES = ('scheduled_ttft_s', 'tpot_s', 'scheduled_e2e_s', 'dispatch_lag_s')


def hashed(value):
    return historical.stable_hash(value)


def raw_bytes(path, ledger):
    path = path.resolve()
    raw = path.read_bytes()
    info = {'sha256': capacity.digest(raw), 'bytes': len(raw)}
    capacity.require(str(path) not in ledger or ledger[str(path)]['sha256'] == info['sha256'],
                     'Input changed while being read: '+str(path))
    ledger[str(path)] = info
    return raw


def read_json(path, ledger):
    return json.loads(raw_bytes(path, ledger))


def read_prom(item, directory, ledger):
    name = item if isinstance(item, str) else item.get('raw_file') or item.get('path')
    capacity.require(name, 'Missing raw Prometheus path')
    raw = raw_bytes(capacity.resolve_raw(name, directory), ledger)
    if isinstance(item, dict) and item.get('sha256'):
        capacity.require(capacity.digest(raw) == item['sha256'], 'Raw Prometheus hash mismatch: '+str(name))
    parsed = historical.parse_metrics(raw.decode())
    if isinstance(item, dict) and item.get('values') is not None:
        embedded = historical.parse_metrics('\n'.join(f'{key} {value}' for key, value in item['values'].items()))
        capacity.require(all(parsed.get(key) == value for key, value in embedded.items()),
                         'Raw and embedded Prometheus metrics differ: '+str(name))
        capacity.require({k: v for k, v in embedded.items() if k[0] in GAUGES}
                         == {k: v for k, v in parsed.items() if k[0] in GAUGES},
                         'Raw and embedded Prometheus gauge coverage differs: '+str(name))
    return parsed


def raw_gauges(run, directory, ledger):
    """Reparse every successful polling snapshot; do not sum KV percentages."""
    per_series = defaultdict(list)
    aggregate = defaultdict(list)
    expected_series = defaultdict(set)
    snapshots = []
    errors = []
    valid = 0
    for sample in run.get('metric_samples', []):
        if sample.get('http_status') != 200 or sample.get('error'):
            errors.append({'offset_s': sample.get('offset_s'), 'error': sample.get('error', 'HTTP metrics failure')})
            continue
        capacity.require(capacity.finite(sample.get('offset_s')), 'Invalid metric sample timestamp')
        values = read_prom(sample, directory, ledger)
        valid += 1
        grouped = defaultdict(list)
        for (name, labels), value in values.items():
            if name in GAUGES:
                per_series[(name, labels)].append(value)
                grouped[name].append(value)
                expected_series[name].add(labels)
        snapshots.append(grouped)
        for name, values_at_time in grouped.items():
            # Cache fractions belong to different engine pools. Their maximum
            # represents the busiest pool; adding fractions has no capacity meaning.
            aggregate[name].append(max(values_at_time) if name == GAUGES[0] else sum(values_at_time))
    def describe(values):
        return dict(capacity.dist(values), sample_mean=sum(values)/len(values) if values else None)
    series = [{'name': name, 'labels': dict(labels), **describe(values)}
              for (name, labels), values in sorted(per_series.items())]
    missing = []
    for name in GAUGES:
        if not expected_series[name]:
            missing.append(name+': no raw samples')
        elif any(len(s.get(name, [])) != len(expected_series[name]) for s in snapshots):
            missing.append(name+': engine/label series missing in some samples')
    return {'status': 'COMPLETE_SAMPLES' if valid and not errors and not missing else 'INCOMPLETE',
            'valid_samples': valid, 'poll_errors': errors, 'missing_series': missing,
            'per_series': series,
            'aggregate': {name: describe(aggregate[name]) for name in GAUGES},
            'basis': 'Arithmetic mean and maximum of raw periodic gauge snapshots; not time-weighted.',
            'kv_basis': 'kv_cache_usage_perc is a fraction (0..1). Aggregate is per-snapshot maximum across engine pools, not a sum or pooled capacity.'}


def request_stats(rows, wall):
    stats = capacity.group(rows)
    good = [row for row in rows if row.get('ok') and row['slo_pass'] is True]
    ok = [row for row in rows if row.get('ok')]
    stats.update(cohort_requests_s=len(ok)/wall,
                 cohort_output_tokens_s=sum(r['usage']['completion_tokens'] for r in ok)/wall,
                 cohort_slo_goodput_requests_s=len(good)/wall,
                 cohort_slo_goodput_output_tokens_s=sum(r['usage']['completion_tokens'] for r in good)/wall)
    return stats


def outstanding_peak(rows, planned=False):
    events = []
    for row in rows:
        if not planned and not row.get('admitted'):
            continue
        left = row['planned_offset_s'] if planned else row['dispatch_offset_s']
        right = row['end_offset_s']
        if right > left:
            events.extend(((left, 1), (right, -1)))
    active = peak = 0
    for _, change in sorted(events):  # half-open intervals: departures before arrivals
        active += change
        peak = max(peak, active)
    return peak


def associations(rows, wall):
    short = [r for r in rows if r['input_tokens'] == 512]
    dispatched = [r for r in short if r.get('admitted')]
    groups = {}
    for tokens in (16384, 32768, 65536):
        relevant = [r for r in rows if r['input_tokens'] == tokens and r.get('admitted')
                    and capacity.finite(r.get('ttft_s'))]
        for part in ('pre_first_text', 'post_first_text'):
            yes, no = [], []
            for row in dispatched:
                instant = row['dispatch_offset_s']
                def active(other):
                    first = other['dispatch_offset_s'] + other['ttft_s']
                    left, right = ((other['dispatch_offset_s'], first) if part == 'pre_first_text'
                                   else (first, other['end_offset_s']))
                    return left <= instant < right
                (yes if any(active(other) for other in relevant) else no).append(row)
            groups[f'p{tokens}_{part}'] = {
                'long_input_tokens': tokens, 'interval': part,
                'relevant_long_requests': len(relevant),
                'with_existing_overlap': request_stats(yes, wall),
                'without_existing_overlap': request_stats(no, wall)}
    return {'short_definition': 'input_tokens == 512; all output lengths retained and disclosed by shape',
            'undispatched_short_requests': len(short)-len(dispatched), 'groups': groups,
            'interpretation': 'Groups use only long requests already active at the short request POST time. Pre-first-text includes queue and prefill; post-first-text is a client response interval, not GPU decode time. Correlated batch, queue, arrival and request-length changes prevent causal attribution. Group goodput uses the whole phase wall, not exposure time.'}


def arrival_identity(report, phase, rows):
    records = []
    for row in sorted(rows, key=lambda r: r['index']):
        capacity.require(row.get('prompt_sha256'), 'Missing request prompt hash')
        records.append({key: row.get(key) for key in
                        ('index', 'shape_name', 'input_tokens', 'requested_output_tokens',
                         'planned_offset_s', 'prompt_sha256', 'ignore_eos')})
    identity = {'tokenizer_sha256': report.get('tokenizer_sha256'), 'mode': phase['mode'],
                'duration_s': phase.get('duration_s'), 'requests': records}
    return {'sha256': hashed(identity), 'requests': len(records),
            'definition': 'Sorted index, planned arrival, shape, input/output lengths, exact prompt SHA and EOS policy; tokenizer hash and mode/horizon. Priority is a treatment and excluded. HTTP dispatch/completion times are outcomes and excluded.'}


def analyze_run(report, run, path, architecture, devices, ledger):
    phase = run['phase']
    for endpoint in ('metrics_before', 'metrics_after'):
        if run.get(endpoint) and not (isinstance(run[endpoint], dict) and run[endpoint].get('error')):
            read_prom(run[endpoint], path.parent, ledger)
    # The historical recomputer knows burst's scheduled SLO semantics. Adapt
    # only in memory, then replace the horizon fields for an explicit trace.
    adapted = deepcopy(run) if phase['mode'] == 'trace' else run
    if phase['mode'] == 'trace':
        adapted['phase']['mode'] = 'burst'
    rows = [capacity.recompute_row(row, adapted['phase']['mode']) for row in run['requests']]
    stage = capacity.analyze_phase(report, adapted, path, architecture, devices, ledger)
    wall = stage['total_wall_s']
    capacity.require(len({r['index'] for r in rows}) == len(rows), 'Duplicate request index')
    capacity.require(sorted(r['index'] for r in rows) == list(range(len(rows))), 'Request index sequence is incomplete')
    if run.get('raw_requests_jsonl'):
        raw_path = capacity.resolve_raw(run['raw_requests_jsonl'], path.parent)
        streamed = [json.loads(line) for line in raw_bytes(raw_path, ledger).decode().splitlines() if line.strip()]
        capacity.require(sorted(streamed, key=lambda r: r['index']) == sorted(run['requests'], key=lambda r: r['index']),
                         'Streaming request JSONL differs from capacity.json')
    if phase['mode'] == 'trace':
        horizon = phase['duration_s']
        capacity.require(capacity.finite(horizon) and 0 < horizon <= wall+1e-5, 'Invalid fixed trace horizon')
        arrivals = phase['arrivals']
        capacity.require(len(arrivals) == len(rows), 'Trace plan/request count mismatch')
        for row in sorted(rows, key=lambda r: r['index']):
            planned = arrivals[row['index']]
            capacity.require(row['planned_offset_s'] == planned['offset_s'] and row['shape_name'] == planned['shape_name'],
                             'Trace row differs from explicit planned arrival')
            capacity.require(0 <= row['planned_offset_s'] < horizon, 'Trace arrival outside fixed horizon')
        within = [r for r in rows if r.get('ok') and r['end_offset_s'] <= horizon]
        good = [r for r in rows if r.get('ok') and r['slo_pass'] is True]
        stage.update(mode='trace', measurement_window_s=horizon, drain_s=max(0., wall-horizon),
                     in_flight_at_window_end=sum(r.get('admitted') and r['dispatch_offset_s'] <= horizon < r['end_offset_s'] for r in rows))
        stage['throughput'].update(
            completed_in_window_requests_s=len(within)/horizon,
            completion_credited_in_window_output_tokens_s=sum(r['usage']['completion_tokens'] for r in within)/horizon,
            completed_in_window_slo_goodput_requests_s=sum(r['slo_pass'] is True for r in within)/horizon,
            offered_window_slo_goodput_requests_s=len(good)/horizon)
        stage['stability'] = {'classification': 'NOT_SUSTAINED_ARRIVAL', 'trend_verdict': None,
                              'reasons': ['Fixed explicit trace describes interference, not sustained offered-rate capacity.']}
    stage['by_shape'] = {name: request_stats([r for r in rows if r['shape_name'] == name], wall)
                         for name in sorted({r['shape_name'] for r in rows})}
    for name, stats in stage['by_shape'].items():
        selected = [r for r in rows if r['shape_name'] == name]
        stats['input_tokens'] = sorted({r['input_tokens'] for r in selected})
        stats['requested_output_tokens'] = sorted({r['requested_output_tokens'] for r in selected})
        stats['priorities'] = sorted({r.get('priority', 0) for r in selected})
    stage['short512'] = request_stats([r for r in rows if r['input_tokens'] == 512], wall)
    stage['client_peak_dispatched_outstanding'] = outstanding_peak(rows)
    stage['client_peak_planned_outstanding'] = outstanding_peak(rows, planned=True)
    stage['raw_gauges'] = raw_gauges(run, path.parent, ledger)
    stage['preemptions'] = stage['prometheus'].get('totals', {}).get('vllm:num_preemptions_total')
    stage['arrival_identity'] = arrival_identity(report, phase, rows)
    stage['comparison_id'] = phase.get('comparison_id', phase['id'])
    stage['associations'] = associations(rows, wall)
    return stage


def identity_checks(stages):
    groups = defaultdict(list)
    for stage in stages:
        if stage['included_as_measurement'] and stage['mode'] != 'closed-loop':
            groups[(stage['comparison_id'], stage['seed'])].append(stage)
    checks = []
    for (name, seed), members in sorted(groups.items()):
        identities = {m['arrival_identity']['sha256'] for m in members}
        checks.append({'comparison_id': name, 'seed': seed,
                       'status': ('SINGLE_OBSERVATION' if len(members) == 1 else 'PASS' if len(identities) == 1 else 'MISMATCH'),
                       'members': [{'experiment': m['experiment'], 'phase_id': m['phase_id'],
                                    'arrival_sha256': m['arrival_identity']['sha256']} for m in members]})
    return checks


def analyze(directories):
    ledger, stages, excluded, errors, contexts = {}, [], [], [], []
    for script in (Path(__file__), OLD/'scripts/summarize_capacity.py', OLD/'scripts/summarize_scenarios.py'):
        raw_bytes(script, ledger)
    for directory in directories:
        path = directory.resolve()/'capacity.json'
        try:
            report = read_json(path, ledger)
            capacity.require(report.get('schema_version') == 1, 'Unsupported capacity schema')
            capacity.require(isinstance(report.get('plan'), dict), 'Missing plan')
            if report.get('status') not in ('PASSED', 'FAILED', 'CENSORED'):
                excluded.append({'source': str(path), 'reason': 'Client is nonterminal: '+str(report.get('status'))})
                continue
            config = read_json(path.parent/'config.json', ledger)
            architecture = config.get('name', path.parent.name)
            deployment = hashed(config)
            supervisor = read_json(path.parent/'status.json', ledger) if (path.parent/'status.json').exists() else None
            experiment = read_json(path.parent/'experiment.json', ledger) if (path.parent/'experiment.json').exists() else None
            context = capacity.completion_context(report, supervisor, experiment)
            contexts.append({'source': str(path), **context})
            devices, warnings = historical.load_gpu(path.parent/'gpu.csv', ZoneInfo('UTC'), ledger)
            for run in report.get('runs', []):
                try:
                    stage = analyze_run(report, run, path, architecture, devices, ledger)
                    stage.update(deployment_key=deployment, gpu_warnings=warnings,
                                 owned_run_terminal_success=context['eligible_for_capacity_qualification'],
                                 eligible_for_capacity_qualification=False)
                    stages.append(stage)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    errors.append({'source': str(path), 'phase_id': run.get('phase', {}).get('id'), 'error': str(exc)})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'source': str(path), 'error': str(exc)})
    identities = identity_checks(stages)
    for path, info in ledger.items():
        if capacity.digest(Path(path).read_bytes()) != info['sha256']:
            errors.append({'source': path, 'error': 'Evidence changed during analysis'})
    if any(x['status'] == 'MISMATCH' for x in identities):
        errors.append({'error': 'Declared comparison groups contain different arrival/prompt identities'})
    if not stages:
        errors.append({'error': 'No analyzable terminal phases'})
    return {'schema_version': 1, 'status': 'FAIL' if errors else 'PASS',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Stage B descriptive long/short scheduling comparison; no new stable-capacity or causal kernel claim.',
            'short_slo': 'Input 512: scheduled TTFT<=1s, TPOT<=50ms, scheduled E2E<=1s+50ms*(actual output tokens-1). Failures remain in denominator.',
            'long_slo': '16K retains TTFT<=8s; >16K TTFT/E2E SLO not defined. Latencies and TPOT still reported.',
            'limitations': ['TPOT is average post-first-text time per output token, not per-token ITL.',
                            'P95/P99 describe finite samples; small cohorts cannot establish precise tails.',
                            'GPU utilization is sampled device busy, not SM occupancy or PCIe bandwidth.',
                            'Allocated GPU memory is not occupied KV; KV gauges are per-engine fractions.',
                            'Preemption zero and low KV gauge do not prove unlimited cache capacity.',
                            'Fixed arrival traces and closed-loop runs cannot certify sustained service capacity.'],
            'stages': stages, 'arrival_identity_checks': identities, 'run_contexts': contexts,
            'excluded': excluded, 'errors': errors, 'evidence_ledger': ledger}


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def quantiles(stats, key, scale=1):
    values = stats['metrics'][key]
    return ' / '.join(number(values[q]*scale if values[q] is not None else None) for q in ('p50', 'p95', 'p99'))


def markdown(report):
    lines = ['# 阶段 B：长上下文混合调度离线复算', '',
             f"分析状态：{report['status']}；{len(report['stages'])} 个 phase。原始输入 SHA256、逐 engine gauge、GPU 积分和完整分组见同名 JSON。", '',
             '吞吐与 Goodput 分母为完整 cohort wall（固定到达窗口加 drain）。trace 使用计划到达作为 TTFT/E2E 起点，不能据此认证持续容量。失败、超时和保护停止保留；warmup/profile 不进入下表。', '',
             '| 实验 / phase | mode | window / drain s | short 成功 / 全部 | short SLO 满足率 | short Goodput req/s | total tok/s | peak outstanding |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    stages = [s for s in report['stages'] if s['included_as_measurement']]
    for s in stages:
        g = s['short512']
        lines.append(f"| {s['experiment']} / {s['phase_id']} | {s['mode']} | {number(s['measurement_window_s'])} / {number(s['drain_s'])} | {g['successful']} / {g['requests']} | {number(g['slo_joint_pass_rate'])} | {number(g['cohort_slo_goodput_requests_s'])} | {number(s['throughput']['cohort_output_tokens_s'])} | {s['client_peak_dispatched_outstanding']} |")
    lines += ['', '## 每种请求形状的延迟', '',
              '每格为 P50 / P95 / P99；TTFT、E2E 均按计划到达，单位秒，TPOT 单位毫秒。>16K 不套用未定义的 TTFT/E2E 阈值。', '',
              '| 实验 / phase / shape | 成功 / 全部 | TTFT s | TPOT ms | E2E s |',
              '| --- | ---: | --- | --- | --- |']
    for s in stages:
        for name, g in s['by_shape'].items():
            lines.append(f"| {s['experiment']} / {s['phase_id']} / {name} | {g['successful']} / {g['requests']} | {quantiles(g, 'scheduled_ttft_s')} | {quantiles(g, 'tpot_s', 1000)} | {quantiles(g, 'scheduled_e2e_s')} |")
    lines += ['', '## KV、队列与资源', '',
              'KV 是 periodic snapshot 中最忙 engine 的使用比例，表内换算百分比；其 mean/max 不是多个 engine 容量之和。waiting/running 为各 engine 请求数之和。功耗由每卡 GPU.csv 按实际 phase 时窗积分；显存 max 是采样值。', '',
              '| 实验 / phase | KV mean / max % | waiting mean / max | running mean / max | 抢占差分 | GPU power mean W | 显存 max MiB / GPU |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for s in stages:
        gauges = s['raw_gauges']['aggregate']
        values = []
        for key, scale in ((GAUGES[0], 100), (GAUGES[1], 1), (GAUGES[2], 1)):
            values.append(' / '.join(number(gauges[key][field]*scale if gauges[key][field] is not None else None) for field in ('sample_mean', 'max')))
        gpu = s['gpu'].get('devices', [])
        powers = [d['measurements']['power_w']['time_weighted_mean'] for d in gpu]
        power = sum(powers) if len(powers) == 4 and all(v is not None for v in powers) else None
        memory = ', '.join(f"{d['gpu']}:{number(d['measurements']['memory_mib']['observed_max'], 0)}" for d in gpu)
        lines.append(f"| {s['experiment']} / {s['phase_id']} | {' | '.join(values)} | {number(s['preemptions'], 0)} | {number(power, 1)} | {memory} |")
    lines += ['', '## 短请求与长请求区间的关联', '',
              'short 为精确 512 输入 tokens。只按其实际 POST 时刻已存在的 16K/32K/64K 请求分组，不使用未来请求。到首文本区间包含排队与 Prefill，首文本后也不是独占 Decode kernel；这些是描述性关联，不能认定 kernel 因果。', '',
              '| 实验 / phase | 长请求区间 | N 有 / 无 | short TTFT P95 有 / 无 s | short TPOT P95 有 / 无 ms |',
              '| --- | --- | ---: | --- | --- |']
    for s in stages:
        for name, a in s['associations']['groups'].items():
            if not a['relevant_long_requests']:
                continue
            yes, no = a['with_existing_overlap'], a['without_existing_overlap']
            ttft = ' / '.join(number(g['metrics']['scheduled_ttft_s']['p95']) for g in (yes, no))
            tpot = ' / '.join(number(g['metrics']['tpot_s']['p95']*1000 if g['metrics']['tpot_s']['p95'] is not None else None) for g in (yes, no))
            lines.append(f"| {s['experiment']} / {s['phase_id']} | {name} | {yes['requests']} / {no['requests']} | {ttft} | {tpot} |")
    lines += ['', '## 到达身份与证据状态', '']
    for item in report['arrival_identity_checks']:
        lines.append(f"- {item['comparison_id']} / seed {item['seed']}: {item['status']}，{len(item['members'])} 个观察值。")
    for context in report['run_contexts']:
        if not context['eligible_for_capacity_qualification']:
            lines.append(f"- 非完整成功实验，仅描述：{context['source']}；{' '.join(context['reasons'])}")
    for item in report['excluded'] + report['errors']:
        lines.append('- '+json.dumps(item, ensure_ascii=False))
    lines += ['', '局限：'+ ' '.join(report['limitations'])]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(suffix) for suffix in ('.json', '.md')]
    capacity.require(not any((path.exists() for path in outputs)), 'Use a new output prefix; existing analysis is preserved')
    report = analyze(args.result_dirs)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, body in zip(outputs, (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', markdown(report))):
        with open_output(path, 'x') as handle:
            handle.write(body)
    print(json.dumps({'status': report['status'], 'stages': len(report['stages']), 'errors': report['errors'], 'output_prefix': str(args.output_prefix)}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
