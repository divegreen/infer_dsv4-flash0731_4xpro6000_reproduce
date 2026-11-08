#!/usr/bin/env python3
"""Read-only detail tables from an independently recomputed capacity summary.

Copies per-phase/per-tier percentiles and qualifications without pooling,
recomputing SLO thresholds, or defining stability rules. No inference imports.
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
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
TIERS = ('le1k', 'le4k', 'le16k', 'gt16k')
TIER_LABELS = {'le1k': '<=1K', 'le4k': '(1K,4K]', 'le16k': '(4K,16K]', 'gt16k': '>16K'}
QUANTILES = ('p50', 'p95', 'p99')
LATENCIES = ('ttft_s', 'scheduled_ttft_s', 'tpot_s', 'e2e_s', 'scheduled_e2e_s',
             'ttft_ratio', 'e2e_ratio')
THROUGHPUT = ('cohort_requests_s', 'cohort_output_tokens_s', 'cohort_slo_goodput_requests_s',
              'cohort_slo_goodput_output_tokens_s', 'completed_in_window_requests_s',
              'completion_credited_in_window_output_tokens_s',
              'completed_in_window_slo_goodput_requests_s', 'offered_window_slo_goodput_requests_s')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def division(a, b):
    return a/b if finite(a) and finite(b) and b > 0 else None


def relative_path(value, destination, source):
    path = Path(value)
    if not path.is_absolute():
        working = Path.cwd()/path
        adjacent = source.parent/path
        # Summarizer CLI retains workspace-relative excluded/source paths;
        # relocated bundles may instead contain paths relative to the summary.
        if working.exists():
            path = working
        elif adjacent.exists():
            path = adjacent
        elif path.parts and path.parts[0] not in ('.', '..') and (Path.cwd()/path.parts[0]).exists():
            path = working
        else:
            path = adjacent
    return os.path.relpath(path.resolve(), destination.resolve())


def workload_label(workload):
    return '; '.join(f'{name}={p}/{o}@{100*w:g}%' for name, p, o, w in workload['shape_mix'])


def extract(stage, destination, source):
    stats, throughput, stability, gpu = (stage.get(k, {}) for k in ('stats', 'throughput', 'stability', 'gpu'))
    offered = stage['offered_requests']
    mode = stage['mode']
    basis = 'scheduled' if mode in ('poisson', 'burst') else 'request'
    row = {
        'workload_key': stage['workload_key'], 'workload': workload_label(stage['workload']),
        'architecture': stage['architecture'], 'deployment_key': stage.get('deployment_key', ''),
        'experiment': stage['experiment'], 'phase_id': stage['phase_id'], 'mode': mode,
        'role': stage.get('role'), 'seed': stage.get('seed'),
        'source': relative_path(stage['source'], destination, source),
        'latency_basis': basis, 'nominal_offered_rate_req_s': stage.get('rate'),
        'actual_scheduled_arrivals': offered,
        'admitted_requests': offered-stats.get('censored', 0),
        'realized_scheduled_arrival_rate_req_s': division(offered, stage['measurement_window_s']) if mode == 'poisson' else None,
        'expected_poisson_arrivals': stage['rate']*stage['measurement_window_s'] if mode == 'poisson' else None,
        'measurement_window_s': stage['measurement_window_s'], 'drain_s': stage['drain_s'],
        'total_wall_s': stage['total_wall_s'], 'successful_requests': stats.get('successful'),
        'failed_requests': stats.get('failed'), 'timeouts': stats.get('timeouts'),
        'censored_requests': stats.get('censored'), 'slo_eligible_requests': stats.get('slo_eligible_requests'),
        'slo_joint_pass_rate': stats.get('slo_joint_pass_rate'),
        'legacy_slo_joint_pass_rate': stats.get('legacy_slo_joint_pass_rate'),
        'p95_slo_pass': stats.get('p95_slo_pass'),
        'eligible_for_capacity_qualification': stage.get('eligible_for_capacity_qualification'),
        'stability_classification': stability.get('classification'),
        'queue_trend_verdict': stability.get('trend_verdict'),
        'queue_analysis_arrivals': stability.get('analysis_arrivals'),
        'queue_analysis_completions': stability.get('analysis_completions'),
        'queue_completion_deficit_requests': stability.get('completion_deficit_requests'),
        'client_mean_outstanding_increase': stability.get('client_mean_outstanding_increase'),
        'server_mean_waiting_increase': stability.get('server_mean_waiting_increase'),
        'server_sample_coverage_fraction': stability.get('server_sample_coverage_fraction'),
        'dispatch_lag_p95_s': stability.get('dispatch_lag_p95_s'),
        'in_flight_at_window_end': stage.get('in_flight_at_window_end'),
        'stability_reasons': ' | '.join(stability.get('reasons', [])),
        'gpu_status': gpu.get('status'), 'gpu_window_s': gpu.get('window_s'),
        'gpu_energy_j': gpu.get('energy_j'),
        'total_gpu_mean_power_w': division(gpu.get('energy_j'), gpu.get('window_s')),
        'successful_output_tokens_per_j': gpu.get('successful_output_tokens_per_j'),
        'slo_output_tokens_per_j': gpu.get('slo_output_tokens_per_j'),
        'gpu_warnings': ' | '.join(str(x) for x in stage.get('gpu_warnings', [])),
        'prometheus_status': stage.get('prometheus', {}).get('status'),
    }
    for name in THROUGHPUT:
        row[name] = throughput.get(name)
    for metric in LATENCIES:
        for quantile in QUANTILES:
            row[f'all_{metric}_{quantile}'] = stats.get('metrics', {}).get(metric, {}).get(quantile)
    for tier in TIERS:
        group = stage.get('by_input_tier', {}).get(tier, {})
        for name in ('requests', 'successful', 'failed', 'timeouts', 'censored', 'slo_joint_pass_rate',
                     'legacy_slo_joint_pass_rate', 'overall_ratio_p95_slo_pass'):
            row[f'{tier}_{name}'] = group.get(name)
        for metric in LATENCIES:
            distribution = group.get('metrics', {}).get(metric, {})
            row[f'{tier}_{metric}_n'] = distribution.get('n')
            for quantile in QUANTILES:
                row[f'{tier}_{metric}_{quantile}'] = distribution.get(quantile)
    devices = {str(device['gpu']): device for device in gpu.get('devices', [])}
    for index in range(4):
        device = devices.get(str(index), {})
        row[f'gpu{index}_uuid'] = device.get('uuid')
        row[f'gpu{index}_samples_inside'] = device.get('samples_inside')
        row[f'gpu{index}_max_sample_gap_s'] = device.get('max_overlapping_sample_gap_s')
        for metric in ('memory_mib', 'utilization_pct', 'power_w'):
            measure = device.get('measurements', {}).get(metric, {})
            for field in ('time_weighted_mean', 'observed_max', 'coverage_fraction'):
                row[f'gpu{index}_{metric}_{field}'] = measure.get(field)
    return row


def fmt(value, digits=3):
    if value is None:
        return '—'
    if type(value) is bool:
        return 'PASS' if value else 'FAIL'
    if finite(value):
        return f'{value:.{digits}f}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def percentage(value):
    return '—' if not finite(value) else f'{value*100:.1f}%'


def latency_triplet(group, key, scale=1):
    distribution = group.get('metrics', {}).get(key, {})
    return ' / '.join(fmt(distribution.get(q)*scale if finite(distribution.get(q)) else None) for q in QUANTILES)


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + [
        '| '+' | '.join(str(x) for x in row)+' |' for row in rows]


def phase_link(row):
    label = f"{row['experiment']} / {row['phase_id']}"
    return f'[{fmt(label)}]({quote(row["source"], safe="/._-")})'


def render(report, rows, stages, source, output_dir):
    relative_source = relative_path(source, output_dir, source)
    lines = ['# 阶段 A 分桶尾延迟与资源查数表', '',
             f'来源：[{source.name}]({quote(relative_source, safe="/._-")})；独立汇总状态 **{report["status"]}**；规则版本 `{report.get("rules", {}).get("version", "unspecified")}`。', '',
             '本表只转录独立汇总中的逐 phase、逐输入档位数据；不重新定义 SLO 或稳定性，不平均或合并不同 phase 的 P95/P99。容量排名继续使用汇总报告。', '',
             'Poisson / burst 的 TTFT、E2E 主列从计划到达起算；closed-loop 从 POST 起算。CSV 另保留 request-start 和 scheduled 两套延迟。尾延迟仅覆盖成功请求；联合满足率保留失败/censored 到达作为分母。', '',
             '吞吐 `cohort` 分母是测量窗口加 drain；`window completion` 统计在窗口内完成的整个请求及其 token，并非逐 token 时间归属。实际到达数是预生成的到达事件数；admitted 数排除客户端保护上限拒收。', '',
             '资源均值来自已有采样积分；整卡功耗不含 CPU/整机。显存与利用率 max 是采样最大值。缺测保留空值，不按零填充。', '']
    grouped = {}
    for row, stage in zip(rows, stages):
        grouped.setdefault((row['workload_key'], row['architecture'], row['deployment_key']), []).append((row, stage))
    for (key, architecture, deployment), pairs in grouped.items():
        lines += [f'## {fmt(architecture)} · `{deployment[:12] or "no-deployment-key"}`', '',
                  f'负载：`{pairs[0][0]["workload"]}`；workload key `{key}`。', '',
                  '### 到达、排空与吞吐口径', '']
        traffic = []
        for row, stage in pairs:
            traffic.append([phase_link(row), fmt(row['nominal_offered_rate_req_s']),
                            f"{row['actual_scheduled_arrivals']} / {row['admitted_requests']}",
                            fmt(row['realized_scheduled_arrival_rate_req_s']),
                            f"{fmt(row['measurement_window_s'], 1)} / {fmt(row['drain_s'], 1)}",
                            fmt(row['cohort_slo_goodput_requests_s']),
                            f"{fmt(row['cohort_requests_s'])} / {fmt(row['cohort_output_tokens_s'], 1)}",
                            f"{fmt(row['completed_in_window_requests_s'])} / {fmt(row['completion_credited_in_window_output_tokens_s'], 1)}"])
        lines += table(['phase', 'offered req/s', '到达 / admitted', '实到 req/s', 'window / drain s',
                        'good req/s', 'cohort req/s / tok/s', 'window completion req/s / tok/s'], traffic)+['',
                       '### 输入档位尾延迟', '',
                       '每个三元组均为 **P50 / P95 / P99**。TTFT、E2E 单位秒，TPOT 单位毫秒；normalized E2E-P95 为逐请求 E2E/其阈值的 P95。', '']
        tails = []
        for row, stage in pairs:
            basis = 'scheduled_' if row['latency_basis']=='scheduled' else ''
            for tier in TIERS:
                group = stage.get('by_input_tier', {}).get(tier)
                if not group:
                    continue
                tails.append([phase_link(row), TIER_LABELS[tier], f"{group['successful']} / {group['requests']}",
                              latency_triplet(group, basis+'ttft_s'), latency_triplet(group, 'tpot_s', 1000),
                              latency_triplet(group, basis+'e2e_s'),
                              fmt(group.get('metrics', {}).get('e2e_ratio', {}).get('p95')),
                              f"{percentage(group.get('slo_joint_pass_rate'))} / {percentage(group.get('legacy_slo_joint_pass_rate'))}",
                              fmt(group.get('overall_ratio_p95_slo_pass')),
                              f"{group.get('failed', 0)} / {group.get('timeouts', 0)} / {group.get('censored', 0)}"])
        lines += table(['phase', 'input tier', '成功 / 总数', 'TTFT s', 'TPOT ms', 'E2E s',
                        'E2E ratio P95', 'joint / legacy', 'tier P95 SLO', 'fail / timeout / censor'], tails)+['',
                       '### 资源与队列证据', '',
                       'GPU 向量按 **0 / 1 / 2 / 3** 顺序；每卡 mean/max 明确保留，不能从四卡平均值推断单卡瓶颈。', '']
        resources = []
        for row, stage in pairs:
            memory = ' / '.join(f"{fmt(row[f'gpu{i}_memory_mib_time_weighted_mean'],0)},{fmt(row[f'gpu{i}_memory_mib_observed_max'],0)}" for i in range(4))
            util = ' / '.join(f"{fmt(row[f'gpu{i}_utilization_pct_time_weighted_mean'],1)},{fmt(row[f'gpu{i}_utilization_pct_observed_max'],0)}" for i in range(4))
            power = ' / '.join(fmt(row[f'gpu{i}_power_w_time_weighted_mean'],1) for i in range(4))
            coverage = ' / '.join(percentage(row[f'gpu{i}_power_w_coverage_fraction']) for i in range(4))
            resources.append([phase_link(row), fmt(row['total_gpu_mean_power_w'],1), power, memory, util, coverage,
                              f"{fmt(row['client_mean_outstanding_increase'])} / {fmt(row['server_mean_waiting_increase'])}",
                              fmt(row['queue_completion_deficit_requests'],0),
                              f"{fmt(row['stability_classification'])} / {fmt(row['queue_trend_verdict'])}"])
        lines += table(['phase', '总卡均功耗 W', '每卡均功耗 W', '每卡mem mean,max MiB',
                        '每卡util mean,max %', '功耗覆盖', 'Δoutstanding / Δwaiting', '完成缺口 req', 'classification / trend'], resources)+['']
    lines += ['## 失败、负面结果与确认限制', '',
              '以下原因直接取自汇总数据。SCREEN_ONLY 仅表示筛选证据不够资格确认稳定容量，不等于服务故障。无异常的筛选点仅在上表保留，不在此重复。', '']
    negatives=[]
    for row in rows:
        failure = any((row[k] or 0) > 0 for k in ('failed_requests', 'timeouts', 'censored_requests'))
        notable = failure or row['p95_slo_pass'] is False or row['eligible_for_capacity_qualification'] is False
        notable = notable or row['queue_trend_verdict'] in ('GROWING_BACKLOG', 'INCONCLUSIVE')
        notable = notable or row['gpu_status'] not in ('COMPLETE_SAMPLED_WINDOW',) or bool(row['gpu_warnings'])
        if notable:
            negatives.append([phase_link(row), fmt(row['p95_slo_pass']),
                              f"{row['failed_requests']} / {row['timeouts']} / {row['censored_requests']}",
                              f"{percentage(row['slo_joint_pass_rate'])} / {percentage(row['legacy_slo_joint_pass_rate'])}",
                              fmt(row['stability_classification']), fmt(row['queue_trend_verdict']),
                              fmt(row['eligible_for_capacity_qualification']),
                              fmt(' | '.join(x for x in (row['stability_reasons'], row['gpu_warnings']) if x))])
    lines += table(['phase', 'P95 SLO', 'fail / timeout / censor', 'joint / legacy', 'classification',
                    'trend', '完整终态可参与确认', '汇总原因 / 资源限制'], negatives) if negatives else ['本次输入没有符合上述条件的阶段记录。']
    lines += ['', '### 未解决尝试与未纳入证据', '']
    blockers = report.get('unresolved_attempt_blockers', [])
    excluded = report.get('excluded', [])
    if not blockers and not excluded:
        lines += ['独立汇总未列出未解决尝试或未纳入文件。']
    for item in blockers:
        local = dict(item)
        if local.get('source'):
            local['source'] = relative_path(local['source'], output_dir, source)
        lines.append('- unresolved attempt: `'+fmt(json.dumps(local, ensure_ascii=False, sort_keys=True))+'`')
    for item in excluded:
        raw_source = item.get('source')
        local_source = relative_path(raw_source, output_dir, source) if raw_source else None
        lines.append('- excluded: `'+fmt(local_source)+'` — '+fmt(item.get('reason')))
    omitted = sum(not s.get('included_as_measurement') for s in report['stages'])
    lines += ['', f'输出 {len(rows)} 个 measurement phase；排除 {omitted} 个 warmup/profile phase。所有原始数据链接均为相对本表的本地路径。', '']
    return '\n'.join(lines)


def build(source, prefix):
    source, prefix = (Path(source).resolve(), Path(prefix).absolute())
    raw = source.read_bytes()
    script_hash = digest(Path(__file__).read_bytes())
    report = json.loads(raw)
    if report.get('schema_version') != 1 or report.get('status') != 'PASS':
        raise ValueError('Requires schema-1 PASS independent capacity summary')
    stages = sorted((s for s in report['stages'] if s.get('included_as_measurement')), key=lambda s: (s['workload_key'], s['architecture'], s.get('deployment_key', ''), s.get('rate') if s.get('rate') is not None else -1, s['experiment'], s['phase_id']))
    if not stages:
        raise ValueError('No measurement stages in summary')
    rows = [extract(s, prefix.parent, source) for s in stages]
    paths = [Path(str(prefix) + suffix) for suffix in ('-tables.md', '-phase-table.csv', '-tables.manifest.json')]
    if any((p.exists() for p in paths)):
        raise FileExistsError('Output prefix already used; select a new prefix')
    md = render(report, rows, stages, source, prefix.parent)
    csv_buffer = io.StringIO(newline='')
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    text = csv_buffer.getvalue()
    if source.read_bytes() != raw or digest(Path(__file__).read_bytes()) != script_hash:
        raise RuntimeError('Summary or table source changed while building tables')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, (md, text)):
        with open_output(path, 'x') as handle:
            handle.write(content)
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), source=relative_path(source, prefix.parent, source), source_sha256=digest(raw), script_sha256=script_hash, rules=report.get('rules'), measurement_phases=len(rows), columns=len(rows[0]), percentile_policy='Each CSV row is one phase; every tier quantile copied from the same phase. No pooled or averaged quantiles.', derived_fields={'realized_scheduled_arrival_rate_req_s': 'offered_requests / window, Poisson only', 'admitted_requests': 'offered_requests - censored_requests', 'expected_poisson_arrivals': 'nominal Poisson rate * fixed window', 'total_gpu_mean_power_w': 'full-coverage GPU energy / sampled GPU window; missing stays blank'}, files={p.name: dict(sha256=digest(p.read_bytes()), bytes=p.stat().st_size) for p in paths[:2]})
    with open_output(paths[2], 'x') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write('\n')
    return dict(status='PASS', phases=len(rows), csv_columns=len(rows[0]), files=[str(p) for p in paths])


def self_test():

    def distribution(x):
        return dict(n=10, p50=x, p95=2 * x, p99=3 * x)
    group = dict(requests=12, successful=10, failed=1, timeouts=1, censored=1, slo_eligible_requests=12, slo_joint_pass_rate=0.5, legacy_slo_joint_pass_rate=0.25, overall_ratio_p95_slo_pass=False, metrics={m: distribution(0.02 if m == 'tpot_s' else 0.7) for m in LATENCIES})

    def phase(key, x):
        return dict(experiment='a01-fixture', phase_id='p' + key, architecture='same-name', deployment_key=key, workload_key='mix', workload=dict(shape_mix=[['short', 512, 128, 1.0]]), source='/tmp/fixture-results/a01/capacity.json', included_as_measurement=True, role='measure', seed=1, mode='poisson', rate=2.0, offered_requests=12, measurement_window_s=5.0, drain_s=1.0, total_wall_s=6.0, eligible_for_capacity_qualification=False, stats=dict(group, p95_slo_pass=False), by_input_tier={'le1k': group}, throughput={k: x for k in THROUGHPUT}, stability=dict(classification='SCREEN_ONLY', trend_verdict='GROWING_BACKLOG', reasons=['fixture growth'], completion_deficit_requests=3), gpu=dict(status='PARTIAL_OR_MISSING', energy_j=None, window_s=6.0, devices=[]))
    report = dict(schema_version=1, status='PASS', rules={'version': 'fixture-policy'}, stages=[phase('configA', 12.0), phase('configB', 99.0)], excluded=[], unresolved_attempt_blockers=[])
    with tempfile.TemporaryDirectory(prefix='capacity-tables-fixture-') as temporary:
        directory = Path(temporary)
        source = directory / 'summary.json'
        write_text(source, json.dumps(report))
        workspace_relative = Path(os.path.relpath(source, Path.cwd()))
        assert relative_path(workspace_relative, directory, source) == 'summary.json'
        adjacent = directory / 'adjacent.json'
        write_text(adjacent, '{}')
        assert relative_path('adjacent.json', directory, source) == 'adjacent.json'
        result = build(source, directory / 'capacity')
        with (directory / 'capacity-phase-table.csv').open() as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(csv_rows) == 2 and {r['deployment_key'] for r in csv_rows} == {'configA', 'configB'}
        assert csv_rows[0]['le1k_scheduled_ttft_s_p95'] == '1.4'
        assert csv_rows[0]['le1k_tpot_s_p95'] == '0.04'
        assert csv_rows[0]['stability_classification'] == 'SCREEN_ONLY'
        assert csv_rows[0]['admitted_requests'] == '11'
        assert csv_rows[0]['realized_scheduled_arrival_rate_req_s'] == '2.4'
        assert csv_rows[0]['total_gpu_mean_power_w'] == '' and csv_rows[0]['gpu0_power_w_time_weighted_mean'] == ''
        md = path_at(directory, 'capacity-tables.md').read_text()
        assert '20.000 / 40.000 / 60.000' in md
        assert 'http://' not in md and '](/tmp/' not in md
        manifest = load_document(directory / 'capacity-tables.manifest.json')
        for name, meta in manifest['files'].items():
            assert digest((directory / name).read_bytes()) == meta['sha256']
        return dict(status='PASS', checks=['different deployment keys remain separate', 'stage/tier quantiles copied without pooling', 'CSV seconds and Markdown TPOT milliseconds', 'screen classification remains screen', 'actual/admitted arrival denominators', 'missing resource telemetry stays blank', 'workspace-relative and summary-relative source paths', 'local relative raw evidence links', 'source/table hashes preserved'], columns=result['csv_columns'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=path_at(ROOT, 'summary/capacity-summary.json'))
    parser.add_argument('--output-prefix', type=Path, default=path_at(ROOT, 'summary/capacity'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else build(args.input, args.output_prefix), indent=2))


if __name__=='__main__':
    main()
