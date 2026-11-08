#!/usr/bin/env python3
"""Descriptive CPU analysis of terminal capacity metrics and request timelines.

No causal attribution: the pre-first-text interval includes server queue and
prefill, and all requests share a changing batch/scheduler state.
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
from pathlib import Path

import summarize_capacity as summary

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def small_summary(rows):
    return {'n': len(rows), 'scheduled_ttft_s': summary.dist([r['scheduled_ttft_s'] for r in rows]),
            'tpot_s': summary.dist([r['tpot_s'] for r in rows]),
            'fraction_ttft_over_1s': sum(r['scheduled_ttft_s'] > 1 for r in rows)/len(rows) if rows else None}


def associations(rows):
    short = [r for r in rows if r.get('ok') and r['input_tokens'] <= 1024 and r['requested_output_tokens'] <= 128]
    others = [r for r in rows if r.get('admitted') and r.get('ttft_s') is not None]
    result = {}
    for label, relevant, part in (
        ('input_ge4k_pre_first_text', lambda r: r['input_tokens'] >= 4096, 'pre'),
        ('input_ge16k_pre_first_text', lambda r: r['input_tokens'] >= 16384, 'pre'),
        ('output_ge1024_after_first_text', lambda r: r['requested_output_tokens'] >= 1024, 'post'),
    ):
        selected = [r for r in others if relevant(r)]
        overlap, no_overlap = [], []
        for row in short:
            instant = row['dispatch_offset_s']
            def active(other):
                if other['request_id'] == row['request_id']:
                    return False
                first = other['dispatch_offset_s']+other['ttft_s']
                if part == 'pre':
                    return other['dispatch_offset_s'] <= instant < first
                return first <= instant < other['end_offset_s']
            (overlap if any(active(other) for other in selected) else no_overlap).append(row)
        result[label] = {'with_existing_overlap': small_summary(overlap),
                         'without_existing_overlap': small_summary(no_overlap),
                         'relevant_requests_in_phase': len(selected),
                         'definition': 'Group short requests by other relevant requests already active at their POST dispatch; no future-arrival lookahead.'}
    return result


def analyze(report, summary_path, historical_profile):
    initial_summary_bytes = summary_path.read_bytes()
    if json.loads(initial_summary_bytes) != report:
        raise ValueError('Summary bytes differ from the parsed analysis input')
    initial_summary_hash = sha(initial_summary_bytes)
    initial_script_hash = sha(Path(__file__).read_bytes())
    if report.get('status') != 'PASS':
        raise ValueError('A PASS independent summary is required')
    loaded, stages, excluded, ledger = {}, [], [], {}
    for stage in report['stages']:
        if not stage.get('included_as_measurement') or stage['mode'] != 'poisson':
            continue
        if not stage.get('eligible_for_capacity_qualification'):
            excluded.append({'experiment': stage['experiment'], 'phase': stage['phase_id'],
                             'reason': 'Whole owned experiment is not terminal success; do not analyze mutable/partial sources.'})
            continue
        path = Path(stage['source'])
        if path not in loaded:
            raw = path.read_bytes()
            expected = report['evidence_ledger'][str(path.resolve())]['sha256']
            if sha(raw) != expected:
                raise ValueError('Capacity source changed after summary: '+str(path))
            loaded[path] = json.loads(raw)
            ledger[str(path)] = {'sha256': sha(raw), 'bytes': len(raw)}
        raw_run = next(r for r in loaded[path]['runs'] if r['phase']['id'] == stage['phase_id'])
        rows = raw_run['requests']
        means = stage['prometheus'].get('derived', {}).get('histogram_means', {})
        sampled_waiting = []
        for sample in raw_run['metric_samples']:
            if sample.get('http_status') != 200 or sample.get('error'):
                continue
            values = [v for k, v in sample.get('values', {}).items() if k.split('{')[0] == 'vllm:num_requests_waiting']
            if values:
                sampled_waiting.append(sum(values))
        devices = stage['gpu'].get('devices', [])
        powers = [d['measurements']['power_w']['time_weighted_mean'] for d in devices]
        group = {
            'experiment': stage['experiment'], 'architecture': stage['architecture'],
            'deployment_key': stage.get('deployment_key'), 'phase_id': stage['phase_id'],
            'workload': stage['workload'], 'workload_key': stage['workload_key'],
            'rate': stage['rate'], 'offered_requests': stage['offered_requests'],
            'window_s': stage['measurement_window_s'], 'drain_s': stage['drain_s'],
            'p95_slo_pass': stage['stats']['p95_slo_pass'], 'throughput': stage['throughput'],
            'by_input_tier': stage['by_input_tier'], 'by_shape': stage['by_shape'],
            'stability': stage['stability'],
            'server_mean_queue_s': means.get('vllm:request_queue_time_seconds'),
            'server_mean_prefill_s': means.get('vllm:request_prefill_time_seconds'),
            'server_mean_decode_s': means.get('vllm:request_decode_time_seconds'),
            'sampled_max_waiting_requests': max(sampled_waiting) if sampled_waiting else None,
            'preemptions': stage['prometheus'].get('totals', {}).get('vllm:num_preemptions_total'),
            'spec_accepted_fraction': stage['prometheus'].get('derived', {}).get('spec_accepted_fraction'),
            'spec_mean_accepted_plus_one': stage['prometheus'].get('derived', {}).get('spec_mean_accepted_plus_one'),
            'mean_four_gpu_power_w': sum(powers) if len(powers) == 4 and all(p is not None for p in powers) else None,
            'gpu_energy_j': stage['gpu'].get('energy_j'),
            'gpu_output_tokens_per_j': stage['gpu'].get('successful_output_tokens_per_j'),
            'per_gpu_peak_memory_mib': {str(d['gpu']): d['measurements']['memory_mib']['observed_max'] for d in devices},
            'per_gpu_time_weighted_device_busy_pct': {str(d['gpu']): d['measurements']['utilization_pct']['time_weighted_mean'] for d in devices},
            'associations': associations(rows),
        }
        stages.append(group)
    profile = None
    if historical_profile.exists():
        raw = historical_profile.read_bytes()
        old = json.loads(raw)
        if old.get('status') != 'PASS':
            raise ValueError('Historical profile summary is not PASS')
        profile = {'path': str(historical_profile.resolve()), 'sha256': sha(raw), 'status': old['status'],
                   'scope': old['scope'], 'limitations': old['limitations'],
                   'rank0_traces': [{'identity': t['identity'], 'path': t['path'], 'sha256': t['sha256'],
                                    'kernel_events': t['cuda_kernel_events'], 'categories': t['categories']}
                                   for t in old['traces'] if t['identity']['rank'] == 0]}
        ledger[str(historical_profile.resolve())] = {'sha256': sha(raw), 'bytes': len(raw)}
    if sha(summary_path.read_bytes()) != initial_summary_hash or sha(Path(__file__).read_bytes()) != initial_script_hash:
        raise ValueError('Summary or analyzer source changed during analysis')
    return {'schema_version': 1, 'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Descriptive terminal-run bottleneck evidence; not causal attribution or new CUDA profiles.',
            'all_measurements_screen_only': bool(stages) and all(s['stability']['classification'] == 'SCREEN_ONLY' for s in stages),
            'source_summary': str(summary_path.resolve()), 'source_summary_sha256': initial_summary_hash,
            'script_sha256': initial_script_hash, 'stages': stages, 'excluded': excluded,
            'historical_s06_profile': profile, 'evidence_ledger': ledger}


def n(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(report):
    architectures = sorted({r['architecture'] for r in report['stages']})
    lines = ['# 阶段 A：容量瓶颈证据', '',
             f"已纳入 {len(architectures)} 种部署、{len(report['stages'])} 个终态 Poisson phase。只读成功完成且清理完毕的实验；完整数据、hash与每档位分位见同名JSON。", '',
             '以下是描述性证据。服务端 queue/prefill/decode 来自显式前后 Prometheus histogram sum/count 差分的请求均值，不能相加成某个请求的 P95，也不是 GPU kernel 独占时间。cohort吞吐包含 arrival window 与 drain。60秒粗扫不认证稳定容量。', '',
             '| 实验 / phase | rate | tok/s | SLO good req/s | queue / prefill / decode 均值 s | max sampled waiting | preemptions | mean GPU W |',
             '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    if report.get('all_measurements_screen_only'):
        lines[2:2] = ['**本文件全部测量均为 SCREEN_ONLY 粗扫，不包含稳定容量确认点，也不是阶段 A 最终容量结论。**', '']
    for r in report['stages']:
        lines.append(f"| {r['experiment']} / {r['phase_id']} | {n(r['rate'])} | {n(r['throughput']['cohort_output_tokens_s'], 1)} | {n(r['throughput']['cohort_slo_goodput_requests_s'])} | {n(r['server_mean_queue_s'])} / {n(r['server_mean_prefill_s'])} / {n(r['server_mean_decode_s'])} | {n(r['sampled_max_waiting_requests'], 0)} | {n(r['preemptions'], 0)} | {n(r['mean_four_gpu_power_w'], 0)} |")
    spec_rows = [r for r in report['stages'] if r['spec_accepted_fraction'] is not None]
    if spec_rows:
        lines += ['', '## DSpark 的接受计数', '',
                  '下表使用各phase独立前后counter差分，排除预热。接受率=接受草稿tokens/草稿tokens；含bonus均长=1+接受草稿tokens/草稿次数。它们随文本内容、shape和调度变化，不能单独证明某个速度差的因果来源。', '',
                  '| 实验 / phase | 草稿接受率 | 含bonus平均接受长度 | Decode请求均值 s | cohort tok/s |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for r in spec_rows:
            lines.append(f"| {r['experiment']} / {r['phase_id']} | {n(r['spec_accepted_fraction'])} | {n(r['spec_mean_accepted_plus_one'])} | {n(r['server_mean_decode_s'])} | {n(r['throughput']['cohort_output_tokens_s'], 1)} |")
    lines += ['', '## 短请求到达时的长请求重叠', '',
              'short指输入≤1K且输出预算≤128。仅按其实际POST时刻已经存在的其他请求分组，不根据随后到达选择样本。长输入的“到首文本区间”包含服务器排队和Prefill，不是精确GPU Prefill；长生成的“首文本后区间”是客户端观察到的响应阶段，也不是独占Decode kernel。共同队列、总并发、请求形状和到达时间均是混杂因素，以下关联不能证明因果。', '',
              '表内“有 / 无”为到达时已有该类重叠。小N的P95尤其不稳，完整P50/P95/P99和TPOT在JSON中；不跨不同rate把关联合成一个效应。', '',
              '| 实验 / phase | 重叠类型 | N 有 / 无 | 短请求 TTFT P50 有 / 无 s | TTFT P95 有 / 无 s | TTFT>1s 比例 有 / 无 |',
              '| --- | --- | ---: | --- | --- | --- |']
    for r in report['stages']:
        for label, a in r['associations'].items():
            if a['relevant_requests_in_phase'] == 0:
                continue
            yes, no = a['with_existing_overlap'], a['without_existing_overlap']
            lines.append(f"| {r['experiment']} / {r['phase_id']} | {label} | {yes['n']} / {no['n']} | {n(yes['scheduled_ttft_s']['p50'])} / {n(no['scheduled_ttft_s']['p50'])} | {n(yes['scheduled_ttft_s']['p95'])} / {n(no['scheduled_ttft_s']['p95'])} | {n(yes['fraction_ttft_over_1s'])} / {n(no['fraction_ttft_over_1s'])} |")
    lines += ['', '## 既有 CUDA Profile 的适用边界', '',
              '复用旧 s06 TP4 独立 Profile，未为阶段 A 开启 profiler。它使用有限约20个engine iteration，覆盖16K Prefill、短Decode与实际64K混合请求；并非本轮Poisson负载，也不是TP2PP2、DP4EP或DSpark的Profile。下面只列rank0的kernel累计时间分类比例，跨stream时间可重叠，绝不能读作E2E占比、PCIe带宽占比或跨架构瓶颈归因。', '',
              '| 旧Profile case / rank0 | communication | MoE/Marlin | explicit Attention | CUDA kernels |',
              '| --- | ---: | ---: | ---: | ---: |']
    profile = report.get('historical_s06_profile')
    if profile:
        for trace in profile['rank0_traces']:
            cats = {c['category']: c['fraction_of_kernel_sum'] for c in trace['categories']}
            lines.append(f"| {trace['identity']['case']} | {n(100*cats.get('communication', 0), 1)}% | {n(100*cats.get('moe_explicit_or_marlin', 0), 1)}% | {n(100*cats.get('attention_explicit', 0), 1)}% | {trace['kernel_events']} |")
    lines += ['', '历史Profile表明通信、MoE/Marlin与Attention均有可观察成本，不能据此把本轮尾延迟归咎于唯一kernel。对新架构作kernel级因果判断仍需对应负载的独立匹配Profile；容量主线先用当前请求与队列证据选有价值的后续问题。', '',
              '## 解释限制与下一步', '',
              '- 所有架构保持APC off、原权重/量化、Marlin与同版vLLM；DP4EP的4096 token调度预算是每engine口径，整体预算不同，属于完整部署比较。',
              '- 1秒GPU利用率是device busy指标，不是SM占用率、PCIe饱和度或有效计算比例；NCCL等待也可能让GPU被计为busy。功率与tokens/J用于资源效率参考，不代表整机能耗。固定显存分配包括预留KV缓存，不能仅据显存接近90%断言有效KV满载。',
              '- 有明显queue增长时，吞吐上升不能等同SLO容量增加；无queue增长也可能因Prefill/Decode共享调度而使TPOT或短请求TTFT失标。',
              '- 旧budget1024已经验证为负面，不能因本轮长请求干扰就未经新证据推荐更小chunk。优先考虑可分离服务池/调度隔离或按场景选架构，具体收益仍需匹配实验。',
              '- 任何未出现在本次终态输入集合的架构/点均未由本文验证；容量认证与推荐以独立容量汇总和阶段A报告为准。']
    return '\n'.join(lines)+'\n'


def self_test():
    long = {'request_id': 'long', 'input_tokens': 4096, 'requested_output_tokens': 128,
            'admitted': True, 'ok': True, 'dispatch_offset_s': 1., 'ttft_s': 2., 'end_offset_s': 10.,
            'scheduled_ttft_s': 2., 'tpot_s': .01}
    def short(name, time):
        return dict(long, request_id=name, input_tokens=512, dispatch_offset_s=time, ttft_s=.1,
                    end_offset_s=time+1, scheduled_ttft_s=.1)
    result = associations([long, short('before', .5), short('during', 2.), short('after', 3.5)])
    row = result['input_ge4k_pre_first_text']
    assert row['with_existing_overlap']['n'] == 1 and row['without_existing_overlap']['n'] == 2
    assert result['input_ge16k_pre_first_text']['relevant_requests_in_phase'] == 0
    return {'status': 'PASS', 'checks': ['dispatch-time existing overlap only; no future overlap selection', 'input threshold and short-request eligibility']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=path_at(ROOT, 'summary/capacity-summary.json'))
    parser.add_argument('--output-prefix', type=Path, default=ROOT / 'capacity-bottlenecks')
    parser.add_argument('--historical-profile', type=Path, default=path_at(ROOT, 'summary/profile-baseline.json'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    report = analyze(load_document(args.summary), args.summary, args.historical_profile)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.output_prefix.with_suffix('.json'), json.dumps(report, indent=2, sort_keys=True) + '\n')
    write_text(args.output_prefix.with_suffix('.md'), markdown(report))
    print(json.dumps({'status': report['status'], 'stages': len(report['stages']), 'excluded': len(report['excluded']), 'output_prefix': str(args.output_prefix)}))


if __name__ == '__main__':
    main()
