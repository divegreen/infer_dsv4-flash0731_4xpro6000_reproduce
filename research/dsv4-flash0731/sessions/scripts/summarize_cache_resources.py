#!/usr/bin/env python3
"""Condense verified offline cache analyses; no HTTP, GPU, or service imports."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {str(p.resolve()): sha(p) for p in args.inputs}
    rows, checks, excluded = ([], [], [])
    for path in args.inputs:
        report = load_document(path)
        if report['status'] != 'PASS_TERMINAL_CACHE_ANALYSIS':
            raise ValueError('Healthy terminal cache analysis required')
        for p in report['phases']:
            if not p['included_as_measurement']:
                excluded.append(dict(experiment=Path(report['run']).name, phase=p['id'], role=p['role']))
                continue
            spec = p['speculative']
            a, t, n = (spec['accepted_tokens'], spec['draft_tokens'], spec['drafts'])
            positions = {}
            for key, value in p['raw_counter_deltas'].items():
                if key.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{'):
                    position = re.search('position="(\\d+)"', key)[1]
                    positions[position] = positions.get(position, 0) + value
            if n is not None:
                if not (0 <= a <= t and n > 0 and math.isclose(a / t, spec['accepted_fraction']) and math.isclose(1 + a / n, spec['mean_accepted_plus_one']) and math.isclose(sum(positions.values()), a)):
                    raise ValueError('Speculative counter identity failed')
                checks.append(Path(report['run']).name + '/' + p['id'] + ' speculative identities')
            gpu = p['gpu']
            complete = gpu['status'] == 'COMPLETE_SAMPLED_WINDOW'
            device_rows = []
            for device in gpu['devices']:
                m = device['measurements']
                device_rows.append(dict(gpu=device['gpu'], utilization_time_mean_pct=m['utilization_pct']['time_weighted_mean'], power_time_mean_w=m['power_w']['time_weighted_mean'], peak_memory_mib=m['memory_mib']['observed_max'], max_sample_gap_s=device['max_overlapping_sample_gap_s']))
            engines = {}
            for key, engine in p['per_engine'].items():
                totals = engine['totals']
                engines[key] = dict(requests=totals.get('vllm:request_inference_time_seconds_count'), prompt_tokens=totals.get('vllm:prompt_tokens_total'), cached_tokens=totals.get('vllm:prefix_cache_hits_total'), output_tokens=totals.get('vllm:generation_tokens_total'), preemptions=totals.get('vllm:num_preemptions_total'), timings=engine['server_timings'])
            rows.append(dict(experiment=Path(report['run']).name, phase=p['id'], role=p['role'], kind=p['kind'], seed=p['seed'], concurrency=p['concurrency'], scope=p['cache_scope'], routing=p['dp_routing'], apc=p['apc_enabled'], workload_sha256=p['workload_sha256'], sessions=p['sessions'], requests=p['requests'], session_rate=p['sessions'] / p['wall_s'], output_token_rate=p['output_tokens_s'], successful_tasks=p['successful_tasks'], successful_task_rate=p['successful_tasks'] / p['wall_s'] if p['successful_tasks'] is not None else None, prompt_tokens=p['known_prompt_tokens'], cached_tokens=p['known_cached_tokens'], logical_computed_tokens=p['logical_computed_prompt_tokens'], output_tokens=p['known_output_tokens'], session_latency_s=p['session_latency_s'], ttft_text_s=p['request_ttft_text_s'], tpot_output_s=p['request_tpot_output_s'], e2e_s=p['request_e2e_s'], server_timings=p['server_timings'], peaks=p['observed_gauge_peaks'], preemptions=sum((x['preemptions'] for x in engines.values())) if all((x['preemptions'] is not None for x in engines.values())) else None, speculative=dict(**spec, accepted_per_position=positions, acceptance_by_position={k: v / n for k, v in positions.items()} if n else {}), resource=dict(complete_four_gpu_window=complete, energy_j=gpu['energy_j'], output_tokens_per_joule=p['output_tokens_per_joule'], mean_board_power_sum_w=sum((d['power_time_mean_w'] for d in device_rows)) if complete else None, devices=device_rows), per_engine=engines))
    sources = {}
    dist = importlib.metadata.distribution('vllm')
    for rel in ('vllm/v1/spec_decode/metrics.py', 'vllm/v1/core/sched/scheduler.py', 'vllm/v1/metrics/stats.py'):
        path = Path(dist.locate_file(rel))
        record = next((f for f in dist.files if str(f) == rel))
        digest = hashlib.sha256(path.read_bytes()).digest()
        if record.hash.value != base64.urlsafe_b64encode(digest).decode().rstrip('='):
            raise ValueError('Installed counter source differs from wheel RECORD')
        sources[rel] = dict(sha256=sha(path), wheel_record_match=True)
    if not all((sha(Path(p)) == h for p, h in inputs.items())):
        raise ValueError('Input analysis changed while reading')
    result = dict(status='PASS_OFFLINE_CACHE_RESOURCE_SUMMARY', created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)), inputs=inputs, metric_source_identity=sources, checks=checks, rows=rows, excluded=excluded, limits=['No cross-workload ranking; use compare_cache_screens.py for matched identities.', 'Decode/Prefill metrics are overlapping request intervals including contention, not exclusive GPU kernel time.', 'Accepted/draft tokens and 1+accepted/drafts follow framework definitions; drafts aggregate per-request observations, not global GPU steps.', 'Speculative acceptance counters may differ from client emitted token counts; no exact equality asserted.', 'NVML utilization is sampled activity, not SM arithmetic utilization, PCIe bandwidth utilization, or bottleneck proof.', 'Board energy excludes CPU/tools and does not assign energy causally to individual requests.', 'Dynamic Agent equal cohorts can have different actual work and quality.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], rows=len(rows), checks=len(checks), output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
