#!/usr/bin/env python3
"""Build final report tables exclusively from complete audited Stage D evidence."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--output-prefix', type=Path, required=True)
    a = ap.parse_args()
    outputs = [a.output_prefix.with_suffix(x) for x in ('.md', '.json')]
    if any((p.exists() or p.resolve().is_relative_to(path_at(D, 'results').resolve()) for p in outputs)):
        raise ValueError('New output outside raw results required')
    raw = a.summary.read_bytes()
    data = json.loads(raw)
    assert data['status'] == 'PASS' and (not data['errors'])
    stages = [s for s in data['stages'] if s['included_as_measurement']]
    assert len(stages) == 12 and all((s['ranking_eligible'] for s in stages))
    pairs = data['paired_comparisons']
    assert len(pairs) == 6 and all((p['status'] == 'PASS' for p in pairs))
    header = ['# Stage D final report tables', '', 'All rows are unprofiled, terminal, independently audited; direct comparison only within each registered restart pair.', '', '| block / policy | short n | short TTFT P95 / P99 s | short TPOT P95 ms | short E2E P95 s | short SLO | short Goodput req/s | 64K TTFT P95 s | total tok/s | drain s | stability |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    lines = header[:]
    numeric = []
    for s in sorted(stages, key=lambda x: (x['pair_block'], x['policy'])):
        if s['mode'] != 'poisson':
            continue
        short = s['short512']
        m = short['metrics']
        long = s['by_shape']['p65536_o128']
        g = s['gpu']
        row = dict(experiment=s['experiment'], pair_block=s['pair_block'], policy=s['policy'], short_n=short['requests'], short_slo_passes=short['slo_passes'], short_slo_rate=short['slo_joint_pass_rate'], ttft_p95=m['scheduled_ttft_s']['p95'], ttft_p99=m['scheduled_ttft_s']['p99'], tpot_p95_ms=1000 * m['tpot_s']['p95'], e2e_p95=m['scheduled_e2e_s']['p95'], short_goodput=short['cohort_slo_goodput_requests_s'], long_ttft_p95=long['metrics']['scheduled_ttft_s']['p95'], tokens_s=s['throughput']['cohort_output_tokens_s'], drain=s['drain_s'], stability=s['stability']['classification'], energy_kj=g['energy_j'] / 1000 if g['energy_j'] is not None else None, mean_aggregate_board_w=g['energy_j'] / g['window_s'] if g['energy_j'] is not None else None, kv_peak_pct=100 * s['raw_gauges']['aggregate']['vllm:kv_cache_usage_perc']['max'], waiting_peak=s['raw_gauges']['aggregate']['vllm:num_requests_waiting']['max'], preemptions=s['preemptions'])
        numeric.append(row)
        lines.append(f'| {row['pair_block']} / {row['policy']} | {row['short_n']} | {num(row['ttft_p95'])} / {num(row['ttft_p99'])} | {num(row['tpot_p95_ms'])} | {num(row['e2e_p95'])} | {row['short_slo_passes']}/{row['short_n']} ({row['short_slo_rate']:.2%}) | {row['short_goodput']:.5f} | {num(row['long_ttft_p95'])} | {num(row['tokens_s'])} | {num(row['drain'])} | {row['stability']} |')
    lines += ['', '## Burst/interference boundaries', '', '| block / phase / policy | short TTFT P95 s | short TPOT P95 ms | short SLO | 64K TTFT P95 s | complete wall s | energy kJ |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in sorted(stages, key=lambda x: (x['phase_id'], x['pair_block'], x['policy'])):
        if s['mode'] == 'poisson':
            continue
        g = s['short512']
        m = g['metrics']
        energy = s['gpu']['energy_j']
        lines.append(f'| {s['pair_block']} / {s['phase_id']} / {s['policy']} | {num(m['scheduled_ttft_s']['p95'])} | {num(1000 * m['tpot_s']['p95'])} | {g['slo_passes']}/{g['requests']} | {num(s['by_shape']['p65536_o128']['metrics']['scheduled_ttft_s']['p95'])} | {num(s['total_wall_s'])} | {num(None if energy is None else energy / 1000)} |')
    lines += ['', '## Matched effects (FairShare relative to Observe)', '', '| block / phase | short TTFT P95 % | short TPOT P95 % | short Goodput % | 64K TTFT P95 % | total tok/s % | GPU energy % |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in pairs:
        keys = ['p512_o128.scheduled_ttft_s.p95', 'p512_o128.tpot_s.p95', 'p512_o128.cohort_slo_goodput_requests_s', 'p65536_o128.scheduled_ttft_s.p95', 'cohort_output_tokens_s', 'gpu_energy_j']
        values = [num(p['deltas'][key]['percent_change'], 2) for key in keys]
        lines.append('| ' + p['pair_block'] + ' / ' + p['comparison_id'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'A zero-control Goodput has no finite relative percentage (—); report absolute change instead. Normal quality is separate from these phases.', '', '## Poisson resources', '', '| block / policy | four-board kJ | aggregate board W | KV peak % | waiting peak | preemptions |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in numeric:
        lines.append(f'| {r['pair_block']} / {r['policy']} | {num(r['energy_kj'])} | {num(r['mean_aggregate_board_w'])} | {num(r['kv_peak_pct'])} | {num(r['waiting_peak'], 0)} | {num(r['preemptions'], 0)} |')
    lines += ['', '## Quality', '', '| experiment | natural-EOS requests | protocol successes | strict passed |', '| --- | ---: | ---: | ---: |']
    for r in data['runs']:
        q = r['quality']
        lines.append(f'| {r['experiment']} | {q['requests']} | {q['protocol_successes']} | {q['quality_passed']} |')
    assert sha(a.summary) == hashlib.sha256(raw).hexdigest()
    a.output_prefix.parent.mkdir(exist_ok=True, parents=True)
    with open_output(outputs[0], 'x') as f:
        f.write('\n'.join(lines) + '\n')
    with open_output(outputs[1], 'x') as f:
        json.dump(dict(status='PASS', summary_sha256=sha(a.summary), source_sha256=sha(Path(__file__)), tables_sha256=sha(outputs[0]), poisson=numeric, paired_comparisons=pairs, normal_requests=sum((s['offered_requests'] for s in data['stages'])), formal_requests=sum((s['offered_requests'] for s in stages)), quality_requests=sum((r['quality']['requests'] for r in data['runs']))), f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', formal_phases=len(stages), pairs=len(pairs))))
if __name__=='__main__':main()
