"""Standalone PNG/PDF figures from audited scenario aggregates."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#2563a6','#24957f','#df8b32','#9b59a6','#c85454','#6f7c87','#677c36']


def label(row):
    return row['experiment_id'].replace('scenarios-tp4-nocache','TP4 reference').replace('scenarios-tp4-apc','TP4 APC').replace('scenarios-tp4-dspark3-retry2','TP4 DSpark3').replace('scenarios-tp2pp2','TP2 PP2').replace('scenarios-tp4-b1024','TP4 chunk1024').replace('scenarios-dp4ep','DP4 EP').replace('scenarios-tp4-b8192','TP4 chunk8192')


def main():
    source = path_at(ROOT, 'summary/summary.json')
    report = load_document(source)
    if report['audit']['status'] != 'PASS':
        raise RuntimeError('Audit must pass before plotting')
    rows = [r for r in report['combined'] if r['final_execution_verified']]
    out = path_at(ROOT, 'figures')
    out.mkdir(parents=True, exist_ok=True)
    panels = {'long-generation': [('decode', 'p1024_o1024_c1', 'measure', 'throughput', '1024 in / 1024 out, C1'), ('decode', 'p1024_o1024_c8', 'measure', 'throughput', '1024 in / 1024 out, C8')], 'actual-64k-context': [('context', 'p65536_o128_c1', 'measure', 'ttft50', '64K in / 128 out, C1'), ('context', 'p65536_o128_c4', 'measure', 'ttft50', '64K in / 128 out, C4'), ('context', 'p65536_o128_c1', 'measure', 'tpot95', '64K in / 128 out, C1'), ('context', 'p65536_o128_c4', 'measure', 'tpot95', '64K in / 128 out, C4')], 'arrival-and-concurrency': [('arrival', 'mixed_poisson_r2', 'measure', 'scheduled95', 'Mixed Poisson 2 req/s'), ('arrival', 'mixed_poisson_r6', 'measure', 'scheduled95', 'Mixed Poisson 6 req/s'), ('arrival', 'p512_o128_c64', 'measure', 'throughput', '512 in / 128 out, C64'), ('arrival', 'mixed_poisson_r2', 'measure', 'goodput', 'Mixed Poisson 2 req/s'), ('arrival', 'mixed_poisson_r6', 'measure', 'goodput', 'Mixed Poisson 6 req/s'), ('arrival', 'p512_o128_c64', 'measure', 'tpot95', '512 in / 128 out, C64')], 'history-reuse': [('cache', 'prefix8192_o64', 'warm_serial', 'ttft50', '8K shared prefix, serial reuse'), ('cache', 'prefix32768_o64', 'warm_serial', 'ttft50', '32K shared prefix, serial reuse'), ('cache', 'prefix8192_o64', 'warm_parallel', 'ttft50', '8K shared prefix, 8 concurrent forks'), ('cache', 'prefix32768_o64', 'warm_parallel', 'ttft50', '32K shared prefix, 8 concurrent forks')]}
    units = dict(throughput='Output tokens/s (higher is better)', ttft50='TTFT p50 (s, lower is better)', tpot95='TPOT p95 (ms, lower is better)', scheduled95='Scheduled TTFT p95 (s, lower is better)', goodput='Exploratory goodput (req/s, higher is better)')

    def value(row, key):
        s = row['summary']
        if key == 'throughput':
            return s['output_tokens_per_s']
        if key == 'ttft50':
            return s['latencies']['ttft_s']['p50']
        if key == 'tpot95':
            return s['latencies']['tpot_s']['p95'] * 1000
        if key == 'scheduled95':
            return s['scheduled_latencies']['scheduled_ttft_s']['p95']
        return s['exploratory_slo']['goodput_requests_per_s']
    generated = []
    for name, items in panels.items():
        cols = 3 if len(items) == 6 else 2
        nrows = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(cols * 6.2, nrows * 4.3), squeeze=False)
        for ax, (suite, case, phase, key, title) in zip(axes.flat, items):
            selected = [r for r in rows if (r['suite'], r['case'], r['phase']) == (suite, case, phase)]
            y = list(range(len(selected)))
            values = [value(r, key) for r in selected]
            bars = ax.barh(y, values, color=COLORS[:len(selected)])
            ax.set_yticks(y, [label(r) for r in selected], fontsize=9)
            ax.invert_yaxis()
            ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=8)
            ax.set_xlim(0, max(values + [0.1]) * 1.24)
            ax.set_xlabel(units[key], fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.grid(axis='x', alpha=0.18)
            ax.set_axisbelow(True)
        fig.suptitle('DeepSeek-V4-Flash-0731 | 4 x RTX PRO 6000 | PCIe | measured scenarios', fontsize=12)
        note = 'Combined two rounds; percentiles from individual requests. Synthetic finite workloads, not a steady-state SLA.'
        if name == 'arrival-and-concurrency':
            note += ' Goodput: scheduled TTFT <= 1s, TPOT <= 50ms, scheduled E2E <= 10s.'
        if name == 'history-reuse':
            note += ' Warm stages only; cold/JIT results preserved separately.'
        fig.text(0.5, 0.01, note, ha='center', fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.045, 1, 0.95))
        for ext in ['png', 'pdf']:
            target = out / f'{name}.{ext}'
            fig.savefig(target, dpi=180)
            generated.append(target)
        plt.close(fig)
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), figures={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in generated})
    write_text(out / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    print('Generated', len(generated), 'figures in', out)


if __name__=='__main__':main()
