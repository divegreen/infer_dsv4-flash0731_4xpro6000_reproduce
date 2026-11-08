"""Generate publication-exportable plots from build_analysis.py output."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = load_document(path_at(ROOT, 'results/analysis.json'))
    output = path_at(ROOT, 'figures')
    output.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for row in data['screen']:
        experiments.setdefault(row['experiment'], {})[row['case']] = row
    names = [name for name, values in experiments.items() if 'p512_o64_c1' in values and 'p512_o64_c32' in values and name.endswith('-screen')]
    labels = [name.removesuffix('-screen').replace('marlin-graph', 'marlin') for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(names) * 0.38)), sharey=True)
    y = list(range(len(names)))
    axes[0].barh(y, [experiments[name]['p512_o64_c1']['tpot_p50_ms_by_round'][0] for name in names], color='#3173a6')
    axes[1].barh(y, [experiments[name]['p512_o64_c32']['output_tokens_per_s'] for name in names], color='#23886c')
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('C1 TPOT p50 (ms, lower is better)')
    axes[1].set_xlabel('C32 output tokens/s (higher is better)')
    axes[0].set_title('Single-request decoding latency')
    axes[1].set_title('One burst of 32 requests')
    for axis in axes:
        axis.grid(axis='x', alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle('DSv4 Flash 0731 | 4 x RTX PRO 6000 PCIe | screening 512 input / 64 output', fontsize=12)
    fig.text(0.5, 0.005, 'Single screening runs, not steady-state SLA. Outliers retained. Sources: analysis.json and per-run raw JSON.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'screening.{ext}', dpi=180)
    plt.close(fig)
    full = [row for row in data['full'] if row['report_status'] == 'PASSED']
    if any((row['connection_policy'] == 'fresh_per_request' for row in full)):
        full = [row for row in full if row['connection_policy'] == 'fresh_per_request']
    if full:
        cases = list(dict.fromkeys((row['case'] for row in full)))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, case in zip(axes.flat, cases):
            rows = [row for row in full if row['case'] == case]
            ys = range(len(rows))
            values = [row['output_tokens_per_s'] for row in rows]
            errors = [[v - r['throughput_min'] for v, r in zip(values, rows)], [r['throughput_max'] - v for v, r in zip(values, rows)]]
            ax.barh(list(ys), values, xerr=errors, color='#3173a6', capsize=3)
            ax.set_yticks(list(ys), [row['experiment'].removesuffix('-full') for row in rows], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(case)
            ax.set_xlabel('Output tokens/s; bars = combined rounds, whiskers = range')
            ax.grid(axis='x', alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle('Completed repeated workloads | ' + full[0]['connection_policy'] + ' | range is not a confidence interval')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'repeated-workloads.{ext}', dpi=180)
        plt.close(fig)
    long_rows = [row for row in data['long'] if row['report_status'] == 'PASSED']
    if long_rows:
        cases = list(dict.fromkeys((row['case'] for row in long_rows)))
        fig, axes = plt.subplots(1, len(cases), figsize=(17, 5), squeeze=False)
        for ax, case in zip(axes.flat, cases):
            rows = [row for row in long_rows if row['case'] == case]
            ax.barh(range(len(rows)), [sum(row['ttft_p50_ms_by_round']) / len(row['ttft_p50_ms_by_round']) / 1000 for row in rows], color='#23886c')
            ax.set_yticks(range(len(rows)), [row['experiment'].removesuffix('-long') for row in rows], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(case)
            ax.set_xlabel('Mean of round TTFT medians (s)')
            ax.grid(axis='x', alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle('Long-input latency | 64K server limit; requests tested at 16K and 32K only')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'long-inputs.{ext}', dpi=180)
        plt.close(fig)
    print('Figures saved to', output)


if __name__ == "__main__":
    main()
