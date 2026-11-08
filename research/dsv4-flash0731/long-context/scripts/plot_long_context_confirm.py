#!/usr/bin/env python3
"""Publication figures for the two matched Stage B 450-second confirmations.

Only b10/b11 formal Poisson phases are eligible. Inspection can validate an
incomplete real input; rendering always requires both experiments and two seeds.
Matplotlib/NumPy are imported only for rendering from the caller's environment.
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
import math
from pathlib import Path

EXPERIMENTS = ('context-tp2pp2-confirm', 'context-tp4-th2048-cad8-confirm')
LABELS = {'context-tp2pp2-confirm': 'TP2×PP2',
          'context-tp4-th2048-cad8-confirm': 'TP4 cap2048 + cadence8'}
SHAPES = ('p512_o128', 'p16384_o128', 'p32768_o128', 'p65536_o128')
TOKENS = (512, 16384, 32768, 65536)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(report, complete=True):
    require(report.get('status') == 'PASS' and not report.get('errors'), 'Plot requires PASS offline analysis')
    selected = [s for s in report['stages'] if s.get('experiment') in EXPERIMENTS
                and s.get('included_as_measurement') and s.get('role') not in ('warmup', 'profile')]
    require(selected, 'No b10/b11 formal confirmation windows')
    by_experiment = {name: [] for name in EXPERIMENTS}
    for stage in selected:
        require(stage.get('role') == 'confirm' and stage.get('mode') == 'poisson', 'Only formal Poisson confirmation windows')
        require(stage.get('measurement_window_s') == 450 and stage.get('rate') == .5,
                'Confirmation plot requires exactly 450 seconds at 0.5 req/s; screens cannot be mixed')
        require(stage.get('owned_run_terminal_success') is True, 'Every selected experiment must be terminal successful')
        require(stage['stability']['classification'] in ('STABLE_IN_WINDOW', 'INCONCLUSIVE', 'GROWING_BACKLOG'),
                'Invalid confirmation stability classification')
        require(set(stage['by_shape']) == set(SHAPES), 'Expected exactly four input shapes at output length 128')
        for shape, tokens in zip(SHAPES, TOKENS):
            group = stage['by_shape'][shape]
            require(group.get('input_tokens') == [tokens] and group.get('requested_output_tokens') == [128],
                    'Shape descriptor differs from matched confirmation protocol')
            require(group['requests'] > 0 and group['successful'] == group['requests'], 'Missing or failed shape cohort')
            for key in ('scheduled_ttft_s', 'tpot_s'):
                require(finite(group['metrics'][key]['p95']) and group['metrics'][key]['p95'] >= 0, 'Missing/nonfinite P95')
        short = stage['short512']
        require(finite(short['cohort_slo_goodput_requests_s']) and short['cohort_slo_goodput_requests_s'] >= 0,
                'Missing/nonfinite short Goodput')
        require(finite(short['slo_joint_pass_rate']) and 0 <= short['slo_joint_pass_rate'] <= 1,
                'Missing/nonfinite short SLO satisfaction')
        require(finite(stage['drain_s']) and stage['drain_s'] >= 0, 'Invalid drain')
        require(finite(stage['throughput']['cohort_output_tokens_s']) and stage['throughput']['cohort_output_tokens_s'] >= 0,
                'Missing/nonfinite cohort token throughput')
        require(stage.get('arrival_identity', {}).get('sha256'), 'Missing exact workload arrival identity')
        by_experiment[stage['experiment']].append(stage)
    missing = []
    for name, rows in by_experiment.items():
        if not rows:
            missing.append(name)
            continue
        require(len(rows) == 2 and len({s['seed'] for s in rows}) == 2, 'Each available experiment needs exactly two distinct confirmation seeds')
        require(len({s['deployment_key'] for s in rows}) == 1, 'Deployment changed between confirmation windows')
        rows.sort(key=lambda s: s['seed'])
    if complete:
        require(not missing, 'Formal figure requires both complete experiments; missing: '+', '.join(missing))
    present = [rows for rows in by_experiment.values() if rows]
    seeds = [s['seed'] for s in present[0]]
    require(all([s['seed'] for s in rows] == seeds for rows in present), 'Experiments use different confirmation seeds')
    require(len({s['workload_key'] for s in selected}) == 1, 'Shape distribution changed between windows')
    if not missing:
        for left, right in zip(by_experiment[EXPERIMENTS[0]], by_experiment[EXPERIMENTS[1]]):
            require(left['arrival_identity']['sha256'] == right['arrival_identity']['sha256'],
                    'Paired seed arrival/prompt identities differ')
            require(all(left['by_shape'][shape]['requests'] == right['by_shape'][shape]['requests'] for shape in SHAPES),
                    'Paired shape sample counts differ')
    return by_experiment, seeds, missing


def render(report, summary, output):
    rows, seeds, _ = select(report)
    require(not output.exists(), 'Output directory already exists; use a unique figure directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.4))
    fig.subplots_adjust(left=0.065, right=0.95, top=0.82, bottom=0.12, hspace=0.48, wspace=0.32)
    colors = ('#2166ac', '#c75b12')
    hatches = ('', '///')
    width = 0.19
    x = np.arange(len(SHAPES))
    bar_labels = []
    for ei, name in enumerate(EXPERIMENTS):
        for si, seed in enumerate(seeds):
            bar_labels.append(Patch(facecolor=colors[ei], edgecolor='white', hatch=hatches[si], label=f'{LABELS[name]} · seed {seed}'))
    for ax, key, scale, title, ylabel in ((axes[0, 0], 'scheduled_ttft_s', 1, 'A  Time to first token, P95', 'Scheduled TTFT (s)'), (axes[0, 1], 'tpot_s', 1000, 'B  Average output-token interval, P95', 'TPOT (ms)')):
        maxima = []
        for ei, name in enumerate(EXPERIMENTS):
            for si, stage in enumerate(rows[name]):
                values = [stage['by_shape'][shape]['metrics'][key]['p95'] * scale for shape in SHAPES]
                maxima.extend(values)
                offsets = x + (ei * 2 + si - 1.5) * width
                bars = ax.bar(offsets, values, width=width, color=colors[ei], edgecolor='white', linewidth=0.7, hatch=hatches[si], alpha=0.96 if si == 0 else 0.72, zorder=3)
                ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=3, fontsize=7.8, rotation=90)
        ax.set_xticks(x, ['512', '16K', '32K', '64K'])
        ax.set_xlabel('Input tokens (output: 128 tokens)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=0.22, zorder=0)
        ax.set_ylim(0, max(maxima) * 1.25 if maxima and max(maxima) > 0 else 1)
        if key == 'tpot_s':
            ax.axhline(50, color='#555555', linestyle='--', linewidth=1.0, label='50 ms Research reference', zorder=4)
            ax.text(0.98, 0.97, 'Research TPOT reference: 50 ms', transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='#555555')
        else:
            ax.text(0.02, 0.97, 'Research TTFT limits: 512 = 1 s; 16K = 8 s\n32K / 64K: no TTFT SLO defined', transform=ax.transAxes, va='top', fontsize=8.5, color='#555555')
    ax = axes[1, 0]
    right = ax.twinx()
    sx = np.arange(2)
    max_goodput = 0.0
    for ei, name in enumerate(EXPERIMENTS):
        values = [s['short512']['cohort_slo_goodput_requests_s'] for s in rows[name]]
        rates = [100 * s['short512']['slo_joint_pass_rate'] for s in rows[name]]
        xpos = sx + (ei - 0.5) * 0.28
        bars = ax.bar(xpos, values, width=0.26, color=colors[ei], alpha=0.85, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4, fontsize=9)
        right.scatter(xpos, rates, s=65, marker='D', facecolors='white', edgecolors=colors[ei], linewidths=1.6, zorder=5)
        for xx, rate in zip(xpos, rates):
            right.annotate(f'{rate:.1f}%', (xx, rate), xytext=(0, -15), textcoords='offset points', ha='center', va='top', color=colors[ei], fontsize=8.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.94, pad=1.5))
        max_goodput = max(max_goodput, *values)
    ax.set_xticks(sx, [f'Seed {s}' for s in seeds])
    ax.set_ylabel('Short-request Goodput (req/s); bars')
    right.set_ylabel('Joint short Research SLO satisfied (%); diamonds')
    ax.set_ylim(0, max_goodput * 1.4 if max_goodput else 0.1)
    right.set_ylim(0, 110)
    right.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', alpha=0.22, zorder=0)
    ax.set_title('C  Short 512-token request utility', loc='left', fontweight='bold')
    ax.text(0.02, 0.97, 'Goodput includes drain; failures remain in SLO denominator', transform=ax.transAxes, va='top', fontsize=8.5, color='#555555')
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('D  Finite-window stability and total service output', loc='left', fontweight='bold', pad=12)
    table_rows = []
    for name in EXPERIMENTS:
        for si, stage in enumerate(rows[name]):
            state = stage['stability']['classification']
            state_label = {'STABLE_IN_WINDOW': 'Stable in window', 'INCONCLUSIVE': 'Inconclusive', 'GROWING_BACKLOG': 'Growing backlog'}[state]
            table_rows.append([LABELS[name].replace(' + ', '\n+ '), f'S{si + 1}', state_label, f'{stage['drain_s']:.2f}', f'{stage['throughput']['cohort_output_tokens_s']:.1f}', str(stage['offered_requests'])])
    table = ax.table(cellText=table_rows, colLabels=['Deployment', 'Seed', 'Stability', 'Drain s', 'Output\ntok/s', 'N'], colWidths=[0.28, 0.07, 0.26, 0.12, 0.15, 0.08], cellLoc='center', bbox=[-0.02, 0.29, 1.05, 0.62])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if row == 0:
            cell.set_facecolor('#eef1f4')
            cell.set_text_props(weight='bold')
        elif col == 0:
            cell.set_text_props(color=colors[(row - 1) // 2], weight='bold')
    ax.text(0, 0.16, 'S1 / S2 use the same arrivals and token-ID prompts across deployments.\nStability uses the frozen Stage A finite-window rule; it is separate\nfrom latency SLO and does not establish a production SLA.', transform=ax.transAxes, fontsize=9, va='top', linespacing=1.4)
    fig.suptitle('DeepSeek-V4-Flash-0731 · long-context mixed-service confirmation', fontsize=16, fontweight='bold', y=0.985)
    fig.text(0.5, 0.945, 'GPU 0–3 · PCIe · APC off · Poisson 0.5 req/s · 450 s × 2 paired seeds per deployment', ha='center', fontsize=11)
    fig.legend(handles=bar_labels, loc='upper center', bbox_to_anchor=(0.5, 0.918), ncol=2, frameon=False, fontsize=9.5)
    fig.text(0.065, 0.045, 'Only completed b10 / b11 confirmation windows. No screening, warmup or Profile rows. Each bar is one window, not a confidence interval.\nCohort throughput / Goodput includes drain. TPOT is an average interval, not per-token ITL. No joint TTFT/E2E SLO is defined above 16K.', fontsize=9, color='#444444', linespacing=1.4)
    figures = []
    for extension in ('png', 'pdf'):
        path = output / ('long-context-confirmation.' + extension)
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        figures.append(dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size))
    plt.close(fig)
    selected_rows = []
    for name in EXPERIMENTS:
        for stage in rows[name]:
            selected_rows.append({key: stage[key] for key in ('experiment', 'phase_id', 'seed', 'rate', 'measurement_window_s', 'total_wall_s', 'drain_s', 'arrival_identity', 'stability', 'throughput', 'short512', 'by_shape')})
    manifest = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), summary=str(summary.resolve()), summary_sha256=sha(summary), script_sha256=sha(Path(__file__)), numpy=np.__version__, matplotlib=matplotlib.__version__, figures=figures, selected_phases=selected_rows, scope='Exactly b10/b11, each two matched 450-second Poisson windows; no new capacity or quality certification.')
    with open_output(output / 'manifest.json', 'x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    return dict(status='PASS', selected_windows=4, output_dir=str(output), figures=figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--inspect-only', action='store_true', help='Validate actual available windows; permit a missing counterpart, never render')
    args = parser.parse_args()
    if args.inspect_only and args.output_dir:
        parser.error('--inspect-only does not create a figure directory')
    if not args.inspect_only and (not args.output_dir):
        parser.error('Rendering requires --output-dir')
    initial_sha = sha(args.summary)
    report = load_document(args.summary)
    if args.inspect_only:
        rows, seeds, missing = select(report, complete=False)
        result = dict(status='PARTIAL_INPUT_VALID' if missing else 'COMPLETE_INPUT_VALID', seeds=seeds, observed={name: len(items) for name, items in rows.items()}, missing_experiments=missing, rendering_permitted=not missing, summary_sha256=initial_sha)
    else:
        result = render(report, args.summary, args.output_dir)
    require(sha(args.summary) == initial_sha, 'Summary changed while plotting')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
