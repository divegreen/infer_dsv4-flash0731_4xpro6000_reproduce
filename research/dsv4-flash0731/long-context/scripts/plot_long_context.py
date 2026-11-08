"""Publication artifacts from an already audited Stage B analysis JSON."""

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    report = load_document(args.summary)
    if report['status'] != 'PASS':
        raise ValueError('Plot requires analysis PASS; not a performance/SLO PASS')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stages = [s for s in report['stages'] if s['included_as_measurement']]
    experiments = list(dict.fromkeys((s['experiment'] for s in stages)))
    phases = list(dict.fromkeys((s['phase_id'] for s in stages)))
    lookup = {(s['experiment'], s['phase_id']): s for s in stages}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), summary=str(args.summary.resolve()), summary_sha256=sha(args.summary), script_sha256=sha(Path(__file__)), matplotlib=matplotlib.__version__, numpy=np.__version__, figures=[])

    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            path = args.output_dir / (name + '.' + ext)
            fig.savefig(path, dpi=170, bbox_inches='tight')
            manifest['figures'].append(dict(file=path.name, sha256=sha(path)))
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(max(13, len(phases) * 2), max(7, len(experiments) * 1.25)))
    measures = [('Short 512-input TTFT P95 (s)', lambda s: s['short512']['metrics']['scheduled_ttft_s']['p95'], True), ('Short joint Research SLO satisfaction (%)', lambda s: 100 * s['short512']['slo_joint_pass_rate'] if s['short512']['slo_joint_pass_rate'] is not None else None, False), ('64K-input TTFT P95 (s; descriptive, no SLO)', lambda s: s['by_shape'].get('p65536_o128', {}).get('metrics', {}).get('scheduled_ttft_s', {}).get('p95'), True), ('Peak KV usage of busiest engine (%)', lambda s: 100 * s['raw_gauges']['aggregate']['vllm:kv_cache_usage_perc']['max'] if s['raw_gauges']['aggregate']['vllm:kv_cache_usage_perc']['max'] is not None else None, False)]
    for ax, (title, getter, log) in zip(axes.flat, measures):
        values = np.full((len(experiments), len(phases)), np.nan)
        for i, experiment in enumerate(experiments):
            for j, phase in enumerate(phases):
                row = lookup.get((experiment, phase))
                if row:
                    value = getter(row)
                    if value is not None:
                        values[i, j] = value
        positive = values[np.isfinite(values) & (values > 0)]
        norm = LogNorm(vmin=max(0.01, positive.min()), vmax=max(positive.max(), positive.min() * 1.01)) if log and len(positive) else None
        im = ax.imshow(values, aspect='auto', cmap='viridis', norm=norm, **{} if norm else dict(vmin=0, vmax=100))
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(j, i, f'{value:.2f}' if log else f'{value:.1f}', ha='center', va='center', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1))
        ax.set_xticks(range(len(phases)), [p.replace('context_mix-', 'mix ').replace('-screen', '') for p in phases], rotation=30, ha='right')
        ax.set_yticks(range(len(experiments)), experiments)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle('Stage B matched workloads: each cell is a finite observed phase, not a production SLA', y=1.015)
    save(fig, 'matched-scheduling')
    manifest['notes'] = ['No warmup/Profile rows; do not compare unmatched workloads as causal evidence.', '32K/64K TTFT/E2E SLO undefined; small-sample tails are descriptive.', 'KV gauge is max across per-engine fractions, not sum.']
    write_text(args.output_dir / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
