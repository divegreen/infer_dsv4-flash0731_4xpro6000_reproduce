#!/usr/bin/env python3
"""Standalone CPU figures of verified D scheduler decisions, not GPU ownership."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use unique figure directory')
    raw = args.input.read_bytes()
    report = json.loads(raw)
    if report['status'] != 'PASS' or report['matched_conditions']['status'] != 'PASS':
        raise ValueError('Requires accepted matched Profile report')
    for name, entry in report['evidence_ledger'].items():
        content = Path(name).read_bytes()
        if sha(content) != entry['sha256'] or len(content) != entry['bytes']:
            raise ValueError('Evidence changed: ' + name)
    experiments = report['experiments']
    if len(experiments) != 2 or {e['scheduler']['identity']['policy'] for e in experiments} != {'observe-v1', 'fairshare-v1'}:
        raise ValueError('Requires one validated observation per policy')
    if not all((e['scheduler']['request_position_continuity'].startswith('PASS:') for e in experiments)):
        raise ValueError('Position-continuity audit missing')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    args.output_dir.mkdir(parents=True)
    experiments.sort(key=lambda e: e['scheduler']['identity']['policy'] == 'fairshare-v1')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True, layout='constrained')
    values = []
    maximum = 0
    for ax, experiment in zip(axes, experiments):
        scheduler = experiment['scheduler']
        origin = min((t['step'] for t in scheduler['timeline']))
        maximum = max(maximum, max((t['step'] for t in scheduler['timeline'])) - origin)
        entries = sorted(scheduler['binding'].items(), key=lambda kv: kv[1]['index'])
        labels = []
        for y, (key, binding) in enumerate(entries):
            record = scheduler['requests'][key]
            allocations = record['allocations']
            context = [x for x in allocations if x['context_tokens']]
            generation = [x for x in allocations if x['generation_tokens']]
            ax.scatter([x['step'] - origin for x in context], [y] * len(context), s=[14 + 30 * x['context_tokens'] / 2048 for x in context], marker='s', color='#d17b22', edgecolors='none')
            ax.scatter([x['step'] - origin for x in generation], [y] * len(generation), s=6, color='#176f91', edgecolors='none')
            observed = record['first_observed_step'] - origin
            first = record['first_allocation_step'] - origin
            ax.plot([observed, first], [y, y], color='#555555', linewidth=1, zorder=0)
            ax.scatter([observed], [y], s=35, marker='|', color='black', zorder=4)
            labels.append(f'{binding['index']}: {('64K' if binding['request_class'] == 'long' else '512')}')
            values.append(dict(experiment=experiment['experiment'], policy=scheduler['identity']['policy'], request_index=binding['index'], context_tokens=record['context_tokens'], generation_tokens=record['generation_tokens'], first_observed_relative_step=observed, first_allocation_relative_step=first, observation_to_allocation_steps=record['observed_to_first_allocation_steps'], allocated_calls=record['allocated_calls']))
        ax.set_yticks(range(8), labels)
        ax.invert_yaxis()
        ax.set_ylabel('Request index / input tokens')
        ax.set_title(scheduler['identity']['policy'] + f' | {scheduler['profile_active_steps']} active steps')
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('Scheduler call index relative to first profile observation (not wall time)')
    axes[-1].set_xlim(-3, maximum + 3)
    fig.suptitle('Profile diagnostic: exact request-ID token allocation decisions\nTwo 64K/128 and six 512/128 requests; APC off; not normal performance timing', fontsize=12)
    axes[0].legend(handles=[Line2D([], [], marker='s', color='#d17b22', linestyle='', label='Context input allocation (size = count)'), Line2D([], [], marker='.', color='#176f91', linestyle='', label='Generation input allocation'), Line2D([], [], marker='|', color='black', linestyle='-', label='First observed to first allocated')], loc='upper right', fontsize=8)
    for ext in ('svg', 'png'):
        fig.savefig(args.output_dir / ('request-allocations.' + ext), dpi=150)
    plt.close(fig)
    totals = []
    for e in experiments:
        scheduler = e['scheduler']
        trace = e['traces'][0]
        communications = [c for c in trace['categories'] if c['category'] == 'communication']
        totals.append(dict(policy=scheduler['identity']['policy'], active_steps=scheduler['profile_active_steps'], context_tokens=scheduler['context_tokens_sum'], generation_tokens=scheduler['generation_tokens_sum'], long_short_context_steps=scheduler['long_short_context_steps'], long_context_short_generation_steps=scheduler['long_context_short_generation_steps'], rank0_communication_category_calls=sum((c['calls'] for c in communications)) if communications else None, rank0_cuda_kernel_calls=trace['cuda_kernel_events']))
    if args.input.read_bytes() != raw:
        raise RuntimeError('Input report changed while plotting')
    paths = sorted(args.output_dir.iterdir())
    result = dict(status='PASS', scope='Profile-only request allocation visualization; no GPU request ownership or capacity/normal timing claim', input=str(args.input.resolve()), input_sha256=sha(raw), script_sha256=sha(Path(__file__).read_bytes()), python_executable=sys.executable, matplotlib=matplotlib.__version__, requests=values, totals=totals, outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in paths})
    with open_output(args.output_dir / 'figure-data.json', 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', requests=len(values), output_dir=str(args.output_dir))))
if __name__=='__main__':main()
