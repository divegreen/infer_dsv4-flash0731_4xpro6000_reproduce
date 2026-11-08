#!/usr/bin/env python3
"""Standalone capacity PNG/PDF curves from the independent capacity summary.

No inference dependencies. Points are finite measurement windows; screening is
never relabeled as confirmed stable capacity. Use an isolated matplotlib env.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#2563a6', '#24957f', '#d58521', '#9256a3', '#c85454', '#5c6f85']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short_arch(value):
    low = value.lower().replace('_', '-')
    if 'dspark' in low:
        return 'TP4 + DSpark3'
    if ('dp4' in low and 'ep' in low):
        return 'DP4 + EP'
    if 'pp2' in low:
        return 'TP2 x PP2'
    if 'tp4' in low:
        return 'TP4'
    return value


def workload_name(workload):
    shapes = workload['shape_mix']
    names = {row[0] for row in shapes}
    if names == {'p512_o128', 'p4096_o256', 'p16384_o128'}:
        title = 'Mixed prefill'
    elif names == {'p512_o128', 'p1024_o1024'}:
        title = 'Mixed interaction / long generation'
    else:
        title = 'Mixed workload'
    detail = ', '.join(f'{int(p)}/{int(o)} ({100*float(w):g}%)' for _, p, o, w in shapes)
    return title, detail


def series(report):
    if report.get('status') != 'PASS' or report.get('schema_version') != 1:
        raise ValueError('Plotting requires a schema-1 PASS independent summary')
    rows = [r for r in report['stages'] if r.get('included_as_measurement')
            and r.get('mode') == 'poisson' and r.get('rate') is not None]
    if not rows:
        raise ValueError('No measured Poisson stages to plot')
    marks = {}
    for item in report.get('capacities', []):
        for rate in item.get('rates', []):
            marks[(item['architecture'], item.get('deployment_key', ''), item['workload_key'], rate['offered_rate_s'])] = rate
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not math.isfinite(row['rate']) or row['rate'] <= 0:
            raise ValueError('Nonpositive or invalid offered rate')
        data[row['workload_key']][(row['architecture'], row.get('deployment_key', ''))].append(row)
    return data, marks


def plot(report, source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    source_bytes = Path(source).read_bytes()
    if json.loads(source_bytes) != report:
        raise ValueError('Loaded summary differs from the source file')
    source_hash, script_hash = (hashlib.sha256(source_bytes).hexdigest(), sha(__file__))
    data, marks = series(report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metric_specs = [('cohort_slo_goodput_requests_s', 'Research-SLO goodput (req/s)', lambda r: r['throughput']['cohort_slo_goodput_requests_s']), ('scheduled_ttft_p95_s', 'Scheduled TTFT P95 (s)', lambda r: r['stats']['metrics']['scheduled_ttft_s']['p95']), ('tpot_p95_ms', 'TPOT P95 (ms)', lambda r: 1000 * r['stats']['metrics']['tpot_s']['p95'] if r['stats']['metrics']['tpot_s']['p95'] is not None else None), ('cohort_output_tokens_s', 'Cohort output (tok/s)', lambda r: r['throughput']['cohort_output_tokens_s'])]
    architectures = sorted({a for archs in data.values() for a in archs}, key=lambda a: (short_arch(a[0]), a[1]))

    def legend_label(identity):
        label = short_arch(identity[0])
        duplicate = sum((short_arch(a[0]) == label for a in architectures)) > 1
        return label + (' [' + (identity[1][:8] or identity[0]) + ']' if duplicate else '')
    colors = {a: COLORS[i % len(COLORS)] for i, a in enumerate(architectures)}
    fig, axes = plt.subplots(len(data), 4, figsize=(17, 3.6 * len(data) + 1.6), squeeze=False)
    plotted = []
    for wi, (workload, archs) in enumerate(sorted(data.items())):
        example = next(iter(archs.values()))[0]
        title, detail = workload_name(example['workload'])
        for mi, (key, ylabel, value) in enumerate(metric_specs):
            ax = axes[wi][mi]
            for arch, rows in sorted(archs.items()):
                grouped = defaultdict(list)
                for row in rows:
                    y = value(row)
                    if y is None or not math.isfinite(y):
                        continue
                    rate = row['rate']
                    grouped[rate].append(y)
                    qualification = marks.get((*arch, workload, rate), {})
                    long_window = row['measurement_window_s'] >= report.get('rules', {}).get('min_confirmation_duration_s', 180.0)
                    confirmed = long_window and qualification.get('finite_stable_confirmed', False)
                    slo_confirmed = confirmed and qualification.get('finite_p95_slo_confirmed', False)
                    marker = '*' if slo_confirmed else 'D' if confirmed else 's' if long_window else 'o'
                    ax.scatter([rate], [y], marker=marker, s=75 if marker == '*' else 30, facecolors=colors[arch] if confirmed else 'none', edgecolors=colors[arch], linewidths=1.1, zorder=4)
                    plotted.append(dict(experiment=row['experiment'], phase_id=row['phase_id'], architecture=arch[0], deployment_key=arch[1], workload_key=workload, offered_rate_s=rate, metric=key, value=y, measurement_window_s=row['measurement_window_s'], marker=marker, finite_stable_confirmed=confirmed, finite_p95_slo_confirmed=slo_confirmed, source_stability=row['stability']['classification']))
                xs = sorted(grouped)
                ys = [statistics.median(grouped[x]) for x in xs]
                if xs:
                    ax.plot(xs, ys, color=colors[arch], linestyle='--', linewidth=1.0, alpha=0.75)
            ax.set_xlabel('Offered Poisson arrival rate (req/s)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f'{title}\n{detail}' if mi == 0 else title, fontsize=10)
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            if key == 'tpot_p95_ms':
                ax.set_ylim(0, max(ax.get_ylim()[1], 55.0))
                ax.axhline(50, color='#555555', linestyle=':', linewidth=1)
                ax.text(0.99, 50, '50 ms Research SLO', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color='#555555')
    handles = [Line2D([0], [0], color=colors[a], linestyle='--', label=legend_label(a)) for a in architectures]
    handles += [Line2D([0], [0], marker=m, linestyle='', color='#444444', markerfacecolor=f, label=label, markersize=7 if m != '*' else 10) for m, f, label in [('o', 'none', 'Screen / not qualified'), ('s', 'none', 'Long window / unconfirmed'), ('D', '#444444', 'Finite stable confirmed'), ('*', '#444444', 'Finite stable + P95 SLO confirmed')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.947), ncol=4, fontsize=9, frameon=False)
    prefix = 'SYNTHETIC FIXTURE | ' if report.get('fixture') else ''
    fig.suptitle(prefix + 'DeepSeek-V4-Flash-0731 | 4 x RTX PRO 6000 | PCIe | online capacity', fontsize=13, y=0.995)
    note = 'Each point is one finite window. Dashed guides = median per-rate window values, not pooled request percentiles.\nGoodput/output throughput use window + drain. Mixed TTFT/TPOT P95 are descriptive; SLO qualification is per input tier.\nScreen points do not establish stable capacity; confirmed points follow the summary rules, not a production SLA.'
    fig.text(0.5, 0.014, note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 0.88))
    generated = []
    for extension in ('png', 'pdf'):
        path = output / f'capacity-curves.{extension}'
        fig.savefig(path, dpi=180, metadata={'Title': 'Phase A finite online capacity'})
        generated.append(path)
    plt.close(fig)
    if sha(source) != source_hash or sha(__file__) != script_hash:
        raise RuntimeError('Summary or plotting source changed during rendering; no manifest produced')
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), source=str(Path(source).resolve()), source_sha256=source_hash, script_sha256=script_hash, summary_rules=report.get('rules'), matplotlib_version=matplotlib.__version__, fixture=bool(report.get('fixture')), point_interpretation=note, plotted_points=plotted, figures={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in generated})
    write_text(output / 'manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return dict(status='PASS', output=str(output.resolve()), points=len(plotted), figures=[str(p.resolve()) for p in generated])


def self_test(output=None):
    rows = []
    mix = dict(shape_mix=[['p512_o128', 512, 128, 0.7], ['p1024_o1024', 1024, 1024, 0.3]])
    for rate in (0.5, 1.0, 2.0):
        rows.append(dict(included_as_measurement=True, mode='poisson', rate=rate, experiment='fixture-screen', phase_id=f'screen-{rate}', architecture='tp4-fixture', deployment_key='aaaaaaaa11111111', workload_key='fixture-mix', workload=mix, measurement_window_s=60.0, throughput=dict(cohort_slo_goodput_requests_s=0.6 * rate, cohort_output_tokens_s=150 * rate), stats=dict(metrics=dict(scheduled_ttft_s=dict(p95=rate), tpot_s=dict(p95=0.02 * rate))), stability=dict(classification='SCREEN_ONLY')))
    confirm = dict(rows[1], experiment='fixture-confirm', phase_id='confirm-1', measurement_window_s=180.0, stability=dict(classification='STABLE_IN_WINDOW'))
    rows.append(confirm)
    rows.append(dict(confirm, deployment_key='bbbbbbbb22222222', experiment='fixture-other-deployment', throughput=dict(cohort_slo_goodput_requests_s=0.3, cohort_output_tokens_s=110.0)))
    report = dict(schema_version=1, status='PASS', fixture=True, rules=dict(min_confirmation_duration_s=180), stages=rows, capacities=[dict(architecture='tp4-fixture', deployment_key='aaaaaaaa11111111', workload_key='fixture-mix', rates=[dict(offered_rate_s=1.0, finite_stable_confirmed=True, finite_p95_slo_confirmed=True)])])
    with tempfile.TemporaryDirectory(prefix='capacity-plot-fixture-') as temporary:
        root = Path(temporary)
        source = root / 'summary.json'
        write_text(source, json.dumps(report))
        result = plot(report, source, Path(output) if output else path_at(root, 'figures'))
        manifest = load_document(Path(result['output']) / 'manifest.json')
        assert all((not p['finite_stable_confirmed'] for p in manifest['plotted_points'] if p['experiment'] == 'fixture-screen'))
        assert all((p['marker'] == '*' for p in manifest['plotted_points'] if p['experiment'] == 'fixture-confirm'))
        assert all((p['marker'] == 's' and (not p['finite_stable_confirmed']) for p in manifest['plotted_points'] if p['deployment_key'] == 'bbbbbbbb22222222'))
        assert len({p['deployment_key'] for p in manifest['plotted_points']}) == 2
        assert manifest['fixture'] and len(manifest['figures']) == 2
        for name, expected in manifest['figures'].items():
            path = Path(result['output']) / name
            assert path.stat().st_size > 1000 and sha(path) == expected['sha256']
        original_sha = globals()['sha']
        original_bytes = source.read_bytes()
        changed = False

        def mutate_at_final_source_check(path):
            nonlocal changed
            if Path(path) == source and (not changed):
                write_bytes(source, original_bytes + b' ')
                changed = True
            return original_sha(path)
        globals()['sha'] = mutate_at_final_source_check
        try:
            try:
                plot(report, source, root / 'mutated-figures')
            except RuntimeError as exc:
                assert 'changed during rendering' in str(exc)
            else:
                raise AssertionError('Source mutation during rendering was not rejected')
            assert not (root / 'mutated-figures/manifest.json').exists()
        finally:
            globals()['sha'] = original_sha
            write_bytes(source, original_bytes)
        return dict(status='PASS', scope='synthetic CPU-only plot fixture, not performance data', checks=['screen never relabeled confirmed', 'same-name deployments never share curves or qualifications', 'source mutation during rendering rejected without a false manifest', 'qualified marker uses capacity aggregate', 'PNG/PDF artifacts and hashes', 'input/script hashes and all plotted values retained'], output=result['output'] if output else 'temporary fixture removed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=path_at(ROOT, 'summary/capacity-summary.json'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test(args.output_dir)
    else:
        if not args.output_dir:
            parser.error('--output-dir is required and must not already exist')
        result = plot(load_document(args.input), args.input, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
