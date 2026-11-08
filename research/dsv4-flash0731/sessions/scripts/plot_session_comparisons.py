#!/usr/bin/env python3
"""Render only real completed Stage C comparisons; preserve individual seeds.

No averaging across seeds, no model calls, no fake values and no overwrites.
Warmup/Profile/pilot and failed owned runs are disclosed but never plotted.
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
from datetime import datetime,timezone
import csv
import hashlib
import json
import math
from pathlib import Path


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def option(args,name,default):
    return args[args.index(name)+1] if name in args else default


def deployment(config):
    args=config['args']
    tp=int(option(args,'--tensor-parallel-size',1));pp=int(option(args,'--pipeline-parallel-size',1));dp=int(option(args,'--data-parallel-size',1))
    if dp>1:base=f'DP{dp}'+('+EP' if '--enable-expert-parallel' in args else '')
    elif pp>1:base=f'TP{tp}×PP{pp}'
    else:base=f'TP{tp}'
    if '--speculative-config' in args:
        speculative=json.loads(option(args,'--speculative-config','{}'))
        base+=' + '+('DSpark' if 'dspark' in json.dumps(speculative).lower() else 'speculative')+str(speculative.get('num_speculative_tokens',''))
    apc='APC on' if '--enable-prefix-caching' in args else 'APC off' if '--no-enable-prefix-caching' in args else 'APC unspecified'
    return base,apc


def collect(report):
    need(report.get('status')=='PASS' and not report.get('errors'),'Require evidence-analysis PASS, not partial/error summary')
    experiments={e['id']:e for e in report['experiments']}
    rows=[];excluded=[];seen=set()
    for stage in report['stages']:
        key=(stage['experiment'],stage['phase']['id']);need(key not in seen,'Duplicate experiment/phase');seen.add(key)
        phase=stage['phase'];stats=stage['summary']
        if not stage.get('included_as_measurement') or not stage.get('terminal_success') or phase['role'] not in ('screen','measure','confirm'):
            excluded.append(dict(experiment=stage['experiment'],phase_id=phase['id'],role=phase['role'],reason=stage.get('exclusion_reason')));continue
        need(experiments[stage['experiment']]['terminal_success'],'Included stage from failed owned run')
        if phase['arrival']!='closed':
            excluded.append(dict(experiment=stage['experiment'],phase_id=phase['id'],role=phase['role'],
                                 reason='Open-arrival phases are reported separately; this comparison uses fixed concurrency.'))
            continue
        need(stats['sessions']>0 and stage['total_wall_s']>0,'Empty/invalid measured cohort')
        architecture,apc=deployment(experiments[stage['experiment']]['config'])
        cost=stats.get('tokens_per_successful_task') or {}
        row=dict(experiment=stage['experiment'],phase_id=phase['id'],role=phase['role'],kind=phase['kind'],
                 architecture=architecture,apc=apc,concurrency=phase['concurrency'],seed=phase['seed'],
                 cache_scope=phase['cache_scope'],dp_routing=phase['dp_routing'],sessions=stats['sessions'],
                 protocol_completed=stats['protocol_completed'],successful_tasks=stats['successful_tasks'],
                 protocol_sessions_s=stats['completed_sessions_s'],successful_tasks_s=stats['successful_tasks_s'],
                 task_success_rate=stats['task_success_rate'],visible_latency_p95_s=stats['user_visible_latency_s']['p95'],
                 visible_latency_samples=stats['user_visible_latency_s']['n'],cohort_latency_p95_s=stats['session_latency_s']['p95'],
                 output_tokens_s=stats['output_tokens_s'],cache_fraction=stats['cache_fraction'],
                 known_prompt_tokens_per_success=cost.get('prompt'),known_output_tokens_per_success=cost.get('output'),
                 token_cost_complete=cost.get('token_usage_complete'),gpu_j_per_successful_task=stats.get('gpu_j_per_successful_task'),
                 actual_model_calls=stats['model_calls'],actual_tool_calls=stats['tool_calls'],
                 total_wall_s=stage['total_wall_s'],arrival_workload_sha256=stage['arrival_workload_sha256'],
                 workload_identity_scope=stage['arrival_workload_identity']['scope'],terminal_reasons=stats['terminal_reasons'])
        if phase['kind']=='replay':need(stats['successful_tasks'] is None and stats['task_success_rate'] is None,'Replay cannot have task quality score')
        for field in ('protocol_sessions_s','total_wall_s'):
            need(type(row[field]) in (int,float) and math.isfinite(row[field]) and row[field]>=0,'Invalid plotted metric')
        rows.append(row)
    return rows,excluded


def deployment_label(row):
    label=row['architecture']+'\n'+row['apc']
    if row['architecture'].startswith('DP'):label+='\ndefault routing'
    return label


def groups(rows):
    result=[]
    # Only the actually matched treatment groups are compared. A single
    # available treatment remains in tables, not a fabricated comparison.
    definitions=(
        ('tp4_cache_scope',lambda r:r['kind']=='replay' and r['architecture']=='TP4' and r['apc']=='APC on' and r['concurrency']==8 and r['dp_routing']=='none',lambda r:r['cache_scope'],
         'Exact replay · TP4 APC on · C8 · cache namespace',('protocol_sessions_s','visible_latency_p95_s','cache_fraction')),
        ('replay_deployment_c1',lambda r:r['kind']=='replay' and r['concurrency']==1 and r['cache_scope']=='session' and r['dp_routing']=='none',deployment_label,
         'Exact replay · C1 · deployment comparison',('protocol_sessions_s','visible_latency_p95_s','cache_fraction')),
        ('replay_deployment_c8',lambda r:r['kind']=='replay' and r['concurrency']==8 and r['cache_scope']=='session' and r['dp_routing']=='none',deployment_label,
         'Exact replay · C8 · deployment comparison',('protocol_sessions_s','visible_latency_p95_s','cache_fraction')),
        ('agent_deployment_c1',lambda r:r['kind']=='agent' and r['concurrency']==1 and r['cache_scope']=='session' and r['dp_routing']=='none',deployment_label,
         'Model-driven local authored tasks · C1 · deployment comparison',('successful_tasks_s','task_success_rate','visible_latency_p95_s')),
        ('agent_deployment_c6',lambda r:r['kind']=='agent' and r['concurrency']==6 and r['cache_scope']=='session' and r['dp_routing']=='none',deployment_label,
         'Model-driven local authored tasks · C6 · deployment comparison',('successful_tasks_s','task_success_rate','visible_latency_p95_s')),
        ('dp_replay_routing',lambda r:r['kind']=='replay' and r['architecture'].startswith('DP4') and r['concurrency']==8 and r['cache_scope']=='session',lambda r:r['dp_routing'],
         'Exact replay · DP4+EP APC on · C8 · routing',('protocol_sessions_s','visible_latency_p95_s','cache_fraction')),
        ('dp_agent_routing',lambda r:r['kind']=='agent' and r['architecture'].startswith('DP4') and r['concurrency']==6 and r['cache_scope']=='session',lambda r:r['dp_routing'],
         'Model-driven local authored tasks · DP4+EP APC on · C6 · routing',('successful_tasks_s','task_success_rate','visible_latency_p95_s')))
    for identifier,predicate,label,title,metrics in definitions:
        selected=[dict(r,condition=label(r)) for r in rows if predicate(r)]
        if len({r['condition'] for r in selected})<2:continue
        identity=defaultdict(set)
        counts=defaultdict(set)
        conditions=sorted({r['condition'] for r in selected})
        for r in selected:
            identity[r['seed']].add(r['arrival_workload_sha256']);counts[r['seed']].add(r['sessions'])
        need(all(len(v)==1 for v in identity.values()),identifier+': within-seed workload identities do not match')
        need(all(len(v)==1 for v in counts.values()),identifier+': within-seed sample counts differ')
        pairs={(r['condition'],r['seed']) for r in selected};need(len(pairs)==len(selected),identifier+': duplicate condition/seed')
        complete=len(pairs)==len(conditions)*len(identity)
        result.append(dict(id=identifier,title=title,metrics=metrics,rows=selected,conditions=conditions,seeds=sorted(identity),
                           complete_paired_grid=complete,missing_pairs=[dict(condition=c,seed=s) for c in conditions for s in sorted(identity) if (c,s) not in pairs]))
    return result


LABELS={'protocol_sessions_s':'Protocol completed sessions / cohort second',
        'successful_tasks_s':'Successful tasks / cohort second',
        'visible_latency_p95_s':'User-visible Session latency P95 (s)',
        'task_success_rate':'Objective task success (%)', 'cache_fraction':'Known prompt cache fraction (%)'}


def render(report, summary, output):
    rows, excluded = collect(report)
    views = groups(rows)
    need(rows and views, 'No real matched normal-performance comparison is available')
    need(not output.exists(), 'Output already exists; use a new artifact ID')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    output.mkdir(parents=True, exist_ok=False)
    figures = []
    colors = ['#2166ac', '#c75b12', '#4d9221', '#762a83', '#008b8b']
    markers = ['o', 'D', '^', 's', 'v']
    for view in views:
        fig, axes = plt.subplots(1, 3, figsize=(max(14, len(view['conditions']) * 2.8), 6.6))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.36, top=0.79, wspace=0.28)
        for metric, ax in zip(view['metrics'], axes):
            percent = metric in ('cache_fraction', 'task_success_rate')
            factor = 100 if percent else 1
            values = []
            for si, seed in enumerate(view['seeds']):
                selected = [r for r in view['rows'] if r['seed'] == seed]
                x = [view['conditions'].index(r['condition']) + (si - (len(view['seeds']) - 1) / 2) * 0.13 for r in selected]
                valid = [(xx, r) for xx, r in zip(x, selected) if r[metric] is not None]
                y = [r[metric] * factor for _, r in valid]
                values.extend(y)
                ax.scatter([xx for xx, _ in valid], y, label=f'Seed {seed}', s=65, color=colors[si % len(colors)], marker=markers[si % len(markers)], zorder=4)
                for (xx, r), value in zip(valid, y):
                    label = f'{value:.3f}' if metric.endswith('_s') and metric != 'visible_latency_p95_s' else f'{value:.2f}'
                    if metric == 'task_success_rate':
                        label += f'\n{r['successful_tasks']}/{r['sessions']}'
                    if metric == 'visible_latency_p95_s':
                        label += f'\nn={r['visible_latency_samples']}'
                    peers = sorted([q for q in view['rows'] if q['condition'] == r['condition'] and q[metric] is not None], key=lambda q: (q[metric], q['seed']))
                    rank = next((j for j, q in enumerate(peers) if q['seed'] == r['seed']))
                    offset = 9 if rank == len(peers) - 1 else -25 * (len(peers) - 1 - rank)
                    ax.annotate(label, (xx, value), xytext=(0, offset), textcoords='offset points', ha='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.6))
            ax.set_xticks(range(len(view['conditions'])), [c.replace(' + ', '\n+ ') for c in view['conditions']])
            ax.set_ylabel(LABELS[metric])
            ax.grid(axis='y', alpha=0.2, zorder=0)
            ax.set_xlim(-0.6, len(view['conditions']) - 0.4)
            if percent:
                ax.set_ylim(-18, 118)
            elif values:
                ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)
            ax.tick_params(axis='x', labelrotation=0, labelsize=9)
        axes[0].legend(loc='upper left', fontsize=8, frameon=False)
        fig.suptitle('DeepSeek-V4-Flash-0731 · ' + view['title'], fontsize=14, fontweight='bold', y=0.96)
        counts = '; '.join((f'seed {s}: N={next((r['sessions'] for r in view['rows'] if r['seed'] == s))}' for s in view['seeds']))
        fig.text(0.5, 0.88, 'GPU 0–3 · PCIe · ' + counts + ' · each marker is one completed cohort, not a confidence interval', ha='center', fontsize=9)
        sources = '; '.join((f'{r['experiment']}/{r['phase_id']}' for r in view['rows']))
        source_experiments = ', '.join(sorted({r['experiment'] for r in view['rows']}))
        caveat = 'Authored local tasks, not production data or SWE-bench. Dynamic histories/call counts differ; assignments/budgets match.' if view['rows'][0]['kind'] == 'agent' else 'Within each seed, exact token inputs/output budgets/think waits match.'
        quality_note = 'Strict task success uses all assigned attempts; latency uses natural completions (n shown).' if view['rows'][0]['kind'] == 'agent' else 'Fixed replay has no task-success grade; latency uses protocol-complete replay sessions (n shown).'
        fig.text(0.07, 0.14, caveat + '\nThroughput includes drain/tools/grading; visible latency excludes final grading. No Session latency SLO is defined.\n' + quality_note + '\nWarmup, pilot, Profile and failed experiments excluded. Missing cache values are not plotted as zero.', fontsize=8.3, linespacing=1.4)
        fig.text(0.07, 0.035, 'Sources: ' + source_experiments + '\nExact phase IDs, workload hashes and sample counts are in manifest.json and comparison-rows.csv.', fontsize=8, linespacing=1.4)
        for ext in ('png', 'pdf'):
            path = output / (view['id'] + '.' + ext)
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            figures.append(dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size, comparison=view['id']))
        plt.close(fig)
    csvpath = output / 'comparison-rows.csv'
    with open_output(csvpath, 'x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    manifest = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), summary=str(summary.resolve()), summary_sha256=sha(summary), script_sha256=sha(Path(__file__)), matplotlib=matplotlib.__version__, figures=figures, table=dict(file=csvpath.name, sha256=sha(csvpath)), comparisons=views, excluded=excluded, limitations=['C1 and C8 are separate workloads with different seeds; no claimed causal concurrency speedup from unmatched cohorts.', 'Partial condition/seed grids are explicit in each comparison; no missing data is invented.', 'Finite closed concurrency comparisons are not open-arrival capacity or Session SLO certification.', 'Task success denominators include all assigned attempts; visible latency sample counts only include user-visible natural completions.'])
    with open_output(output / 'manifest.json', 'x') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return dict(status='PASS', comparisons=len(views), eligible_phases=len(rows), plotted_phases=len({(r['experiment'], r['phase_id']) for v in views for r in v['rows']}), excluded_phases=len(excluded), output_dir=str(output), figures=figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    need(args.inspect_only != bool(args.output_dir), 'Choose inspect-only or an output directory')
    before = sha(args.summary)
    report = load_document(args.summary)
    if args.inspect_only:
        rows, excluded = collect(report)
        views = groups(rows)
        result = dict(status='VALID_REAL_INPUT', normal_phases=len(rows), excluded_phases=len(excluded), comparisons=[dict(id=v['id'], phases=len(v['rows']), complete_paired_grid=v['complete_paired_grid'], missing_pairs=v['missing_pairs']) for v in views], rendering_permitted=bool(views), summary_sha256=before)
    else:
        result = render(report, args.summary, args.output_dir)
    need(sha(args.summary) == before, 'Summary changed while rendering')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__=='__main__':main()
