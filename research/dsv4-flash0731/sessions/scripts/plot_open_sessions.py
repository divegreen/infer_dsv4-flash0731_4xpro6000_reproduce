#!/usr/bin/env python3
"""Plot actual Poisson Session windows only; no implicit capacity certification."""

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
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('closed_session_plot_helpers',Path(__file__).with_name('plot_session_comparisons.py'))
base=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(base)
need=base.need


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def collect(report):
    need(report.get('status')=='PASS' and not report.get('errors'),'Require actual evidence-analysis PASS')
    experiments={e['id']:e for e in report['experiments']};plans={};ledger={};rows=[];excluded=[];seen=set()
    for stage in report['stages']:
        phase=stage['phase'];experiment=stage['experiment'];key=(experiment,phase['id'])
        need(key not in seen,'Duplicate experiment/phase');seen.add(key)
        if phase['arrival']!='poisson':continue
        if not stage.get('included_as_measurement') or not stage.get('terminal_success') or phase['role'] not in ('screen','measure','confirm'):
            excluded.append(dict(experiment=experiment,phase_id=phase['id'],reason=stage.get('exclusion_reason')));continue
        e=experiments[experiment];need(e['terminal_success'],'Failed experiment cannot supply normal open points')
        if experiment not in plans:
            matches=[(p,v) for p,v in report['input_ledger'].items() if p.endswith('/'+experiment+'/session-plan.json')]
            need(len(matches)==1,'Missing audited launch-plan identity')
            original,expected=matches[0];path=Path(original)
            if not path.is_file():
                marker='/research/dsv4-flash0731/';need(marker in original,'Cannot relocate audited plan')
                path=ROOT.parent/original.split(marker,1)[1]
            raw=path.read_bytes();info=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            need(info==expected,'Audited launch plan changed')
            plan=json.loads(raw);fields=('model','timeout_s','session_timeout_s','max_active_sessions')
            need(all(k in plan for k in fields),'Global service bounds must be explicit; no invented defaults')
            plans[experiment]={k:plan[k] for k in fields};ledger[str(path.resolve())]=info
        stats=stage['summary'];arch,apc=base.deployment(e['config'])
        config={k:v for k,v in e['config'].items() if k!='name'}
        contract=dict(phase={k:v for k,v in phase.items() if k not in ('id','seed','rate')},
                      service_protocol=plans[experiment],client_sha256=e['client_sha256'])
        stability=stage['stability']
        rows.append(dict(experiment=experiment,phase_id=phase['id'],kind=phase['kind'],role=phase['role'],
            architecture=arch,apc=apc,condition=arch+' / '+apc,config_sha256=digest(config),
            comparison_contract=contract,comparison_sha256=digest(contract),seed=phase['seed'],rate=phase['rate'],
            duration_s=phase['duration_s'],sessions=stats['sessions'],protocol_completed=stats['protocol_completed'],
            successful_tasks=stats['successful_tasks'],protocol_sessions_s=stats['completed_sessions_s'],
            successful_tasks_s=stats['successful_tasks_s'],output_tokens_s=stats['output_tokens_s'],
            task_success_rate=stats['task_success_rate'],protocol_completion_rate=stats['protocol_completion_rate'],
            visible_p95_s=stats['user_visible_latency_s']['p95'],visible_latency_n=stats['user_visible_latency_s']['n'],
            total_wall_s=stage['total_wall_s'],drain_s=stage['drain_s'],cache_fraction=stats['cache_fraction'],
            classification=stability['classification'],trend_verdict=stability.get('trend_verdict'),
            completion_deficit=stability.get('completion_deficit'),outstanding_increase=stability.get('session_outstanding_increase'),
            stability_reasons=stability.get('reasons'),workload_sha256=stage['arrival_workload_sha256']))
    for path,info in ledger.items():need(base.sha(Path(path))==info['sha256'],'Audited plan changed during collection')
    return rows,excluded,ledger


def groups(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['comparison_sha256']].append(dict(row))
    views=[]
    for key,selected in sorted(grouped.items()):
        configs=defaultdict(set)
        for row in selected:configs[row['condition']].add(row['config_sha256'])
        for row in selected:
            if len(configs[row['condition']])>1:row['condition']+=' ['+row['config_sha256'][:7]+']'
        identities=defaultdict(set);counts=defaultdict(set)
        pairs=set()
        for row in selected:
            cell=(row['rate'],row['seed']);identities[cell].add(row['workload_sha256']);counts[cell].add(row['sessions'])
            pair=(row['condition'],*cell);need(pair not in pairs,'Duplicate deployment/rate/seed window');pairs.add(pair)
        need(all(len(v)==1 for v in identities.values()),'Within-rate/seed exact workload identities differ')
        need(all(len(v)==1 for v in counts.values()),'Within-rate/seed assigned session counts differ')
        conditions=sorted({r['condition'] for r in selected});missing=[dict(condition=c,rate=rate,seed=seed) for rate,seed in sorted(identities) for c in conditions if (c,rate,seed) not in pairs]
        views.append(dict(id='open-'+selected[0]['kind']+'-'+selected[0]['role']+'-'+key[:10],kind=selected[0]['kind'],
            role=selected[0]['role'],duration_s=selected[0]['duration_s'],contract=selected[0]['comparison_contract'],
            rows=selected,conditions=conditions,complete_matched_grid=not missing,missing_pairs=missing))
    return views


def build_figure(view):
    import matplotlib.pyplot as plt
    agent=view['kind']=='agent'
    metrics=[('protocol_sessions_s','Protocol sessions / cohort second'),
             ('successful_tasks_s','Successful tasks / cohort second') if agent else ('output_tokens_s','Output tokens / cohort second'),
             ('visible_p95_s','User-visible Session P95 (s)'),
             ('task_success_rate','Strict task success (%)') if agent else ('protocol_completion_rate','Protocol completion (%)'),
             ('completion_deficit','Analysis-window arrivals minus completions'),
             ('outstanding_increase','Last-bin minus first-bin outstanding sessions')]
    fig,axes=plt.subplots(2,3,figsize=(16,10));fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.84,hspace=.32,wspace=.30)
    colors=plt.get_cmap('tab10');markers={'SCREEN_ONLY':'s','STABLE_IN_WINDOW':'o','GROWING_BACKLOG':'^','INCONCLUSIVE':'x'}
    for ax,(metric,label) in zip(axes.flat,metrics):
        percent=metric.endswith('_rate')
        for i,condition in enumerate(view['conditions']):
            data=[r for r in view['rows'] if r['condition']==condition and r.get(metric) is not None]
            for j,row in enumerate(data):
                y=row[metric]*(100 if percent else 1)
                hollow=len(view['conditions'])>1 and i%2==0
                marker=markers.get(row['classification'],'D')
                # Unfilled x markers take their visible stroke from color, not edgecolors.
                style={'color':colors(i)} if marker=='x' else dict(facecolors='none' if hollow else colors(i),edgecolors=colors(i))
                ax.scatter(row['rate'],y,**style,
                    marker=marker,s=100 if hollow else 44,linewidths=1.6,
                    label=condition if j==0 else None,zorder=3)
                if metric in ('visible_p95_s','task_success_rate'):
                    text=('n='+str(row['visible_latency_n'])) if metric=='visible_p95_s' else str(row['successful_tasks'])+'/'+str(row['sessions'])
                    ax.annotate(text,(row['rate'],y),xytext=(-7 if i%2==0 else 7,6+(j%2)*9),ha='right' if i%2==0 else 'left',textcoords='offset points',fontsize=7)
        ax.set_xlabel('Nominal Poisson Session arrival rate (sessions/s)');ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.margins(x=.12,y=.18)
        if percent:ax.set_ylim(-5,112)
        elif metric in ('completion_deficit','outstanding_increase'):ax.axhline(0,color='gray',lw=.8)
        else:ax.set_ylim(bottom=0)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('DeepSeek-V4-Flash-0731 · Poisson '+view['kind']+' · '+view['role']+' · T='+str(view['duration_s'])+'s',fontsize=16,y=.98)
    fig.text(.07,.045,'Each point is an actual seed/window; no means or confidence intervals. Cohort throughput includes drain/tools/grading.\n'
        'Markers: square=screen only, circle=stable in one window, triangle=growing backlog, x=inconclusive.\n'
        'One-window stability is not repeated capacity certification. Session latency SLO is undefined; independent confirmation audit is separate.\n'
        +('Authored local Agent tasks; histories can differ. ' if agent else 'Fixed replay has no task-success grade. ')+
        'Exact phase IDs, counts, missing pairs and workload hashes are in the manifest.',fontsize=9,linespacing=1.35)
    return fig


def render(report, summary, output):
    rows, excluded, ledger = collect(report)
    views = groups(rows)
    need(views, 'No real completed open-arrival windows; no placeholder performance charts')
    need(not output.exists(), 'Use a unique output directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for view in views:
        fig = build_figure(view)
        for ext in ('png', 'pdf'):
            path = output / (view['id'] + '.' + ext)
            fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
            files.append(dict(file=path.name, sha256=base.sha(path), bytes=path.stat().st_size))
        plt.close(fig)
    manifest = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), summary=str(summary.resolve()), summary_sha256=base.sha(summary), script_sha256=base.sha(Path(__file__)), helper_sha256=base.sha(Path(base.__file__)), matplotlib=matplotlib.__version__, audited_plan_inputs=ledger, figures=files, groups=views, excluded_open_phases=excluded, limitations=['No interpolation or averaging across windows; absent condition/rate/seed pairs are explicit.', 'Screen and confirm/duration/workload/cache/routing/global timeout-admission bounds/client source are separate groups.', 'Normal performance figures exclude failed owned runs; their exclusion is explicit and independent confirmation audit must still include every started confirm.', 'Single-window classifications never certify a rate, maximum capacity or Session SLO.'])
    with open_output(output / 'manifest.json', 'x') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return dict(status='PASS', windows=len(rows), comparisons=len(views), excluded_open_phases=len(excluded), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    need(args.inspect_only != bool(args.output_dir), 'Choose inspect-only or unique output directory')
    before = base.sha(args.summary)
    report = load_document(args.summary)
    if args.inspect_only:
        rows, excluded, ledger = collect(report)
        views = groups(rows)
        result = dict(status='REAL_INPUT_INSPECTED', open_windows=len(rows), groups=len(views), excluded=len(excluded), rendering_permitted=bool(views), audited_plan_inputs=ledger)
    else:
        result = render(report, args.summary, args.output_dir)
    need(base.sha(args.summary) == before, 'Summary changed during plotting')
    print(json.dumps(result, indent=2))


if __name__=='__main__':main()
