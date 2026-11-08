#!/usr/bin/env python3
"""Plot verified terminal D comparisons; never read live/partial GPU results."""

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
import traceback
from pathlib import Path

POLICIES=('observe-v1','fairshare-v1')
COLORS=('#4c78a8','#e69f00')
SCENARIOS=('dense64-normal','context_mix','kv_residency-c8long')
TITLES=('Two 64K + six short','Poisson context mix, 0.5 req/s','Eight 64K + sixteen short')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def collect(report):
    assert report['status']=='PASS' and not report['errors']
    stages=[s for s in report['stages'] if s['included_as_measurement']]
    assert len(stages)==12 and all(s['ranking_eligible'] for s in stages)
    assert len(report['paired_comparisons'])==6 and all(p['status']=='PASS' for p in report['paired_comparisons'])
    coverage=[]
    for pair in report['paired_comparisons']:
        assert len(pair['members'])==2
        for member in pair['members']:
            key=(member['experiment'],member['phase'])
            hits=[s for s in stages if (s['experiment'],s['phase_id'])==key]
            assert len(hits)==1 and hits[0]['pair_block']==pair['pair_block']
            coverage.append(key)
    assert len(coverage)==len(set(coverage))==12
    rows=[]
    for s in stages:
        scenario='context_mix' if s['mode']=='poisson' else s['phase_id']
        assert scenario in SCENARIOS and s['policy'] in POLICIES
        assert s['pair_block'] in ('trial670','trial671')
        block=s['pair_block'].removeprefix('trial')
        short=s['by_shape']['p512_o128'];long=s['by_shape']['p65536_o128']
        gpu=s['gpu'];energy=gpu.get('energy_j')
        if energy is not None:
            assert gpu['status']=='COMPLETE_SAMPLED_WINDOW' and len(gpu['devices'])==4
            assert {d['gpu'] for d in gpu['devices']}=={0,1,2,3}
            assert all(d['measurements']['power_w']['coverage_fraction']>=1-1e-7 for d in gpu['devices'])
            assert abs(gpu['window_s']-s['total_wall_s'])<1e-3
        rows.append(dict(experiment=s['experiment'],phase=s['phase_id'],scenario=scenario,block=block,policy=s['policy'],
            short_requests=short['requests'],long_requests=long['requests'],
            short_ttft_p95_s=short['metrics']['scheduled_ttft_s']['p95'],long_ttft_p95_s=long['metrics']['scheduled_ttft_s']['p95'],
            short_tpot_p95_ms=1000*short['metrics']['tpot_s']['p95'],short_e2e_p95_s=short['metrics']['scheduled_e2e_s']['p95'],
            short_slo_rate=short['slo_joint_pass_rate'],short_goodput=short['cohort_slo_goodput_requests_s'],
            output_tokens_s=s['throughput']['cohort_output_tokens_s'],wall_s=s['total_wall_s'],drain_s=s['drain_s'],
            energy_kj=None if energy is None else energy/1000,
            aggregate_four_board_mean_w=None if energy is None else energy/s['total_wall_s'],
            kv_peak_pct=100*s['raw_gauges']['aggregate']['vllm:kv_cache_usage_perc']['max'],
            gpu_status=s['gpu']['status'],stability=s['stability']['classification'],preemptions=s['preemptions']))
    assert len({(r['scenario'],r['block'],r['policy']) for r in rows})==12
    return rows

def figures(rows):
    import matplotlib.pyplot as plt
    import numpy as np
    lookup={(r['scenario'],r['block'],r['policy']):r for r in rows}
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    x=np.arange(2);width=.34
    for row,(scenario,title) in enumerate(zip(SCENARIOS,TITLES)):
        for col,(metric,label) in enumerate((('short_ttft_p95_s','Short TTFT P95 (s)'),('long_ttft_p95_s','64K TTFT P95 (s)'))):
            ax=axes[row,col]
            for index,policy in enumerate(POLICIES):
                values=[lookup[scenario,block,policy][metric] for block in ('670','671')]
                bars=ax.bar(x+(index-.5)*width,values,width,label=('Observe control','FairShare')[index],color=COLORS[index])
                ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
            ax.set_xticks(x,['Trial 670','Trial 671']);ax.set_ylabel(label);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.margins(y=.18)
            if col==0:ax.axhline(1,color='#555555',linestyle='--',linewidth=1,label='Short Research SLO')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Unprofiled matched trials: short admission versus long-request cost\nBurst uses the same workload seed in two independent restart blocks',fontsize=12)
    resource,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    metrics=(('short_tpot_p95_ms','Short TPOT P95 (ms)'),('short_goodput','Short Goodput (req/s)'),('output_tokens_s','Cohort output (token/s)'),('energy_kj','Four-board energy (kJ)'),('kv_peak_pct','KV peak (%)'),('drain_s','Drain after 450 s (s)'))
    for ax,(metric,label) in zip(axes.flat,metrics):
        for index,policy in enumerate(POLICIES):
            values=[lookup['context_mix',block,policy][metric] for block in ('670','671')]
            if any(v is None for v in values):raise ValueError('Missing measured resource coverage: '+metric)
            bars=ax.bar(x+(index-.5)*width,values,width,label=('Observe control','FairShare')[index],color=COLORS[index]);ax.bar_label(bars,fmt='%.3g',fontsize=8,padding=2)
        ax.set_xticks(x,['Seed 670','Seed 671']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.18)
        if metric=='short_tpot_p95_ms':ax.axhline(50,color='#555555',ls='--',lw=1)
    axes[0,0].legend(fontsize=8)
    resource.suptitle('Poisson context mix: performance and resource tradeoffs\nComplete cohort includes drain; power is sampled GPU boards, not wall-plug energy',fontsize=12)
    return {'matched-latency':fig,'poisson-resources':resource}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--log', type=Path)
    a = ap.parse_args()
    log = a.log or a.output_dir.with_suffix('.operation.json')
    raw_root = path_at(Path(__file__).resolve().parents[1], 'results')
    if any((p.resolve().is_relative_to(raw_root.resolve()) for p in (a.output_dir, log))):
        raise ValueError('Never write plots or logs into raw results')
    if log.exists():
        raise FileExistsError('Use unique operation log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with open_output(log, 'x') as f:
        json.dump(dict(status='IN PROGRESS'), f)
    operation = dict(status='FAILED', summary=str(a.summary), output_dir=str(a.output_dir), source_sha256=sha(Path(__file__)))
    try:
        raw = a.summary.read_bytes()
        report = json.loads(raw)
        rows = collect(report)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        import matplotlib
        matplotlib.use('Agg')
        outputs = {}
        for name, fig in figures(rows).items():
            for suffix in ('.png', '.pdf'):
                p = a.output_dir / (name + suffix)
                fig.savefig(p, dpi=160)
                outputs[str(p)] = sha(p)
        data = dict(status='PASS', summary=str(a.summary.resolve()), summary_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=sha(Path(__file__)), matplotlib=matplotlib.__version__, rows=rows, outputs=outputs, scope='12 terminal unprofiled phases only; P95 descriptive; no profile latency or maximum-capacity claim.')
        assert sha(a.summary) == data['summary_sha256']
        assert sha(Path(__file__)) == operation['source_sha256']
        with open_output(a.output_dir / 'figure-data.json', 'x') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        operation.update(status='PASS', rows=len(rows), outputs=outputs, summary_sha256=data['summary_sha256'])
    except Exception as exc:
        operation.update(error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        write_text(log, json.dumps(operation, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', rows=len(rows), outputs=len(outputs))))
if __name__=='__main__':main()
