#!/usr/bin/env python3
"""D-only terminal unprofiled comparisons using frozen A/B measurement rules.

CPU/read-only: independent raw/source audit precedes B per-phase recomputation.
Failed terminal runs are retained descriptively, never ranked as successes.
No global capacity certificate, automatic baseline replacement, GPU or HTTP.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile
from zoneinfo import ZoneInfo
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

b=module('d_frozen_b_performance',ROOT.parent/'long-context/scripts/analyze_long_context.py')
aud=module('d_evidence_auditor',ROOT/'scripts/audit_study.py')
need=b.capacity.require
POLICIES={'research_scheduler.ObserveResearchScheduler':'observe-v1',
          'research_scheduler.FairShareResearchScheduler':'fairshare-v1'}


def conditions(config,identity):
    """Exclude only experiment label and selected policy from paired conditions."""
    config=deepcopy(config);config.pop('name',None);args=config['args']
    need(args.count('--scheduler-cls')==1,'Exactly one explicit scheduler class required')
    index=args.index('--scheduler-cls');cls=args[index+1]
    need(cls in POLICIES,'Unsupported policy for matched D comparison')
    del args[index:index+2]
    need(not any(x.startswith('--profiler-config') or x=='--enable-logging-iteration-details' for x in args),
         'Profile/iteration tracing cannot enter normal performance')
    sources={key:value['sha256'] for key,value in identity['files'].items()
             if '/scripts/' in key or key.startswith('scripts/') or key.startswith('installed/')
             or key.startswith(('metadata/', 'project/'))
             or key in ('environment.json','model-inventory.json','project_layout.py','experiments/catalog.json')}
    need(sources,'Missing execution sources for matched conditions')
    return POLICIES[cls],dict(config_sha256=b.hashed(config),runtime_sources_sha256=b.hashed(sources),
                             basis='All config except name/scheduler class; executed script/installed/environment source hashes')



def registered_block(directory, report, identity, ledger, root):
    """Bind restart pairs to the preregistered exact launch config/plan hashes."""
    registry_key='metadata/scheduler-pair-blocks.json'
    portable=any(key.startswith('experiments/') for key in identity['files'])
    if portable or registry_key in identity['files']:
        need(registry_key in identity['files'],'Portable run lacks a captured scheduler-pair registration')
        path=directory/'source-snapshot.zip'
        raw=b.raw_bytes(path,ledger)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            registered=archive.read(registry_key)
        entry=identity['files'][registry_key]
        need(hashlib.sha256(registered).hexdigest()==entry['sha256'] and len(registered)==entry['bytes'],
             'Captured pair registration differs from runtime identity')
        decision=json.loads(registered)
        decision_source=str(path.resolve())+'!'+registry_key
    else:
        # Original evidence and self-contained legacy CPU fixtures remain readable.
        path=root/'decisions/normal-pair-blocks-r1.json'
        decision=b.read_json(path,ledger)
        decision_source=str(path.resolve())
    groups=decision['groups'];members=[x for group in groups for x in group['experiments']]
    need(len(members)==len(set(members)),'Duplicate experiment in registered pair blocks')
    blocks=[group['block'] for group in groups]
    need(len(blocks)==len(set(blocks)),'Duplicate registered block ID')
    selected=[group for group in groups if directory.name in group['experiments']]
    need(len(selected)==1,'Experiment missing from registered normal pair blocks')
    need(len(selected[0]['experiments'])==2,'Each registered block requires exactly two experiments')
    matches={}
    for key,expected in decision['input_sha256'].items():
        if Path(key).name!=directory.name+'.json':continue
        category_name = 'configs' if '/configs/' in key else 'workloads' if '/workloads/' in key else None
        captured = aud.base.snapshot_source_key(key,root,category_name) if category_name else None
        need(captured is not None,'Registered input outside scheduler source namespace')
        need(captured in identity['files'] and identity['files'][captured]['sha256']==expected,
             'Registered launch input SHA differs from actual snapshot: '+captured)
        category='config' if '/configs/' in captured else 'plan' if '/workloads/' in captured else None
        need(category and category not in matches,'Ambiguous registered launch input')
        matches[category]=expected
    need(set(matches)=={'config','plan'} and report['plan_sha256']==matches['plan'],
         'Registered config/plan identity incomplete')
    return dict(block=selected[0]['block'],decision_path=decision_source,registered_experiments=selected[0]['experiments'],input_sha256=matches)


def scalars(stage):
    result={
        'wall_s':stage['total_wall_s'],'drain_s':stage['drain_s'],
        'cohort_requests_s':stage['throughput']['cohort_requests_s'],
        'cohort_output_tokens_s':stage['throughput']['cohort_output_tokens_s'],
        'cohort_slo_goodput_requests_s':stage['throughput']['cohort_slo_goodput_requests_s'],
        'gpu_energy_j':stage['gpu'].get('energy_j'),
        'gpu_output_tokens_per_j':stage['gpu'].get('successful_output_tokens_per_j'),
        'preemptions':stage['preemptions'],
        'client_peak_outstanding':stage['client_peak_dispatched_outstanding']}
    for name,value in stage['raw_gauges']['aggregate'].items():
        for stat in ('sample_mean','max'):result[name+'.'+stat]=value[stat]
    for shape,stats in stage['by_shape'].items():
        for metric in ('scheduled_ttft_s','tpot_s','scheduled_e2e_s','dispatch_lag_s'):
            for quantile in ('p50','p95','p99'):result[shape+'.'+metric+'.'+quantile]=stats['metrics'][metric][quantile]
        for key in ('requests','successful','failed','timeouts','censored','slo_passes',
                    'slo_joint_pass_rate','cohort_slo_goodput_requests_s','cohort_output_tokens_s'):
            result[shape+'.'+key]=stats[key]
    return result


def compare(stages):
    groups=defaultdict(list)
    for stage in stages:
        if stage['included_as_measurement']:
            groups[(stage.get('pair_block','UNREGISTERED'),stage['comparison_id'],stage['seed'])].append(stage)
    comparisons=[]
    for (block,name,seed),members in sorted(groups.items()):
        item=dict(pair_block=block,comparison_id=name,seed=seed,members=[dict(experiment=s['experiment'],phase=s['phase_id'],
            policy=s['policy'],ranking_eligible=s['ranking_eligible'],arrival_sha256=s['arrival_identity']['sha256']) for s in members])
        if len(members)==1:
            item.update(status='SINGLE_OBSERVATION',deltas={})
        elif len(members)!=2 or {s['policy'] for s in members}!=set(POLICIES.values()):
            item.update(status='NOT VERIFIED',reason='Require one observation of each policy; duplicate attempts retained',deltas={})
        elif len({s['arrival_identity']['sha256'] for s in members})!=1:
            item.update(status='MISMATCH',reason='Prompt/token/arrival identity differs',deltas={})
        elif len({b.hashed(s['matched_conditions']) for s in members})!=1:
            item.update(status='MISMATCH',reason='Non-treatment configuration/runtime sources differ',deltas={})
        elif not all(s['ranking_eligible'] for s in members):
            item.update(status='DESCRIPTIVE_ONLY',reason='Failed or incomplete owned execution cannot enter performance ranking',deltas={})
        else:
            reference=next(s for s in members if s['policy']=='observe-v1')
            candidate=next(s for s in members if s['policy']=='fairshare-v1')
            left,right=scalars(reference),scalars(candidate)
            need(set(left)==set(right),'Matched requests have different scalar/shape inventories')
            delta={key:dict(observe=left[key],fairshare=right[key],
                difference=right[key]-left[key] if left[key] is not None and right[key] is not None else None,
                percent_change=100*(right[key]/left[key]-1) if left[key] not in (None,0) and right[key] is not None else None)
                for key in left}
            item.update(status='PASS',reference=reference['experiment'],candidate=candidate['experiment'],
                deltas=delta,interpretation='Within-seed paired observations; no IID-request significance claim or winner selection')
        comparisons.append(item)
    return comparisons


def repeat_observations(stages):
    """Describe every supplied long window; never claim all-started certification."""
    groups=defaultdict(list)
    for s in stages:
        if s['included_as_measurement'] and s['mode']=='poisson' and s['measurement_window_s']>=b.capacity.RULES['min_confirmation_duration_s']:
            groups[(s['policy'],s['workload_key'],s['rate'],b.hashed(s['matched_conditions']))].append(s)
    return [dict(policy=policy,workload_key=workload,rate=rate,matched_conditions_sha256=conditions_sha,distinct_seeds=len({s['seed'] for s in members}),
        experiments=[s['experiment']+'/'+s['phase_id'] for s in members],
        classifications=[s['stability']['classification'] for s in members],
        all_supplied_long_windows_stable=all(s['ranking_eligible'] and s['stability']['classification']=='STABLE_IN_WINDOW' for s in members),
        repeated_supplied_evidence=len({s['seed'] for s in members})>=b.capacity.RULES['min_distinct_confirmation_seeds'],
        short_p95_slo_all_supplied=all(s['short512']['overall_ratio_p95_slo_pass'] is True for s in members),
        capacity_certified=False,reason='Supplied-run descriptive aggregation; no global all-started archive closure or rate scan certificate')
        for (policy,workload,rate,conditions_sha),members in sorted(groups.items())]



def quality_details(directory, audit, ledger):
    """Present already independently audited natural-EOS sidecar separately."""
    summaries=[entry.get('quality_probe',{'status':'NOT ASSESSED'}) for entry in audit['runtime_reports']]
    need(len(summaries)==1,'Expected one quality scope per owned experiment')
    result=dict(summaries[0])
    path=directory/'quality.json'
    if path.exists() and result['status'] not in ('NOT_REQUESTED','NOT_STARTED','NO_REPORT'):
        raw=b.read_json(path,ledger);rows=[];by_case=defaultdict(lambda:dict(requests=0,protocol_successes=0,quality_passed=0))
        for run in raw.get('runs',[]):
            row=dict(round=run['round'],case_id=run['case_id'],protocol_ok=run['http']['protocol_ok'],
                quality=run['quality'],e2e_s=run['http']['e2e_s'],finish_reason=run['http'].get('finish_reason'))
            rows.append(row);group=by_case[run['case_id']];group['requests']+=1
            group['protocol_successes']+=int(row['protocol_ok']);group['quality_passed']+=int(row['quality']['passed'])
        result.update(source=str(path),by_case=dict(by_case),requests_detail=rows)
    result.update(excluded_from_performance=True,quality_scores_gate_performance=False,
        interpretation='Independent evidence audit and answer scoring; distinct from random-prompt throughput and not a broad model-quality/Agent score')
    return result


def analyze(directories, root=ROOT):
    ledger = {}
    stages = []
    errors = []
    runs = []
    excluded = []
    sources = (Path(__file__), Path(b.__file__), Path(b.capacity.__file__), Path(b.historical.__file__), Path(aud.__file__), Path(aud.base.__file__), Path(aud.quality_audit.__file__))
    for p in sources:
        b.raw_bytes(p, ledger)
    for directory in directories:
        directory = directory.resolve()
        path = directory / 'capacity.json'
        try:
            need(directory.parent == path_at(root.resolve(), 'results'), 'Input must be within D results; historical inputs remain excluded')
            report = b.read_json(path, ledger)
            config = b.read_json(directory / 'config.json', ledger)
            status = b.read_json(directory / 'status.json', ledger)
            experiment = b.read_json(directory / 'experiment.json', ledger)
            need(status.get('end_utc') and report.get('status') in ('PASSED', 'FAILED', 'CENSORED'), 'Only terminal owned runs may be analyzed')
            need(experiment.get('profile') is False and report.get('profiling', {}).get('excluded_from_performance') is False, 'Profile reports are excluded from unprofiled comparison')
            need(not any((p['role'] == 'profile' for p in report['plan']['phases'])), 'Profile phase in normal plan')
            need(not list(directory.glob('scheduler-steps-*.jsonl')), 'Per-step tracing present in normal performance')
            verifier = aud.Auditor(root)
            verifier.report(path)
            audit = verifier.finish()
            need(audit['status'] == 'PASS', 'Independent raw/source audit failed: ' + json.dumps(audit['errors'] + audit['incomplete']))
            for name, digest in audit['source_sha256'].items():
                raw = b.raw_bytes(Path(name), ledger)
                need(hashlib.sha256(raw).hexdigest() == digest, 'Audit/analysis evidence changed: ' + name)
            identity = b.read_json(directory / 'runtime-identity.json', ledger)
            policy, matching = conditions(config, identity)
            registration = registered_block(directory, report, identity, ledger, root)
            context = b.capacity.completion_context(report, status, experiment)
            success = context['eligible_for_capacity_qualification'] and all((x['supervisor_success'] for x in audit['runtime_reports']))
            devices, warnings = b.historical.load_gpu(directory / 'gpu.csv', ZoneInfo('UTC'), ledger)
            runs.append(dict(experiment=directory.name, policy=policy, status=report['status'], successful_owned_execution=success, context=context, audit=audit, matched_conditions=matching, pair_registration=registration, quality=quality_details(directory, audit, ledger)))
            for run in report['runs']:
                s = b.analyze_run(report, run, path, policy, devices, ledger)
                s.update(policy=policy, pair_block=registration['block'], matched_conditions=matching, gpu_warnings=warnings, ranking_eligible=success and s['included_as_measurement'], owned_run_terminal_success=success, eligible_for_capacity_qualification=False)
                stages.append(s)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
            errors.append(dict(source=str(path), error=type(exc).__name__ + ': ' + str(exc)))
            excluded.append(dict(source=str(path), reason='Not eligible for normal verified ranking; failed validation preserved'))
    paired = compare(stages)
    if any((p['status'] == 'MISMATCH' for p in paired)):
        errors.append(dict(error='Declared matched comparison conditions differ'))
    for path, entry in ledger.items():
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
            errors.append(dict(source=path, error='Evidence changed during analysis'))
    return dict(schema_version=1, status='PASS' if stages and (not errors) else 'NOT VERIFIED', created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, python_executable=sys.executable, scope='Stage D unprofiled scheduler comparison; frozen per-window finite-capacity-v1, no global capacity certificate', rules=b.capacity.RULES, stages=stages, runs=runs, paired_comparisons=paired, repeated_window_observations=repeat_observations(stages), errors=errors, excluded=excluded, evidence_ledger=ledger, limitations=['Profile and step-logged runs never enter performance comparisons.', 'observe-v1 delegates to the installed parent through an external research wrapper; wrapper bookkeeping remains in normal timing, so this is not a new unwrapped-native baseline.', 'Goodput includes only requests with a defined Research SLO; 32K/64K joint TTFT/E2E SLO remains undefined.', 'TPOT is mean post-first-text latency, not per-token ITL; successful-request percentiles retain failure counts separately.', 'Throughput and Goodput denominators include the complete arrival horizon plus drain.', 'GPU power/energy describe sampled whole boards; utilization is device busy, not SM occupancy or PCIe saturation.', 'Two seeds are shown separately; request-level IID bootstrap/significance claims are not justified.', 'Stable-in-window classification is separate from SLO and does not establish maximum throughput or a production SLA.', 'Failed terminal experiments remain descriptive; they never silently disappear from a declared comparison.', 'Functional/quality probes require a separate source-bound task analysis; capacity prompts alone do not establish model quality.'])


def markdown(report):
    lines=['# 阶段 D 正常性能配对复算','',f"状态：{report['status']}。逐窗规则 `{report['rules']['version']}`；不做全局容量认证。",'',
        '| 实验 / phase | policy | mode | 正常排名可用 | window / drain s | 完成 / 全部 | tok/s | short SLO数 | short Goodput req/s | 稳定性 |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for s in report['stages']:
        if not s['included_as_measurement']:continue
        g=s['short512'];lines.append(f"| {s['experiment']} / {s['phase_id']} | {s['policy']} | {s['mode']} | {s['ranking_eligible']} | {s['measurement_window_s']:.3f} / {s['drain_s']:.3f} | {s['stats']['successful']}/{s['offered_requests']} | {s['throughput']['cohort_output_tokens_s']:.3f} | {g['slo_passes']}/{g['slo_eligible_requests']} | {g['cohort_slo_goodput_requests_s']:.4f} | {s['stability']['classification']} |")
    lines+=['','## 每种请求形状','', 'TTFT/E2E以计划到达为起点，单位秒；TPOT单位ms；各格P50 / P95 / P99。32K/64K联合SLO未定义。','',
        '| 实验 / phase / shape | 完成 / 全部 | TTFT | TPOT | E2E | 失败 / 超时 / censored |',
        '| --- | ---: | --- | --- | --- | --- |']
    for s in report['stages']:
        if not s['included_as_measurement']:continue
        for name,g in s['by_shape'].items():lines.append(f"| {s['experiment']} / {s['phase_id']} / {name} | {g['successful']}/{g['requests']} | {b.quantiles(g,'scheduled_ttft_s')} | {b.quantiles(g,'tpot_s',1000)} | {b.quantiles(g,'scheduled_e2e_s')} | {g['failed']} / {g['timeouts']} / {g['censored']} |")
    lines+=['','## 配对状态','']
    for p in report['paired_comparisons']:lines.append(f"- {p['pair_block']} / {p['comparison_id']} seed {p['seed']}: {p['status']}。"+p.get('reason','逐项绝对差和百分比见JSON。'))
    lines+=['','## 资源','', '| 实验 / phase | GPU能量J | tok/J | KV peak % | waiting peak | 抢占 | GPU积分状态 |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for s in report['stages']:
        if not s['included_as_measurement']:continue
        g=s['raw_gauges']['aggregate'];kv=g[b.GAUGES[0]]['max']
        lines.append(f"| {s['experiment']} / {s['phase_id']} | {b.number(s['gpu'].get('energy_j'))} | {b.number(s['gpu'].get('successful_output_tokens_per_j'))} | {b.number(None if kv is None else kv*100)} | {b.number(g[b.GAUGES[1]]['max'])} | {b.number(s['preemptions'])} | {s['gpu']['status']} |")
    lines+=['','## 独立质量探针','', '| 实验 | 状态 | 协议完成 / 请求 | 答案通过 |', '| --- | --- | ---: | ---: |']
    for run in report['runs']:
        q=run['quality'];lines.append(f"| {run['experiment']} | {q['status']} | {q.get('protocol_successes','—')} / {q.get('requests','—')} | {q.get('quality_passed','—')} |")
    lines+=['','质量探针为8类合成问题重复2轮，自然EOS/严格格式评分；答错不撤销独立完成的性能测量，不代表广泛模型/Agent能力。协议失败与owned执行状态另行保留。','', '限制：']+['- '+v for v in report['limitations']]
    if report['errors']:lines+=['','校验失败：']+['- '+json.dumps(v,ensure_ascii=False) for v in report['errors']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(s) for s in ('.json', '.md')]
    need(not any((p.exists() for p in outputs)), 'Use unique output revision')
    need(all((path_at(ROOT, 'results').resolve() not in p.resolve().parents for p in outputs)), 'Cannot write summaries into raw results')
    report = analyze(args.result_dirs)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, body in zip(outputs, (json.dumps(report, indent=2, ensure_ascii=False) + '\n', markdown(report))):
        with open_output(path, 'x') as f:
            f.write(body)
    print(json.dumps(dict(status=report['status'], stages=len(report['stages']), errors=report['errors'], output_prefix=str(args.output_prefix))))
    return 0 if report['status'] == 'PASS' else 1
if __name__=='__main__':raise SystemExit(main())
