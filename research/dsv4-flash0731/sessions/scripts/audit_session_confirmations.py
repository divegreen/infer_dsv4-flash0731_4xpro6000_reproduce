#!/usr/bin/env python3
"""Discover every started Stage C confirmation; aggregate without seed selection.

CPU/read-only. Uses the independent raw auditor, never production analysis or
serving code. Scans all c* experiment directories, including failed/running
ones, and explicitly separates incompatible runtime/config/workload cohorts.
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
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('independent_session_audit',Path(__file__).with_name('audit_agent_sessions.py'))
independent=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(independent)


def need(value,message):
    if not value:raise ValueError(message)


def signature(config,phase,command,report,task_catalog=None):
    # Experiment/phase IDs and seeds are identities, not workload treatments.
    cfg={key:value for key,value in config.items() if key!='name'}
    workload={key:value for key,value in phase.items() if key not in ('id','seed','role')}
    sources=command.get('source_sha256',{})
    if not isinstance(sources,dict):sources={}
    required=['scenarios/scripts/agent_session_client.py','scenarios/scripts/capacity_client.py',
              'scripts/benchmark_client.py','scenarios/scripts/serve_agent_sessions.py',
              'scenarios/scripts/run_logged.py','environment.json','model-inventory.json']
    portable='path_context' in command
    if portable:
        required=[key for key in required if key not in ('environment.json','model-inventory.json')]
        required.extend(('metadata/environment.json','metadata/model-inventory.json','project_layout.py',
                         'experiments/catalog.json','project/tools/reproduce.py'))
    missing=[]
    if phase['kind']=='agent':
        required.append('scenarios/scripts/agent_session_tasks.py')
        fixture=('experiments/' if portable else '')+'scenarios/workloads/sessions/tasks/'
        required.append(fixture+'catalog.json')
        required.extend(key for key in sources if key.startswith(fixture))
        # The required set must not shrink when a source-map entry disappears.
        # Derive selected-task inputs from the hash-checked launch catalog,
        # never from the current workspace's possibly newer task definitions.
        if task_catalog is None:missing.append('launch_task_catalog')
        else:
            tasks={task['id']:task for task in task_catalog.get('tasks',[])}
            for task_id in phase.get('tasks',[]):
                task=tasks.get(task_id)
                if task is None:missing.append('launch_task_catalog.task.'+task_id);continue
                required.extend(fixture+task_id+'/public/'+name for name in task['readable'])
                private=['grading_cases.json','reference.json'] if task['kind']=='coding' else ['answers.json']
                required.extend(fixture+task_id+'/private/'+name for name in private)
    valid_hash=lambda value:isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None
    missing.extend(key for key in required if not valid_hash(sources.get(key)))
    if not config:missing.append('config')
    args=config.get('args')
    if not isinstance(args,list) or not args or not all(isinstance(a,str) and a for a in args):
        missing.append('config.args')
    env=config.get('env')
    if not isinstance(env,dict) or env.get('VLLM_SERVER_DEV_MODE')!='1':missing.append('config.env')
    if 'port' in config and (type(config['port'])is not int or not 1<=config['port']<=65535):missing.append('config.port')
    if not isinstance(command.get('environment'),dict) or not command['environment']:
        missing.append('command.environment')
    else:
        if not all(isinstance(k,str) and isinstance(v,str) for k,v in command['environment'].items()):
            missing.append('command.environment')
        if not isinstance(command['environment'].get('CUDA_VISIBLE_DEVICES'),str) or not command['environment']['CUDA_VISIBLE_DEVICES']:
            missing.append('command.environment.CUDA_VISIBLE_DEVICES')
    if not valid_hash(report.get('tokenizer_sha256')):missing.append('report.tokenizer_sha256')
    if not isinstance(report.get('transport'),str) or not report['transport']:missing.append('report.transport')
    runtime={key:sources.get(key) for key in sorted(required)}
    # These bounds apply across phases and change censoring/admission/timeout
    # semantics even when the individual workload phase is otherwise identical.
    # Use the executed normalized plan; do not invent defaults for missing data.
    plan=report.get('plan') if isinstance(report.get('plan'),dict) else {}
    protocol_fields=('timeout_s','session_timeout_s','max_active_sessions','model')
    service_protocol={key:plan.get(key) for key in protocol_fields}
    for key in ('timeout_s','session_timeout_s'):
        value=plan.get(key)
        if type(value)not in (int,float) or not math.isfinite(value) or value<=0:missing.append('report.plan.'+key)
    if type(plan.get('max_active_sessions'))is not int or plan['max_active_sessions']<=0:
        missing.append('report.plan.max_active_sessions')
    if not isinstance(plan.get('model'),str) or not plan['model']:missing.append('report.plan.model')
    data=dict(config=cfg,workload=workload,runtime_sha256=runtime,environment=command.get('environment'),
              service_protocol=service_protocol,tokenizer_sha256=report.get('tokenizer_sha256'),transport=report.get('transport'))
    return independent.sha(independent.canonical(data)),data,sorted(set(missing))


def aggregate(windows):
    seeds={w['seed'] for w in windows}
    blockers=[]
    for window in windows:
        identity=window['experiment']+'/'+window['phase_id']
        if not window['run_terminal_success']:blockers.append(identity+': entire run not successful terminal')
        if not window['raw_integrity_pass']:blockers.append(identity+': independent raw integrity incomplete/failed')
        if window['classification']!='STABLE_IN_WINDOW':blockers.append(identity+': '+window['classification'])
        if window.get('missing_identity_fields'):blockers.append(identity+': missing runtime or service protocol identity')
    if len(seeds)<2:blockers.append('Fewer than two distinct confirmation seeds')
    return dict(started_windows=len(windows),distinct_seeds=len(seeds),seeds=sorted(seeds),
                finite_protocol_stable_confirmed=bool(windows and not blockers),blockers=blockers,
                session_latency_slo=None,session_slo_capacity_confirmed=None,
                task_success_note='Natural terminal task failure may count protocol completion; successful task throughput remains separately quality-qualified.')


def scan(results,ledger):
    grouped=defaultdict(list);details={};run_results=[];planned_unstarted=[]
    for directory in sorted(results.iterdir()):
        if not directory.is_dir() or not (directory/'session-plan.json').is_file():continue
        plan=independent.load(directory/'session-plan.json',ledger)
        confirmations=[p for p in plan.get('phases',[]) if p.get('role')=='confirm']
        if not confirmations:continue
        def optional(name):
            path=directory/name
            return independent.load(path,ledger) if path.is_file() else {}
        report=optional('sessions.json');command=optional('command.json');config=optional('config.json')
        status=optional('status.json');experiment=optional('experiment.json')
        terminal=(status.get('end_utc') is not None and type(status.get('exit_code'))is int)
        success=(terminal and status['exit_code']==0 and not status.get('timed_out') and not status.get('supervisor_error')
                 and report.get('status')=='COMPLETED' and not report.get('active_phase') and experiment.get('status')=='PASS')
        complete_runs={p['phase']['id']:p for p in report.get('phases',[])}
        audit_result=None;audit_error=None
        if terminal:
            try:audit_result=independent.audit(directory,ledger)
            except Exception as exc:audit_error=repr(exc)
        task_catalog=None;catalog_error=None
        if any(p.get('kind')=='agent' for p in confirmations):
            try:
                member=('experiments/' if 'path_context' in command else '')+'scenarios/workloads/sessions/tasks/catalog.json'
                with zipfile.ZipFile(io.BytesIO(independent.read(directory/'source-snapshot.zip',ledger))) as archive:
                    catalog_raw=archive.read(member)
                need(independent.sha(catalog_raw)==command.get('source_sha256',{}).get(member),'Launch task catalog SHA mismatch')
                task_catalog=json.loads(catalog_raw)
            except Exception as exc:catalog_error=repr(exc)
        run_results.append(dict(experiment=directory.name,terminal=terminal,terminal_success=bool(success),
                                raw_integrity_pass=audit_result is not None,raw_integrity_error=audit_error,
                                task_catalog_error=catalog_error))
        for phase in confirmations:
            phase_id=phase['id'];raw=directory/('sessions-'+phase_id)
            started=phase_id in complete_runs or report.get('active_phase')==phase_id or raw.is_dir()
            if not started:
                planned_unstarted.append(dict(experiment=directory.name,phase_id=phase_id,seed=phase.get('seed')));continue
            need(phase.get('arrival')=='poisson','Confirm role requires Poisson session arrival')
            key,data,missing=signature(config,phase,command,report,task_catalog);details[key]=data
            # A failed terminal raw audit cannot prove that its recorded
            # treatment identity is genuine, even when strings look valid.
            if terminal and audit_result is None:missing.append('terminal_raw_identity_not_verified')
            observed=complete_runs.get(phase_id)
            if observed:
                stable=independent.stability(observed,success)
                classification=stable['classification']
            else:
                stable=None;classification='INCOMPLETE_STARTED_WINDOW'
            grouped[key].append(dict(experiment=directory.name,phase_id=phase_id,seed=phase['seed'],
                offered_rate_s=phase['rate'],duration_s=phase['duration_s'],classification=classification,
                run_terminal_success=bool(success),raw_integrity_pass=audit_result is not None,
                missing_identity_fields=missing,stability=stable,
                protocol_completed=sum(independent.protocol_complete(s) for s in observed['sessions']) if observed else None,
                task_successes=sum(s.get('task_success') is True for s in observed['sessions']) if observed and phase['kind']=='agent' else None,
                actual_sessions=len(observed['sessions']) if observed else None))
    groups=[]
    for key,windows in sorted(grouped.items()):
        groups.append(dict(cohort_sha256=key,identity=details[key],windows=windows,confirmation=aggregate(windows)))
    unresolved=[dict(experiment=w['experiment'],phase_id=w['phase_id'],missing_identity_fields=w['missing_identity_fields'])
                for group in groups for w in group['windows'] if w['missing_identity_fields']]
    # Unknown identity is not evidence of a different treatment. Conservatively
    # withhold all confirmations in this scan until every started window has
    # enough identity to be included or shown to belong to another cohort.
    for group in groups:
        if unresolved:
            group['confirmation']['finite_protocol_stable_confirmed']=False
            group['confirmation']['blockers'].extend(
                'Unresolved started confirmation identity: '+w['experiment']+'/'+w['phase_id'] for w in unresolved)
    return dict(groups=groups,examined_runs=run_results,planned_but_unstarted=planned_unstarted,
                unresolved_started_identity=unresolved,
                started_confirmation_windows=sum(len(group['windows']) for group in groups),
                confirmed_protocol_points=sum(g['confirmation']['finite_protocol_stable_confirmed'] for g in groups))


def self_test():
    checks = []
    good = dict(experiment='synthetic', phase_id='one', seed=1, run_terminal_success=True, raw_integrity_pass=True, classification='STABLE_IN_WINDOW', missing_identity_fields=[])
    two = {**good, 'phase_id': 'two', 'seed': 2}
    need(aggregate([good, two])['finite_protocol_stable_confirmed'], 'Two stable seeds')
    checks.append('two_stable_seeds')
    for label, rows in [('single', [good]), ('same_seed', [good, {**two, 'seed': 1}]), ('bad_first', [{**good, 'classification': 'INCONCLUSIVE'}, two]), ('bad_extra', [good, two, {**good, 'phase_id': 'three', 'seed': 3, 'classification': 'GROWING_BACKLOG'}]), ('in_progress', [good, {**two, 'run_terminal_success': False}]), ('integrity', [good, {**two, 'raw_integrity_pass': False}]), ('source_missing', [good, {**two, 'missing_identity_fields': ['client']}])]:
        need(not aggregate(rows)['finite_protocol_stable_confirmed'], label)
        checks.append(label)
    config = {'name': 'synthetic', 'args': ['--tp', '4']}
    phase = dict(id='one', kind='replay', arrival='poisson', role='confirm', seed=1, rate=0.1, duration_s=300, profiles=['coding'], cache_scope='session', dp_routing='none')
    command = dict(source_sha256={}, environment={'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    global_plan = dict(timeout_s=120, session_timeout_s=300, max_active_sessions=64, model='dsv4', tokenizer_path='/mock/tokenizer')
    report = dict(tokenizer_sha256='x', transport='fresh_tcp_no_retry', plan=global_plan)
    key, _, _ = signature(config, phase, command, report)
    need(signature({**config, 'name': 'two'}, {**phase, 'id': 'two', 'seed': 2}, command, report)[0] == key, 'IDs/seed excluded')
    checks.append('seed_and_experiment_identity_not_treatment')
    for field, value in [('cache_scope', 'shared'), ('dp_routing', 'sticky4'), ('rate', 0.2), ('profiles', ['research']), ('duration_s', 450)]:
        need(signature(config, {**phase, field: value}, command, report)[0] != key, 'Separate ' + field)
        checks.append('separate_' + field)
    changed = {**command, 'source_sha256': {'scenarios/scripts/agent_session_client.py': 'changed'}}
    need(signature(config, phase, changed, report)[0] != key, 'Runtime changed')
    checks.append('separate_runtime')
    for field, value in [('timeout_s', 180), ('session_timeout_s', 600), ('max_active_sessions', 128), ('model', 'other-model')]:
        changed_report = {**report, 'plan': {**global_plan, field: value}}
        need(signature(config, phase, command, changed_report)[0] != key, 'Separate service protocol ' + field)
        checks.append('separate_global_' + field)
        absent = {k: v for k, v in global_plan.items() if k != field}
        _, _, missing = signature(config, phase, command, {**report, 'plan': absent})
        need('report.plan.' + field in missing, 'Missing global identity ' + field)
        need(not aggregate([good, {**two, 'missing_identity_fields': missing}])['finite_protocol_stable_confirmed'], 'Missing global identity blocks confirmation')
        checks.append('missing_global_' + field + '_blocks_confirmation')
    need(signature(config, phase, command, {**report, 'plan': {**global_plan, 'tokenizer_path': '/other/tokenizer'}})[0] == key, 'Tokenizer byte identity, not requested path, controls grouping')
    checks.append('tokenizer_path_not_treatment')
    with tempfile.TemporaryDirectory(prefix='session-confirm-discovery-') as temp:
        directory = Path(temp) / 'c99-simulation'
        directory.mkdir()
        write_text(directory / 'session-plan.json', json.dumps({'phases': [phase, {**phase, 'id': 'not-started', 'seed': 2}]}))
        write_text(directory / 'sessions.json', json.dumps({**report, 'status': 'RUNNING', 'active_phase': 'one', 'phases': []}))
        write_text(directory / 'command.json', json.dumps(command))
        write_text(directory / 'config.json', json.dumps(config))
        result = scan(Path(temp), {})
        need(result['started_confirmation_windows'] == 1 and len(result['planned_but_unstarted']) == 1, 'Started/unstarted separation')
        need(not result['groups'][0]['confirmation']['finite_protocol_stable_confirmed'], 'Running window blocks confirmation')
        checks.extend(['discover_started_incomplete', 'planned_is_not_executed', 'running_window_not_selected_away'])
        second = Path(temp) / 'c100-simulation'
        second.mkdir()
        write_text(second / 'session-plan.json', json.dumps({'phases': [phase]}))
        write_text(second / 'sessions.json', json.dumps({**report, 'plan': {**global_plan, 'max_active_sessions': 128}, 'status': 'RUNNING', 'active_phase': 'one', 'phases': []}))
        write_text(second / 'command.json', json.dumps(command))
        write_text(second / 'config.json', json.dumps(config))
        result = scan(Path(temp), {})
        need(result['started_confirmation_windows'] == 2 and len(result['groups']) == 2, 'Global admission bounds separate discovered cohorts')
        checks.append('discovered_runs_with_different_global_bounds_not_pooled')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=independent.RESULTS)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a new confirmation audit ID')
    ledger = {}
    for path in (Path(__file__), Path(independent.__file__), path_at(ROOT, 'finite-session-v1.md')):
        independent.read(path, ledger)
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), rule='finite-session-v1', scope='Independent CPU aggregation of every discovered started confirm window; no maximum capacity or session SLO claims')
    try:
        if args.self_test:
            result['self_test_checks'] = self_test()
        result.update(scan(args.results.resolve(), ledger))
        for path, info in ledger.items():
            need(hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'], 'Evidence changed during aggregation; retry after terminal under a new output ID')
    except Exception as exc:
        import traceback
        result.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    result['evidence_ledger'] = ledger
    result['limitations'] = ['Only directories present in the recorded local results root are discoverable; external/deleted experiments cannot be ruled out.', 'All started windows in each exact runtime/config/task/cache/routing cohort are included; incompatible cohorts are visibly separate, never pooled.', 'Any started window with unresolved runtime/config/protocol/tokenizer/transport/task identity conservatively blocks every confirmation in this scan until resolved; unknown is not proof of an incompatible cohort.', 'A snapshot during live writing may fail unchanged-evidence checks; it never upgrades ongoing windows to confirmation.', 'Protocol stability is separate from task success, session latency targets and production SLA; session SLO remains undefined.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'started_confirmation_windows', 'confirmed_protocol_points', 'error') if k in result}, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__=='__main__':main()
