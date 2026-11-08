#!/usr/bin/env python3
"""CPU-only synthetic negative controls for D offline Profile interpretation.

All fixtures are MOCK, not measured timing or independent implementation proof.
No vLLM/CUDA/HTTP import or process start. Unique output path required.
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
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile
sys.dont_write_bytecode=True
PATH=Path(__file__).with_name('analyze_profiles.py')
spec=importlib.util.spec_from_file_location('d_profile_analysis_under_test', PATH)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    """Conserve 134144 context + 1016 generation, include unrelated warmup."""
    rows=[]; ids=[]
    for i in range(8):
        wire=f'd-MOCK-profile-{i}'; key='cmpl-'+wire+'-0-'+f'{i:08x}'
        ids.append(key)
        rows.append(dict(index=i,wire_request_id=wire,server_response_ids=['cmpl-'+wire],
            input_tokens=65536 if i<2 else 512,usage=dict(completion_tokens=128),
            dispatch_offset_s=a.OFFSETS[i],ttft_s=12.,scheduled_ttft_s=12.,
            tpot_s=.02,scheduled_e2e_s=14.54))
    identity=dict(type='scheduler_identity',schema_version=1,policy='observe-v1',
        step_trace_enabled=True,async_scheduler=True,class_name='research_scheduler.ObserveResearchScheduler',
        config=dict(token_budget=4096,input_budget=4096))
    records=[identity]; computed={key:0 for key in ids}
    def state(key,prompt):
        return dict(num_prompt_tokens=prompt,num_tokens=prompt+128,num_tokens_with_spec=prompt+128,
            num_computed_tokens=computed[key],is_prefill_chunk=computed[key]<prompt,
            num_output_placeholders=0,num_in_flight_tokens=0,next_decode_eligible_step=0,
            num_preemptions=0,status='RUNNING')
    def add(allocation, *, warm=False, finish=False):
        serial=len(records); stamp=90. if warm else 100.+serial*.2
        requests=({'MOCK-warm':dict(num_prompt_tokens=16)} if warm else
                  {} if finish else {key:state(key,rows[i]['input_tokens']) for i,key in enumerate(ids)})
        before=dict(current_step=serial,num_executed_steps=serial-1,running=list(requests),waiting=[],
            kv_holding_waiting=[],deferred_waiting=[],prefill_capacity_bound=False,
            last_prefill_step=0,local_prefill_interval=8,requests=requests)
        after=deepcopy(before)
        if not warm:
            for key,n in allocation.items():
                computed[key]+=n;after['requests'][key]['num_computed_tokens']=computed[key]
        record=dict(type='step',schema_version=1,policy='observe-v1',scheduler_step=serial,
            begin_monotonic_ns=int(stamp*1e9),end_monotonic_ns=int(stamp*1e9)+1000,
            begin_unix_ns=int(stamp*1e9),end_unix_ns=int(stamp*1e9)+1000,
            throttle_prefills=False,before=before,after=after,decision=None,
            output=dict(num_scheduled_tokens=allocation,total_num_scheduled_tokens=sum(allocation.values()),
                admitted_request_ids=[],preempted_request_ids=[],finished_request_ids=list(ids) if finish else []))
        records.append(record)
    add({'MOCK-warm':16},warm=True)
    add({}) # valid idle scheduler record, excluded from active counts
    for _ in range(32):add({ids[0]:2048,ids[1]:2048})
    add({key:512 for key in ids[2:]})
    for _ in range(127):add({key:1 for key in ids})
    add({},finish=True) # late finished IDs don't imply allocation / GPU work
    controls=dict(controls=[dict(started_unix=99.),dict(finished_unix=300.)])
    return records,rows,dict(start_unix=100.),controls


def source_fixture(directory):
    raw = dict(schema_version=1, experiment_id=directory.name, base_url='http://127.0.0.1:8260', model='dsv4', tokenizer_path='/MOCK/tokenizer', seed=1, timeout_s=180, max_inflight=16, metrics_interval_s=1, phases=[dict(id='MOCK', mode='closed-loop', requests=1, shape_mix=[dict(name='p512', input_tokens=512, output_tokens=128)])])
    config = dict(name=directory.name, args=['--MOCK'])
    prefix = a.ROOT.name
    members = {prefix + '/configs/' + directory.name + '.json': json.dumps(config).encode(), prefix + '/workloads/' + directory.name + '.json': json.dumps(raw).encode(), prefix + '/scripts/study_client.py': b'# MOCK client\n', prefix + '/scripts/serve_study.py': b'# MOCK harness\n', prefix + '/scripts/research_scheduler.py': b'# MOCK scheduler\n'}
    for name in ('v1/engine/input_processor.py', 'v1/engine/async_llm.py', 'entrypoints/openai/completion/serving.py', 'v1/core/sched/async_scheduler.py', 'v1/core/sched/scheduler.py'):
        members['installed/vllm/' + name] = b'# MOCK installed source ' + name.encode()
    files = {name: dict(path='/MOCK/' + name, sha256=a.sha(data), bytes=len(data)) for name, data in members.items()}
    identity = dict(files=files)
    identity_bytes = json.dumps(identity).encode()
    write_bytes(directory / 'runtime-identity.json', identity_bytes)
    with zipfile.ZipFile(directory / 'source-snapshot.zip', 'w') as z:
        for key, value in members.items():
            z.writestr(key, value)
    hashes = {name: value['sha256'] for name, value in files.items()}
    command = dict(runtime_identity_sha256=a.sha(identity_bytes), source_snapshot_sha256=a.sha((directory / 'source-snapshot.zip').read_bytes()), source_sha256=hashes)
    status = dict(source_identity_unchanged=True, changed_source_members=[], supervisor_exit_code=0, cleanup_error=None, exit_code=0, timed_out=False, interrupted=False, source_sha256=deepcopy(hashes))
    report = dict(client_sha256=hashes[prefix + '/scripts/study_client.py'], profile_harness_sha256=hashes[prefix + '/scripts/serve_study.py'])
    records = {'command.json': command, 'status.json': status, 'config.json': config, 'capacity-plan.json': a.normalize_launch_plan(raw), 'capacity.json': report}
    return (records, identity)


def run_checks():
    checks=[]
    def check(name,fn):
        try:
            detail=fn();checks.append(dict(name=name,status='PASS',detail=detail))
        except Exception as e:checks.append(dict(name=name,status='FAIL',error=type(e).__name__+': '+str(e)))
    def reject(call):
        try:call()
        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as e:return str(e)
        raise AssertionError('Invalid MOCK evidence was accepted')
    def mutation(change):
        records,rows,run,controls=fixture();change(records,rows,run,controls)
        return reject(lambda:a.scheduler_analysis(records,rows,run,controls))
    def accepted():
        result=a.scheduler_analysis(*fixture())
        assert result['context_tokens_sum']==134144 and result['generation_tokens_sum']==1016
        assert result['profile_active_steps']==160 and result['total_logged_steps']==163
        assert len(result['binding'])==8 and all(x['finished_observed'] for x in result['requests'].values())
        assert all(x['generation_tokens']==127 for x in result['requests'].values())
        assert len(result['requests'][next(iter(result['binding']))]['allocations'])==159
        return 'Exact totals; warmup/idle/late cleanup excluded correctly; all 8 IDs bound'
    check('MOCK valid mixed-request ledger',accepted)
    cases=[
      ('missing identity',lambda r,c,u,v:r.pop(0)),
      ('duplicate identity',lambda r,c,u,v:r.append(deepcopy(r[0]))),
      ('schedule error',lambda r,c,u,v:r[-1].update(type='schedule_error')),
      ('missing step',lambda r,c,u,v:r.pop(3)),
      ('duplicate step',lambda r,c,u,v:r[3].update(scheduler_step=2)),
      ('wrong schema',lambda r,c,u,v:r[3].update(schema_version=2)),
      ('policy changes',lambda r,c,u,v:r[3].update(policy='fairshare-v1')),
      ('false identity class',lambda r,c,u,v:r[0].update(class_name='other')),
      ('trace disabled',lambda r,c,u,v:r[0].update(step_trace_enabled=False)),
      ('different token budget',lambda r,c,u,v:r[0]['config'].update(token_budget=8192)),
      ('time reversal',lambda r,c,u,v:r[3].update(end_monotonic_ns=1)),
      ('overlapping calls',lambda r,c,u,v:r[3].update(begin_monotonic_ns=r[2]['begin_monotonic_ns'])),
      ('missing queue snapshot',lambda r,c,u,v:r[3]['before'].pop('waiting')),
      ('queue unknown ID',lambda r,c,u,v:r[3]['before']['waiting'].append('unknown')),
      ('wrong output total',lambda r,c,u,v:r[3]['output'].update(total_num_scheduled_tokens=4095)),
      ('boolean output total',lambda r,c,u,v:r[2]['output'].update(total_num_scheduled_tokens=False)),
      ('zero token map entry',lambda r,c,u,v:r[2]['output']['num_scheduled_tokens'].update(unknown=0)),
      ('negative token map entry',lambda r,c,u,v:r[2]['output']['num_scheduled_tokens'].update(unknown=-1)),
      ('missing client wire ID',lambda r,c,u,v:c[0].pop('wire_request_id')),
      ('duplicate wire ID',lambda r,c,u,v:c[1].update(wire_request_id=c[0]['wire_request_id'])),
      ('wrong SSE ID',lambda r,c,u,v:c[0].update(server_response_ids=['wrong'])),
      ('multiple SSE IDs',lambda r,c,u,v:c[0]['server_response_ids'].append('wrong')),
      ('wrong prompt length',lambda r,c,u,v:r[3]['before']['requests'][next(iter(r[3]['before']['requests']))].update(num_prompt_tokens=512)),
      ('negative computed tokens',lambda r,c,u,v:r[3]['before']['requests'][next(iter(r[3]['before']['requests']))].update(num_computed_tokens=-1)),
      ('duplicate context interval unchanged totals',lambda r,c,u,v:r[4]['before']['requests'][next(iter(r[4]['before']['requests']))].update(num_computed_tokens=0)),
      ('wrong after computed delta',lambda r,c,u,v:r[4]['after']['requests'][next(iter(r[4]['after']['requests']))].update(num_computed_tokens=0)),
      ('broken context conservation',lambda r,c,u,v:r[3]['before']['requests'][next(iter(r[3]['before']['requests']))].update(num_computed_tokens=65536)),
      ('wrong actual output count',lambda r,c,u,v:c[0]['usage'].update(completion_tokens=127)),
      ('preemption output',lambda r,c,u,v:r[3]['output']['preempted_request_ids'].append('MOCK')),
      ('preemption state',lambda r,c,u,v:r[3]['before']['requests'][next(iter(r[3]['before']['requests']))].update(num_preemptions=1)),
      ('allocation outside capture',lambda r,c,u,v:v['controls'][1].update(finished_unix=101)),
    ]
    for name,change in cases:check('MOCK rejects '+name,lambda change=change:mutation(change))
    def foreign(r,c,u,v):
        r[3]['output']['num_scheduled_tokens']['foreign']=1
        r[3]['output']['num_scheduled_tokens'][next(iter(r[3]['output']['num_scheduled_tokens']))]-=1
    check('MOCK rejects concurrent foreign request',lambda:mutation(foreign))
    def missing_before(r,c,u,v):
        key=next(iter(r[3]['output']['num_scheduled_tokens']))
        for slot in ['before']:
            r[3][slot]['running'].remove(key);del r[3][slot]['requests'][key]
    check('MOCK rejects allocation absent before snapshot',lambda:mutation(missing_before))
    def mappings():
        r,c,u,v=fixture();ids=list(a.scheduler_analysis(r,c,u,v)['binding'])
        reject(lambda:a.bind_requests(c,ids+[ids[0][:-8]+'abcdef01']))
        reject(lambda:a.bind_requests(c,[key+'tail' for key in ids]))
        reject(lambda:a.bind_requests(c,[key.replace('-0-','-1-') for key in ids]))
        reject(lambda:a.bind_requests(c,[key[:-8]+'zzzzzzzz' for key in ids]))
        reject(lambda:a.bind_requests(c,ids[1:]))
        c[0]['wire_request_id']='d.[MOCK]';c[0]['server_response_ids']=['cmpl-d.[MOCK]']
        ids[0]='cmpl-d.[MOCK]-0-aaaaaaaa'
        assert len(a.bind_requests(c,ids))==8
        return '5 absent/ambiguous/anchor/suffix rejections; regex metacharacter literal join accepted'
    check('MOCK strict request mapping',mappings)
    with tempfile.TemporaryDirectory(prefix='d-profile-MOCK-') as tmp:
        directory=Path(tmp)/'d99-MOCK';directory.mkdir();records,identity=source_fixture(directory)
        check('MOCK source archive and normalized plan',lambda:a.snapshot_identity(directory,records,{})==identity or (_ for _ in ()).throw(AssertionError()))
        source_cases=[
            ('changed source flag',lambda r:r['status.json'].update(source_identity_unchanged=False)),
            ('nonzero supervisor',lambda r:r['status.json'].update(supervisor_exit_code=1)),
            ('child failure',lambda r:r['status.json'].update(exit_code=1)),
            ('timeout',lambda r:r['status.json'].update(timed_out=True)),
            ('interrupted',lambda r:r['status.json'].update(interrupted=True)),
            ('cleanup failure',lambda r:r['status.json'].update(cleanup_error='MOCK')),
            ('runtime hash',lambda r:r['command.json'].update(runtime_identity_sha256='bad')),
            ('ZIP hash',lambda r:r['command.json'].update(source_snapshot_sha256='bad')),
            ('status inventory',lambda r:r['status.json']['source_sha256'].pop(next(iter(r['status.json']['source_sha256'])))),
            ('client execution hash',lambda r:r['capacity.json'].update(client_sha256='bad')),
            ('saved config mutation',lambda r:r['config.json'].update(name='other')),
            ('saved default mutation',lambda r:r['capacity-plan.json']['phases'][0].update(ignore_eos=False)),
        ]
        for name,change in source_cases:
            def call(change=change):
                changed=deepcopy(records);change(changed)
                return reject(lambda:a.snapshot_identity(directory,changed,{}))
            check('MOCK rejects '+name,call)
        def self_identity():
            logs,client,run,controls=fixture();module=identity['files'][a.ROOT.name+'/scripts/research_scheduler.py']
            parent=identity['files']['installed/vllm/v1/core/sched/async_scheduler.py']
            logs[0].update(module_file=module['path'],module_sha256=module['sha256'],parent_file=parent['path'],parent_sha256=parent['sha256'])
            assert a.scheduler_analysis(logs,client,run,controls,identity)['status']=='PASS'
            logs[0]['module_sha256']='bad'
            return reject(lambda:a.scheduler_analysis(logs,client,run,controls,identity))
        check('MOCK actual scheduler import binding',self_identity)
    def comparison_fixture():
        key=a.ROOT.name+'/scripts/serve_study.py'
        left=dict(experiment='d01-MOCK',same_workload_identity={'sha256':'MOCK'},
            comparison_conditions=dict(config_sha256='MOCK same config',profile_control_timeout_s=120,
                runtime_sources={key:a.TIMEOUT_HARNESS_PAIR[0],'installed/MOCK':'same'}))
        right=deepcopy(left);right['experiment']='scheduler-tp4-fairshare-dense-profile-retry'
        right['comparison_conditions']['profile_control_timeout_s']=600
        right['comparison_conditions']['runtime_sources'][key]=a.TIMEOUT_HARNESS_PAIR[1]
        return left,right
    def mock_comparison(experiments, *, historical_log=True, alter_decision=None):
        # Exercise the frozen comparison rules with explicitly synthetic
        # provenance files. These fixtures do not verify a historical export.
        original_root = a.ROOT
        decision = (original_root / 'decisions/profile-export-retry-r1.json').read_bytes()
        if alter_decision is not None:
            changed = json.loads(decision); alter_decision(changed)
            decision = json.dumps(changed).encode()
        with tempfile.TemporaryDirectory(prefix='profile-timeout-MOCK-') as temp:
            mock_root = Path(temp) / original_root.name
            write_bytes(mock_root / 'decisions/profile-export-retry-r1.json', decision)
            if historical_log:
                write_text(mock_root / 'logs/profile-timeout-harness-diff-r1.txt',
                           'MOCK CPU fixture: timeout decision branch only; not historical source diff evidence.\n')
            a.ROOT = mock_root
            try:
                return a.matched_profile_conditions(experiments, {})
            finally:
                a.ROOT = original_root
    def accepted_timeout():
        result=mock_comparison(list(comparison_fixture()))
        assert result['status']=='PASS' and len(result['declared_differences'])==1
        return 'Only the declared120-to600 deadline and exact reviewed harness SHA pair accepted'
    check('MOCK controlled timeout revision explicitly accepted',accepted_timeout)
    changes=[('other deadline',lambda r:r['comparison_conditions'].update(profile_control_timeout_s=300)),
             ('other non-treatment config',lambda r:r['comparison_conditions'].update(config_sha256='different')),
             ('other runtime source',lambda r:r['comparison_conditions']['runtime_sources'].update({'installed/MOCK':'different'})),
             ('unreviewed harness',lambda r:r['comparison_conditions']['runtime_sources'].update({a.ROOT.name+'/scripts/serve_study.py':'unknown'})),
             ('undeclared retry ID',lambda r:r.update(experiment='other-retry')),
             ('changed prompt/arrival hash',lambda r:r.update(same_workload_identity={'sha256':'different'}))]
    for name,change in changes:
        def call(change=change):
            left,right=comparison_fixture();change(right)
            return reject(lambda:mock_comparison([left,right]))
        check('MOCK timeout exception rejects '+name,call)
    def same_source_fixture():
        # Use the real native configs so this test covers the fresh-reproduction
        # path, not just the original pair of historical harness revisions.
        names = ('scheduler-tp4-observe-dense-profile', 'scheduler-tp4-fairshare-dense-profile-retry')
        identity = {'files': {a.ROOT.name+'/scripts/serve_study.py': {'sha256': a.sha(b'MOCK current harness')},
                              'installed/MOCK': {'sha256': a.sha(b'MOCK installed source')}}}
        return [dict(experiment=name, same_workload_identity={'sha256':'MOCK same prepared prompt/arrival'},
                     comparison_conditions=a.comparison_conditions(
                         load_document(path_at(a.ROOT, 'configs')/(name+'.json')), identity))
                for name in names]
    def accepted_same_source(reverse=False):
        pair = same_source_fixture()
        if reverse: pair.reverse()
        result = mock_comparison(pair, historical_log=False)
        assert result['status'] == 'PASS' and len(result['declared_differences']) == 1
        difference = result['declared_differences'][0]
        assert difference['runtime_source_identity'] == 'identical' and 'source_diff' not in difference
        assert difference['before'] == 120 and difference['after'] == 600
        return 'Native Observe/retry configs; same captured sources; no historical log required'
    check('MOCK native same-source export deadline pair accepted without historical log', accepted_same_source)
    check('MOCK native same-source pair ordering is immaterial', lambda: accepted_same_source(True))
    same_source_changes = changes + [
        ('wrong policy class', lambda r:r['comparison_conditions'].update(policy_class='other.Scheduler'))]
    for name, change in same_source_changes:
        def call(change=change):
            left, right = same_source_fixture(); change(right)
            return reject(lambda:mock_comparison([left,right], historical_log=False))
        check('MOCK same-source timeout pair rejects '+name, call)
    def wrong_anchor():
        left, right = same_source_fixture(); left['experiment'] = 'other-control'
        return reject(lambda:mock_comparison([left,right],historical_log=False))
    check('MOCK same-source timeout pair rejects unregistered control', wrong_anchor)
    def missing_harness():
        left, right = same_source_fixture()
        for row in (left,right): del row['comparison_conditions']['runtime_sources'][a.ROOT.name+'/scripts/serve_study.py']
        return reject(lambda:mock_comparison([left,right],historical_log=False))
    check('MOCK same-source timeout pair rejects absent harness identity', missing_harness)
    check('MOCK same-source timeout pair rejects altered retry protocol',
          lambda:reject(lambda:mock_comparison(same_source_fixture(),historical_log=False,
              alter_decision=lambda d:d.update(controlled_change={'profile_control_timeout_s':{'before':120,'after':300}}))))
    def historical_log_still_required():
        try: mock_comparison(list(comparison_fixture()),historical_log=False)
        except FileNotFoundError: return 'Different historical sources still require original review log'
        raise AssertionError('Historical source revision accepted without its review log')
    check('MOCK historical harness revision still requires review log', historical_log_still_required)
    check('no GPU or HTTP client imported',lambda: all(x not in sys.modules for x in ('torch','vllm','aiohttp','http.client')) or (_ for _ in ()).throw(AssertionError()))
    return dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
        scope='MOCK CPU negative controls, not GPU execution or independent review',
        utc=datetime.now(timezone.utc).isoformat(),checks=checks,count=len(checks),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PATH,Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use unique output revision')
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=result['count'], failures=[c for c in result['checks'] if c['status'] != 'PASS'], output=str(args.output))))
    return 0 if result['status'] == 'PASS' else 1
if __name__=='__main__':raise SystemExit(main())
