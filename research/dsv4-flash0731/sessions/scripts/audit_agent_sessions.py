#!/usr/bin/env python3
"""Independent CPU terminal audit and finite-session-v1 recomputation.

Uses only the standard library. Reads evidence, never imports the runtime,
analyzer, task executor, model, or GPU libraries. Refuses output replacement.
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
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile

STAGE=Path(__file__).resolve().parents[1]
RESULTS = path_at(STAGE.parent, 'scenarios/results')


def need(condition,message):
    if not condition:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def finite(v):return type(v) in (int,float) and math.isfinite(v)


def close(a,b,message,tolerance=.003):
    need(a is b if a is None or b is None else finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-8,abs_tol=tolerance),message)


def p95(values):
    values=sorted(v for v in values if finite(v))
    if not values:return None
    pos=.95*(len(values)-1);i=int(pos)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(pos-i)


def read(path,ledger):
    path=path.resolve();raw=path.read_bytes();info={'sha256':sha(raw),'bytes':len(raw)}
    need(str(path) not in ledger or ledger[str(path)]==info,'Evidence changed: '+str(path))
    ledger[str(path)]=info
    return raw


def load(path,ledger):return json.loads(read(path,ledger))
def lines(path,ledger):return [json.loads(row) for row in read(path,ledger).decode().splitlines() if row.strip()]


def protocol_complete(s):
    if s.get('admitted') is not True or s.get('censored') or s.get('protocol_ok') is not True:return False
    if not s.get('requests') or any(r.get('ok') is not True or r.get('timed_out') or r.get('cancelled') for r in s['requests']):return False
    if s['kind']=='replay':return s.get('terminal_reason')=='replay_complete' and s.get('task_success') is None
    grade=s.get('grade')
    return (s.get('terminal_reason')=='natural_stop' and isinstance(grade,dict)
            and type(grade.get('success')) is bool and type(s.get('task_success')) is bool and grade['success']==s['task_success'])


def trend(arrivals,completions,means,eligible):
    epsilon=max(2.,.05*arrivals)
    if not eligible:return 'INCONCLUSIVE'
    deficit=arrivals-completions;delta=means[-1]-means[0]
    if deficit<=epsilon and delta<=epsilon:return 'STABLE_IN_WINDOW'
    if deficit>epsilon and delta>epsilon and all(a<b for a,b in zip(means,means[1:])):return 'GROWING_BACKLOG'
    return 'INCONCLUSIVE'


def stability(phase,terminal=True):
    f=phase['phase'];rows=phase['sessions'];T=phase['measurement_window_s']
    if f['arrival']!='poisson':return {'classification':'NOT_OPEN_ARRIVAL','version':'finite-session-v1'}
    need(finite(T) and T>0,'Invalid arrival horizon')
    interval=phase.get('metrics_interval_s',1.)
    need(finite(interval) and interval>0,'Invalid metrics interval')
    valid=[];slots=set();start=T/4
    for sample in phase['metric_samples']:
        when=sample.get('offset_s')
        if not finite(when) or not start<=when<T or sample.get('error') or sample.get('http_status')!=200:continue
        values=[v for key,v in sample.get('values',{}).items() if key.split('{',1)[0]=='vllm:num_requests_waiting']
        if not values or not all(finite(v) and v>=0 for v in values):continue
        valid.append((when,sum(values)));slots.add(math.floor((when-start)/interval))
    bins=[]
    for i in range(3):
        left=start+i*T/4;right=left+T/4
        values=[v for t,v in valid if left<=t<right]
        bins.append(dict(start_s=left,end_s=right,
            arrivals=sum(left<=s['planned_offset_s']<right for s in rows),
            protocol_completions=sum(protocol_complete(s) and left<=s['end_offset_s']<right for s in rows),
            mean_session_outstanding=sum(max(0.,min(right,s['end_offset_s'])-max(left,s['planned_offset_s'])) for s in rows)/(right-left),
            server_mean_waiting=sum(values)/len(values) if values else None,server_samples=len(values)))
    A=sum(b['arrivals'] for b in bins);C=sum(b['protocol_completions'] for b in bins)
    means=[b['mean_session_outstanding'] for b in bins];coverage=min(1.,len(slots)*interval/(.75*T))
    lag=p95([s.get('dispatch_lag_s') for s in rows if s.get('admitted')]);reasons=[]
    if not terminal:reasons.append('Experiment not successful terminal')
    if phase.get('status')!='COMPLETED' or phase.get('error') or phase.get('protection_triggered'):reasons.append('Phase failure or protection')
    if not rows or any(not protocol_complete(s) for s in rows):reasons.append('Request/protocol/budget/censor/grade failure')
    if len(rows)!=phase.get('planned_sessions',len(rows)):reasons.append('Missing planned session records')
    if A<20:reasons.append('Fewer than 20 analysis session arrivals')
    if coverage<.8 or any(b['server_mean_waiting'] is None for b in bins):reasons.append('Insufficient waiting observations')
    if lag is None or lag>.1:reasons.append('Session dispatch P95 above 100 ms or missing')
    verdict=trend(A,C,means,not reasons)
    if verdict=='INCONCLUSIVE' and not reasons:reasons.append('Between frozen stability and growth rules')
    return dict(version='finite-session-v1',classification='SCREEN_ONLY' if T<180 or f.get('role')!='confirm' else verdict,
                trend_verdict=verdict,bins=bins,analysis_arrivals=A,analysis_protocol_completions=C,
                completion_deficit=A-C,epsilon=max(2.,.05*A),session_mean_outstanding_increase=means[-1]-means[0],
                server_waiting_coverage=coverage,dispatch_lag_p95_s=lag,reasons=reasons,session_latency_slo=None)


def request_check(r,tool_limit=4):
    need(sha(canonical(r['payload']))==r['payload_sha256'],'Request semantic payload SHA')
    need(sha(r['wire_body'].encode())==r['wire_body_sha256'] and json.loads(r['wire_body'])==r['payload'],'Exact wire body SHA/content')
    need(r['attempts']==1,'Request retries changed')
    close(r['end_offset_s']-r['dispatch_offset_s'],r['elapsed_until_return_s'],'Request elapsed interval')
    need(r['end_offset_s']>=r['dispatch_offset_s'],'Negative request duration')
    parsed=[];text=[];reason=[];calls={};first=None;text_first=None;tool_first=None;done=None;finish=None;usage=None;last=-1
    chat='messages' in r['payload']
    for event in r['raw_events']:
        t=event['elapsed_s'];need(finite(t) and last<=t<=r['elapsed_until_return_s']+.003,'SSE time order');last=t
        if event['data'].strip()=='[DONE]':need(done is None,'Duplicate DONE');done=t;continue
        try:obj=json.loads(event['data'])
        except ValueError:need(not r['ok'],'Malformed successful SSE');continue
        if not isinstance(obj,dict) or obj.get('error'):need(not r['ok'],'Error successful SSE');continue
        parsed.append(dict(elapsed_s=t,data=obj))
        if obj.get('usage') is not None:usage=obj['usage']
        if not r['ok']:continue
        for choice in obj.get('choices',[]):
            need(choice.get('index',0)==0,'Wrong stream choice')
            delta=choice.get('delta',{}) if chat else dict(content=choice.get('text'))
            value=delta.get('content') or '';why=delta.get('reasoning_content') or delta.get('reasoning') or ''
            need(isinstance(value,str) and isinstance(why,str),'Nontext stream fields')
            if value:
                text.append(value)
                if text_first is None:text_first=t
                if first is None:first=t
            if why:
                reason.append(why)
                if first is None:first=t
            for call in delta.get('tool_calls') or []:
                index=call['index'];need(type(index)is int and 0<=index<tool_limit,'Tool index')
                dst=calls.setdefault(index,dict(id='',type='function',function=dict(name='',arguments='')))
                dst['id']+=call.get('id') or ''
                function=call.get('function') or {}
                for key in ('name','arguments'):dst['function'][key]+=function.get(key) or ''
                if function.get('name') or function.get('arguments'):
                    if tool_first is None:tool_first=t
                    if first is None:first=t
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    need(parsed==r['events'] and usage==r.get('usage'),'Raw and parsed SSE/usage disagree')
    if not r['ok']:return
    need(r['http_status']==200 and done is not None and r['done_received'] and not r.get('error') and not r.get('timed_out') and not r.get('cancelled') and not r.get('raw_stream_truncated'),'False successful request')
    need(isinstance(usage,dict) and all(type(usage.get(k))is int and usage[k]>0 for k in ('prompt_tokens','completion_tokens','total_tokens')),'Usage types')
    need(usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens'] and usage['completion_tokens']<=r['payload']['max_tokens'],'Usage totals')
    need(''.join(text)==r['full_text'] and ''.join(reason)==r['reasoning'],'SSE text content')
    tools=[calls[k] for k in sorted(calls)];need(tools==r['tool_calls'],'SSE tool content')
    need(finish==r['finish_reason'] and finish in ('stop','length','tool_calls'),'Finish reason')
    if tools:
        need(finish=='tool_calls' and sorted(calls)==list(range(len(calls))),'Tool terminal/indexes')
        need(len({c['id'] for c in tools})==len(tools),'Duplicate tool IDs')
        need(all(c['id'] and c['function']['name'] and isinstance(json.loads(c['function']['arguments']),dict) for c in tools),'Tool arguments')
    else:need(finish!='tool_calls','Missing terminal tools')
    for key,value in [('first_output_s',first),('ttft_s',text_first),('ttft_tool_s',tool_first),('e2e_s',done)]:close(r.get(key),value,'SSE timing '+key,1e-9)
    count=usage['completion_tokens'];eligible=not tools and not reason and text_first is not None
    need(r['slo_text_eligible']==eligible,'Text SLO eligibility')
    close(r.get('tpot_output_s'),(done-first)/(count-1) if count>1 else None,'Output TPOT',1e-9)
    close(r.get('tpot_s'),(done-text_first)/(count-1) if count>1 and eligible else None,'Text TPOT',1e-9)
    detail=usage.get('prompt_tokens_details');need(detail is None or isinstance(detail,dict),'Cache usage details')
    cached=(detail or {}).get('cached_tokens');need(cached is None or type(cached)is int and 0<=cached<=usage['prompt_tokens'],'Cache token count')
    need(r.get('cached_tokens')==cached,'Derived cache count')


def audit(run,ledger):
    need(run.is_dir(),'Missing experiment directory')
    command,status,experiment,report,prepared,plan=[load(run/name,ledger) for name in ('command.json','status.json','experiment.json','sessions.json','sessions-prepared.json','session-plan.json')]
    need(status.get('end_utc') and type(status.get('exit_code'))is int,'Supervisor terminal required')
    need(experiment.get('status') in ('PASS','FAILED') and all(type(c.get('exit_code'))is int for c in experiment['clients']),'Terminal client required')
    need(report.get('status') in ('COMPLETED','FAILED') and report.get('finished_utc'),'Terminal session report required')
    terminal_success=(status['exit_code']==0 and status.get('timed_out') is False and not status.get('supervisor_error')
                      and experiment['status']=='PASS' and all(c['exit_code']==0 for c in experiment['clients'])
                      and report['status']=='COMPLETED' and not report.get('active_phase'))
    need(report['plan']==plan and report['prepared_sha256']==sha(canonical(prepared)),'Plan/prepared identity')
    need(report['tokenizer_sha256']==prepared['tokenizer_sha256'],'Tokenizer metadata')
    need(command['command']==status['command'],'Supervisor command mismatch')
    snapshot=run/'source-snapshot.zip';read(snapshot,ledger)
    source_bytes={}
    with zipfile.ZipFile(snapshot) as archive:
        need(set(archive.namelist())==set(command['source_sha256']),'Snapshot member mismatch')
        for name in archive.namelist():
            raw=archive.read(name);source_bytes[name]=raw
            need(sha(raw)==command['source_sha256'][name]==status['source_sha256'][name],'Snapshot SHA '+name)
    prefix='scenarios/scripts/'
    need(report['client_sha256']==sha(source_bytes[prefix+'agent_session_client.py']),'Executed client SHA')
    tree=ast.parse(source_bytes[prefix+'agent_session_client.py']);limits=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_TOOL_CALLS_PER_ROUND' for t in node.targets):limits.append(ast.literal_eval(node.value))
    tool_limit=limits[0] if len(limits)==1 else 4
    need(type(tool_limit)is int and tool_limit in (4,8),'Unknown frozen runtime tool limit')
    if 'tool_calls_per_round_limit' in report:need(report['tool_calls_per_round_limit']==tool_limit,'Reported tool limit differs from executed source')
    for key,expected in report['helper_sha256'].items():
        name='scripts/benchmark_client.py' if key=='benchmark_client' else prefix+key+'.py'
        need(sha(source_bytes[name])==expected,'Executed helper SHA '+key)
    need(len(report['phases'])==len(plan['phases'])==len(prepared['phases']),'Missing phases')
    totals=Counter();phases=[]
    for phase,bank,planned in zip(report['phases'],prepared['phases'],plan['phases']):
        need(phase['phase']==bank['phase']==planned,'Phase plan differs')
        f=phase['phase'];rows=phase['sessions'];specs=bank['sessions'];raw=run/Path(phase['raw_dir']).name
        need(phase['planned_sessions']==len(specs)==len(rows),'Missing session rows')
        need(sorted(lines(raw/'sessions.jsonl',ledger),key=lambda s:s['index'])==rows,'Session JSONL mismatch')
        events=lines(raw/'events.jsonl',ledger)
        starts=[e for e in events if e['event']=='session_start'];ends=[e for e in events if e['event']=='session_end']
        need(len(ends)==len(rows) and len(starts)==sum(s['admitted'] for s in rows),'Session event counts')
        need(len({e['session_index'] for e in ends})==len(rows),'Duplicate session end')
        reset=load(raw/'reset.json',ledger);need(reset['http_status']==200 and reset['response'].get('success') is True,'Cache reset failed')
        if f['arrival']=='poisson':
            rng=random.Random(f['seed']);offset=0.;offsets=[]
            while True:
                offset+=rng.expovariate(f['rate'])
                if offset>=f['duration_s']:break
                offsets.append(offset)
            need([s['planned_offset_s'] for s in specs]==offsets,'Poisson schedule mismatch')
        for row,spec in zip(rows,specs):
            need(row['index']==spec['index'] and row['kind']==f['kind'],'Session identity')
            if f['arrival']=='poisson':need(row['planned_offset_s']==spec['planned_offset_s'],'Planned session timing drift')
            need(finite(row['end_offset_s']) and row['end_offset_s']<=phase['total_wall_s']+.003,'Session terminal time')
            if row['admitted']:
                need(not row['censored'] and row['end_offset_s']>=row['start_offset_s'],'Admitted session timing')
                close(row['scheduled_latency_s'],row['end_offset_s']-row['planned_offset_s'],'Session latency')
                close(row['dispatch_lag_s'],max(0.,row['start_offset_s']-row['planned_offset_s']),'Session dispatch')
            # Tool-result string serialization preserves insertion order; the
            # final report sorts object keys. Raw event JSONL is authoritative.
            raw_tools=[e['tool'] for e in events if e['event']=='tool' and e['session_index']==row['index']]
            if f['kind']=='agent':need(raw_tools==row['tools'],'Raw tool event records differ from checkpoint')
            previous=row.get('start_offset_s') or 0;history=None;tools=iter(raw_tools)
            for turn,r in enumerate(row['requests']):
                request_check(r,tool_limit);need(r['turn']==turn and r['session_index']==row['index'],'Turn identity')
                need(previous-.003<=r['dispatch_offset_s']<=r['end_offset_s']<=row['end_offset_s']+.003,'Turn/session timing')
                previous=r['end_offset_s'];label='c-'+sha(canonical([f['seed'],f['id']]))[:24]
                if f['cache_scope']!='shared':label+=f'-s{row["index"]}'
                if f['cache_scope']=='none':label+=f'-t{turn}'
                need(r['payload']['cache_salt']==label,'Session cache namespace')
                logged=[e['request'] for e in events if e['event']=='model_request' and e['request']['request_id']==r['request_id']]
                need(logged==[r],'Request event mismatch')
                if f['kind']=='replay':
                    need(r['payload']['prompt']==spec['prompts'][turn] and r['payload']['max_tokens']==spec['outputs'][turn],'Replay workload mismatch')
                    if r['ok']:need(r['usage']['prompt_tokens']==len(spec['prompts'][turn]) and r['output_tokens']==spec['outputs'][turn] and r['finish_reason']=='length','Replay exact output mismatch')
                else:
                    if history is not None:need(r['payload']['messages']==history,'Dynamic tool feedback/history drift')
                    history=list(r['payload']['messages']);assistant=dict(role='assistant',content=r['full_text'] or None)
                    if r['tool_calls']:assistant['tool_calls']=r['tool_calls']
                    if r['reasoning']:assistant['reasoning_content']=r['reasoning']
                    history.append(assistant)
                    if r['ok']:
                        for call in r['tool_calls']:
                            action=next(tools,None)
                            if action is None:
                                need(row['terminal_reason'] in ('tool_budget','session_timeout','cancelled'),'Missing executed tool');break
                            need(action['call']==call,'Executed tool changed model call')
                            history.append(dict(role='tool',tool_call_id=call['id'],content=json.dumps(action['result'],ensure_ascii=False)))
            close(row['model_wall_s'],sum(r['elapsed_until_return_s'] for r in row['requests']),'Session model sum')
            close(row['tool_wall_s'],sum(t['wall_s'] for t in row['tools']),'Session tool sum')
            if f['kind']=='agent' and row.get('initial_state'):
                initial,final=row['initial_state'],row['final_state'];workspace=raw/f'session-{row["index"]:04d}'
                states=load(workspace/'task-state.json',ledger)
                need(states['initial']==initial and states['current']==final,'Task-state snapshot mismatch')
                need(initial['tool_sequence']==0 and final['tool_sequence']==len(row['tools']),'Task tool sequence')
                need({k:v for k,v in initial.items() if k!='tool_sequence'}=={k:v for k,v in final.items() if k!='tool_sequence'},'Unexpected task identity mutation')
                owner=load(workspace/'.session-owner.json',ledger)
                need(owner==dict(task_id=row['task_id'],token=initial['owner_token']),'Task workspace ownership')
                prefix='experiments/' if any(key.startswith('experiments/scenarios/workloads/sessions/tasks/') for key in source_bytes) else ''
                base=prefix+'scenarios/workloads/sessions/tasks/'+row['task_id']+'/'
                manifest={key[len(base):]:sha(value) for key,value in source_bytes.items() if key.startswith(base)}
                need(sha(canonical(manifest))==initial['source_manifest_sha256'],'Task source manifest')
                for relative,digest in initial['initial_public_sha256'].items():need(manifest['public/'+relative]==digest,'Initial public source identity')
                totals['agent_tasks']+=1;totals['successful_tasks']+=row.get('task_success') is True
            totals['sessions']+=1;totals['requests']+=len(row['requests']);totals['tools']+=len(row['tools'])
        for sample in [phase['metrics_before'],*phase['metric_samples'],phase['metrics_after']]:
            if sample.get('raw_file'):
                body=read(raw/Path(sample['raw_file']).name,ledger)
                need(sha(body)==sample['sha256'],'Metrics file SHA')
                values={}
                for line in body.decode().splitlines():
                    if not line.strip() or line.startswith('#'):continue
                    try:key,value=line.rsplit(None,1);number=float(value)
                    except ValueError:continue
                    if key in sample.get('values',{}):values[key]=number
                need(values==sample.get('values',{}),'Embedded metrics differ from raw Prometheus')
        totals['phases']+=1;totals['metrics_samples']+=len(phase['metric_samples'])
        phases.append(dict(id=f['id'],kind=f['kind'],role=f['role'],sessions=len(rows),requests=sum(len(s['requests']) for s in rows),
                           protocol_complete=sum(protocol_complete(s) for s in rows),task_successes=sum(s.get('task_success') is True for s in rows),
                           terminal_reasons=dict(Counter(s['terminal_reason'] for s in rows)),stability=stability(phase,terminal_success)))
    return dict(experiment=run.name,git_commit=command['git_commit'],runtime_terminal_success=terminal_success,
                supervisor_exit_code=status['exit_code'],report_status=report['status'],active_phase_label=report.get('active_phase'),
                tool_calls_per_round_limit=tool_limit,totals=dict(totals),phases=phases)


def self_test():
    checks=[]
    for name,A,C,means,eligible,expected in [
        ('stable',100,98,[3,4,5],True,'STABLE_IN_WINDOW'),
        ('growth',100,80,[1,10,20],True,'GROWING_BACKLOG'),
        ('deficit_only',100,80,[1,2,3],True,'INCONCLUSIVE'),
        ('nonmonotonic',100,80,[1,30,20],True,'INCONCLUSIVE'),
        ('invalid',100,100,[1,1,1],False,'INCONCLUSIVE'),
        ('exact_epsilon',413,392,[1,2,3],True,'INCONCLUSIVE'),
        ('within_epsilon',413,393,[1,2,3],True,'STABLE_IN_WINDOW'),
        ('carryover_completion',50,60,[5,2,0],True,'STABLE_IN_WINDOW')]:
        need(trend(A,C,means,eligible)==expected,name);checks.append(name)
    need(p95([0,1,2,3])==2.8499999999999996,'Linear p95');checks.append('linear_p95')
    base=dict(admitted=True,censored=False,protocol_ok=True,kind='agent',terminal_reason='natural_stop',grade={'success':False},task_success=False,requests=[{'ok':True}])
    need(protocol_complete(base),'Quality failure counts protocol terminal');checks.append('quality_false_not_protocol_false')
    for update in ({'grade':None},{'terminal_reason':'turn_budget'},{'task_success':0},{'censored':True}):
        need(not protocol_complete({**base,**update}),'Invalid protocol completion '+str(update));checks.append('reject_'+str(update))
    rows=[{**copy.deepcopy(base),'index':i,'planned_offset_s':61+3*i,'start_offset_s':61+3*i,
           'end_offset_s':62+3*i,'dispatch_lag_s':0.} for i in range(60)]
    samples=[dict(offset_s=i,http_status=200,values={'vllm:num_requests_waiting':0.,'vllm:num_requests_waiting_by_reason{reason="capacity"}':99.}) for i in range(240)]
    phase=dict(phase={'arrival':'poisson','role':'confirm'},measurement_window_s=240.,metrics_interval_s=1.,
               planned_sessions=60,status='COMPLETED',protection_triggered=False,error=None,sessions=rows,metric_samples=samples)
    result=stability(phase)
    need(result['classification']=='STABLE_IN_WINDOW' and result['analysis_arrivals']==60 and result['analysis_protocol_completions']==60,'Integrated stable phase')
    need(result['server_waiting_coverage']==1 and all(b['server_mean_waiting']==0 for b in result['bins']),'Exact metric names avoid breakdown double counting')
    checks.extend(['integrated_stable_phase','waiting_by_reason_not_double_counted'])
    for label,change in [
        ('missing_grade',lambda p:p['sessions'][0].update(grade=None)),
        ('dispatch_lag',lambda p:[s.update(dispatch_lag_s=.101) for s in p['sessions']]),
        ('monitor_http_failure',lambda p:[s.update(http_status=500) for s in p['metric_samples']]),
        ('duplicate_monitor_times',lambda p:[s.update(offset_s=61.) for s in p['metric_samples']]),
        ('tool_budget',lambda p:p['sessions'][0].update(terminal_reason='tool_budget'))]:
        bad=copy.deepcopy(phase);change(bad)
        need(stability(bad)['classification']=='INCONCLUSIVE',label);checks.append(label)
    low=copy.deepcopy(phase);low['sessions']=low['sessions'][:19];low['planned_sessions']=19
    need(stability(low)['classification']=='INCONCLUSIVE','Minimum analysis arrivals');checks.append('minimum_analysis_arrivals')
    for n,expected in [(144,'STABLE_IN_WINDOW'),(143,'INCONCLUSIVE')]:
        observed=copy.deepcopy(phase);observed['metric_samples']=observed['metric_samples'][60:60+n]
        need(stability(observed)['classification']==expected,'Coverage threshold '+str(n));checks.append('coverage_'+str(n))
    carry=copy.deepcopy(phase);carry['sessions'].append({**copy.deepcopy(rows[0]),'index':60,'planned_offset_s':1.,'end_offset_s':61.5});carry['planned_sessions']=61
    need(stability(carry)['analysis_protocol_completions']==61 and stability(carry)['analysis_arrivals']==60,'Carryover count');checks.append('carryover_completions_integrated')
    boundary=copy.deepcopy(phase);boundary['sessions'][0]['end_offset_s']=120.
    result=stability(boundary)
    need(result['bins'][0]['protocol_completions']==19 and result['bins'][1]['protocol_completions']==21,'Half-open completion bins');checks.append('half_open_bins')
    growing=copy.deepcopy(phase)
    for row in growing['sessions']:row['end_offset_s']=300.
    need(stability(growing)['classification']=='GROWING_BACKLOG','Integrated sustained growth');checks.append('integrated_growth')
    screen=copy.deepcopy(phase);screen['phase']['role']='pilot'
    need(stability(screen)['classification']=='SCREEN_ONLY','Pilot is not capacity confirmation');checks.append('pilot_excluded_from_confirmation')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a new audit output ID')
    ledger = {}
    read(Path(__file__), ledger)
    read(path_at(STAGE, 'finite-session-v1.md'), ledger)
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='Independent CPU terminal/raw identity audit and finite-session-v1; not task semantic regrading or GPU profiling', experiments=[])
    try:
        if args.self_test:
            result['self_test_checks'] = self_test()
        for name in args.run:
            need(Path(name).name == name, 'Simple run ID required')
            result['experiments'].append(audit(RESULTS / name, ledger))
        need(args.self_test or args.run, 'Provide --run or --self-test')
        for path, info in ledger.items():
            need(sha(Path(path).read_bytes()) == info['sha256'], 'Evidence changed during audit: ' + path)
    except Exception as exc:
        import traceback
        result.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    result['evidence_ledger'] = ledger
    result['limitations'] = ['PASS means evidence consistency, not performance or task quality success.', 'Dynamic task grades are checked for provenance and consistency; this tool does not re-execute model-written code.', 'Pilot/screen/closed phases cannot confirm session capacity; session SLO remains undefined.', 'Source snapshots and recorded tokenizer/model metadata do not newly hash all model weights.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output), experiments=[dict(experiment=e['experiment'], totals=e['totals']) for e in result['experiments']], self_test_checks=len(result.get('self_test_checks', [])), error=result.get('error')), indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__=='__main__':main()
