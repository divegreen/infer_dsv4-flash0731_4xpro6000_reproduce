"""CPU-only terminal Session evidence analysis, preserving quality/workload scope."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import ast
from copy import deepcopy
from collections import Counter,defaultdict
from datetime import timezone,datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'scenarios'
spec=importlib.util.spec_from_file_location('historical',OLD/'scripts/summarize_scenarios.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
path_spec=importlib.util.spec_from_file_location('session_path_audit',OLD/'scripts/capacity_audit.py')
path_audit=importlib.util.module_from_spec(path_spec);path_spec.loader.exec_module(path_audit)

def need(x,msg):
    if not x:raise ValueError(msg)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(v):
    v=[x for x in v if type(x) in (float,int) and math.isfinite(x)]
    return dict(n=len(v),p50=h.percentile(v,.5),p95=h.percentile(v,.95),p99=h.percentile(v,.99),max=max(v) if v else None)
def ratio(a,b):return a/b if b else None

def finite(value):return type(value) in (float,int) and math.isfinite(value)
def close(a,b,label,tolerance=1e-7):
    need(a is b if a is None or b is None else finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-8,abs_tol=tolerance),label)
def hashed(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def named_values(sample,name):
    # waiting_by_reason is a breakdown of waiting, never another pool.
    return [v for key,v in sample.get('values',{}).items() if key.split('{',1)[0]==name and finite(v)]

def workload_identity(bank,tokenizer_sha256):
    phase=bank['phase']
    context={key:phase.get(key) for key in ('kind','arrival','duration_s','rate','concurrency')}
    rows=[]
    for row in bank['sessions']:
        fields=('index','planned_offset_s','seed','profile','prompts','outputs','tool_waits_s') if phase['kind']=='replay' else ('index','planned_offset_s','seed','task_id')
        rows.append({key:row.get(key) for key in fields})
    if phase['kind']=='agent':context.update({key:phase.get(key) for key in ('max_turns','max_tools','max_output_tokens')})
    return dict(sha256=hashed(dict(tokenizer_sha256=tokenizer_sha256,arrival=context,sessions=rows)),
                scope='Exact replay tokens/output budgets/think waits and arrival plan' if phase['kind']=='replay' else 'Matched task assignment/arrival/budget only; actual dynamic request histories are not matched',
                exclusions=['phase ID','role','cache_scope','dp_routing','model outcomes','actual closed-loop dispatch times'],
                closed_arrival_note='Closed-loop sessions have no preplanned offsets; concurrency and task order are matched, actual dispatch is an outcome.')

def complete_protocol(session):
    if not session.get('admitted') or session.get('censored') or session.get('protocol_ok') is not True:return False
    if not session.get('requests') or any(r.get('ok') is not True or r.get('timed_out') or r.get('cancelled') for r in session['requests']):return False
    if session['kind']=='replay':return session.get('terminal_reason')=='replay_complete' and session.get('task_success') is None
    return session.get('terminal_reason')=='natural_stop' and type((session.get('grade') or {}).get('success'))is bool and type(session.get('task_success'))is bool and session['task_success']==session['grade']['success']

def stability(run):
    f=run['phase'];rows=run['sessions'];window=run['measurement_window_s']
    if f['arrival']!='poisson':return dict(classification='NOT_OPEN_ARRIVAL',version='finite-session-v1')
    begin=.25*window;width=.25*window;bins=[];observed_slots=set()
    for i in range(3):
        a,b=begin+i*width,begin+(i+1)*width
        waiting=[];slots=set()
        for m in run['metric_samples']:
            if a<=m['offset_s']<b and m.get('http_status')==200 and not m.get('error'):
                values=named_values(m,'vllm:num_requests_waiting')
                if values and all(v>=0 for v in values):
                    waiting.append(sum(values))
                    interval=run.get('metrics_interval_s',1.)
                    need(finite(interval) and interval>0,'Invalid monitoring interval')
                    slots.add(math.floor((m['offset_s']-begin)/interval))
        observed_slots.update(slots)
        bins.append(dict(start_s=a,end_s=b,arrivals=sum(a<=r['planned_offset_s']<b for r in rows),
                         completions=sum(r['protocol_ok'] and a<=r['end_offset_s']<b for r in rows),
                         mean_session_outstanding=sum(max(0.,min(b,r['end_offset_s'])-max(a,r['planned_offset_s'])) for r in rows)/width,
                         server_mean_waiting=sum(waiting)/len(waiting) if waiting else None,server_samples=len(waiting),server_observed_interval_slots=len(slots)))
    A=sum(b['arrivals'] for b in bins);C=sum(b['completions'] for b in bins);eps=max(2.,.05*A)
    means=[b['mean_session_outstanding'] for b in bins];increase=means[-1]-means[0]
    interval=run.get('metrics_interval_s',1.)
    need(finite(interval) and interval>0,'Invalid monitoring interval')
    coverage=min(1.,len(observed_slots)/(.75*window/interval))
    lag=dist([s['dispatch_lag_s'] for s in rows])['p95'];reasons=[]
    if any(not complete_protocol(s) for s in rows):reasons.append('Protocol/request failure, missing grade, timeout, budget/truncation, or censored session')
    if run.get('status')!='COMPLETED' or run.get('error') or run.get('protection_triggered'):reasons.append('Phase did not complete without protection/runtime errors')
    if run.get('owned_run_terminal_success') is False:reasons.append('Entire owned experiment is not terminal successful')
    if A<20:reasons.append('Fewer than 20 analysis session arrivals')
    if coverage<.8:reasons.append('Insufficient server waiting observation')
    if any(b['server_mean_waiting'] is None for b in bins):reasons.append('At least one analysis bin lacks valid waiting observations')
    if lag is None or lag>.1:reasons.append('Session dispatch P95 exceeds 100ms')
    verdict='INCONCLUSIVE'
    if not reasons:
        if A-C<=eps and increase<=eps:verdict='STABLE_IN_WINDOW'
        elif A-C>eps and means[0]<means[1]<means[2] and increase>eps:verdict='GROWING_BACKLOG'
        else:reasons.append('Between frozen stable and growth rules')
    return dict(version='finite-session-v1',classification='SCREEN_ONLY' if window<180 or f.get('role')!='confirm' else verdict,trend_verdict=verdict,
                analysis_bins=bins,analysis_arrivals=A,analysis_completions=C,completion_deficit=A-C,tolerance=eps,
                session_outstanding_increase=increase,server_sample_coverage=coverage,monitoring_interval_s=interval,dispatch_lag_p95_s=lag,reasons=reasons,
                server_waiting_units='model requests, not sessions; observation only, not compared to session epsilon',
                session_latency_slo=None,scope='Protocol processing stability; task success and latency are separate')

def summarize(rows,wall):
    calls=[r for s in rows for r in s['requests']];ok=[r for r in calls if r['ok']]
    done=[s for s in rows if s['protocol_ok']];graded=any(s['kind']=='agent' for s in rows)
    wins=[s for s in rows if s.get('task_success') is True];n=len(wins)
    p=sum(r['usage']['prompt_tokens'] for r in ok);o=sum(r['usage']['completion_tokens'] for r in ok)
    cached=[r.get('cached_tokens') for r in ok];known=[v for v in cached if v is not None]
    known_usage=[r['usage'] for r in calls if isinstance(r.get('usage'),dict) and
                 all(type(r['usage'].get(k))is int and r['usage'][k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')) and
                 r['usage']['prompt_tokens']+r['usage']['completion_tokens']==r['usage']['total_tokens']]
    known_p=sum(u['prompt_tokens'] for u in known_usage);known_o=sum(u['completion_tokens'] for u in known_usage)
    return dict(sessions=len(rows),protocol_completed=len(done),successful_tasks=n if graded else None,
                task_success_rate=ratio(n,len(rows)) if graded else None,protocol_completion_rate=ratio(len(done),len(rows)),
                completed_sessions_s=ratio(len(done),wall),successful_tasks_s=ratio(n,wall) if graded else None,
                model_calls=len(calls),successful_model_calls=len(ok),tool_calls=sum(len(s['tools']) for s in rows),
                prompt_tokens=p,output_tokens=o,output_tokens_s=ratio(o,wall),cached_tokens=sum(known) if known else None,
                cached_usage_calls=len(known),cache_fraction=ratio(sum(known),p) if len(known)==len(ok) else None,
                all_calls_known_usage=dict(prompt_tokens=known_p,output_tokens=known_o,known_calls=len(known_usage),total_calls=len(calls),complete=len(known_usage)==len(calls)),
                tokens_per_successful_task=dict(prompt=ratio(known_p,n),output=ratio(known_o,n),model_calls=ratio(len(calls),n),tools=ratio(sum(len(s['tools']) for s in rows),n),
                    token_usage_complete=len(known_usage)==len(calls),basis='All cohort attempts including failed tasks and valid known usage from failed calls; missing failed-call usage makes token cost a lower bound.') if graded else None,
                session_latency_s=dist([s['scheduled_latency_s'] for s in done]),successful_task_latency_s=dist([s['scheduled_latency_s'] for s in wins]),
                user_visible_latency_s=dist([s.get('user_visible_latency_s') for s in done]),
                model_wall_s=dist([s.get('model_wall_s') for s in rows]),tool_wall_s=dist([s.get('tool_wall_s') for s in rows]),
                grading_wall_s=dist([s.get('grading_wall_s') for s in rows]),
                calls_per_session=dist([len(s['requests']) for s in rows]),tools_per_session=dist([len(s['tools']) for s in rows]),
                terminal_reasons=dict(Counter(s['terminal_reason'] for s in rows)),
                request_metrics={k:dist([r.get(k) for r in ok]) for k in ('ttft_s','ttft_tool_s','first_output_s','e2e_s','tpot_s','tpot_output_s')},
                text_slo_eligible=sum(r.get('slo_applicable') is True and r.get('slo_text_eligible') for r in calls),
                text_slo_passes=sum(r.get('slo_pass') is True for r in calls))


def recorded_tool_limit(source):
    """Read the bound from the executed source without importing that runtime."""
    tree=ast.parse(source)
    constants={node.targets[0].id:node.value.value for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Constant) and type(node.value.value)is int}
    function=next(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='stream_request')
    bounds=[]
    for node in ast.walk(function):
        if isinstance(node,ast.Compare) and len(node.ops)==2 and isinstance(node.ops[0],ast.LtE) and isinstance(node.ops[1],ast.Lt) and isinstance(node.left,ast.Constant) and node.left.value==0 and isinstance(node.comparators[0],ast.Name) and node.comparators[0].id=='ix':
            bound=node.comparators[1]
            value=bound.value if isinstance(bound,ast.Constant) else constants.get(bound.id) if isinstance(bound,ast.Name) else None
            if type(value)is int:bounds.append(value)
    need(len(bounds)==1 and 1<=bounds[0]<=64,'Cannot establish executed per-round tool bound')
    return bounds[0]


def check_request(request,tool_limit=4):
    """Reconstruct successful stream claims from recorded raw SSE, no client import."""
    r=request;need(hashed(r['payload'])==r['payload_sha256'],'Payload SHA mismatch')
    if 'wire_body' in r or 'wire_body_sha256' in r:
        need(isinstance(r.get('wire_body'),str) and hashlib.sha256(r['wire_body'].encode()).hexdigest()==r.get('wire_body_sha256'),'Wire body SHA mismatch')
        need(json.loads(r['wire_body'])==r['payload'],'Wire request differs semantically from payload')
    need(r['attempts']==1 and finite(r['dispatch_offset_s']) and finite(r['end_offset_s']),'Invalid request attempts/timing')
    need(r['end_offset_s']>=r['dispatch_offset_s'],'Negative request elapsed')
    close(r['elapsed_until_return_s'],r['end_offset_s']-r['dispatch_offset_s'],'Request wall mismatch',.003)
    last=-1.;parsed=[];text=[];reasoning=[];tools={};first=ttft=tool_first=done=None;usage=None;finish=None
    chat='messages' in r['payload']
    for event in r['raw_events']:
        t=event['elapsed_s'];need(finite(t) and last<=t<=r['elapsed_until_return_s']+.003,'Raw SSE event time is invalid');last=t
        if event['data'].strip()=='[DONE]':
            need(done is None,'Multiple DONE events');done=t;continue
        try:obj=json.loads(event['data'])
        except ValueError:
            need(not r['ok'],'Successful stream contains malformed raw JSON');continue
        if not isinstance(obj,dict) or obj.get('error'):
            need(not r['ok'],'Successful stream contains error object');continue
        parsed.append(dict(elapsed_s=t,data=obj))
        if obj.get('usage') is not None:usage=obj['usage']
        if not r['ok']:continue
        choices=obj.get('choices',[]);need(isinstance(choices,list) and len(choices)<=1,'Invalid number of choices')
        for choice in choices:
            need(choice.get('index',0)==0,'Unexpected choice index')
            delta=choice.get('delta',{}) if chat else {'content':choice.get('text')}
            content=delta.get('content') or '';why=delta.get('reasoning_content') or delta.get('reasoning') or ''
            need(isinstance(content,str) and isinstance(why,str),'Nonstring stream delta')
            if content:
                text.append(content)
                if ttft is None:ttft=t
                if first is None:first=t
            if why:
                reasoning.append(why)
                if first is None:first=t
            for call in delta.get('tool_calls') or []:
                index=call['index'];need(type(index)is int and 0<=index<tool_limit,'Invalid tool index under executed runtime source')
                item=tools.setdefault(index,dict(id='',type='function',function=dict(name='',arguments='')))
                if call.get('id'):item['id']+=call['id']
                for key in ('name','arguments'):item['function'][key]+=(call.get('function') or {}).get(key) or ''
                if (call.get('function') or {}).get('name') or (call.get('function') or {}).get('arguments'):
                    if tool_first is None:tool_first=t
                    if first is None:first=t
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    need(parsed==r['events'],'Parsed SSE events differ from raw SSE')
    need(usage==r.get('usage'),'Recorded usage differs from raw final usage')
    if not r['ok']:return
    need(r['http_status']==200 and r['done_received'] and done is not None and not r.get('raw_stream_truncated'),'False successful stream state')
    need(not r.get('error') and not r.get('timed_out') and not r.get('cancelled'),'Successful stream carries error/timeout/cancel')
    need(isinstance(usage,dict) and all(type(usage.get(k))is int and usage[k]>=1 for k in ('prompt_tokens','completion_tokens','total_tokens')),'Invalid successful usage types')
    need(usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],'Usage total mismatch')
    need(usage['completion_tokens']<=r['payload']['max_tokens'],'Usage exceeds output budget')
    need(r['input_tokens']==usage['prompt_tokens'] and r['output_tokens']==usage['completion_tokens'],'Derived token lengths differ from usage')
    need(''.join(text)==r['full_text'] and ''.join(reasoning)==r['reasoning'],'Stream text reconstruction mismatch')
    reconstructed=[tools[k] for k in sorted(tools)]
    need(reconstructed==r['tool_calls'],'Stream tool reconstruction mismatch')
    need(finish==r['finish_reason'] and finish in ('stop','length','tool_calls'),'Invalid stream finish')
    if reconstructed:
        need(sorted(tools)==list(range(len(tools))) and finish=='tool_calls','Noncontiguous/incomplete tool calls')
        need(len({c['id'] for c in reconstructed})==len(reconstructed),'Duplicate tool call IDs')
        need(all(c['id'] and c['function']['name'] and isinstance(json.loads(c['function']['arguments']),dict) for c in reconstructed),'Incomplete function calls')
        close(r['tool_arguments_ready_s'],done,'Tool completion timing mismatch')
    else:need(finish!='tool_calls','Missing terminal tool calls')
    for key,value in (('first_output_s',first),('ttft_s',ttft),('ttft_tool_s',tool_first),('e2e_s',done)):close(r.get(key),value,'SSE '+key+' mismatch')
    need(first is not None and 0<=first<=done,'No generated output')
    count=usage['completion_tokens'];eligible=not reconstructed and not reasoning and ttft is not None
    close(r.get('tpot_output_s'),(done-first)/(count-1) if count>1 else None,'Output TPOT mismatch')
    close(r.get('tpot_s'),(done-ttft)/(count-1) if count>1 and eligible else None,'Text TPOT mismatch')
    need(r.get('slo_text_eligible')==eligible,'Text SLO eligibility mismatch')
    detail=usage.get('prompt_tokens_details') or {};cached=detail.get('cached_tokens')
    need(cached is None or type(cached)is int and 0<=cached<=usage['prompt_tokens'],'Invalid cached usage')
    need(r.get('cached_tokens')==cached,'Cached usage reconstruction mismatch')
    if eligible:
        limit=next((bound for tokens,bound in ((1024,1.),(4096,3.),(16384,8.)) if usage['prompt_tokens']<=tokens),None)
        expected=None if limit is None else ttft<=limit and done<=limit+.05*(count-1) and (count==1 or r['tpot_s']<=.05)
        need(r.get('slo_pass') is expected,'Request Research SLO mismatch')
    else:need(r.get('slo_pass') is None,'Tool/reasoning request has text SLO claim')


def check_session(s,b,phase,wall,tool_limit=4):
    need(s['kind']==phase['kind'] and s.get('task_id')==b.get('task_id') and s.get('profile')==b.get('profile'),'Task/profile assignment drift')
    need(finite(s['planned_offset_s']) and s['planned_offset_s']>=0 and finite(s['end_offset_s']) and s['end_offset_s']<=wall+.003,'Invalid session boundary')
    if phase['arrival']=='poisson':need(s['planned_offset_s']==b['planned_offset_s'],'Arrival plan drift')
    if s['admitted']:
        need(not s['censored'] and 0<=s['start_offset_s']<=s['end_offset_s'],'Invalid admitted interval')
        close(s['dispatch_lag_s'],max(0.,s['start_offset_s']-s['planned_offset_s']),'Session dispatch lag mismatch')
        close(s['scheduled_latency_s'],s['end_offset_s']-s['planned_offset_s'],'Session scheduled latency mismatch',.003)
        close(s['elapsed_s'],s['end_offset_s']-s['start_offset_s'],'Session elapsed mismatch',.003)
    else:
        need(s['censored'] and not s['protocol_ok'] and not s['requests'] and s['scheduled_latency_s'] is None,'False censored session')
    need([r['turn'] for r in s['requests']]==list(range(len(s['requests']))),'Request turn sequence mismatch')
    previous=s.get('start_offset_s') or 0
    for r in s['requests']:
        check_request(r,tool_limit)
        label='c-'+hashed([phase['seed'],phase['id']])[:24]
        if phase['cache_scope']!='shared':label+='-s'+str(s['index'])
        if phase['cache_scope']=='none':label+='-t'+str(r['turn'])
        need(r['payload'].get('cache_salt')==label,'Actual cache namespace differs from treatment')
        rank=r.get('headers',{}).get('X-data-parallel-rank')
        need(rank==(str(s['index']%4) if phase['dp_routing']=='sticky4' else None),'Actual DP routing differs from treatment')
        need(r['payload'].get('seed')==b['seed'],'Request seed differs from prepared session')
        need(r['session_index']==s['index'] and previous-.003<=r['dispatch_offset_s']<=r['end_offset_s']<=s['end_offset_s']+.003,'Session request ordering/bounds mismatch')
        previous=r['end_offset_s']
        if phase['kind']=='replay':
            turn=r['turn'];need(turn<len(b['prompts']) and r['payload']['prompt']==b['prompts'][turn],'Replay prompt drift')
            need(r['payload']['max_tokens']==b['outputs'][turn] and r['payload']['ignore_eos'] is True and r['payload']['add_special_tokens'] is False,'Replay output/EOS treatment drift')
            if r['ok']:need(r['usage']['prompt_tokens']==len(b['prompts'][turn]) and r['usage']['completion_tokens']==b['outputs'][turn] and r['finish_reason']=='length','Exact replay usage differs')
    close(s.get('model_wall_s',0),sum(r['elapsed_until_return_s'] for r in s['requests']),'Model wall sum differs')
    close(s.get('tool_wall_s',0),sum(t['wall_s'] for t in s['tools']),'Tool wall sum differs')
    close(s.get('grading_wall_s',0),(s.get('grade') or {}).get('wall_s',0),'Grading wall differs')
    if s.get('user_visible_latency_s') is not None:
        need(s['planned_offset_s']<=s['user_visible_end_offset_s']<=s['end_offset_s'],'User-visible end bounds')
        close(s['user_visible_latency_s'],s['user_visible_end_offset_s']-s['planned_offset_s'],'User-visible latency mismatch')
    if s['protocol_ok']:
        need(complete_protocol(s),'Claimed protocol completion lacks complete requests/natural terminal/grade')
        if phase['kind']=='replay':need(len(s['requests'])==len(b['prompts']),'Incomplete replay marked complete')
    if s.get('task_success') is not None:
        need(type(s['task_success'])is bool and s['kind']=='agent' and s['protocol_ok'] and s.get('grade',{}).get('success') is s['task_success'],'False task grade')


def raw_gauge_values(text):
    result={}
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        fields=line.rsplit(None,1)
        if len(fields)!=2:continue
        label,value=fields;name=label.split('{',1)[0]
        if any(token in name for token in ('num_requests_running','num_requests_waiting','kv_cache_usage_perc','prefix_cache_hits_total','prefix_cache_queries_total')):
            number=float(value)
            if finite(number):
                need(label not in result,'Duplicate raw Prometheus gauge')
                result[label]=number
    return result


def check_phase_files(run,raw,read,read_bytes):
    reset=read(raw/'reset.json');need(reset.get('http_status')==200 and reset.get('response',{}).get('success') is True,'Unverified APC reset')
    log=[json.loads(line) for line in read_bytes(raw/'events.jsonl').decode().splitlines() if line]
    rows=run['sessions']
    for s in rows:
        events=[e for e in log if e.get('session_index')==s['index']]
        need([e['request'] for e in events if e['event']=='model_request']==s['requests'],'Model request event log mismatch')
        need([e['tool'] for e in events if e['event'] in ('tool','synthetic_tool_wait')]==s['tools'],'Tool event log mismatch')
        need([e['operation'] for e in events if e['event']=='host_operation']==s.get('host_operations',[]),'Host operation event log mismatch')
        terminals=[e for e in events if e['event']=='session_end'];need(len(terminals)==1,'Missing/duplicate session end event')
        close(terminals[0]['offset_s'],s['end_offset_s'],'Event session end differs')
        need(terminals[0]['terminal_reason']==s['terminal_reason'],'Event terminal reason differs')
        starts=[e for e in events if e['event']=='session_start'];need(len(starts)==int(s['admitted']),'Session start event admission mismatch')
        if starts:
            close(starts[0]['offset_s'],s['start_offset_s'],'Session start event timing')
            close(starts[0]['planned_offset_s'],s['planned_offset_s'],'Session start planned timing')
        if s['kind']=='agent' and s.get('final_state') is not None:
            state=read(raw/f"session-{s['index']:04d}"/'task-state.json')
            need(state==dict(initial=s.get('initial_state'),current=s['final_state']),'Final session state differs from saved task state')
    need(all(e.get('session_index') in range(len(rows)) for e in log),'Event log names an unplanned session')
    offsets=[m['offset_s'] for m in run['metric_samples']]
    need(all(finite(t) and 0<=t<=run['total_wall_s']+.003 for t in offsets) and offsets==sorted(offsets),'Metric sample time order/bounds invalid')
    for sample in [run['metrics_before'],*run['metric_samples'],run['metrics_after']]:
        if sample.get('error'):continue
        need(sample.get('http_status')==200 and sample.get('raw_file') and sample.get('sha256'),'Missing raw Prometheus evidence')
        path=raw/Path(sample['raw_file']).name;body=read_bytes(path)
        need(hashlib.sha256(body).hexdigest()==sample['sha256'],'Prometheus sample SHA mismatch')
        need(raw_gauge_values(body.decode())==sample.get('values'),'Embedded gauges differ from raw Prometheus')


def analyze(dirs):
    ledger={};stages=[];experiments=[];errors=[]
    def read_bytes(p):
        p=p.resolve();raw=p.read_bytes();info=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        need(str(p) not in ledger or ledger[str(p)]==info,'Evidence changed between reads')
        ledger[str(p)]=info;return raw
    def read(p):
        return json.loads(read_bytes(p))
    read_bytes(Path(__file__))
    read_bytes(Path(h.__file__))
    read_bytes(Path(path_audit.__file__))
    for directory in dirs:
        d=directory.resolve()
        try:
            command=read(d/'command.json');status=read(d/'status.json');exp=read(d/'experiment.json');report=read(d/'sessions.json')
            need(status.get('exit_code') is not None and status.get('end_utc'),'Nonterminal run')
            need(command.get('experiment_id')==status.get('experiment_id')==d.name,'Supervisor experiment identity differs')
            terminal=status['exit_code']==0 and status.get('timed_out') is False and not status.get('supervisor_error') and exp['status']=='PASS' and report['status']=='COMPLETED' and not report.get('active_phase')
            terminal=terminal and bool(exp.get('clients')) and all(c.get('exit_code')==0 for c in exp['clients'])
            cfg=read(d/'config.json');plan=read(d/'session-plan.json')
            need(plan==report['plan'],'Runtime plan differs from report plan')
            with zipfile.ZipFile(d/'source-snapshot.zip') as z:
                need(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(command['source_sha256']),'Snapshot member/hash map set differs')
                for name,digest in command['source_sha256'].items():need(hashlib.sha256(z.read(name)).hexdigest()==digest,'Snapshot mismatch '+name)
                client='scenarios/scripts/agent_session_client.py'
                need(command['source_sha256'][client]==report['client_sha256'],'Runtime client mismatch')
                for helper,digest in report['helper_sha256'].items():
                    member='scripts/benchmark_client.py' if helper=='benchmark_client' else 'scenarios/scripts/'+helper+'.py'
                    need(command['source_sha256'][member]==digest,'Runtime helper mismatch '+helper)
                wire_required=b'wire_body_sha256' in z.read(client)
                tool_limit=recorded_tool_limit(z.read(client).decode())
                if 'tool_calls_per_round_limit' in report:need(type(report['tool_calls_per_round_limit'])is int and report['tool_calls_per_round_limit']==tool_limit,'Reported per-round tool bound differs from executed source')
                argv=command['command']
                need(len(argv)==6 and Path(argv[1]).name=='serve_agent_sessions.py' and argv[2]=='--config' and argv[4]=='--plan','Unexpected session launcher command')
                config_member=path_audit.snapshot_source_key(argv[3],OLD,'configs')
                if config_member is None:
                    matches=[name for name in z.namelist()
                             if name.startswith(('scenarios/configs/','experiments/scenarios/configs/'))
                             and Path(name).name==Path(argv[3]).name]
                    need(len(matches)==1,'Missing/ambiguous launch config')
                    config_member=matches[0]
                config_value=json.loads(z.read(config_member))
                if config_member.startswith('experiments/'):
                    need(status.get('path_context')==command.get('path_context'),'Launch/terminal path context differs')
                    for member in ('project_layout.py','experiments/catalog.json','project/tools/reproduce.py',
                                   'metadata/environment.json','metadata/model-inventory.json'):
                        need(member in z.namelist(),'Portable runtime source missing: '+member)
                    config_value=path_audit.expand_captured_document(config_value,command.get('path_context',{}))
                need(config_value==cfg,'Runtime config differs from launch snapshot')
                source_plans=[n for n in z.namelist() if n.startswith(('experiments/scenarios/workloads/sessions/','scenarios/workloads/sessions/')) and Path(n).name==Path(argv[5]).name]
                need(len(source_plans)==1,'Missing/ambiguous launch workload')
                original_plan=json.loads(z.read(source_plans[0]))
                if source_plans[0].startswith('experiments/'):
                    original_plan=path_audit.expand_captured_document(original_plan,command.get('path_context',{}))
                # Validation adds defaults, but cannot alter supplied fields.
                def contains(actual,expected):
                    if isinstance(expected,dict):return isinstance(actual,dict) and all(k in actual and contains(actual[k],v) for k,v in expected.items())
                    if isinstance(expected,list):return isinstance(actual,list) and len(actual)==len(expected) and all(contains(a,b) for a,b in zip(actual,expected))
                    return type(actual)is type(expected) and actual==expected
                need(contains(plan,original_plan),'Runtime plan changed supplied launch values')
                task_manifests={}
                for run in report['phases']:
                    for session in run['sessions']:
                        task_id=session.get('task_id')
                        if task_id and task_id not in task_manifests:
                            prefix=('experiments/' if config_member.startswith('experiments/') else '')+'scenarios/workloads/sessions/tasks/'+task_id+'/'
                            values={name[len(prefix):]:digest for name,digest in command['source_sha256'].items() if name.startswith(prefix)}
                            need(values,'Task fixture absent from launch snapshot')
                            task_manifests[task_id]=hashed(values)
                        if session.get('initial_state'):
                            need(session['initial_state']['source_manifest_sha256']==task_manifests[task_id],'Actual task source changed since snapshot')
            ledger[str(d/'source-snapshot.zip')]=dict(sha256=sha(d/'source-snapshot.zip'),bytes=(d/'source-snapshot.zip').stat().st_size)
            prepared=read(d/'sessions-prepared.json');need(hashed(prepared)==report['prepared_sha256'],'Prepared bank hash mismatch')
            need(prepared['tokenizer_sha256']==report['tokenizer_sha256'],'Tokenizer identity mismatch')
            need([p['phase'] for p in prepared['phases']]==plan['phases'],'Prepared phases differ from runtime plan')
            ids=[p['phase']['id'] for p in report['phases']];expected_ids=[p['id'] for p in plan['phases']]
            need(ids==expected_ids[:len(ids)] and len(ids)==len(set(ids)),'Completed phase ordering differs from plan')
            if terminal:need(ids==expected_ids,'Successful run has incomplete planned phases')
            devices,warnings=h.load_gpu(d/'gpu.csv',timezone.utc,ledger)
            experiments.append(dict(id=d.name,terminal_success=terminal,execution_status=report['status'],supervisor_exit_code=status['exit_code'],tool_calls_per_round_limit=tool_limit,git_commit=command['git_commit'],client_sha256=report['client_sha256'],prepared_sha256=report['prepared_sha256'],config=cfg))
            for run in report['phases']:
                f=run['phase'];raw=d/Path(run['raw_dir']).name
                rows=run['sessions'];recorded=[json.loads(l) for l in read_bytes(raw/'sessions.jsonl').decode().splitlines() if l]
                need(sorted(recorded,key=lambda s:s['index'])==rows,'Session JSONL mismatch')
                bank=next(p for p in prepared['phases'] if p['phase']['id']==f['id'])
                need(bank['phase']==f,'Actual phase descriptor differs from prepared phase')
                need(len(rows)==len(bank['sessions']),'Missing planned session rows')
                need([s['index'] for s in rows]==list(range(len(rows))),'Session sequence mismatch')
                wall=run['total_wall_s'];need(finite(wall) and wall>0,'Invalid phase wall')
                close(run['end_unix']-run['start_unix'],wall,'Phase UNIX/monotonic wall mismatch',.01)
                horizon=f['duration_s'] if f['arrival']=='poisson' else wall
                close(run['measurement_window_s'],horizon,'Measurement horizon differs')
                close(run['drain_s'],max(0.,wall-horizon),'Drain differs')
                if terminal:need(wall+.003>=horizon,'Successful open window did not elapse')
                for s,b in zip(rows,bank['sessions']):
                    if f['kind']=='replay':need([hashed(p) for p in b['prompts']]==b['prompt_hashes'],'Prepared prompt hashes differ')
                    check_session(s,b,f,wall,tool_limit)
                    if wire_required:need(all('wire_body' in req and 'wire_body_sha256' in req for req in s['requests']),'This runtime source requires exact HTTP wire body evidence')
                check_phase_files(run,raw,read,read_bytes)
                before=h.parse_metrics(read_bytes(raw/'before.prom').decode());after=h.parse_metrics(read_bytes(raw/'after.prom').decode())
                metrics=h.metrics_delta(before,after);gpu=h.integrate_gpu(devices,run['start_unix'],run['end_unix'],[0,1,2,3],2.5)
                summary=summarize(rows,run['total_wall_s'])
                summary['gpu_j_per_successful_task']=ratio(gpu['energy_j'],summary['successful_tasks']) if gpu['energy_j'] is not None else None
                gauges=defaultdict(list)
                for m in run['metric_samples']:
                    for name in ('vllm:kv_cache_usage_perc','vllm:num_requests_running','vllm:num_requests_waiting'):
                        if m.get('error'):continue
                        values=named_values(m,name)
                        if values:gauges[name].append(max(values) if name.endswith('_perc') else sum(values))
                cohorts={}
                for kind in sorted({s.get('profile') or s.get('task_id') for s in rows}):
                    chosen=[s for s in rows if (s.get('profile') or s.get('task_id'))==kind];cohorts[kind]=summarize(chosen,run['total_wall_s'])
                turns={}
                for turn in sorted({r['turn'] for s in rows for r in s['requests']}):
                    calls=[r for s in rows for r in s['requests'] if r['turn']==turn and r['ok']]
                    turns[str(turn)]=dict(calls=len(calls),prompt_tokens=dist([r['usage']['prompt_tokens'] for r in calls]),
                                         cached_tokens=dist([r.get('cached_tokens') for r in calls]),first_output_s=dist([r['first_output_s'] for r in calls]),e2e_s=dist([r['e2e_s'] for r in calls]))
                identity=workload_identity(bank,report['tokenizer_sha256'])
                included=terminal and f['role'] in ('measure','screen','confirm')
                stages.append(dict(experiment=d.name,phase=f,terminal_success=terminal,summary=summary,by_workload=cohorts,by_turn=turns,
                                   included_as_measurement=included,exclusion_reason=None if included else 'Nonterminal/failed experiment or role excluded from normal performance: '+f['role'],
                                   arrival_workload_sha256=identity['sha256'],arrival_workload_identity=identity,prepared_phase_sha256=hashed(bank),
                                   treatment={key:f.get(key) for key in ('cache_scope','dp_routing')},
                                   total_wall_s=run['total_wall_s'],measurement_window_s=run['measurement_window_s'],drain_s=run['drain_s'],
                                   stability=stability(dict(run,owned_run_terminal_success=terminal)),metrics=metrics,gpu=gpu,gpu_warnings=warnings,gauges={k:dist(v) for k,v in gauges.items()}))
        except Exception as exc:errors.append(dict(experiment=d.name,error=repr(exc)))
    for p,r in ledger.items():need(sha(Path(p))==r['sha256'],'Evidence changed during analysis')
    return dict(schema_version=2,status='PASS' if not errors else 'FAIL',utc=datetime.now(timezone.utc).isoformat(),
                analyzer_sha256=sha(Path(__file__)),historical_metric_helper_sha256=sha(Path(h.__file__)),
                experiments=experiments,stages=stages,input_ledger=ledger,errors=errors,
                session_slo=None,limitations=['Fixed token replay has no task success grade.','Dynamic tasks may produce different histories/call counts.','No Session latency SLO; task success/s is quality-qualified throughput.','GPU energy is sampled board energy, not whole-system or per-session attribution.',
                    'Warmup, Profile, pilot and failed runs are preserved but excluded from normal-performance tables.',
                    'Source/stream/log consistency does not independently rerun task grading or prove model quality.',
                    'No repeated-rate capacity aggregation is implemented here; single-window STABLE_IN_WINDOW is not a confirmed maximum.',
                    'Known token costs include failed attempts when usage is available; missing failed-call usage is explicitly incomplete.'])


def fmt(v):return '—' if v is None else f'{v:.3f}'
def markdown(r):
    lines=['# 阶段 C Session 离线复算','',f"一致性：{r['status']}；{len(r['experiments'])}实验、{len(r['stages'])}phase。所有吞吐分母包含drain；回放无task成功率，闭环不假设相同工作量。",'',
           '| experiment / phase | sessions | success tasks | completed session/s | success task/s | Session P50/P95/P99 s | cache fraction | output tok/s | stability |',
           '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |']
    for s in r['stages']:
        if not s['included_as_measurement']:continue
        x=s['summary'];q=x['session_latency_s']
        lines.append(f"| {s['experiment']} / {s['phase']['id']} | {x['sessions']} | {x['successful_tasks']} | {fmt(x['completed_sessions_s'])} | {fmt(x['successful_tasks_s'])} | {' / '.join(fmt(q[k]) for k in ('p50','p95','p99'))} | {fmt(x['cache_fraction'])} | {fmt(x['output_tokens_s'])} | {s['stability']['classification']} |")
    lines+=['','## 按任务/轨迹类型','', '| experiment / phase / type | N | success | calls mean | tools mean | Session P95 s | model P50 s | tools P50 s |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in r['stages']:
        if not s['included_as_measurement']:continue
        for name,x in s['by_workload'].items():
            lines.append(f"| {s['experiment']} / {s['phase']['id']} / {name} | {x['sessions']} | {x['successful_tasks']} | {fmt(x['model_calls']/x['sessions'])} | {fmt(x['tool_calls']/x['sessions'])} | {fmt(x['session_latency_s']['p95'])} | {fmt(x['model_wall_s']['p50'])} | {fmt(x['tool_wall_s']['p50'])} |")
    lines+=['','## 保留但不进入正常性能表','']
    for s in r['stages']:
        if not s['included_as_measurement']:lines.append(f"- {s['experiment']} / {s['phase']['id']}: {s['exclusion_reason']}; {s['summary']['sessions']} sessions，协议完成{s['summary']['protocol_completed']}，任务成功{s['summary']['successful_tasks']}。")
    lines+=['','详细request/turn、缓存计数、GPU能量/利用率/显存、稳定性各子窗和全部输入SHA见同名JSON。失败/预算/截断/质量错误不可从所属cohort分母删除。','']
    return '\n'.join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result-dirs', nargs='+', type=Path, required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    a = p.parse_args()
    j = a.output_prefix.with_suffix('.json')
    m = a.output_prefix.with_suffix('.md')
    need(not j.exists() and (not m.exists()), 'Use new output prefix')
    result = analyze(a.result_dirs)
    j.parent.mkdir(exist_ok=True, parents=True)
    write_text(j, json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    write_text(m, markdown(result))
    print(json.dumps(dict(status=result['status'], phases=len(result['stages']), errors=result['errors'])))
    if result['errors']:
        raise SystemExit(1)
if __name__=='__main__':main()
