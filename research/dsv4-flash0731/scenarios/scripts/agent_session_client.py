"""Bounded Agent Session benchmark: exact token replay and real tool feedback.

Owned localhost only. Fresh connections, no retries, complete per-call evidence.
Replay never claims task success; closed-loop never assumes matching token counts.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import aiohttp
import capacity_client as cap

ROOT = Path(__file__).resolve().parents[1]
MAX_TOOL_CALLS_PER_ROUND = 8
PROFILES = {
    'coding': dict(inputs=[8192, 12288, 16384, 20480], outputs=[128, 256, 128, 256], waits=[.1,.2,.1]),
    'research': dict(inputs=[16384, 24576, 32768, 40960], outputs=[128,128,128,512], waits=[.5,1.,.5]),
    'data': dict(inputs=[4096,6144,8192,10240], outputs=[128,256,128,256], waits=[.1,.5,.1]),
}

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low < value <= high


def validate_plan(value):
    if not isinstance(value, dict):
        raise ValueError('Plan must be an object')
    p = copy.deepcopy(value)
    if p.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    p['base_url'] = cap.base_url(p.get('base_url', 'http://127.0.0.1:8260'))
    p.setdefault('model', 'dsv4')
    p.setdefault('tokenizer_path', str(model_path()))
    p.setdefault('timeout_s', 180)
    p.setdefault('session_timeout_s', 600)
    p.setdefault('max_active_sessions', 128)
    if p['model'] != 'dsv4' or not _number(p['timeout_s'], 0, 600) or (not _number(p['session_timeout_s'], 0, 1800)):
        raise ValueError('Invalid model/time bounds')
    if not _integer(p['max_active_sessions'], 1, 256):
        raise ValueError('Invalid protection limit')
    if not isinstance(p.get('phases'), list) or not 1 <= len(p['phases']) <= 100:
        raise ValueError('Invalid phases')
    seen = set()
    for f in p['phases']:
        if not isinstance(f, dict):
            raise ValueError('Phase must be an object')
        n = f.get('id', '')
        if not isinstance(n, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', n) or n in seen:
            raise ValueError('Invalid/duplicate phase ID')
        seen.add(n)
        if f.get('kind') not in ('replay', 'agent') or f.get('arrival') not in ('closed', 'poisson'):
            raise ValueError('Invalid workload/arrival')
        f.setdefault('seed', 20261106)
        f.setdefault('role', 'measure')
        f.setdefault('cache_scope', 'session')
        f.setdefault('dp_routing', 'none')
        f.setdefault('max_turns', 12)
        f.setdefault('max_tools', 32)
        f.setdefault('max_output_tokens', 768)
        if not _integer(f['seed'], 0, 2 ** 63 - 257) or f['role'] not in ('warmup', 'measure', 'profile', 'pilot', 'screen', 'confirm'):
            raise ValueError('Invalid seed/role')
        if f['cache_scope'] not in ('none', 'session', 'shared') or f['dp_routing'] not in ('none', 'sticky4'):
            raise ValueError('Invalid cache/routing')
        if not _integer(f['max_turns'], 1, 16) or not _integer(f['max_tools'], 1, 64) or (not _integer(f['max_output_tokens'], 16, 4096)):
            raise ValueError('Unbounded agent budget')
        if f['arrival'] == 'closed':
            if not _integer(f.get('sessions'), 1, 256) or not _integer(f.get('concurrency'), 1, p['max_active_sessions']):
                raise ValueError('Invalid closed concurrency')
            if any((k in f for k in ('duration_s', 'rate'))):
                raise ValueError('Closed phases must not declare open arrivals')
        else:
            if not _number(f.get('duration_s'), 0, 1200) or not _number(f.get('rate'), 0, 8):
                raise ValueError('Invalid Poisson bounds')
            if any((k in f for k in ('sessions', 'concurrency'))):
                raise ValueError('Poisson phases must not have closed-loop limits')
        if f['kind'] == 'replay':
            if not isinstance(f.get('profiles'), list) or not f['profiles'] or any((not isinstance(n, str) or n not in PROFILES for n in f['profiles'])):
                raise ValueError('Unknown replay profile')
        else:
            import agent_session_tasks as tasks
            allowed = {t['id'] for t in tasks.task_catalog()}
            if not isinstance(f.get('tasks'), list) or not f['tasks'] or any((not isinstance(n, str) or n not in allowed for n in f['tasks'])):
                raise ValueError('Unknown task')
    if not seen:
        raise ValueError('No phases')
    return p


def arrivals(f):
    if f['arrival']=='closed':return [None]*f['sessions']
    rng=random.Random(f['seed']);t=0.;out=[]
    while True:
        t+=rng.expovariate(f['rate'])
        if t>=f['duration_s']:return out
        out.append(t)
        if len(out)>2000:raise ValueError('Too many planned sessions')


def prepare(plan):
    from tokenizers import Tokenizer
    path=Path(plan['tokenizer_path']).expanduser().resolve()/'tokenizer.json';tok=Tokenizer.from_file(str(path));tok.no_padding();tok.no_truncation()
    phases=[]
    for f in plan['phases']:
        rows=[]
        for i,offset in enumerate(arrivals(f)):
            row=dict(index=i,planned_offset_s=offset,seed=f['seed']+i)
            if f['kind']=='replay':
                name=f['profiles'][i%len(f['profiles'])];spec=PROFILES[name]
                # Same public preamble across sessions; unique suffix/history per session.
                shared=cap.base.make_prompt(tok,dict(input_tokens=2048,name='shared-agent-context'), f['seed'],0,'agent-shared')
                tail=cap.base.make_prompt(tok,dict(input_tokens=max(spec['inputs'])-2048,name='session-context'),f['seed'],i,'agent-'+name)
                full=shared+tail
                prompts=[full[:n] for n in spec['inputs']]
                row.update(profile=name,prompts=prompts,prompt_hashes=[digest(v) for v in prompts],outputs=spec['outputs'],tool_waits_s=spec['waits'])
            else:row['task_id']=f['tasks'][i%len(f['tasks'])]
            rows.append(row)
        phases.append(dict(phase=f,sessions=rows))
    return dict(schema_version=1,tokenizer_sha256=cap.sha_file(path),phases=phases)


def salt_for(f,index,turn):
    label='c-'+digest([f['seed'],f['id']])[:24]
    if f['cache_scope']=='shared':return label
    label+=f'-s{index}'
    return label+f'-t{turn}' if f['cache_scope']=='none' else label


async def stream_request(http,url,payload,origin,request_id,timeout,chat=False,expected=None,headers=None,record_callback=None):
    start=time.perf_counter();parts=[];reasoning=[];calls={};events=[];raw_events=[];stream_bytes=0
    # Exact wire order matters to tool-schema rendering. Sorted semantic hashes
    # and sorted checkpoint JSON alone cannot reconstruct this HTTP body.
    wire_body=json.dumps(payload,ensure_ascii=True,allow_nan=False)
    wire_headers={**(headers or {}),'Content-Type':'application/json'}
    r=dict(request_id=request_id,ok=False,payload=payload,payload_sha256=digest(payload),headers=wire_headers,
           wire_body=wire_body,wire_body_sha256=hashlib.sha256(wire_body.encode()).hexdigest(),
           dispatch_offset_s=start-origin,dispatch_unix=time.time(),ready_offset_s=start-origin,
           first_output_s=None,ttft_s=None,ttft_tool_s=None,e2e_s=None,tpot_s=None,
           usage=None,finish_reason=None,done_received=False,http_status=None,error=None,timed_out=False,cancelled=False,
           attempts=1,raw_stream_truncated=False,tool_arguments_ready_s=None)
    try:
        async with http.post(url+('/v1/chat/completions' if chat else '/v1/completions'),data=wire_body.encode(),headers=wire_headers,
                             allow_redirects=False,timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            r['http_status']=response.status
            if response.status!=200:
                r['error_body']=(await response.content.read(8192)).decode(errors='replace')
                raise RuntimeError(f'HTTP {response.status}: '+r['error_body'])
            if 'text/event-stream' not in response.headers.get('Content-Type',''):raise ValueError('Missing SSE content type')
            async for event in cap.base.sse_events(response.content):
                t=time.perf_counter()-start
                stream_bytes+=len(event.encode())
                if stream_bytes>8*1024*1024:
                    r['raw_stream_truncated']=True
                    raise ValueError('Bounded stream bytes exceeded')
                raw_events.append(dict(elapsed_s=t,data=event))
                if event.strip()=='[DONE]':r.update(done_received=True,e2e_s=t);break
                obj=json.loads(event)
                if not isinstance(obj,dict) or obj.get('error'):raise ValueError('Invalid/error streaming object')
                if obj.get('usage') is not None:r['usage']=obj['usage']
                events.append(dict(elapsed_s=t,data=obj))
                if len(events)>20000:raise ValueError('Bounded stream event count exceeded')
                choices=obj.get('choices',[])
                if not isinstance(choices,list) or len(choices)>1:raise ValueError('Expected at most one choice')
                for c in choices:
                    if not isinstance(c,dict):raise ValueError('Malformed choice')
                    if c.get('index',0)!=0:raise ValueError('Unexpected choice index')
                    delta=c.get('delta',{}) if chat else dict(content=c.get('text'))
                    if not isinstance(delta,dict):raise ValueError('Malformed delta')
                    text=delta.get('content') or ''
                    if not isinstance(text,str):raise ValueError('Nonstring content')
                    if text:
                        parts.append(text)
                        if r['ttft_s'] is None:r['ttft_s']=t
                        if r['first_output_s'] is None:r['first_output_s']=t
                    reason=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if not isinstance(reason,str):raise ValueError('Nonstring reasoning')
                    if reason:
                        reasoning.append(reason)
                        if r['first_output_s'] is None:r['first_output_s']=t
                    tool_deltas=delta.get('tool_calls') or []
                    if not isinstance(tool_deltas,list):raise ValueError('Malformed tool deltas')
                    for tc in tool_deltas:
                        if not isinstance(tc,dict) or tc.get('type','function')!='function':raise ValueError('Malformed tool delta')
                        ix=tc.get('index')
                        if type(ix)is not int or not 0<=ix<MAX_TOOL_CALLS_PER_ROUND:raise ValueError('Tool index outside per-round bound')
                        dst=calls.setdefault(ix,dict(id='',type='function',function=dict(name='',arguments='')))
                        if tc.get('id'):
                            if not isinstance(tc['id'],str):raise ValueError('Nonstring tool ID')
                            dst['id']+=tc['id']
                        fun=tc.get('function') or {}
                        if not isinstance(fun,dict):raise ValueError('Malformed tool function')
                        for key in ('name','arguments'):
                            val=fun.get(key) or ''
                            if not isinstance(val,str):raise ValueError('Nonstring tool delta')
                            dst['function'][key]+=val
                        if fun.get('name') or fun.get('arguments'):
                            if r['ttft_tool_s'] is None:r['ttft_tool_s']=t
                            if r['first_output_s'] is None:r['first_output_s']=t
                    if c.get('finish_reason') is not None:r['finish_reason']=c['finish_reason']
            u=r['usage']
            if not r['done_received'] or not isinstance(u,dict) or r['first_output_s'] is None:raise ValueError('Incomplete stream, usage or generated output')
            a,b=u.get('prompt_tokens'),u.get('completion_tokens')
            if type(a)is not int or type(b)is not int or type(u.get('total_tokens'))is not int or a<1 or not 1<=b<=payload['max_tokens'] or u.get('total_tokens')!=a+b:raise ValueError('Invalid usage')
            if expected is not None and (a!=expected or b!=payload['max_tokens'] or r['finish_reason']!='length'):raise ValueError('Exact replay token contract violated')
            if r['finish_reason'] not in ('stop','length','tool_calls'):raise ValueError('Unknown terminal reason')
            if calls:
                if r['finish_reason']!='tool_calls':raise ValueError('Tool deltas without terminal tool_calls')
                if sorted(calls)!=list(range(len(calls))):raise ValueError('Tool indexes not contiguous')
                if len({call['id'] for call in calls.values()})!=len(calls):raise ValueError('Duplicate tool IDs')
                for call in calls.values():
                    if not call['id'] or not call['function']['name'] or not isinstance(json.loads(call['function']['arguments']),dict):raise ValueError('Incomplete tool call')
                r['tool_arguments_ready_s']=r['e2e_s']
            elif r['finish_reason']=='tool_calls':raise ValueError('Missing tool deltas')
            r['input_tokens']=a;r['output_tokens']=b
            detail=u.get('prompt_tokens_details')
            if detail is not None and not isinstance(detail,dict):raise ValueError('Malformed prompt_tokens_details')
            r['cached_tokens']=(detail or {}).get('cached_tokens')
            if r['cached_tokens'] is not None and (type(r['cached_tokens'])is not int or not 0<=r['cached_tokens']<=a):raise ValueError('Invalid cached_tokens')
            r['tpot_output_s']=(r['e2e_s']-r['first_output_s'])/(b-1) if b>1 else None
            # Existing text TTFT/SLO only on text-only rounds. Tool delta latency is separate.
            r['slo_text_eligible']=not calls and not reasoning and r['ttft_s'] is not None
            r['tpot_s']=(r['e2e_s']-r['ttft_s'])/(b-1) if b>1 and r['slo_text_eligible'] else None
            r['ok']=True
            if r['slo_text_eligible']:cap.add_slo(r,'closed-loop')
        r['slo_pass']=r.get('slo_pass') if r.get('slo_text_eligible') else None
    except asyncio.CancelledError:
        r.update(cancelled=True,error=dict(type='CancelledError',message='Owning session cancelled request'))
        raise
    except asyncio.TimeoutError as exc:r.update(timed_out=True,error=dict(type='TimeoutError',message=str(exc)))
    except Exception as exc:r.update(ok=False,error=dict(type=type(exc).__name__,message=str(exc)))
    finally:
        r.update(end_offset_s=time.perf_counter()-origin,end_unix=time.time(),elapsed_until_return_s=time.perf_counter()-start,
                 full_text=''.join(parts),reasoning=''.join(reasoning),tool_calls=[calls[k] for k in sorted(calls)],events=events,raw_events=raw_events)
        if record_callback is not None:record_callback(r)
    return r


def phase_summary(rows,wall):
    requests=[r for s in rows for r in s['requests']];ok=[r for r in requests if r['ok']]
    passed=[s for s in rows if s.get('task_success') is True]
    return dict(sessions=len(rows),completed=sum(s['protocol_ok'] for s in rows),task_graded=sum(s.get('task_success') is not None for s in rows),
                successful_tasks=len(passed),session_completion_rate=sum(s['protocol_ok'] for s in rows)/len(rows) if rows else None,
                task_success_rate=len(passed)/len(rows) if rows and any(s.get('task_success') is not None for s in rows) else None,
                protocol_complete_sessions_s=sum(s['protocol_ok'] for s in rows)/wall,successful_tasks_s=len(passed)/wall if passed or any(s.get('task_success') is not None for s in rows) else None,
                requests=len(requests),successful_requests=len(ok),output_tokens=sum(r['usage']['completion_tokens'] for r in ok),
                prompt_tokens=sum(r['usage']['prompt_tokens'] for r in ok),cached_tokens=sum(r.get('cached_tokens') or 0 for r in ok),
                cached_usage_available=sum(r.get('cached_tokens') is not None for r in ok),
                output_tokens_s=sum(r['usage']['completion_tokens'] for r in ok)/wall,
                session_latency_s=cap.stats([s.get('scheduled_latency_s') for s in rows if s['protocol_ok']]),
                successful_task_latency_s=cap.stats([s['scheduled_latency_s'] for s in passed]),
                request_ttft_text_s=cap.stats([r['ttft_s'] for r in ok]),request_first_output_s=cap.stats([r['first_output_s'] for r in ok]),
                request_e2e_s=cap.stats([r['e2e_s'] for r in ok]),request_tpot_output_s=cap.stats([r.get('tpot_output_s') for r in ok]))


async def execute_phase(item, plan, out):
    f = item['phase']
    raw = out.parent / (out.stem + '-' + f['id'])
    raw.mkdir(exist_ok=False)
    events = open_output(raw / 'events.jsonl', 'x')
    sessions_file = open_output(raw / 'sessions.jsonl', 'x')

    def emit(x):
        events.write(json.dumps(x, ensure_ascii=False) + '\n')
        events.flush()
    connector = aiohttp.TCPConnector(force_close=True, limit=0)
    rows = []
    jobs = set()
    active = set()
    peak = 0
    phase_error = None
    protection = False
    try:
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as http:
            async with http.post(plan['base_url'] + '/reset_prefix_cache?reset_running_requests=false', allow_redirects=False, timeout=aiohttp.ClientTimeout(total=plan['timeout_s'])) as r:
                control = dict(http_status=r.status, response=await r.json())
                write_text(raw / 'reset.json', json.dumps(control, indent=2))
                if r.status != 200 or not isinstance(control['response'], dict) or control['response'].get('success') is not True:
                    raise RuntimeError('Cache reset failed')
            before = await cap.metric_snapshot(http, plan['base_url'], raw / 'before.prom')
            origin = time.perf_counter()
            start_unix = time.time()
            stop = asyncio.Event()
            samples = []
            monitor = asyncio.create_task(cap.poll_metrics(http, plan['base_url'], origin, stop, 1.0, raw, samples))

            def base_row(spec, admitted):
                begin = time.perf_counter()
                scheduled = spec['planned_offset_s']
                if scheduled is None:
                    scheduled = begin - origin
                return dict(index=spec['index'], kind=f['kind'], profile=spec.get('profile'), task_id=spec.get('task_id'), planned_offset_s=scheduled, start_offset_s=begin - origin if admitted else None, dispatch_lag_s=max(0.0, begin - origin - scheduled) if admitted else None, admitted=admitted, censored=not admitted, requests=[], tools=[], host_operations=[], task_success=None, protocol_ok=False, terminal_reason=None, grade=None)

            def write_row(sr):
                rows.append(sr)
                sessions_file.write(json.dumps(sr, ensure_ascii=False) + '\n')
                sessions_file.flush()
                emit(dict(event='session_end', session_index=sr['index'], terminal_reason=sr['terminal_reason'], offset_s=sr['end_offset_s']))

            async def one(spec):
                i = spec['index']
                begin = time.perf_counter()
                sr = base_row(spec, True)
                state = None
                emit(dict(event='session_start', session_index=i, offset_s=sr['start_offset_s'], planned_offset_s=sr['planned_offset_s']))

                def record_request(req, turn):
                    req.update(session_index=i, turn=turn)
                    sr['requests'].append(req)
                    emit(dict(event='model_request', session_index=i, request=req))

                def save_state(label):
                    if state is not None:
                        sr[label] = copy.deepcopy(state)
                        cap.save_json(raw / f'session-{i:04d}' / 'task-state.json', dict(initial=sr.get('initial_state'), current=state))

                async def host_call(name, func, *args, on_complete=None):
                    op = dict(name=name, ready_offset_s=time.perf_counter() - origin, cancelled_while_waiting=False)

                    def invoke():
                        op['worker_start_offset_s'] = time.perf_counter() - origin
                        try:
                            return func(*args)
                        finally:
                            op['worker_end_offset_s'] = time.perf_counter() - origin
                    job = asyncio.create_task(asyncio.to_thread(invoke))
                    value = None
                    completed = False
                    try:
                        value = await asyncio.shield(job)
                        completed = True
                    except asyncio.CancelledError:
                        op['cancelled_while_waiting'] = True
                        op['cleanup_start_offset_s'] = time.perf_counter() - origin
                        try:
                            value = await asyncio.shield(job)
                            completed = True
                        finally:
                            op['cleanup_end_offset_s'] = time.perf_counter() - origin
                        raise
                    finally:
                        op['end_offset_s'] = time.perf_counter() - origin
                        op['queue_s'] = op.get('worker_start_offset_s', op['end_offset_s']) - op['ready_offset_s']
                        op['worker_s'] = op.get('worker_end_offset_s', op['end_offset_s']) - op.get('worker_start_offset_s', op['end_offset_s'])
                        sr['host_operations'].append(op)
                        if completed and on_complete is not None:
                            on_complete(value)
                        emit(dict(event='host_operation', session_index=i, operation=op))
                    return value

                async def body():
                    nonlocal state
                    if f['kind'] == 'agent':
                        import agent_session_tasks as tasks

                        def created(value):
                            nonlocal state
                            state = value
                            save_state('initial_state')
                        await host_call('create_session', tasks.create_session, spec['task_id'], raw / f'session-{i:04d}', on_complete=created)
                        messages = tasks.initial_messages(state)
                        schemas = tasks.tools_for(state)
                        rounds = f['max_turns']
                        final = None
                    else:
                        rounds = len(spec['prompts'])
                    for turn in range(rounds):
                        common = dict(model=plan['model'], temperature=0, seed=spec['seed'], stream=True, stream_options=dict(include_usage=True), cache_salt=salt_for(f, i, turn))
                        if f['kind'] == 'replay':
                            payload = dict(**common, prompt=spec['prompts'][turn], max_tokens=spec['outputs'][turn], ignore_eos=True, add_special_tokens=False)
                        else:
                            payload = dict(**common, messages=copy.deepcopy(messages), tools=schemas, tool_choice='auto', max_tokens=f['max_output_tokens'], reasoning_effort='none', chat_template_kwargs=dict(enable_thinking=False))
                        headers = {'X-data-parallel-rank': str(i % 4)} if f['dp_routing'] == 'sticky4' else None
                        req = await stream_request(http, plan['base_url'], payload, origin, f'{f['id']}-s{i}-t{turn}', plan['timeout_s'], chat=f['kind'] == 'agent', expected=len(spec['prompts'][turn]) if f['kind'] == 'replay' else None, headers=headers, record_callback=lambda row, t=turn: record_request(row, t))
                        if not req['ok']:
                            sr['terminal_reason'] = 'request_error'
                            return
                        if f['kind'] == 'replay':
                            if turn < len(spec['tool_waits_s']):
                                started = time.perf_counter()
                                tool = dict(synthetic_wait=True, planned_s=spec['tool_waits_s'][turn], start_offset_s=started - origin, completed=False)
                                try:
                                    await asyncio.sleep(spec['tool_waits_s'][turn])
                                    tool['completed'] = True
                                finally:
                                    tool.update(wall_s=time.perf_counter() - started, end_offset_s=time.perf_counter() - origin)
                                    sr['tools'].append(tool)
                                    emit(dict(event='synthetic_tool_wait', session_index=i, tool=tool))
                        else:
                            calls = req['tool_calls']
                            msg = dict(role='assistant', content=req['full_text'] or None)
                            if calls:
                                msg['tool_calls'] = calls
                            if req['reasoning']:
                                msg['reasoning_content'] = req['reasoning']
                            messages.append(msg)
                            if calls:
                                if len(sr['tools']) + len(calls) > f['max_tools']:
                                    sr['terminal_reason'] = 'tool_budget'
                                    return
                                for call in calls:

                                    def record_tool(value):
                                        sr['tools'].append(value)
                                        save_state('current_state')
                                        emit(dict(event='tool', session_index=i, tool=value))
                                    tool = await host_call('execute_tool', tasks.execute_tool, state, call, on_complete=record_tool)
                                    messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(tool['result'], ensure_ascii=False)))
                            elif req['finish_reason'] == 'stop':
                                final = req['full_text']
                                sr['user_visible_end_offset_s'] = time.perf_counter() - origin
                                break
                            else:
                                sr['terminal_reason'] = 'output_truncated'
                                return
                    if f['kind'] == 'replay':
                        sr.update(protocol_ok=True, terminal_reason='replay_complete', user_visible_end_offset_s=time.perf_counter() - origin)
                        return
                    if final is None:
                        sr['terminal_reason'] = 'turn_budget'
                        return

                    def graded(value):
                        sr['grade'] = value
                    await host_call('grade', tasks.grade, state, final, on_complete=graded)
                    if not isinstance(sr['grade'], dict) or type(sr['grade'].get('success')) is not bool:
                        raise ValueError('Invalid task grading result')
                    sr['task_success'] = sr['grade']['success']
                    sr['protocol_ok'] = True
                    sr['terminal_reason'] = 'natural_stop'
                    sr['final_text'] = final
                try:
                    await asyncio.wait_for(body(), timeout=plan['session_timeout_s'])
                except asyncio.TimeoutError:
                    sr['terminal_reason'] = 'session_timeout'
                except asyncio.CancelledError:
                    sr['terminal_reason'] = 'cancelled'
                    raise
                except Exception as exc:
                    sr.update(terminal_reason='client_error', error=repr(exc))
                finally:
                    ended = time.perf_counter()
                    save_state('final_state')
                    sr.update(end_offset_s=ended - origin, elapsed_s=ended - begin, scheduled_latency_s=ended - origin - sr['planned_offset_s'])
                    sr['user_visible_latency_s'] = sr['user_visible_end_offset_s'] - sr['planned_offset_s'] if 'user_visible_end_offset_s' in sr else None
                    sr['model_wall_s'] = sum((r['elapsed_until_return_s'] for r in sr['requests']))
                    sr['tool_wall_s'] = sum((t['wall_s'] for t in sr['tools']))
                    sr['grading_wall_s'] = (sr.get('grade') or {}).get('wall_s', 0.0)
                    write_row(sr)
            try:
                if f['arrival'] == 'closed':
                    pending = iter(item['sessions'])

                    async def worker():
                        for spec in pending:
                            await one(spec)
                    jobs = {asyncio.create_task(worker()) for _ in range(f['concurrency'])}
                    peak = min(f['concurrency'], len(item['sessions']))
                    await asyncio.gather(*jobs)
                else:
                    for spec in item['sessions']:
                        await asyncio.sleep(max(0.0, origin + spec['planned_offset_s'] - time.perf_counter()))
                        active = {t for t in active if not t.done()}
                        if len(active) >= plan['max_active_sessions']:
                            protection = True
                            phase_error = 'Active-session protection triggered; remaining planned sessions censored'
                            break
                        task = asyncio.create_task(one(spec))
                        active.add(task)
                        jobs.add(task)
                        peak = max(peak, len(active))
                    if not protection:
                        await asyncio.sleep(max(0.0, origin + f['duration_s'] - time.perf_counter()))
                        if jobs:
                            await asyncio.gather(*jobs)
            except Exception as exc:
                phase_error = repr(exc)
            finally:
                for task in jobs:
                    if not task.done():
                        task.cancel()
                if jobs:
                    await asyncio.gather(*jobs, return_exceptions=True)
                finished = {row['index'] for row in rows}
                for spec in item['sessions']:
                    if spec['index'] not in finished:
                        sr = base_row(spec, False)
                        ended = time.perf_counter() - origin
                        sr.update(terminal_reason='protection_censored' if protection else 'phase_aborted', end_offset_s=ended, elapsed_s=None, scheduled_latency_s=None, model_wall_s=0.0, tool_wall_s=0.0, grading_wall_s=0.0)
                        write_row(sr)
                stop.set()
                await monitor
            end = time.perf_counter()
            end_unix = time.time()
            after = await cap.metric_snapshot(http, plan['base_url'], raw / 'after.prom')
    finally:
        events.close()
        sessions_file.close()
    wall = end - origin
    return dict(phase=f, status='FAILED' if phase_error else 'COMPLETED', error=phase_error, protection_triggered=protection, start_unix=start_unix, end_unix=end_unix, total_wall_s=wall, measurement_window_s=f.get('duration_s', wall), drain_s=max(0.0, wall - f.get('duration_s', wall)), planned_sessions=len(item['sessions']), peak_active_sessions=peak, raw_dir=str(raw), sessions=sorted(rows, key=lambda x: x['index']), metrics_before=before, metrics_after=after, metric_samples=samples, summary=phase_summary(rows, wall))


async def run(plan,prepared,out):
    import sys
    sources={'capacity_client':cap.sha_file(cap.__file__),'benchmark_client':cap.sha_file(cap.base.__file__)}
    if any(item['phase']['kind']=='agent' for item in prepared['phases']):
        import agent_session_tasks as tasks
        sources['agent_session_tasks']=cap.sha_file(tasks.__file__)
    report=dict(schema_version=1,status='RUNNING',plan=plan,prepared_sha256=digest(prepared),tokenizer_sha256=prepared['tokenizer_sha256'],
                client_sha256=cap.sha_file(__file__),helper_sha256=sources,python=sys.version,aiohttp=aiohttp.__version__,
                transport='fresh_tcp_no_retry',tool_calls_per_round_limit=MAX_TOOL_CALLS_PER_ROUND,
                session_latency_slo=None,started_utc=cap.utc_now(),phases=[])
    cap.save_json(out,report)
    try:
        for item in prepared['phases']:
            report['active_phase']=item['phase']['id'];cap.save_json(out,report)
            phase=await execute_phase(item,plan,out);report['phases'].append(phase);cap.save_json(out,report)
            print(json.dumps(dict(phase=phase['phase']['id'],summary=phase['summary'])),flush=True)
            if phase['status']=='FAILED' or any(s['terminal_reason'] in ('request_error','client_error','session_timeout','cancelled') for s in phase['sessions']):raise RuntimeError('Protocol/runtime failure; no later phases executed')
        report['status']='COMPLETED';report.pop('active_phase',None)
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:report['finished_utc']=cap.utc_now();cap.save_json(out,report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    plan = validate_plan(load_document(args.plan))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with open_output(args.output, 'x') as f:
        f.write('{}\n')
    prepared = prepare(plan)
    bank = args.output.with_name(args.output.stem + '-prepared.json')
    with open_output(bank, 'x') as f:
        json.dump(prepared, f, ensure_ascii=False)
    if args.prepare_only:
        cap.save_json(args.output, dict(status='PREPARED', prepared=str(bank), sha256=cap.sha_file(bank)))
        return
    asyncio.run(run(plan, prepared, args.output))

if __name__=='__main__':main()
