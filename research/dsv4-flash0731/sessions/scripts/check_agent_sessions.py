#!/usr/bin/env python3
"""CPU-only localhost SSE fixtures. No GPU, model HTTP, or benchmark results.

Runs a temporary aiohttp server on an OS-selected loopback port. Agent tools
are explicitly simulated; actual task-package tests are a separate check.
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
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types

from aiohttp import web
import aiohttp

STAGE = Path(__file__).resolve().parents[1]
RUNTIME = STAGE.parent/'scenarios/scripts'
sys.path.insert(0, str(RUNTIME))
import agent_session_client as client


def require(value, detail):
    if not value:
        raise AssertionError(detail)


class FixtureServer:
    def __init__(self):
        self.received = []
        self.delay = .003
        self.reset_success = True

    async def reset(self, request):
        return web.json_response({'success': self.reset_success})

    async def metrics(self, request):
        return web.Response(text='vllm:num_requests_waiting 0\nvllm:num_requests_running 0\nvllm:kv_cache_usage_perc 0\n')

    async def completion(self, request):
        body = await request.text()
        data = json.loads(body)
        case = data.get('fixture_case', 'normal')
        self.received.append(dict(path=request.path, payload=data, wire_body=body,peer=request.transport.get_extra_info('peername'),
                                  time=time.perf_counter(), headers=dict(request.headers)))
        if case == 'http500':
            return web.Response(status=500, text='OFFLINE_SIMULATION_FAILURE')
        if case == 'redirect':
            return web.Response(status=307, headers={'Location': '/v1/completions'})
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        chat = request.path.endswith('/chat/completions')
        try:
            async def send(value):
                raw = ('data: '+(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))+'\r\n\r\n').encode()
                # Split arbitrary byte offsets, including inside UTF-8 characters.
                for start in range(0, len(raw), 11):
                    await response.write(raw[start:start+11])

            def choice(delta, reason=None):
                return {'choices': [dict(index=0, **({'delta': delta} if chat else {'text': delta.get('content')}), finish_reason=reason)]}

            tool = case in ('tool', 'duplicate_ids', 'invalid_arguments') or case.startswith('tools_') or (chat and case == 'normal' and not any(m['role'] == 'tool' for m in data['messages']))
            if case == 'bad_json':
                await send('{malformed')
                await response.write_eof()
                return response
            if case == 'bad_reasoning':
                await send(choice({'reasoning_content': ['not', 'text']}))
            elif tool:
                count=int(case.partition('_')[2]) if case.startswith('tools_') else 1
                await send(choice({'tool_calls': [dict(index=i,id=f'call_{i+1}',type='function',function=dict(name='lookup',arguments='{"v":')) for i in range(count)]}))
                await asyncio.sleep(.002)
                await send(choice({'tool_calls': [dict(index=i,function=dict(arguments='1}' if case != 'invalid_arguments' else 'oops')) for i in range(count)]}))
                if case == 'duplicate_ids':
                    await send(choice({'tool_calls': [dict(index=1,id='call_1',type='function',function=dict(name='lookup',arguments='{}'))]}))
            else:
                await send(choice({'content': '模拟'}))
            await asyncio.sleep(.15 if case in ('timeout', 'cancel') else self.delay)
            finish = 'tool_calls' if tool else ('stop' if chat else 'length')
            await send(choice({}, finish))
            count = 12 if chat else len(data['prompt'])
            output = 3 if chat else data['max_tokens']
            usage = dict(prompt_tokens=count,completion_tokens=output,total_tokens=count+output,prompt_tokens_details={'cached_tokens':0})
            if case == 'bad_usage':usage['total_tokens'] += 1
            if case == 'bad_cached':usage['prompt_tokens_details'] = []
            if case == 'bad_cached_bool':usage['prompt_tokens_details']['cached_tokens'] = True
            await send({'choices': [], 'usage': usage})
            if case != 'missing_done':await send('[DONE]')
            await response.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response


def plan_phase(name='cpu', kind='replay', arrival='closed', **extra):
    phase=dict(id=name,kind=kind,arrival=arrival,role='measure',seed=105,cache_scope='session',dp_routing='none',
               max_turns=3,max_tools=4,max_output_tokens=32)
    phase.update(dict(sessions=4,concurrency=2) if arrival=='closed' else dict(duration_s=.15,rate=1))
    phase.update(dict(profiles=['coding']) if kind=='replay' else dict(tasks=['mock']))
    phase.update(extra)
    return phase


def spec(index, offset=None):
    return dict(index=index,seed=105+index,planned_offset_s=offset,profile='coding',
                prompts=[[1,2,3],[1,2,3,4]],outputs=[2,2],tool_waits_s=[.02])


def mock_tasks():
    module=types.ModuleType('agent_session_tasks');module.__file__=__file__
    module.delay=.002;module.success=True
    module.task_catalog=lambda: [{'id':'mock'}]
    def create(task_id, directory):
        directory.mkdir(exist_ok=False)
        return dict(task_id=task_id,workspace=str(directory),owner_token='OFFLINE_SIMULATION_NOT_MODEL',tool_sequence=0)
    module.create_session=create
    module.initial_messages=lambda state:[dict(role='user',content='OFFLINE_SIMULATION_NOT_MODEL: query then answer.')]
    module.tools_for=lambda state:[dict(type='function',function=dict(name='lookup',parameters={'type':'object'}))]
    def execute(state, call):
        start=time.perf_counter();time.sleep(module.delay);state['tool_sequence']+=1
        return dict(call=call,result={'value':1,'simulation':True},wall_s=time.perf_counter()-start,action_ok=True)
    module.execute_tool=execute
    module.grade=lambda state,text:dict(success=module.success and state['tool_sequence']==1 and text=='模拟',wall_s=0.,simulation=True)
    return module


async def exercise(directory, checks):
    server = FixtureServer()
    app = web.Application()
    app.router.add_post('/reset_prefix_cache', server.reset)
    app.router.add_get('/metrics', server.metrics)
    app.router.add_post('/v1/completions', server.completion)
    app.router.add_post('/v1/chat/completions', server.completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    module = mock_tasks()
    original = sys.modules.get('agent_session_tasks')
    sys.modules['agent_session_tasks'] = module
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True, limit=0), trust_env=False) as http:

            async def request(case='normal', chat=False, timeout=1.0, callback=None):
                payload = dict(model='dsv4', prompt=[1, 2, 3], max_tokens=8, stream=True, fixture_case=case)
                if chat:
                    payload['messages'] = [dict(role='user', content='fixture')]
                return await client.stream_request(http, url, payload, time.perf_counter(), case, timeout, chat=chat, expected=None if chat else 3, record_callback=callback)
            text = await request()
            require(text['ok'] and text['full_text'] == '模拟' and (text['output_tokens'] == 8), 'split SSE text/usage contract')
            require(text['ttft_s'] is not None and text['tpot_s'] is not None and text['slo_text_eligible'], 'text timing/SLO eligibility')
            require(text['raw_events'][-1]['data'] == '[DONE]' and len(text['events']) == 3, 'raw SSE including DONE preserved')
            require(text['wire_body'] == server.received[-1]['wire_body'] and text['wire_body_sha256'] == hashlib.sha256(text['wire_body'].encode()).hexdigest(), 'exact wire body bytes retained')
            checks.append('Split UTF-8/CRLF text SSE, actual usage, raw event/DONE evidence, text SLO')
            tool = await request('tool', True)
            require(tool['ok'] and tool['ttft_s'] is None and (tool['tpot_s'] is None) and (tool['slo_pass'] is None), 'tool-only must not require text')
            require(tool['ttft_tool_s'] is not None and tool['tool_arguments_ready_s'] == tool['e2e_s'], 'tool latency and arguments completion')
            require(tool['tool_calls'][0]['function']['arguments'] == '{"v":1}', 'fragmented arguments assembled')
            require(tool['wire_body'] == server.received[-1]['wire_body'], 'chat wire body matches bytes received')
            checks.append('Tool-only SSE, fragmented arguments, separate tool first event, no text SLO')
            for count in (5, 6, 8):
                many = await request(f'tools_{count}', True)
                require(many['ok'] and len(many['tool_calls']) == count and (many['slo_pass'] is None), 'Legal multi-tool response ' + str(count))
                checks.append('Accept ' + str(count) + ' complete tool calls per round')
            too_many = await request('tools_9', True)
            require(not too_many['ok'] and 'per-round bound' in too_many['error']['message'], 'Reject more than eight tools')
            checks.append('Reject ninth tool with partial raw response retained')
            for case in ('http500', 'redirect', 'missing_done', 'bad_json', 'bad_usage', 'bad_cached', 'bad_cached_bool', 'duplicate_ids', 'invalid_arguments', 'bad_reasoning'):
                before = len(server.received)
                row = await request(case, case in ('duplicate_ids', 'invalid_arguments', 'bad_reasoning'))
                require(not row['ok'] and row['error'] is not None, case + ' rejected')
                require(len(server.received) == before + 1 and row['attempts'] == 1, case + ' no retry/redirect')
                checks.append('Rejected ' + case + ' with failure evidence and one attempt')
            row = await request('timeout', False, 0.02)
            require(row['timed_out'] and (not row['ok']) and (row['full_text'] == '模拟'), 'request timeout preserves partial text')
            recorded = []
            task = asyncio.create_task(request('cancel', False, 1.0, recorded.append))
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            require(len(recorded) == 1 and recorded[0]['cancelled'] and (recorded[0]['full_text'] == '模拟'), 'cancel partial record callback')
            checks.append('Request deadline and cancellation preserve partial output/evidence')
            peers = [row['peer'] for row in server.received]
            require(len(set(peers)) == len(peers), 'fresh TCP per model request')
            checks.append('Real localhost peer ports confirm fresh connections')
        plan = dict(base_url=url, model='dsv4', timeout_s=1.0, session_timeout_s=2.0, max_active_sessions=8)

        async def phase(name, f, rows, overrides=None):
            result = await client.execute_phase(dict(phase=f, sessions=rows), {**plan, **(overrides or {})}, directory / (name + '.json'))
            client.cap.save_json(directory / (name + '.json'), result)
            client.cap.save_json(directory / (name + '-prepared.json'), dict(phase=f, sessions=rows))
            client.cap.save_json(directory / (name + '-plan.json'), {**plan, **(overrides or {}), 'phases': [f]})
            return result
        server.delay = 0.035
        closed = await phase('closed', plan_phase(), [spec(i) for i in range(4)])
        require(closed['status'] == 'COMPLETED' and closed['summary']['completed'] == 4 and (closed['summary']['requests'] == 8), 'closed sessions complete')
        rows = closed['sessions']
        require(max((r['start_offset_s'] for r in rows[:2])) < min((r['end_offset_s'] for r in rows[:2])), 'C2 actually overlaps')
        require(rows[2]['start_offset_s'] >= min((r['end_offset_s'] for r in rows[:2])), 'closed replacement waits for slot')
        require(all((s['requests'][1]['dispatch_offset_s'] >= s['requests'][0]['end_offset_s'] + 0.015 for s in rows)), 'causal next turn includes tool wait')
        checks.append('Closed C2 replenishment, causal turns, tool waits and complete evidence')
        opened = await phase('opened', plan_phase(arrival='poisson'), [spec(i, t) for i, t in enumerate((0.0, 0.01, 0.02, 0.1))])
        require(opened['summary']['completed'] == 4 and opened['drain_s'] > 0.015, 'open horizon plus causal drain')
        require(opened['sessions'][2]['start_offset_s'] < opened['sessions'][0]['end_offset_s'], 'open arrivals not gated on completion')
        require(opened['sessions'][-1]['requests'][-1]['dispatch_offset_s'] > 0.15, 'next turn continues after horizon')
        checks.append('Open scheduled arrivals remain open; later turns execute in drain')
        protected = await phase('protected', plan_phase(arrival='poisson'), [spec(i, t) for i, t in enumerate((0.0, 0.01, 0.08, 0.14))], {'max_active_sessions': 1})
        require(protected['status'] == 'FAILED' and protected['protection_triggered'], 'protection explicit failed phase')
        require(len(protected['sessions']) == 4 and sum((s['censored'] for s in protected['sessions'])) == 3, 'all planned censored sessions retained')
        require(protected['sessions'][0]['terminal_reason'] == 'cancelled' and protected['sessions'][0]['requests'][0]['cancelled'], 'active request/session cancellation recorded')
        require(protected['sessions'][0]['requests'][0]['raw_events'], 'partial stream retained during protection')
        checks.append('Protection records every planned session and cancels/preserves partial active work')
        server.delay = 0.002
        agent_f = plan_phase(kind='agent', sessions=2, concurrency=2)
        agent_rows = [dict(index=i, seed=105 + i, planned_offset_s=None, task_id='mock') for i in range(2)]
        agents = await phase('agents', agent_f, agent_rows)
        require(agents['summary']['successful_tasks'] == 2 and agents['summary']['requests'] == 4, 'simulated dynamic loop')
        for session in agents['sessions']:
            require(session['requests'][0]['ttft_s'] is None and session['requests'][1]['payload']['messages'][-1]['role'] == 'tool', 'real prior tool observation fed to next turn')
            require(session['initial_state']['tool_sequence'] == 0 and session['final_state']['tool_sequence'] == 1, 'initial/final state snapshots')
            state = load_document(Path(session['final_state']['workspace']) / 'task-state.json')
            require(state['initial']['owner_token'] == 'OFFLINE_SIMULATION_NOT_MODEL' and state['current']['tool_sequence'] == 1, 'state persisted for independent grade')
            require(session['user_visible_latency_s'] <= session['scheduled_latency_s'], 'user finish distinct from grading terminal')
            require(all((op['queue_s'] >= 0 and op['worker_s'] >= 0 for op in session['host_operations'])), 'tool executor queue/run timings')
        require(len({s['final_state']['workspace'] for s in agents['sessions']}) == 2, 'separate workspaces')
        checks.append('OFFLINE_SIMULATION_NOT_MODEL dynamic tools, feedback, isolated state, grade and timing')
        module.success = False
        quality_fail = await phase('quality-fail', plan_phase(kind='agent', sessions=1, concurrency=1), agent_rows[:1])
        require(quality_fail['sessions'][0]['protocol_ok'] and quality_fail['summary']['successful_tasks'] == 0, 'quality failure not protocol failure')
        module.success = True
        checks.append('Natural terminal with failed task grade is separate from protocol completion')
        module.delay = 0.08
        host_timeout = await phase('host-timeout', plan_phase(kind='agent', sessions=1, concurrency=1), agent_rows[:1], {'session_timeout_s': 0.025})
        timed = host_timeout['sessions'][0]
        require(timed['terminal_reason'] == 'session_timeout' and len(timed['tools']) == 1 and (timed['final_state']['tool_sequence'] == 1), 'cancelled session waits for bounded tool cleanup')
        require(any((o['cancelled_while_waiting'] for o in timed['host_operations'])) and timed['elapsed_s'] >= 0.075, 'cleanup time preserved')
        checks.append('Session deadline waits for bounded tool-thread cleanup and records resulting state')
        module.delay = 0.002
        budget = await phase('budget', plan_phase(kind='agent', sessions=1, concurrency=1, max_turns=1), agent_rows[:1])
        require(not budget['sessions'][0]['protocol_ok'] and budget['sessions'][0]['terminal_reason'] == 'turn_budget', 'budget exhaustion is not success')
        checks.append('Agent turn budget remains explicit non-success')
        full_plan = {**plan, 'phases': [plan_phase(sessions=2, concurrency=2)]}
        full_prepared = dict(tokenizer_sha256='OFFLINE_FIXTURE', phases=[dict(phase=full_plan['phases'][0], sessions=[spec(i) for i in range(2)])])
        full_out = directory / 'successful-run.json'
        await client.run(full_plan, full_prepared, full_out)
        client.cap.save_json(directory / 'successful-run-prepared.json', full_prepared)
        full_saved = load_document(full_out)
        require(full_saved['status'] == 'COMPLETED' and 'active_phase' not in full_saved and (len(full_saved['phases']) == 1), 'successful run terminal')
        checks.append('Successful top-level run persists terminal report and helper identities')
        report_plan = {**plan, 'max_active_sessions': 1, 'phases': [plan_phase(arrival='poisson')]}
        prepared = dict(tokenizer_sha256='OFFLINE_FIXTURE', phases=[dict(phase=report_plan['phases'][0], sessions=[spec(i, t) for i, t in enumerate((0.0, 0.01, 0.08))])])
        server.delay = 0.035
        out = directory / 'failed-run.json'
        try:
            await client.run(report_plan, prepared, out)
        except RuntimeError:
            pass
        else:
            raise AssertionError('run must fail protection')
        saved = load_document(out)
        require(saved['status'] == 'FAILED' and len(saved['phases'][0]['sessions']) == 3 and saved['helper_sha256'], 'failed phase checkpoint and helper identity')
        checks.append('Failed top-level run retains phase/all planned sessions and helper hashes')
        server.reset_success = False
        try:
            await phase('reset-failure', plan_phase(), [spec(0)])
        except RuntimeError:
            pass
        else:
            raise AssertionError('Failed cache reset accepted')
        require(load_document(directory / 'reset-failure-cpu/reset.json')['response']['success'] is False, 'reset failure evidence')
        checks.append('Failed cache reset aborts before model requests and is preserved')
    finally:
        if original is None:
            sys.modules.pop('agent_session_tasks', None)
        else:
            sys.modules['agent_session_tasks'] = original
        await runner.cleanup()


def validate_and_prepare(checks):
    p=dict(schema_version=1,phases=[plan_phase(sessions=2,concurrency=1)])
    valid=client.validate_plan(p)
    require(valid['session_timeout_s']==600,'default session timeout')
    expanded=copy.deepcopy(p);expanded['phases'][0]['max_output_tokens']=4096
    require(client.validate_plan(expanded)['phases'][0]['max_output_tokens']==4096,'4096 output budget supported')
    checks.append('4096-token budget accepted without changing default budget')
    negatives=[('timeout_nan',lambda x:x.update(timeout_s=float('nan'))),
               ('session_bool',lambda x:x['phases'][0].update(sessions=True)),
               ('concurrency_float',lambda x:x['phases'][0].update(concurrency=1.5)),
               ('max_active_bool',lambda x:x.update(max_active_sessions=True)),
               ('unknown_profile',lambda x:x['phases'][0].update(profiles=['missing'])),
               ('mixed_arrivals',lambda x:x['phases'][0].update(rate=1)),
               ('seed_bool',lambda x:x['phases'][0].update(seed=True))]
    negatives.append(('output_budget_unbounded',lambda x:x['phases'][0].update(max_output_tokens=4097)))
    for name,change in negatives:
        x=copy.deepcopy(p);change(x)
        try:client.validate_plan(x)
        except ValueError:pass
        else:raise AssertionError('Invalid '+name+' accepted')
        checks.append('Plan rejects '+name)
    f=plan_phase(arrival='poisson',duration_s=10,rate=2)
    offsets=client.arrivals(f)
    require(offsets==client.arrivals(f) and all(0<=t<10 for t in offsets) and offsets==sorted(offsets),'deterministic finite Poisson')
    checks.append('Poisson offsets reproducible, increasing and within fixed horizon')
    prepared=client.prepare(valid)
    first,second=prepared['phases'][0]['sessions']
    require([len(v) for v in first['prompts']]==client.PROFILES['coding']['inputs'],'actual tokenizer exact prompt lengths')
    require(all(b[:len(a)]==a for a,b in zip(first['prompts'],first['prompts'][1:])),'nested history exact prefix chain')
    require(first['prompts'][0][:2048]==second['prompts'][0][:2048] and first['prompts'][0]!=second['prompts'][0],'common preamble and session-specific suffix')
    require(first['prompt_hashes']==[client.digest(v) for v in first['prompts']],'prepared hashes match IDs')
    checks.append('Real tokenizer exact replay lengths, nested prefixes, shared preamble and session identity')
    f=valid['phases'][0]
    require(client.salt_for(f,0,0)==client.salt_for(f,0,3)!=client.salt_for(f,1,0),'session salt isolates sessions but preserves turns')
    f={**f,'cache_scope':'none'};require(client.salt_for(f,0,0)!=client.salt_for(f,0,1),'none salt disables cross-turn reuse')
    f={**f,'cache_scope':'shared'};require(client.salt_for(f,0,0)==client.salt_for(f,1,1),'shared salt same across sessions')
    checks.append('Session/none/shared salt semantics; metadata alone does not prove server cache behavior')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, help='Optional new directory retaining explicitly simulated raw fixture evidence')
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new CPU fixture evidence ID')
    started = datetime.now(timezone.utc).isoformat()
    checks = []
    error = None
    try:
        validate_and_prepare(checks)
        if args.artifacts:
            args.artifacts.mkdir(parents=True, exist_ok=False)
            asyncio.run(exercise(args.artifacts.resolve(), checks))
        else:
            with tempfile.TemporaryDirectory(prefix='stage-c-session-cpu-') as temp:
                asyncio.run(exercise(Path(temp), checks))
    except Exception as exc:
        import traceback
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    result = dict(status='FAILED' if error else 'PASS', scope='CPU localhost fixtures; OFFLINE_SIMULATION_NOT_MODEL; not GPU/performance/quality evidence', started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(), checks=checks, error=error, client_sha256=hashlib.sha256(Path(client.__file__).read_bytes()).hexdigest(), fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if error else 0)


if __name__=='__main__':main()
