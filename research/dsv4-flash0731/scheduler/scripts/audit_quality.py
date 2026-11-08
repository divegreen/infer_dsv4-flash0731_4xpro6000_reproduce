#!/usr/bin/env python3
"""Independent CPU checks of the optional D natural-EOS quality sidecar.

Reads captured fixtures and raw responses, never runs the captured client or
Agent tools. Answers may differ/fail; consistency and protocol status are kept
separate from the completed performance dataset.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
from datetime import datetime
import hashlib
import json
from pathlib import Path


def sha(raw):return hashlib.sha256(raw).hexdigest()
def compact(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True)


def audit(a,directory,performance,config,experiment,snapshot,supervisor_success):
    where=str(directory/'quality.json')
    if 'quality_probe' not in config:
        return dict(status='NOT_REQUESTED')
    a.read(Path(__file__))
    a.equal(config['quality_probe'],dict(rounds=2,max_tokens=128,timeout_s=180),where+'.registered_spec')
    side=experiment.get('quality_probe')
    if side is None:
        a.check(not supervisor_success,where,'Successful configured quality run has no sidecar')
        return dict(status='NOT_STARTED',scope='No quality result; performance completion remains separate')
    a.equal(side.get('excluded_from_performance'),True,where+'.excluded')
    a.equal(side.get('quality_scores_gate_performance'),False,where+'.nongating_scores')
    a.equal(side.get('report'),str(directory/'quality.json'),where+'.sidecar_path')
    a.equal(side.get('log'),str(directory/'quality.log'),where+'.sidecar_log')
    if (directory/'quality.log').is_file():a.read(directory/'quality.log')
    elif supervisor_success:a.check(False,where,'Missing quality log')
    if not (directory/'quality.json').is_file():
        a.check(not supervisor_success,where,'Successful quality has no report')
        return dict(status='NO_REPORT',execution_status=side.get('status'),error=side.get('error'))
    raw=a.read(directory/'quality.json');quality=json.loads(raw)
    if side.get('report_sha256') is not None:a.equal(side['report_sha256'],sha(raw),where+'.report_hash')
    if quality.get('status')=='INITIALIZING':
        a.check(not supervisor_success,where,'Successful quality only initialized')
        return dict(status='INITIALIZING',execution_status=side.get('status'))
    prefix=('experiments/scenarios/workloads/agent/'
            if any(key.startswith('experiments/scenarios/workloads/agent/') for key in snapshot)
            else 'scenarios/workloads/agent/')
    selected={key.removeprefix(prefix):value for key,value in snapshot.items() if key.startswith(prefix)}
    a.check(all(name in selected for name in ('tasks.json','quality.json','replay_context.md')),where,'Quality fixture closure incomplete')
    a.equal(quality['workload_sha256'],{name:sha(raw) for name,raw in selected.items()},where+'.fixture_identity')
    clientkey='scenarios/scripts/agent_benchmark.py'
    a.equal(quality['client_sha256'],sha(snapshot[clientkey]),where+'.executed_probe_source')
    a.equal(quality['mode'],'quality',where+'.mode');a.equal(quality['rounds'],2,where+'.rounds')
    plan=performance['plan']
    a.equal(quality['base_url'],plan['base_url'],where+'.owned_base_url')
    a.equal(quality['model'],plan['model'],where+'.model')
    cfg=quality['configuration']
    for key,value in dict(mode='quality',rounds=2,max_tokens=128,timeout=180.,reset_prefix_cache=False,cache_label='unspecified',output=str(directory/'quality.json')).items():
        a.equal(cfg.get(key),value,where+'.config/'+key)
    argv=side['command'];expected=[argv[0],str(a.root.parent/clientkey),'--mode','quality','--base-url',plan['base_url'],
        '--model',plan['model'],'--rounds','2','--max-tokens','128','--timeout','180','--cache-label','unspecified','--output',str(directory/'quality.json')]
    a.equal(argv,expected,where+'.actual_probe_command')
    a.equal(quality.get('reset_events'),[],where+'.no_cache_reset')
    a.check(datetime.fromisoformat(side['started_utc'])>=datetime.fromisoformat(performance['finished_utc']),where,'Quality overlaps performance interval')
    cases=json.loads(selected['quality.json'])['cases'];expected_runs=[(r,c) for r in (1,2) for c in cases]
    runs=quality['runs'];a.check(len(runs)<=len(expected_runs),where,'Extra quality requests')
    complete=quality.get('status') in ('COMPLETED','COMPLETED_WITH_PROTOCOL_ERRORS')
    if complete:a.equal(len(runs),16,where+'.complete_requests')
    good=[];passed=0;elapsed=0.
    for index,(run,(round_index,case)) in enumerate(zip(runs,expected_runs)):
        loc=where+'.run/'+str(index)
        a.equal(run['round'],round_index,loc+'.round');a.equal(run['case_id'],case['id'],loc+'.case')
        messages=case.get('messages',[dict(role='user',content=case.get('prompt',''))])
        a.equal(run['request_messages'],messages,loc+'.messages')
        a.equal(run['request_messages_sha256'],sha(compact(messages).encode()),loc+'.message_hash')
        http=run['http'];payload=dict(model=plan['model'],messages=messages,temperature=0,seed=20261105,stream=False,
            max_tokens=128,reasoning_effort='none',chat_template_kwargs={'enable_thinking':False})
        a.equal(http['request'],payload,loc+'.natural_eos_payload')
        a.equal(http['attempts'],1,loc+'.single_attempt')
        a.equal(http.get('ttft_s'),None,loc+'.nonstream_ttft');a.equal(http.get('tpot_s'),None,loc+'.nonstream_tpot')
        elapsed+=http['e2e_s']
        if not http['protocol_ok']:
            a.equal(run['quality'],dict(passed=False,reason='protocol_failure'),loc+'.protocol_failed_score')
            continue
        a.equal(http['http_status'],200,loc+'.http_status')
        a.check(not http.get('response_truncated') and not http.get('response_incomplete'),loc,'Successful response truncated/incomplete')
        parsed=json.loads(http['raw_response']);a.equal(parsed,http['response'],loc+'.raw_response')
        a.equal(len(parsed['choices']),1,loc+'.choice_count');choice=parsed['choices'][0]
        a.equal(http['message'],choice['message'],loc+'.message');a.equal(http['usage'],parsed['usage'],loc+'.usage')
        a.equal(http['finish_reason'],choice['finish_reason'],loc+'.finish')
        usage=parsed['usage'];a.check(all(type(usage[k]) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')),loc,'Invalid usage')
        a.equal(usage['total_tokens'],usage['prompt_tokens']+usage['completion_tokens'],loc+'.usage_total')
        text=choice['message'].get('content') or ''
        try:
            match=compact(json.loads(text))==compact(case['expected']) if case['kind']=='json' else text.strip().casefold()==case['expected']
        except ValueError:match=False
        natural=choice['finish_reason']=='stop'
        for key,value in dict(passed=match and natural,content_matched=match,naturally_stopped=natural).items():a.equal(run['quality'][key],value,loc+'.score/'+key)
        passed+=match and natural;good.append(http)
    if complete:
        expected=dict(requests=len(runs),protocol_successes=len(good),protocol_failures=len(runs)-len(good),
            prompt_tokens=sum(h['usage']['prompt_tokens'] for h in good),completion_tokens=sum(h['usage']['completion_tokens'] for h in good),
            request_e2e_sum_s=elapsed,ttft_s=None,tpot_s=None,quality_passed=passed,agent_successes=0,task_tests_passed=0)
        a.equal(quality['summary'],expected,where+'.independent_summary')
        a.equal(quality['status'],'COMPLETED' if len(good)==len(runs) else 'COMPLETED_WITH_PROTOCOL_ERRORS',where+'.terminal_protocol_status')
    if supervisor_success:
        a.equal(side['status'],'PASS',where+'.successful_sidecar');a.equal(side['exit_code'],0,where+'.successful_exit')
        a.equal(quality['status'],'COMPLETED',where+'.successful_quality_report')
    return dict(status=quality['status'],execution_status=side['status'],requests=len(runs),protocol_successes=len(good),
        protocol_failures=len(runs)-len(good),quality_passed=passed,quality_scores_gate_performance=False,
        scope='Eight synthetic cases twice; strict formatting score, not broad model/Agent capability or performance metric')
