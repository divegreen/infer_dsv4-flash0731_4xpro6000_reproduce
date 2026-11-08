"""Finite local scenario benchmarks. All errors/outputs retained; no POST retries."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import aiohttp
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('historical_benchmark', ROOT.parent/'scripts/benchmark_client.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
SEED = 20261105


def workloads(suite):
    if suite == 'decode':
        return [dict(name='p1024_o1024_c1',p=1024,o=1024,c=1,n=4),
                dict(name='p1024_o1024_c8',p=1024,o=1024,c=8,n=16)]
    if suite == 'context':
        return [dict(name='p65536_o128_c1',p=65536,o=128,c=1,n=2),
                dict(name='p65536_o128_c4',p=65536,o=128,c=4,n=8)]
    if suite == 'arrival':
        return [dict(name='mixed_poisson_r2',p=512,o=128,c=128,n=48,rate=2.),
                dict(name='mixed_poisson_r6',p=512,o=128,c=128,n=48,rate=6.),
                dict(name='p512_o128_c64',p=512,o=128,c=64,n=128)]
    if suite == 'cache':
        return [dict(name=f'prefix{p}_o64',p=p+128,prefix=p,o=64,c=1,n=8) for p in [8192,32768]]
    if suite == 'profile':
        return [dict(name='profile_prefill_16k',p=16384,o=8,c=1,n=1),
                dict(name='profile_decode',p=1024,o=128,c=1,n=1),
                dict(name='profile_context_mix',p=65536,o=128,c=4,n=4)]
    raise ValueError(suite)


def text_ids(tokenizer, n, marker):
    first=tokenizer.encode(f'Reference {marker}.\n',add_special_tokens=False).ids
    body=tokenizer.encode('\n'.join(base.PARAGRAPHS)+'\n',add_special_tokens=False).ids
    suffix=tokenizer.encode(base.SUFFIX,add_special_tokens=False).ids
    needed=n-len(first)-len(suffix)
    assert needed>0
    return first+(body*math.ceil(needed/len(body)))[:needed]+suffix


def prepare(suite, rounds):
    tokenizer_path = Path(str(model_path() / 'tokenizer.json'))
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    prompts = {}
    banks = {}
    cases = workloads(suite)
    for case in cases:
        for round_id in range(1, rounds + 1):
            phases = ['cold', 'exact_replay', 'fork_first', 'warm_serial', 'warm_parallel'] if suite == 'cache' else ['warmup', 'measure']
            for phase in phases:
                n = 1 if phase in ['cold', 'exact_replay', 'fork_first'] else case['c'] if phase == 'warmup' and 'rate' not in case else 4 if phase == 'warmup' else case['n']
                rows = []
                for i in range(n):
                    p, o = (case['p'], case['o'])
                    if 'rate' in case:
                        choice = random.Random(SEED + i).random()
                        p, o = (512, 128) if choice < 0.7 else (4096, 256) if choice < 0.9 else (16384, 128)
                    marker = f'{SEED}:{(case['name'] if 'rate' not in case else 'mixed')}:{round_id}:{('cold' if phase == 'exact_replay' else phase)}:{i}'
                    if suite == 'cache':
                        prefix = text_ids(tok, case['prefix'], f'cache:{case['name']}:{round_id}')
                        ids = prefix + text_ids(tok, 128, marker)
                    else:
                        ids = text_ids(tok, p, marker)
                    h = base.digest_ids(ids)
                    prompts[h] = dict(token_ids=ids, input_tokens=len(ids), sha256=h)
                    rows.append(dict(prompt_sha256=h, output_tokens=o, seed=SEED + i))
                banks[f'{round_id}/{case['name']}/{phase}'] = rows
    return dict(schema_version=2, created_utc=base.utc_now(), status='PREPARED', suite=suite, rounds=rounds, cases=cases, prompts=prompts, banks=banks, seed=SEED, client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(), runs=[], protocol=dict(connections='fresh_per_request', retries=0, prefix_reset='once per case/round for cache suite; unique prefixes across cases/rounds otherwise', sampling='temperature0; seed recorded; ignore_eos true for controlled performance', ttft='request POST to first nonempty text; includes connection and server queue', tpot='(E2E-TTFT)/(actual output tokens-1); not ITL', scheduled_latency='includes client dispatch lag relative to planned Poisson arrival', percentiles='linear interpolation over all successful measured requests; failures retained separately', example_slo='Exploratory thresholds TTFT from scheduled arrival <=1s, TPOT<=50ms, E2E from scheduled arrival<=10s; not a customer SLA'))


async def control(session,url,path,payload=None):
    async with session.post(url+path,json=payload,allow_redirects=False) as response:
        body=await response.text()
        if response.status!=200: raise RuntimeError(f'{path}: HTTP {response.status} {body[:1000]}')
        if 'reset_prefix_cache' in path and not json.loads(body).get('success'):
            raise RuntimeError('Prefix cache reset did not return success=true')
        return dict(status=response.status,body=body)


async def metrics(session, url, directory, label):
    async with session.get(url + '/metrics', allow_redirects=False) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f'metrics HTTP {response.status}')
    path = directory / f'metrics-{label}.txt'
    write_text(path, body)
    return path.name


async def request(session,url,prompt,item,request_id):
    payload=dict(model='dsv4',prompt=prompt['token_ids'],max_tokens=item['output_tokens'],temperature=0,seed=item['seed'],
                 n=1,stream=True,echo=False,stream_options={'include_usage':True},ignore_eos=True,add_special_tokens=False)
    result=dict(request_id=request_id,prompt_sha256=prompt['sha256'],input_tokens=prompt['input_tokens'],
                requested_output_tokens=item['output_tokens'],seed=item['seed'],started_utc=base.utc_now(),
                start_unix=time.time(),ok=False,http_status=None,usage=None,ttft_s=None,e2e_s=None,tpot_s=None,error=None)
    started=time.perf_counter();parts=[];done=False;events=[]
    try:
        async with session.post(url+'/v1/completions',json=payload,allow_redirects=False) as response:
            result['http_status']=response.status
            if response.status!=200: raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:2000]}')
            async for event in base.sse_events(response.content):
                now=time.perf_counter()-started
                if event.strip()=='[DONE]':
                    done=True;result['e2e_s']=now;break
                obj=json.loads(event)
                if obj.get('error'): raise RuntimeError(str(obj['error']))
                if obj.get('usage') is not None: result['usage']=obj['usage']
                for choice in obj.get('choices',[]):
                    text=choice.get('text') or ''
                    if text:
                        if result['ttft_s'] is None: result['ttft_s']=now
                        parts.append(text);events.append(now)
                    if choice.get('finish_reason'): result['finish_reason']=choice['finish_reason']
        usage=result['usage']
        if not done or not parts or not usage: raise RuntimeError('Missing DONE, text or actual usage')
        if usage['prompt_tokens']!=prompt['input_tokens'] or usage['completion_tokens']!=item['output_tokens']:
            raise RuntimeError('Actual token count mismatch')
        result['tpot_s']=(result['e2e_s']-result['ttft_s'])/(usage['completion_tokens']-1)
        result['ok']=True
    except Exception as e:
        result['error']={'type':type(e).__name__,'message':str(e)}
    result.update(text=''.join(parts),text_event_seconds=events,elapsed_s=time.perf_counter()-started,done_received=done)
    result['text_sha256']=hashlib.sha256(result['text'].encode()).hexdigest()
    return result


def summary(rows,wall):
    success=[r for r in rows if r['ok']]
    tokens=sum(r['usage']['completion_tokens'] for r in success)
    result=dict(requests=len(rows),successful=len(success),failed=len(rows)-len(success),wall_s=wall,
                output_tokens=tokens,output_tokens_per_s=tokens/wall,requests_per_s=len(success)/wall,
                example_slo_pass=sum(r['ttft_s']+r['dispatch_lag_s']<=1 and r['tpot_s']<=.05 and r['e2e_s']+r['dispatch_lag_s']<=10 for r in success))
    result['example_goodput_requests_per_s']=result['example_slo_pass']/wall
    for key in ['ttft_s','e2e_s','tpot_s','dispatch_lag_s','scheduled_ttft_s','scheduled_e2e_s']:
        values=[r[key] for r in success]
        result[key]={'p50':base.percentile(values,.5),'p95':base.percentile(values,.95),'max':max(values) if values else None}
    return result


async def batch(session,args,report,case,round_id,phase):
    items=report['banks'][f'{round_id}/{case["name"]}/{phase}']
    concurrency=8 if phase=='warm_parallel' else 1 if phase in ['cold','exact_replay','fork_first'] else case['c']
    sem=asyncio.Semaphore(concurrency)
    start=time.perf_counter();start_unix=time.time()
    offsets=[0.]*len(items)
    if 'rate' in case and phase=='measure':
        rng=random.Random(SEED+round_id)
        for i in range(1,len(items)): offsets[i]=offsets[i-1]+rng.expovariate(case['rate'])
    async def one(i,item):
        delay=start+offsets[i]-time.perf_counter()
        if delay>0: await asyncio.sleep(delay)
        async with sem:
            send=time.perf_counter(); lag=max(0.,send-start-offsets[i])
            row=await request(session,args.base_url,report['prompts'][item['prompt_sha256']],item,f'{round_id}/{case["name"]}/{phase}/{i}')
            row.update(planned_offset_s=offsets[i],dispatch_lag_s=lag,
                       scheduled_ttft_s=(lag+row['ttft_s']) if row['ttft_s'] is not None else None,
                       scheduled_e2e_s=(lag+row['e2e_s']) if row['e2e_s'] is not None else None)
            return row
    rows=await asyncio.gather(*(one(i,item) for i,item in enumerate(items)))
    wall=time.perf_counter()-start
    return dict(phase=phase,start_unix=start_unix,end_unix=time.time(),requests=rows,summary=summary(rows,wall),
                offered_rate=case.get('rate'),concurrency_limit=concurrency,planned_arrival_span_s=offsets[-1])


async def run(args,report):
    connector=aiohttp.TCPConnector(limit=128,limit_per_host=128,force_close=True)
    timeout=aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector,timeout=timeout,trust_env=False) as session:
        for r in range(1,args.rounds+1):
            for case in report['cases']:
                entry=dict(round=r,case=case['name'],stages=[],status='RUNNING');report['runs'].append(entry)
                if args.suite=='cache':
                    entry['reset']=await control(session,args.base_url,'/reset_prefix_cache?reset_running_requests=false')
                phases=['cold','exact_replay','fork_first','warm_serial','warm_parallel'] if args.suite=='cache' else ['warmup','measure']
                for phase in phases:
                    label=f'{args.suite}-r{r}-{case["name"]}-{phase}'
                    before=await metrics(session,args.base_url,args.output.parent,label+'-before')
                    if args.suite=='profile' and phase=='measure':
                        entry.setdefault('profile_control',[]).append(await control(session,args.base_url,'/start_profile',dict(profile_prefix=case['name'],max_iterations=20,delay_iterations=0)))
                    try:
                        stage=await batch(session,args,report,case,r,phase)
                    finally:
                        if args.suite=='profile' and phase=='measure':
                            entry.setdefault('profile_control',[]).append(await control(session,args.base_url,'/stop_profile'))
                    after=await metrics(session,args.base_url,args.output.parent,label+'-after')
                    stage.update(metrics_before=before,metrics_after=after)
                    entry['stages'].append(stage)
                    base.save_report(args.output,report)
                    print(json.dumps(dict(round=r,case=case['name'],phase=phase,summary=stage['summary'])),flush=True)
                    if stage['summary']['failed']:
                        entry['status']='FAILED';return False
                entry['status']='PASSED'
                base.save_report(args.output,report)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8260')
    parser.add_argument('--suite', choices=['decode', 'context', 'arrival', 'cache', 'profile'], required=True)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    base.endpoints(args.base_url)
    if not 1 <= args.rounds <= 4:
        parser.error('rounds must be 1..4')
    args.base_url = args.base_url.rstrip('/')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        f.write('{}\n')
    report = prepare(args.suite, args.rounds)
    report['base_url'] = args.base_url
    base.save_report(args.output, report)
    if args.prepare_only:
        print('PREPARED', args.output)
        return 0
    report['status'] = 'RUNNING'
    try:
        success = asyncio.run(run(args, report))
        report['status'] = 'PASSED' if success else 'FAILED'
    except BaseException as exc:
        report.update(status='FAILED', fatal_error=repr(exc))
        raise
    finally:
        report['finished_utc'] = base.utc_now()
        base.save_report(args.output, report)
    return 0 if report['status'] == 'PASSED' else 1


if __name__=='__main__':
    raise SystemExit(main())
