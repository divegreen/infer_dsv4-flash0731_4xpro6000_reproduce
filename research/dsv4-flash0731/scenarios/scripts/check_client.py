"""Actual finite CPU HTTP fixture checks; no GPU/model or external network."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from aiohttp import web
import scenario_client as client

async def main():
    seen=[];failed=False
    async def completions(request):
        payload=await request.json();seen.append(request.transport)
        assert payload['temperature']==0 and payload['ignore_eos'] and payload['stream']
        if failed:return web.Response(status=503,text='deliberate test fixture error')
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
        for text in ['one',' two']:
            await response.write(('data: '+json.dumps({'choices':[{'index':0,'text':text}]})+'\n\n').encode())
        await response.write(('data: '+json.dumps({'choices':[{'index':0,'text':'','finish_reason':'length'}],
            'usage':{'prompt_tokens':len(payload['prompt']),'completion_tokens':payload['max_tokens']}})+'\n\ndata: [DONE]\n\n').encode())
        await response.write_eof();return response
    app=web.Application();app.router.add_post('/v1/completions',completions)
    runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    prompt=dict(token_ids=[1,2,3],input_tokens=3,sha256=client.base.digest_ids([1,2,3]))
    case=dict(name='fixture',c=2,n=4);item=dict(prompt_sha256=prompt['sha256'],output_tokens=2,seed=1)
    report=dict(prompts={prompt['sha256']:prompt},banks={'1/fixture/measure':[item]*4})
    try:
        async with client.aiohttp.ClientSession(connector=client.aiohttp.TCPConnector(force_close=True),trust_env=False) as session:
            result=await client.batch(session,SimpleNamespace(base_url=f'http://127.0.0.1:{port}'),report,case,1,'measure')
            assert result['summary']['successful']==4 and result['summary']['output_tokens']==8
            assert len({id(t) for t in seen})==4
            assert all(r['text']=='one two' and len(r['text_event_seconds'])==2 for r in result['requests'])
            failed=True;before=len(seen)
            result=await client.request(session,f'http://127.0.0.1:{port}',prompt,item,'failure')
            assert not result['ok'] and result['http_status']==503 and len(seen)==before+1
    finally:await runner.cleanup()
    for suite in ['decode','context','arrival','cache','profile']:
        a=client.prepare(suite,2);b=client.prepare(suite,2)
        assert a['banks']==b['banks'] and a['prompts']==b['prompts']
        if suite=='cache':
            for case in a['cases']:
                hashes=[a['banks'][f'1/{case["name"]}/{phase}'][0]['prompt_sha256'] for phase in ['cold','warm_serial','warm_parallel']]
                prefixes=[a['prompts'][h]['token_ids'][:case['prefix']] for h in hashes]
                assert prefixes[0]==prefixes[1]==prefixes[2]
                assert len(set(hashes))==3
                assert a['banks'][f'1/{case["name"]}/cold']==a['banks'][f'1/{case["name"]}/exact_replay']
    print('PASS: actual local SSE/usage/full text, fresh connections, failed POST attempted once, deterministic workloads, exact shared prefixes')

asyncio.run(main())
