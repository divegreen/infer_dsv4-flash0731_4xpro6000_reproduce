#!/usr/bin/env python3
"""Independent CPU review of actual D plans, exact token inputs and HTTP lifecycle.

Only tokenizer and a temporary localhost fixture are used; no model/GPU calls.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import ast
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
RESEARCH = PHASE.parent
sys.path.insert(0, str(HERE))
import serve_study as harness
import study_client as client
from aiohttp import web

CHECKS = []
LEDGER = {}


def check(label, value):
    if not value:
        raise AssertionError(label)
    CHECKS.append(label)


def read(path):
    raw = path.read_bytes()
    LEDGER[str(path.resolve())] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return raw


def load(path):
    read(path)
    return load_document(path)


def normalized_client(tree, *, revised):
    tree = copy.deepcopy(tree)
    tree.body = tree.body[1:]  # module docstring is descriptive only
    if revised:
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'request_one')
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
                node.keywords = [k for k in node.keywords if k.arg != 'request_id']
        class RemoveObservation(ast.NodeTransformer):
            def visit_Assign(self, node):
                if (len(node.targets)==1 and isinstance(node.targets[0], ast.Subscript)
                        and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id=='row'
                        and isinstance(node.targets[0].slice, ast.Constant)
                        and node.targets[0].slice.value in ('wire_request_id','server_response_ids')):
                    return None
                return self.generic_visit(node)
            def visit_If(self, node):
                if ('server_response_ids' in ast.unparse(node.test)
                        and len(node.body)==1 and 'server_response_ids' in ast.unparse(node.body[0])):
                    return None
                return self.generic_visit(node)
        RemoveObservation().visit(function)
    return ast.dump(tree, include_attributes=False)


def options(config):
    result = {}
    args = config['args']; i = 0
    while i < len(args):
        key = args[i]; check('unique option '+config['name']+'/'+key, key not in result)
        if i+1 < len(args) and not args[i+1].startswith('--'):
            result[key] = args[i+1]; i += 2
        else:
            result[key] = True; i += 1
    return result


async def fixture(mode, plan, prepared, directory):
    requests, controls = ([], [])

    async def completion(request):
        payload = await request.json()
        requests.append(payload)
        warm = payload['request_id'].startswith('d-warm-p512-')
        if mode == 'warm_failure' and warm or (mode == 'request_failure' and (not warm) and payload['request_id'].endswith('-0')):
            return web.json_response({'error': 'deliberate fixture HTTP503'}, status=503)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        rid = 'cmpl-' + payload['request_id']
        first = dict(id=rid, choices=[dict(index=0, text='synthetic', finish_reason=None)])
        final = dict(id=rid, choices=[dict(index=0, text=' fixture', finish_reason='length')], usage=dict(prompt_tokens=len(payload['prompt']), completion_tokens=payload['max_tokens'], total_tokens=len(payload['prompt']) + payload['max_tokens']))
        for event in (first, final):
            await response.write(('data: ' + json.dumps(event) + '\n\n').encode())
        await response.write(b'data: [DONE]\n\n')
        await response.write_eof()
        return response

    async def metric(request):
        return web.Response(text='vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0\n')

    async def control(request):
        raw = await request.read()
        body = json.loads(raw) if raw else None
        controls.append(dict(path=request.path, payload=body, unix=time.time()))
        if mode == 'start_failure' and request.path == '/start_profile' or (mode == 'stop_failure' and request.path == '/stop_profile'):
            return web.json_response({'error': 'deliberate control failure'}, status=500)
        return web.json_response({'status': 'ok'})
    app = web.Application()
    app.router.add_post('/v1/completions', completion)
    app.router.add_get('/metrics', metric)
    app.router.add_post('/start_profile', control)
    app.router.add_post('/stop_profile', control)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    local = copy.deepcopy(plan)
    local['base_url'] = f'http://127.0.0.1:{port}'
    local['phases'] = [copy.deepcopy(plan['phases'][0]), copy.deepcopy(plan['phases'][-1])]
    chosen = [copy.deepcopy(prepared[0]), copy.deepcopy(prepared[-1])]
    local['metrics_interval_s'] = 0.1
    report = dict(status='PREPARED', plan=local, runs=[], profiling=dict(excluded_from_performance=True, sessions=[]))
    output = directory / (mode + '.json')
    caught = None
    try:
        try:
            await harness.execute_plan(local, chosen, report, output, control_timeout_s=2)
        except RuntimeError as exc:
            caught = repr(exc)
    finally:
        await runner.cleanup()
    saved = load_document(output)
    check(mode + ' report saved exactly', saved == report)
    check(mode + ' unique wire IDs no hidden retries', len({r['request_id'] for r in requests}) == len(requests))
    for run in report['runs']:
        raw = [json.loads(line) for line in Path(run['raw_requests_jsonl']).read_text().splitlines()]
        check(mode + '/' + run['phase']['id'] + ' raw JSONL equals report sorted', sorted(raw, key=lambda r: r['index']) == run['requests'])
        check(mode + '/' + run['phase']['id'] + ' excluded diagnostic', run['excluded_from_performance'] is True)
        for row in run['requests']:
            check(mode + ' wire identity ' + row['request_id'], row['wire_request_id'] == 'd-' + row['request_id'].replace('/', '-'))
            if row['ok']:
                check(mode + ' received response identity ' + row['request_id'], row['server_response_ids'] == ['cmpl-' + row['wire_request_id']])
    if mode == 'warm_failure':
        check(mode + ' no profile controls after failed warmup', controls == [] and len(requests) == 1 and (report['status'] == 'FAILED') and (caught is None))
    else:
        check(mode + ' start followed by stop exactly once', [x['path'] for x in controls] == ['/start_profile', '/stop_profile'])
        if mode == 'start_failure':
            check(mode + ' original error retained, no profile requests', caught and '/start_profile: HTTP 500' in caught and (len(requests) == 1) and (report['status'] == 'FAILED'))
        elif mode == 'stop_failure':
            check(mode + ' stop error and completed data retained', caught and '/stop_profile: HTTP 500' in caught and (len(requests) == 9) and (report['status'] == 'FAILED') and (len(report['runs']) == 2))
        elif mode == 'request_failure':
            check(mode + ' failed row retained without retry', caught is None and report['status'] == 'FAILED' and (len(requests) == 9) and (report['runs'][-1]['summary']['failed'] == 1))
        else:
            check(mode + ' full success', caught is None and report['status'] == 'PASSED' and (len(requests) == 9))
            capture = report['profiling']['sessions'][0]['controls']
            run = report['runs'][-1]
            check(mode + ' profile control encloses phase', capture[0]['finished_unix'] <= run['start_unix'] <= run['end_unix'] <= capture[1]['started_unix'])
    return dict(mode=mode, requests=len(requests), controls=[r['path'] for r in controls], status=report['status'], error=caught, evidence_sha256={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob(mode + '*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = dict(status='IN PROGRESS', utc=datetime.now(timezone.utc).isoformat(), scope='CPU actual tokenizer/plan comparison and temporary synthetic localhost; no GPU/model or real study run.')
    try:
        old = RESEARCH / 'scenarios/scripts/capacity_client.py'
        new = HERE / 'study_client.py'
        read(HERE / 'serve_study.py')
        read(HERE / 'run_logged.py')
        check('Client AST exact except documented request ID observation additions', normalized_client(ast.parse(read(old)), revised=False) == normalized_client(ast.parse(read(new)), revised=True))
        names = ['scheduler-tp4-observe-dense-profile', 'scheduler-tp4-fairshare-dense-profile']
        plans = [harness.validate_study_plan(load(path_at(PHASE, 'workloads') / f'{name}.json')) for name in names]
        configs = [load(path_at(PHASE, 'configs') / f'{name}.json') for name in names]
        p0 = copy.deepcopy(plans[0])
        p1 = copy.deepcopy(plans[1])
        p0.pop('experiment_id')
        p1.pop('experiment_id')
        check('Plans exact except experiment ID', p0 == p1)
        prepared0, tokenizer = client.prepare(plans[0])
        prepared1, _ = client.prepare(plans[1])
        read(tokenizer)
        check('Actual tokenizer prepared workload exact across candidates', prepared0 == prepared1)
        shape_totals = []
        for phase in prepared0:
            check('every planned row accounted ' + phase['spec']['id'], len(phase['items']) == len(phase['planned_offsets_s']))
            for item in phase['items']:
                prompt = phase['prompts'][item['prompt_sha256']]
                check('actual prompt length ' + phase['spec']['id'] + '/' + str(item['index']), len(prompt['token_ids']) == item['input_tokens'])
                check('actual prompt digest ' + phase['spec']['id'] + '/' + str(item['index']), client.base.digest_ids(prompt['token_ids']) == item['prompt_sha256'])
            shape_totals.append(dict(phase=phase['spec']['id'], role=phase['spec']['role'], requests=len(phase['items']), input_tokens=sum((i['input_tokens'] for i in phase['items'])), output_tokens=sum((i['requested_output_tokens'] for i in phase['items']))))
        final = prepared0[-1]
        check('eight exact profile arrivals', final['planned_offsets_s'] == [0.0, 0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        check('profile workload token totals', sum((i['input_tokens'] for i in final['items'])) == 134144 and sum((i['requested_output_tokens'] for i in final['items'])) == 1024)
        op = [options(c) for c in configs]
        for index, (config, plan) in enumerate(zip(configs, plans)):
            out = path_at(PHASE, 'results') / plan['experiment_id']
            command = harness.server_command(config, plan, out)
            check('validated model CLI ' + names[index], command[1:3] == ['serve', str(model_path())])
            profile = json.loads(op[index]['--profiler-config'])
            check('unique profile destination ' + names[index], profile == harness.profiler_options(out))
            op[index].pop('--profiler-config')
            op[index].pop('--scheduler-cls')
        check('All server options equal except class/profile destination', op[0] == op[1] and configs[0]['env'] == configs[1]['env'])
        check('Both retain cap2048 and cadence8', op[0]['--long-prefill-token-threshold'] == '2048' and op[0]['--prefill-schedule-interval'] == '8')
        fixtures = []
        with tempfile.TemporaryDirectory(prefix='D-study-independent-CPU-') as temp:
            for mode in ('success', 'start_failure', 'stop_failure', 'request_failure', 'warm_failure'):
                fixtures.append(asyncio.run(fixture(mode, plans[0], prepared0, Path(temp))))
        result.update(status='PASS', plan_counts=shape_totals, http_fixtures=fixtures, actual_prepared_identity=hashlib.sha256(json.dumps(prepared0, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        check('All consumed source/input bytes unchanged', all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == i['sha256'] and Path(p).stat().st_size == i['bytes'] for p, i in LEDGER.items())))
    except BaseException as exc:
        import traceback
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    result.update(checks=CHECKS, evidence_ledger=LEDGER, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitations=['Synthetic SSE deliberately declares usage; it tests protocol/lifecycle only, not tokenizer/model output quality.', 'Client records response ID evidence; actual EngineCore request attribution must be checked on GPU-run raw and scheduler trace.', 'Temporary fixture artifacts are summarized here and removed on completion; actual research output directories remain untouched.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(CHECKS), output=str(args.output))))
    return 0 if result['status'] == 'PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
