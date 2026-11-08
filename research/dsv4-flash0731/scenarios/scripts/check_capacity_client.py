#!/usr/bin/env python3
"""CPU-only protocol/metric fixture. Never starts or contacts a model service."""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import asyncio
from collections import Counter
import json
from pathlib import Path
import tempfile
import time

from aiohttp import web
import capacity_client as client


def make_prepared(phase, cases=None, offsets=None):
    cases = cases or [(1, 512, 4)] * phase.get('requests', len(offsets or []))
    items, prompts = [], {}
    for index, (marker, p, o) in enumerate(cases):
        ids = [marker] + [7] * (p-1)
        h = client.base.digest_ids(ids)
        prompts[h] = dict(sha256=h, input_tokens=p, token_ids=ids)
        items.append(dict(index=index, shape_name=f'p{p}_o{o}', input_tokens=p,
                          requested_output_tokens=o, seed=index, ignore_eos=marker != 5,
                          prompt_sha256=h))
    return dict(spec=phase, items=items, prompts=prompts,
                planned_offsets_s=offsets if offsets is not None else [0.]*len(items))


def artificial(p=512, o=10, ttft=.1, tpot=.01, lag=0., ok=True, mode='poisson'):
    row = dict(input_tokens=p, shape_name=f'p{p}', ok=ok, admitted=True, censored=False,
               timed_out=False, usage=dict(completion_tokens=o), ttft_s=ttft,
               tpot_s=tpot, e2e_s=ttft+tpot*(o-1), dispatch_lag_s=lag, http_status=200,
               scheduled_ttft_s=lag+ttft, scheduled_e2e_s=lag+ttft+tpot*(o-1))
    return client.add_slo(row, mode)


def metric_checks():
    assert client.stats([1, 2, 3, 4]) == dict(n=4, p50=2.5, p95=3.8499999999999996, p99=3.9699999999999998, max=4.)
    assert client.stats([])['p99'] is None
    assert artificial(p=1024)['ttft_limit_s'] == 1
    assert artificial(p=1025)['ttft_limit_s'] == 3
    assert artificial(p=4096)['ttft_limit_s'] == 3
    assert artificial(p=4097)['ttft_limit_s'] == 8
    assert artificial(p=16384)['ttft_limit_s'] == 8
    assert artificial(p=16385)['slo_pass'] is None
    assert abs(artificial(o=3)['e2e_limit_s']-1.1) < 1e-9
    assert artificial(lag=2)['slo_pass'] is False
    assert artificial(lag=2, mode='closed-loop')['slo_pass'] is True
    assert artificial(tpot=.06)['legacy_slo_pass'] is False
    failure = artificial(ok=False)
    summary = client.group_summary([artificial(), failure])
    assert summary['slo_eligible_requests'] == 2
    assert summary['slo_joint_pass_rate'] == .5
    assert summary['slo_joint_pass_rate_success_only'] == 1
    # E2E uses actual output, not request budget; output=1 leaves TPOT unmeasured.
    one = artificial(o=1)
    one['tpot_s'] = None
    assert client.add_slo(one, 'poisson')['slo_pass'] is True
    a = client.make_offsets(dict(mode='poisson', rate=2, duration_s=10, seed=71))
    b = client.make_offsets(dict(mode='poisson', rate=2, duration_s=10, seed=71))
    assert a == b and a[0] > 0 and all(x < 10 for x in a)
    phase = dict(id='deterministic', mode='poisson', rate=2, duration_s=10,
                 seed=71, shape_mix=client.MIXES['short_mixed'])
    assert client.make_items(phase, 50) == client.make_items(phase, 50)
    try:
        client.validate_plan(dict(schema_version=1, phases=[dict(phase, requests=10)]))
        raise AssertionError('Count-capped Poisson should be rejected')
    except ValueError:
        pass
    try:
        client.base_url('http://example.com:80')
        raise AssertionError('Remote URL should be rejected')
    except ValueError:
        pass
    late = artificial()
    late.update(planned_offset_s=.5, dispatch_offset_s=2., end_offset_s=3.,
                requested_output_tokens=10)
    late_summary = client.summarize([late], dict(mode='poisson', duration_s=1., rate=1), 3., [], 1)
    assert late_summary['in_flight_at_window_end'] == 0
    live = dict(late, dispatch_offset_s=.5)
    live_summary = client.summarize([live], dict(mode='poisson', duration_s=1., rate=1), 3., [], 1)
    assert live_summary['in_flight_at_window_end'] == 1
    return ['window-end in-flight excludes dispatches after window',
            'linear P50/P95/P99', 'Research SLO boundaries and actual output',
            'failed eligible arrivals retained in pass-rate denominator',
            'scheduled vs closed-loop SLO', 'legacy TPOT threshold',
            'one-token TPOT unmeasured', 'deterministic Poisson and shapes',
            'duration-only Poisson and local URL validation']


async def fixture_checks():
    seen, peers = (Counter(), [])
    inflight, peak = (0, 0)

    async def complete(request):
        nonlocal inflight, peak
        payload = await request.json()
        marker = payload['prompt'][0]
        seen[marker] += 1
        peers.append(request.transport.get_extra_info('peername'))
        inflight += 1
        peak = max(peak, inflight)
        try:
            if marker == 3:
                return web.json_response({'error': 'fixture'}, status=500)
            if marker == 9:
                return web.json_response({'error': 'fixture'}, status=429)
            if marker == 4:
                await asyncio.sleep(0.15)
            if marker in (10, 11):
                await asyncio.sleep(0.04 if marker == 10 else 0.1)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(b': comment\r\n\r\ndata: {"choices":[{"index":0,"text":"a"}]}\r\n\r\n')
            await asyncio.sleep(0.004)
            if marker == 6:
                await response.write_eof()
                return response
            output = 3 if marker == 5 else payload['max_tokens']
            usage = dict(prompt_tokens=len(payload['prompt']) + (1 if marker == 7 else 0), completion_tokens=output, total_tokens=len(payload['prompt']) + output)
            event = dict(choices=[dict(index=0, text='b', finish_reason='length')], usage=usage)
            wire = ('data: ' + json.dumps(event) + '\n\n').encode()
            await response.write(wire[:17])
            await response.write(wire[17:])
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        except (ConnectionResetError, RuntimeError):
            return web.Response(status=499)
        finally:
            inflight -= 1

    async def metrics(request):
        return web.Response(text=f'vllm:num_requests_running{{model_name="fixture"}} {inflight}\nvllm:num_requests_waiting{{model_name="fixture"}} 0\n')
    app = web.Application()
    app.router.add_post('/v1/completions', complete)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    plan = dict(base_url=f'http://127.0.0.1:{port}', model='fixture', timeout_s=0.05, metrics_interval_s=0.01)
    checks = []
    try:
        phase = dict(id='errors', mode='burst', requests=8, concurrency=8, max_inflight=16)
        cases = [(1, 512, 5), (2, 4096, 4), (3, 512, 4), (4, 512, 4), (5, 20000, 5), (6, 512, 4), (7, 512, 4), (9, 512, 4)]
        result = await client.execute_phase(make_prepared(phase, cases), plan)
        s, rows = (result['summary'], result['requests'])
        assert (s['requests'], s['successful'], s['failed'], s['timeouts']) == (8, 3, 5, 1)
        assert s['slo_eligible_requests'] == 7 and s['slo_joint_pass_rate'] == 2 / 7
        assert rows[4]['usage']['completion_tokens'] == 3 and rows[4]['slo_pass'] is None
        assert rows[4]['full_text'] == 'ab' and rows[4]['text_sha256']
        assert rows[2]['http_status'] == 500 and rows[7]['http_status'] == 429
        assert seen[3] == 1 and seen[9] == 1
        assert len(set(peers)) == len(peers), 'each request must use fresh TCP'
        assert result['metric_samples'] and all(('values' in x for x in result['metric_samples']))
        checks += ['SSE split chunks and full text/hash', 'actual output usage', '500/429/timeout/missing DONE/usage mismatch preserved', 'fresh TCP and zero retry', 'server metrics sampled']
        await asyncio.sleep(0.16)
        peak = 0
        slow_plan = dict(plan, timeout_s=0.5)
        phase = dict(id='closed', mode='closed-loop', requests=6, concurrency=2, max_inflight=2)
        result = await client.execute_phase(make_prepared(phase, [(10, 512, 4)] * 6), slow_plan)
        assert peak == 2 and result['summary']['successful'] == 6
        assert result['summary']['drain_s'] == 0
        checks += ['closed-loop worker limit and no open-loop claim']
        peak = 0
        phase = dict(id='open', mode='poisson', rate=20, duration_s=0.12, max_inflight=16)
        result = await client.execute_phase(make_prepared(phase, [(11, 512, 4)] * 3, [0.0, 0.01, 0.11]), slow_plan)
        s = result['summary']
        assert s['successful'] == 3 and s['censored'] == 0 and (peak >= 2)
        assert s['drain_s'] >= 0.08 and s['in_flight_at_window_end'] >= 1
        assert all((r['dispatch_lag_s'] < 0.04 for r in result['requests']))
        assert s['throughput']['completed_in_window_requests_s'] < 3 / s['measurement_window_s']
        assert abs(s['throughput']['cohort_requests_s'] * s['total_wall_s'] - 3) < 1e-09
        checks += ['open Poisson dispatch is not worker-limited', 'fixed offered window separate from drain']
        phase = dict(id='censored', mode='poisson', rate=100, duration_s=0.05, max_inflight=1)
        with tempfile.TemporaryDirectory(prefix='capacity-fixture-') as tmp:
            directory = Path(tmp)
            with open_output(directory / 'requests.jsonl', 'x') as sink:
                result = await client.execute_phase(make_prepared(phase, [(11, 512, 4)] * 3, [0.0, 0.001, 0.002]), slow_plan, directory / 'metrics', sink)
            lines = (directory / 'requests.jsonl').read_text().splitlines()
            assert len(lines) == 3
            assert len(list((directory / 'metrics').glob('*.prom'))) > 0
        s = result['summary']
        assert s['censored'] == 2 and s['successful'] == 1 and (not s['client_uncensored'])
        assert s['slo_joint_pass_rate'] == 1 / 3
        assert s['stability_classification'] == 'NOT_ASSESSED_BY_CLIENT'
        checks += ['safety cap censors rather than throttles open arrivals', 'censored arrivals excluded from successful throughput', 'incremental JSONL and raw metric evidence', 'no automatic stable-capacity claim']
        return (checks, dict(error_phase_successes=3, error_phase_failures=5, fixture_port=port, phases=4))
    finally:
        await runner.cleanup()


def main():
    start = time.perf_counter()
    checks = metric_checks()
    more, details = asyncio.run(fixture_checks())
    print(json.dumps(dict(status='PASS', checks=checks+more, details=details,
                          elapsed_s=time.perf_counter()-start), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
