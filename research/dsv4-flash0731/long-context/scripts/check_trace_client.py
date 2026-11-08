#!/usr/bin/env python3
"""CPU-only trace protocol checks against an owned ephemeral localhost fixture.

No GPU imports or model requests. The existing independent auditor recomputes
the actual HTTP rows, time windows, SLOs, and raw metrics.
"""
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
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time

from aiohttp import web

SCRIPTS = Path(__file__).resolve().parents[2] / 'scenarios/scripts'
sys.path.insert(0, str(SCRIPTS))
import capacity_client as client
import capacity_audit as audit


def trace_phase(**updates):
    phase = dict(id='trace-order', mode='trace', duration_s=.14, seed=2026110601,
                 shape_mix=[dict(name='short', input_tokens=512, output_tokens=4),
                            dict(name='long', input_tokens=16384, output_tokens=4)],
                 arrivals=[dict(offset_s=0., shape_name='long'),
                           dict(offset_s=.025, shape_name='short'),
                           dict(offset_s=.025, shape_name='short'),
                           dict(offset_s=.07, shape_name='long'),
                           dict(offset_s=.10, shape_name='short')])
    phase.update(updates)
    return phase


def plan_for(phase, port=8260):
    return dict(schema_version=1, base_url=f'http://127.0.0.1:{port}',
                model='fixture', timeout_s=2., max_inflight=16,
                metrics_interval_s=.1, phases=[phase])


def prepare_fixture(phase):
    offsets = client.make_offsets(phase)
    items = client.make_items(phase, len(offsets))
    prompts = {}
    for item in items:
        ids = [item['index'] + 1] + [7] * (item['input_tokens'] - 1)
        digest = client.base.digest_ids(ids)
        item['prompt_sha256'] = digest
        prompts[digest] = dict(sha256=digest, input_tokens=len(ids), token_ids=ids)
    return dict(spec=phase, planned_offsets_s=offsets, items=items, prompts=prompts)


def validation_checks():
    raw = plan_for(trace_phase())
    normalized = client.validate_plan(raw)
    assert normalized == audit.normalize_plan(raw)
    phase = normalized['phases'][0]
    expected = ['long', 'short', 'short', 'long', 'short']
    offsets = client.make_offsets(phase)
    items = client.make_items(phase, len(offsets))
    assert offsets == [0., .025, .025, .07, .10]
    assert [x['shape_name'] for x in items] == expected
    regenerated_offsets, regenerated_items = audit.regenerate(phase)
    assert (offsets, items) == (regenerated_offsets, regenerated_items)
    alternate = dict(phase, seed=999)
    assert [x['shape_name'] for x in client.make_items(alternate, 5)] == expected
    assert client.make_offsets(alternate) == offsets
    # The seed still controls prompt generation but never changes trace order.
    assert client.make_items(alternate, 5)[0]['seed'] != items[0]['seed']
    checks = ['explicit long/short order and simultaneous offsets retained',
              'independent normalize/regenerate agrees for explicit trace',
              'trace arrival/shape order is independent of sampling seed']

    mutations = {
        'negative offset': lambda p: p['arrivals'][0].update(offset_s=-.1),
        'nonmonotonic offsets': lambda p: p['arrivals'][2].update(offset_s=.01),
        'offset at horizon': lambda p: p['arrivals'][-1].update(offset_s=p['duration_s']),
        'nonfinite offset': lambda p: p['arrivals'][0].update(offset_s=float('nan')),
        'boolean offset': lambda p: p['arrivals'][0].update(offset_s=False),
        'string offset': lambda p: p['arrivals'][0].update(offset_s='0'),
        'unknown shape': lambda p: p['arrivals'][0].update(shape_name='missing'),
        'non-string shape': lambda p: p['arrivals'][0].update(shape_name=[]),
        'non-object arrival': lambda p: p['arrivals'].__setitem__(0, []),
        'empty arrivals': lambda p: p.update(arrivals=[]),
        'too many arrivals': lambda p: p.update(arrivals=[dict(offset_s=0., shape_name='short')] * 100001),
        'zero horizon': lambda p: p.update(duration_s=0),
        'infinite horizon': lambda p: p.update(duration_s=float('inf')),
        'boolean horizon': lambda p: p.update(duration_s=True),
        'named shape mix': lambda p: p.update(shape_mix='short_mixed'),
        'requests conflict': lambda p: p.update(requests=5),
        'rate conflict': lambda p: p.update(rate=1),
        'concurrency conflict': lambda p: p.update(concurrency=1),
    }
    for label, mutate in mutations.items():
        bad = trace_phase()
        mutate(bad)
        for name, validator in [('client', client.validate_plan), ('auditor', audit.normalize_plan)]:
            try:
                validator(plan_for(bad))
            except ValueError:
                pass
            else:
                raise AssertionError(f'{name} accepted {label}')
        checks.append(f'invalid trace rejected independently: {label}')

    delayed = dict(input_tokens=512, ok=True, usage=dict(completion_tokens=4),
                   ttft_s=.1, e2e_s=.13, tpot_s=.01, dispatch_lag_s=1.1,
                   scheduled_ttft_s=1.2, scheduled_e2e_s=1.23)
    measured = client.add_slo(deepcopy(delayed), 'trace')
    assert measured['slo_latency_basis'] == 'scheduled' and measured['slo_pass'] is False
    assert client.add_slo(deepcopy(delayed), 'closed-loop')['slo_pass'] is True
    assert {key: measured[key] for key in audit.row_slo(delayed, 'trace')} == audit.row_slo(delayed, 'trace')
    checks.append('trace SLO includes dispatch lag and independently matches Research SLO')
    return checks


async def http_checks():
    active = peak = 0
    seen, peers = ([], [])

    async def completions(request):
        nonlocal active, peak
        payload = await request.json()
        marker = payload['prompt'][0]
        active += 1
        peak = max(peak, active)
        seen.append(dict(marker=marker, prompt_tokens=len(payload['prompt']), started=time.perf_counter()))
        peers.append(request.transport.get_extra_info('peername'))
        try:
            await asyncio.sleep(0.22 if len(payload['prompt']) > 1024 else 0.03)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"index":0,"text":"a"}]}\n\n')
            await asyncio.sleep(0.005)
            usage = dict(prompt_tokens=len(payload['prompt']), completion_tokens=payload['max_tokens'], total_tokens=len(payload['prompt']) + payload['max_tokens'])
            event = dict(choices=[dict(index=0, text='b', finish_reason='length')], usage=usage)
            await response.write(('data: ' + json.dumps(event) + '\n\n').encode())
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        finally:
            active -= 1

    async def metrics(request):
        return web.Response(text=f'vllm:num_requests_running{{model_name="fixture"}} {active}\nvllm:num_requests_waiting{{model_name="fixture"}} 0\nvllm:kv_cache_usage_perc{{model_name="fixture"}} 0.125\nvllm:num_preemptions_total{{model_name="fixture"}} 0\n')
    app = web.Application()
    app.router.add_post('/v1/completions', completions)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    checks, reports = ([], [])
    try:
        phases = [trace_phase(), trace_phase(id='trace-idle-tail', duration_s=0.15, arrivals=[dict(offset_s=0.0, shape_name='short')]), trace_phase(id='trace-censored', duration_s=0.05, max_inflight=1, arrivals=[dict(offset_s=0.0, shape_name='long'), dict(offset_s=0.001, shape_name='short'), dict(offset_s=0.002, shape_name='short')])]
        for raw_phase in phases:
            peak = 0
            first_seen = len(seen)
            plan = client.validate_plan(plan_for(raw_phase, port))
            phase = plan['phases'][0]
            prepared = prepare_fixture(phase)
            with tempfile.TemporaryDirectory(prefix='trace-fixture-') as temp:
                directory = Path(temp)
                rows_path = directory / f'capacity-{phase['id']}-requests.jsonl'
                with open_output(rows_path, 'x') as sink:
                    result = await client.execute_phase(prepared, plan, directory / f'capacity-{phase['id']}-metrics', sink)
                result['raw_requests_jsonl'] = str(rows_path)
                independent = audit.Auditor(directory)
                independent.run(directory / 'capacity.json', result, prepared)
                assert not independent.errors, independent.errors
                assert len(rows_path.read_text().splitlines()) == len(prepared['items'])
            rows, summary = (result['requests'], result['summary'])
            assert summary['measurement_window_s'] == phase['duration_s']
            assert summary['total_wall_s'] >= phase['duration_s']
            assert summary['planned_rate_s'] is None
            assert all((row['slo_latency_basis'] == 'scheduled' for row in rows))
            assert [row['shape_name'] for row in rows] == [arrival['shape_name'] for arrival in phase['arrivals']]
            reports.append(dict(phase=phase['id'], successful=summary['successful'], censored=summary['censored'], observed_http_peak=peak, measurement_window_s=summary['measurement_window_s'], wall_s=summary['total_wall_s'], drain_s=summary['drain_s']))
            if phase['id'] == 'trace-order':
                assert summary['successful'] == 5 and peak >= 3
                assert rows[1]['dispatch_offset_s'] < rows[0]['end_offset_s']
                assert rows[2]['dispatch_offset_s'] < rows[0]['end_offset_s']
                assert rows[3]['dispatch_offset_s'] < rows[0]['end_offset_s']
                assert rows[1]['planned_offset_s'] == rows[2]['planned_offset_s'] == 0.025
                assert summary['drain_s'] > 0.1 and summary['in_flight_at_window_end'] >= 1
                assert seen[first_seen]['prompt_tokens'] == 16384
                assert Counter((x['marker'] for x in seen[first_seen:])) == Counter(range(1, 6))
                assert abs(summary['throughput']['cohort_requests_s'] * summary['total_wall_s'] - 5) < 1e-09
                checks.extend(['HTTP trace keeps specified long/short sequence and simultaneous arrivals', 'HTTP dispatch proceeds while earlier long requests are incomplete', 'trace horizon and drain separated in client and independent audit'])
            elif phase['id'] == 'trace-idle-tail':
                assert summary['successful'] == 1 and rows[0]['end_offset_s'] < phase['duration_s']
                assert summary['drain_s'] < 0.05
                checks.append('trace waits until fixed horizon even when the last request completes early')
            else:
                assert summary['successful'] == 1 and summary['censored'] == 2 and (peak == 1)
                assert not summary['client_uncensored']
                assert len(seen) - first_seen == 1
                assert all((row['error']['type'] == 'ClientSafetyCap' for row in rows[1:]))
                checks.append('trace safety cap records censored arrivals instead of queuing or retrying')
        assert len(set(peers)) == len(peers)
        checks.extend(['fresh TCP and no hidden retries in trace requests', 'independent audit verifies actual HTTP rows, summaries, JSONL and raw Prometheus for all trace phases'])
        return (checks, reports)
    finally:
        await runner.cleanup()


def main():
    start = time.perf_counter()
    checks = validation_checks()
    http, reports = asyncio.run(http_checks())
    print(json.dumps(dict(status='PASS', scope='CPU-only owned localhost fixture; no GPU/model requests',
                         checks=checks + http, phases=reports, elapsed_s=time.perf_counter()-start),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
