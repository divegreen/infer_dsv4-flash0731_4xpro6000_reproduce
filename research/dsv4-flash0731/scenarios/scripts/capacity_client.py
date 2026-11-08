#!/usr/bin/env python3
"""Bounded local capacity experiments, driven by a reproducible JSON phase plan.

No GPU imports, HTTP retries, or hidden throttling of open arrivals. Every
scheduled arrival has a row, including requests censored by the safety cap.
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
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import urlsplit

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capacity_base', ROOT.parent / 'scripts/benchmark_client.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
DEFAULT_TOKENIZER = str(model_path())
MIXES = {
    'short_mixed': [dict(name='p512_o128', input_tokens=512, output_tokens=128, weight=70),
                    dict(name='p4096_o256', input_tokens=4096, output_tokens=256, weight=20),
                    dict(name='p16384_o128', input_tokens=16384, output_tokens=128, weight=10)],
    'long_generation': [dict(name='p512_o128', input_tokens=512, output_tokens=128, weight=70),
                        dict(name='p1024_o1024', input_tokens=1024, output_tokens=1024, weight=30)],
}
METRIC_KEYS = ('ttft_s', 'scheduled_ttft_s', 'tpot_s', 'e2e_s', 'scheduled_e2e_s',
               'dispatch_lag_s', 'ttft_ratio', 'e2e_ratio')
SLO = dict(name='Phase A Research SLO, not a business SLA',
           input_bounds=[1024, 4096, 16384], ttft_s=[1., 3., 8.], tpot_s=.050,
           e2e='ttft_limit + .050 * (actual_output_tokens - 1)',
           primary='P95(TTFT ratio)<=1 and P95(E2E ratio)<=1 and P95(TPOT)<=.050',
           legacy='TTFT<=1s, TPOT<=.050s, E2E<=10s', long_context='no TTFT/E2E SLO')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    """Atomically checkpoint; the caller claims its output path exclusively."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    write_text(temp, json.dumps(obj, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def base_url(value):
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError('Invalid local port') from exc
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not port
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip('/') not in ('', '/v1')):
        raise ValueError('Only http://127.0.0.1:PORT[/v1] is allowed')
    return f'http://127.0.0.1:{port}'


def percentile(values, probability):
    if not values:
        return None
    values = sorted(values)
    pos = probability * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(n=len(values), p50=percentile(values, .5), p95=percentile(values, .95),
                p99=percentile(values, .99), max=max(values) if values else None)


def input_tier(tokens):
    if tokens <= 1024:
        return 'le1k', 1.
    if tokens <= 4096:
        return 'le4k', 3.
    if tokens <= 16384:
        return 'le16k', 8.
    return 'gt16k', None


def add_slo(row, mode):
    tier, limit = input_tier(row['input_tokens'])
    row['input_tier'] = tier
    usage = row.get('usage') or {}
    actual = usage.get('completion_tokens')
    actual = actual if type(actual) is int and actual >= 1 else None
    row['slo_applicable'] = limit is not None
    row['ttft_limit_s'] = limit
    row['e2e_limit_s'] = limit + .050 * (actual - 1) if limit is not None and actual else None
    # Burst requests are simultaneous offered load: preserve dispatch delay too.
    basis = 'scheduled_' if mode in ('poisson', 'burst', 'trace') else ''
    row['slo_latency_basis'] = 'scheduled' if basis else 'request'
    ttft, e2e = row.get(basis + 'ttft_s'), row.get(basis + 'e2e_s')
    row['ttft_ratio'] = ttft / limit if ttft is not None and limit else None
    row['e2e_ratio'] = e2e / row['e2e_limit_s'] if e2e is not None and row['e2e_limit_s'] else None
    # A one-token completion has no inter-token interval; do not invent TPOT=0.
    tpot_pass = actual == 1 or (row.get('tpot_s') is not None and row['tpot_s'] <= .050)
    row['slo_pass'] = (bool(row.get('ok') and row['ttft_ratio'] is not None
                            and row['ttft_ratio'] <= 1 and row['e2e_ratio'] is not None
                            and row['e2e_ratio'] <= 1 and tpot_pass)
                       if limit is not None else None)
    row['legacy_slo_pass'] = bool(row.get('ok') and ttft is not None and ttft <= 1
                                  and e2e is not None and e2e <= 10 and tpot_pass)
    return row


def validate_plan(plan):
    if plan.get('schema_version') != 1:
        raise ValueError('plan schema_version must be 1')
    plan = json.loads(json.dumps(plan))
    plan['base_url'] = base_url(plan.get('base_url', 'http://127.0.0.1:8260'))
    plan.setdefault('model', 'dsv4')
    plan.setdefault('tokenizer_path', DEFAULT_TOKENIZER)
    plan.setdefault('seed', 20261106)
    plan.setdefault('timeout_s', 300.)
    plan.setdefault('max_inflight', 256)
    plan.setdefault('metrics_interval_s', 1.)
    if not 1 <= plan['max_inflight'] <= 4096 or not 0 < plan['timeout_s'] <= 3600:
        raise ValueError('max_inflight must be 1..4096 and timeout_s in (0,3600]')
    if not .1 <= plan['metrics_interval_s'] <= 60:
        raise ValueError('metrics_interval_s must be .1..60')
    phases = plan.get('phases')
    if not isinstance(phases, list) or not phases:
        raise ValueError('plan requires nonempty phases')
    ids = set()
    for index, phase in enumerate(phases):
        name = phase.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in ids:
            raise ValueError('Phase IDs must be unique safe file labels')
        ids.add(name)
        mode = phase.get('mode')
        if mode not in ('poisson', 'burst', 'closed-loop', 'trace'):
            raise ValueError(f'Unsupported phase mode {mode}')
        phase.setdefault('seed', plan['seed'] + index)
        phase.setdefault('ignore_eos', True)
        phase.setdefault('role', 'measure')
        phase.setdefault('max_inflight', plan['max_inflight'])
        if not 1 <= phase['max_inflight'] <= 4096:
            raise ValueError('Phase max_inflight must be 1..4096')
        if mode == 'trace':
            duration = phase.get('duration_s')
            if (type(duration) not in (int, float) or not math.isfinite(duration)
                    or not 0 < duration <= 7200):
                raise ValueError('Trace duration_s must be finite and in (0,7200]')
            if any(key in phase for key in ('requests', 'rate', 'concurrency')):
                raise ValueError('Trace uses explicit arrivals, not requests/rate/concurrency')
            arrivals = phase.get('arrivals')
            if not isinstance(arrivals, list) or not 1 <= len(arrivals) <= 100000:
                raise ValueError('Trace requires 1..100000 explicit arrivals')
            previous = -1.
            for arrival in arrivals:
                if not isinstance(arrival, dict):
                    raise ValueError('Trace arrivals must be objects')
                offset = arrival.get('offset_s')
                if (type(offset) not in (int, float) or not math.isfinite(offset)
                        or not previous <= offset < duration or offset < 0):
                    raise ValueError('Trace offsets must be finite, nondecreasing and in [0,duration_s)')
                previous = offset
            if not isinstance(phase.get('shape_mix'), list):
                raise ValueError('Trace shape_mix must be an explicit shape list')
        elif mode == 'poisson':
            if not (0 < phase.get('rate', 0) <= 1000 and 0 < phase.get('duration_s', 0) <= 7200):
                raise ValueError('Poisson rate in (0,1000] and duration_s in (0,7200] required')
            if 'requests' in phase:
                raise ValueError('Poisson is duration-based: do not terminate at a request count')
        else:
            if not 1 <= phase.get('requests', 0) <= 100000:
                raise ValueError('Burst/closed-loop requires 1..100000 requests')
            phase.setdefault('concurrency', phase['requests'] if mode == 'burst' else 1)
            if not 1 <= phase['concurrency'] <= 4096:
                raise ValueError('concurrency must be 1..4096')
            if mode == 'burst' and phase['concurrency'] < phase['requests']:
                raise ValueError('Single burst cannot be client-concurrency queued; concurrency must cover requests')
        mix = phase.get('shape_mix', 'short_mixed')
        if isinstance(mix, str):
            mix = MIXES[mix]
        if not isinstance(mix, list) or not mix:
            raise ValueError('shape_mix must be a named mix or a nonempty list')
        names = set()
        for shape in mix:
            if not shape.get('name') or shape['name'] in names:
                raise ValueError('Shape names must be unique and nonempty within a phase')
            names.add(shape['name'])
            if (type(shape.get('input_tokens')) is not int or shape['input_tokens'] < 1
                    or type(shape.get('output_tokens')) is not int or shape['output_tokens'] < 1
                    or not 0 < shape.get('weight', 1) < 1e9):
                raise ValueError('Invalid shape size or weight')
        phase['shape_mix'] = mix
        if mode == 'trace':
            if any(not isinstance(name, str) for name in names):
                raise ValueError('Trace shape names must be strings')
            if any(not isinstance(arrival.get('shape_name'), str)
                   or arrival['shape_name'] not in names for arrival in phase['arrivals']):
                raise ValueError('Every trace shape_name must exist in shape_mix')
    return plan


def make_offsets(phase):
    if phase['mode'] == 'trace':
        return [arrival['offset_s'] for arrival in phase['arrivals']]
    if phase['mode'] != 'poisson':
        return [0.] * phase['requests']
    rng = random.Random(phase['seed'])
    t, offsets = 0., []
    while True:
        t += rng.expovariate(phase['rate'])
        if t >= phase['duration_s']:
            break
        offsets.append(t)
    return offsets


def make_items(phase, count):
    shapes = phase['shape_mix']
    if phase['mode'] == 'trace':
        if count != len(phase['arrivals']):
            raise ValueError('Trace item count must match explicit arrivals')
        by_name = {shape['name']: shape for shape in shapes}
        selected = [by_name[arrival['shape_name']] for arrival in phase['arrivals']]
    else:
        rng = random.Random(phase['seed'] ^ 0x51A9)
        selected = rng.choices(shapes, weights=[s.get('weight', 1) for s in shapes], k=count)
    return [dict(index=i, shape_name=s['name'], input_tokens=s['input_tokens'],
                 requested_output_tokens=s['output_tokens'], seed=phase['seed'] + i,
                 ignore_eos=phase.get('ignore_eos', True)) for i, s in enumerate(selected)]


def prepare_phase(phase, tokenizer):
    offsets = make_offsets(phase)
    items = make_items(phase, len(offsets))
    prompts = {}
    for item in items:
        marker = f"{phase['seed']}:{item['index']}:{item['shape_name']}"
        ids = base.make_prompt(tokenizer, dict(input_tokens=item['input_tokens'], name='capacity'),
                               phase['seed'], item['index'], 'measure')
        h = base.digest_ids(ids)
        prompts[h] = dict(token_ids=ids, input_tokens=len(ids), sha256=h, marker=marker)
        item['prompt_sha256'] = h
    return dict(spec=phase, planned_offsets_s=offsets, items=items, prompts=prompts)


def empty_row(item, phase, planned, start):
    row = dict(item)
    row.update(request_id=f"{phase['id']}/{item['index']}", phase_id=phase['id'],
               planned_offset_s=planned, planned_unix=start + planned,
               dispatch_offset_s=None, dispatch_lag_s=None, end_offset_s=None,
               ok=False, admitted=False, censored=False, timed_out=False, http_status=None,
               usage=None, ttft_s=None, e2e_s=None, tpot_s=None, scheduled_ttft_s=None,
               scheduled_e2e_s=None, error=None, full_text='', text_sha256=None,
               done_received=False, text_event_seconds=[])
    return row


async def request_one(session, url, model, prompt, item, phase, row, origin, timeout):
    start = time.perf_counter()
    row.update(admitted=True, dispatch_offset_s=start-origin,
               dispatch_lag_s=max(0., start-origin-row['planned_offset_s']),
               dispatch_unix=time.time())
    payload = dict(model=model, prompt=prompt['token_ids'], max_tokens=item['requested_output_tokens'],
                   temperature=0, seed=item['seed'], n=1, stream=True, echo=False,
                   stream_options={'include_usage': True}, ignore_eos=item['ignore_eos'],
                   add_special_tokens=False)
    parts = []
    try:
        async with session.post(url + '/v1/completions', json=payload, allow_redirects=False,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            row['http_status'] = response.status
            if response.status != 200:
                body = (await response.content.read(8192)).decode(errors='replace')
                raise RuntimeError(f'HTTP {response.status}: {body[:2000]}')
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                raise ValueError('Response is not text/event-stream')
            async for event in base.sse_events(response.content):
                elapsed = time.perf_counter() - start
                if event.strip() == '[DONE]':
                    row.update(done_received=True, e2e_s=elapsed)
                    break
                obj = json.loads(event)
                if not isinstance(obj, dict) or obj.get('error'):
                    raise ValueError(f'Invalid/error SSE object: {str(obj)[:1000]}')
                if obj.get('usage') is not None:
                    row['usage'] = obj['usage']
                for choice in obj.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise ValueError('Unexpected extra choice')
                    text = choice.get('text') or ''
                    if not isinstance(text, str):
                        raise ValueError('Non-text completion')
                    if text:
                        if row['ttft_s'] is None:
                            row['ttft_s'] = elapsed
                        parts.append(text)
                        row['text_event_seconds'].append(elapsed)
                    if choice.get('finish_reason') is not None:
                        row['finish_reason'] = choice['finish_reason']
            usage = row['usage']
            if not row['done_received'] or row['ttft_s'] is None or not isinstance(usage, dict):
                raise ValueError('Missing DONE, actual usage, or nonempty text')
            p, o = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if type(p) is not int or type(o) is not int or o < 1 or p != item['input_tokens']:
                raise ValueError('Invalid actual usage or prompt count mismatch')
            if usage.get('total_tokens', p+o) != p+o:
                raise ValueError('Actual total_tokens mismatch')
            if o > item['requested_output_tokens'] or (item['ignore_eos'] and o != item['requested_output_tokens']):
                raise ValueError('Actual output length violates requested budget/ignore_eos')
            if row.get('finish_reason') not in ('stop', 'length'):
                raise ValueError('Missing or unsupported terminal finish_reason')
            if item['ignore_eos'] and row['finish_reason'] != 'length':
                raise ValueError('ignore_eos fixed output must terminate with length')
            row['tpot_s'] = (row['e2e_s']-row['ttft_s'])/(o-1) if o > 1 else None
            row['ok'] = True
    except asyncio.TimeoutError as exc:
        row.update(timed_out=True, error=dict(type='TimeoutError', message=str(exc) or 'request timeout'))
    except Exception as exc:
        row['error'] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        row['end_offset_s'] = time.perf_counter()-origin
        row['end_unix'] = time.time()
        row['elapsed_until_return_s'] = time.perf_counter()-start
        row['full_text'] = ''.join(parts)
        row['text_sha256'] = hashlib.sha256(row['full_text'].encode()).hexdigest()
        for key in ('ttft_s', 'e2e_s'):
            row['scheduled_' + key] = row['dispatch_lag_s'] + row[key] if row[key] is not None else None
        add_slo(row, phase['mode'])
    return row


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        try:
            label, raw = line.rsplit(None, 1)
            name = label.split('{')[0]
            if any(s in name for s in ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                                        'prefix_cache_hits_total', 'prefix_cache_queries_total')):
                value = float(raw)
                if math.isfinite(value):
                    result[label] = value
        except (ValueError, IndexError):
            continue
    return result


async def metric_snapshot(session, url, path=None):
    result = dict(unix=time.time())
    try:
        async with session.get(url + '/metrics', allow_redirects=False, timeout=aiohttp.ClientTimeout(total=3)) as response:
            body = await response.text()
            result['http_status'] = response.status
            if response.status != 200:
                raise RuntimeError(f'metrics HTTP {response.status}')
            result['values'] = parse_metrics(body)
            if path:
                write_text(path, body)
                result.update(raw_file=str(path), sha256=sha_file(path))
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
    result['finished_unix'] = time.time()
    return result


async def poll_metrics(session, url, origin, stop, interval, raw_dir, samples):
    seq = 0
    while not stop.is_set():
        row = dict(offset_s=time.perf_counter() - origin, unix=time.time())
        try:
            async with session.get(url + '/metrics', allow_redirects=False, timeout=aiohttp.ClientTimeout(total=3)) as response:
                body = await response.text()
                row['http_status'] = response.status
                if response.status != 200:
                    raise RuntimeError(f'metrics HTTP {response.status}')
                row['values'] = parse_metrics(body)
                if raw_dir:
                    path = raw_dir / f'{seq:05d}.prom'
                    write_text(path, body)
                    row['raw_file'] = str(path)
                    row['sha256'] = sha_file(path)
        except Exception as exc:
            row['error'] = dict(type=type(exc).__name__, message=str(exc))
        samples.append(row)
        seq += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def group_summary(rows):
    success = [r for r in rows if r['ok']]
    eligible = [r for r in rows if r.get('slo_applicable')]
    eligible_success = [r for r in success if r.get('slo_applicable')]
    metrics = {key: stats([r.get(key) for r in success]) for key in METRIC_KEYS}
    passes = sum(r.get('slo_pass') is True for r in rows)
    p95 = None
    if eligible_success:
        tr = stats([r['ttft_ratio'] for r in eligible_success])['p95']
        er = stats([r['e2e_ratio'] for r in eligible_success])['p95']
        tp = stats([r['tpot_s'] for r in eligible_success])['p95']
        p95 = bool(tr is not None and tr <= 1 and er is not None and er <= 1
                   and (tp is None or tp <= .050))
    return dict(requests=len(rows), admitted=sum(r['admitted'] for r in rows), successful=len(success),
                failed=sum(r['admitted'] and not r['ok'] for r in rows),
                censored=sum(r.get('censored', False) for r in rows),
                http_accepted=sum(r.get('http_status') == 200 for r in rows),
                timeouts=sum(r['timed_out'] for r in rows),
                http_status_counts=dict(Counter(str(r['http_status']) for r in rows if r['http_status'])),
                metrics=metrics, slo_eligible_requests=len(eligible), slo_passes=passes,
                slo_joint_pass_rate=passes/len(eligible) if eligible else None,
                slo_joint_pass_rate_success_only=passes/len(eligible_success) if eligible_success else None,
                p95_slo_pass=p95, legacy_slo_passes=sum(r.get('legacy_slo_pass', False) for r in rows),
                legacy_slo_joint_pass_rate=sum(r.get('legacy_slo_pass', False) for r in rows)/len(rows) if rows else None)


def summarize(rows, phase, wall, metric_samples, peak_inflight):
    result = group_summary(rows)
    mode = phase['mode']
    window = phase.get('duration_s', wall) if mode in ('poisson', 'trace') else wall
    success = [r for r in rows if r['ok']]
    within = [r for r in success if r['end_offset_s'] <= window]
    output = lambda seq: sum(r['usage']['completion_tokens'] for r in seq)
    good = [r for r in success if r.get('slo_pass') is True]
    good_within = [r for r in within if r.get('slo_pass') is True]
    result.update(measurement_window_s=window, total_wall_s=wall, drain_s=max(0., wall-window),
                  planned_arrivals=len(rows), planned_rate_s=phase.get('rate'),
                  realized_offered_rate_s=len(rows)/window if window else None,
                  successful_output_tokens=output(success), peak_inflight=peak_inflight,
                  in_flight_at_window_end=sum(r['admitted'] and r['dispatch_offset_s'] <= window
                                             and r['end_offset_s'] > window for r in rows),
                  throughput=dict(
                      cohort_requests_s=len(success)/wall, cohort_output_tokens_s=output(success)/wall,
                      completed_in_window_requests_s=len(within)/window,
                      completed_in_window_output_tokens_s=output(within)/window,
                      cohort_slo_goodput_requests_s=len(good)/wall,
                      cohort_slo_goodput_output_tokens_s=output(good)/wall,
                      completed_in_window_slo_goodput_requests_s=len(good_within)/window,
                      offered_window_slo_goodput_requests_s=len(good)/window),
                  metrics_poll_errors=sum('error' in s for s in metric_samples))
    result['by_shape'] = {s: group_summary([r for r in rows if r['shape_name'] == s])
                          for s in sorted({r['shape_name'] for r in rows})}
    result['by_input_tier'] = {s: group_summary([r for r in rows if r['input_tier'] == s])
                               for s in sorted({r['input_tier'] for r in rows})}
    # Time bins describe queue growth; classification is deliberately left to analysis.
    bins = []
    bin_width = max(1., window/4)
    for i in range(4):
        left, right = i*bin_width, min(window, (i+1)*bin_width)
        if left >= window:
            break
        cohort = [r for r in rows if left <= r['planned_offset_s'] < right]
        bins.append(dict(start_s=left, end_s=right, arrivals=len(cohort),
                         completed=sum(r['ok'] and left <= r['end_offset_s'] < right for r in rows),
                         outstanding_at_end=sum(r['admitted'] and r['dispatch_offset_s'] < right
                                                and r['end_offset_s'] > right for r in rows),
                         cohort_scheduled_ttft_s=stats([r['scheduled_ttft_s'] for r in cohort if r['ok']]),
                         cohort_dispatch_lag_s=stats([r['dispatch_lag_s'] for r in cohort if r['admitted']])))
    result['time_bins'] = bins
    tier_results = [x for x in result['by_input_tier'].values() if x['slo_eligible_requests']]
    result['overall_ratio_p95_slo_pass'] = result['p95_slo_pass']
    result['p95_slo_pass'] = (all(x['p95_slo_pass'] is True for x in tier_results) if tier_results else None)
    result['p95_slo_scope'] = 'each eligible input-length tier; aggregate normalized ratios are auxiliary'
    result['client_uncensored'] = result['censored'] == 0
    result['transport_success'] = result['failed'] == 0
    result['stability_classification'] = 'NOT_ASSESSED_BY_CLIENT'
    return result


async def execute_phase(prepared, plan, raw_dir=None, sink=None):
    phase, offsets, items, prompts = (prepared[k] for k in ('spec', 'planned_offsets_s', 'items', 'prompts'))
    rows, active, samples, all_tasks = [], set(), [], []
    origin, unix = None, None
    stop = asyncio.Event()
    peak = 0
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=False)
    # Unlimited connector prevents its pool from imposing an undocumented queue.
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
        before = await metric_snapshot(session, plan['base_url'], raw_dir/'before.prom' if raw_dir else None)
        origin, unix = time.perf_counter(), time.time()
        monitor = asyncio.create_task(poll_metrics(session, plan['base_url'], origin, stop,
                                                   plan['metrics_interval_s'], raw_dir, samples))
        def record(row):
            rows.append(row)
            if sink:
                sink.write(json.dumps(row, sort_keys=True)+'\n')
                sink.flush()

        async def one(item, planned):
            row = empty_row(item, phase, planned, unix)
            row = await request_one(session, plan['base_url'], plan['model'],
                                    prompts[item['prompt_sha256']], item, phase, row,
                                    origin, plan['timeout_s'])
            record(row)

        try:
            if phase['mode'] == 'closed-loop':
                queue = asyncio.Queue()
                for item in items:
                    queue.put_nowait(item)
                async def worker():
                    while not queue.empty():
                        item = queue.get_nowait()
                        await one(item, time.perf_counter()-origin)
                        queue.task_done()
                peak = min(phase['concurrency'], len(items))
                await asyncio.gather(*(worker() for _ in range(peak)))
            else:
                for item, planned in zip(items, offsets):
                    delay = origin + planned - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    active = {t for t in active if not t.done()}
                    if len(active) >= phase['max_inflight']:
                        row = empty_row(item, phase, planned, unix)
                        row.update(censored=True, end_offset_s=time.perf_counter()-origin,
                                   error=dict(type='ClientSafetyCap', message='open arrival not admitted; max_inflight reached'))
                        record(add_slo(row, phase['mode']))
                        continue
                    task = asyncio.create_task(one(item, planned))
                    active.add(task)
                    all_tasks.append(task)
                    peak = max(peak, len(active))
                if phase['mode'] in ('poisson', 'trace'):
                    await asyncio.sleep(max(0., origin+phase['duration_s']-time.perf_counter()))
                if all_tasks:
                    await asyncio.gather(*all_tasks)
            measured_wall = time.perf_counter()-origin
            measured_end_unix = time.time()
        finally:
            stop.set()
            await monitor
        after = await metric_snapshot(session, plan['base_url'], raw_dir/'after.prom' if raw_dir else None)
    wall = measured_wall
    rows.sort(key=lambda r: r['index'])
    return dict(phase=phase, start_unix=unix, end_unix=measured_end_unix, requests=rows,
                metric_samples=samples, metrics_before=before, metrics_after=after,
                summary=summarize(rows, phase, wall, samples, peak))


def prepare(plan):
    from tokenizers import Tokenizer
    tokenizer_file = Path(plan['tokenizer_path']).expanduser().resolve()/'tokenizer.json'
    tok = Tokenizer.from_file(str(tokenizer_file))
    tok.no_padding()
    tok.no_truncation()
    return [prepare_phase(phase, tok) for phase in plan['phases']], tokenizer_file


async def run_plan(plan, prepared, report, output):
    for phase in prepared:
        name = phase['spec']['id']
        report['active_phase'] = name
        save_json(output, report)
        rows_path = output.parent / f'{output.stem}-{name}-requests.jsonl'
        with open_output(rows_path, 'x') as sink:
            result = await execute_phase(phase, plan, output.parent / f'{output.stem}-{name}-metrics', sink)
        result['raw_requests_jsonl'] = str(rows_path)
        report['runs'].append(result)
        save_json(output, report)
        print(json.dumps(dict(phase=name, summary=result['summary']), sort_keys=True), flush=True)
        if result['summary']['failed'] or result['summary']['censored']:
            report['status'] = 'FAILED' if result['summary']['failed'] else 'CENSORED'
            report['stop_reason'] = 'Do not continue after transport failure or client cap; preserve evidence'
            break
    else:
        report['status'] = 'PASSED'
    report.pop('active_phase', None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan = validate_plan(load_document(args.plan))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as handle:
        handle.write('{}\n')
    report = dict(schema_version=1, status='PREPARING', created_utc=utc_now(), plan=plan, plan_path=str(args.plan.resolve()), plan_sha256=sha_file(args.plan), client_sha256=sha_file(__file__), base_helper_sha256=sha_file(base.__file__), python=sys.version, aiohttp=importlib.metadata.version('aiohttp'), research_slo=SLO, transport=dict(connection_policy='fresh_per_request', retries=0), metric_definitions=dict(tpot='(E2E-TTFT)/(actual output tokens-1), not token ITL', scheduled='planned arrival to response; includes client scheduling lag', throughput='cohort: completions / (window+drain); completed_in_window: completion-credited in fixed window', p95='per-request normalized TTFT/E2E ratios; no P99 or joint pass-rate gate', stability='not decided by client; inspect window trends, server queue, failures and repeats'), runs=[])
    save_json(args.output, report)
    try:
        prepared, tokenizer_file = prepare(plan)
        prompt_path = args.output.parent / (args.output.stem + '-workload.json')
        with open_output(prompt_path, 'x') as handle:
            json.dump(prepared, handle, separators=(',', ':'))
            handle.write('\n')
        report.update(workload_file=str(prompt_path), workload_sha256=sha_file(prompt_path), tokenizer_sha256=sha_file(tokenizer_file), prepared_requests={p['spec']['id']: len(p['items']) for p in prepared}, status='PREPARED')
        save_json(args.output, report)
        if args.prepare_only:
            print(json.dumps(dict(status='PREPARED', prepared_requests=report['prepared_requests'])), flush=True)
            return 0
        report['status'] = 'RUNNING'
        asyncio.run(run_plan(plan, prepared, report, args.output))
        return 0 if report['status'] == 'PASSED' else 1
    except BaseException as exc:
        report.update(status='FAILED', fatal_error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        report['finished_utc'] = utc_now()
        save_json(args.output, report)


if __name__ == '__main__':
    raise SystemExit(main())
