#!/usr/bin/env python3
"""Read terminal local Session evidence; reconcile cache usage with raw metrics.

Failed experiments remain failed. Complete replay phases are described only as
partial mechanism evidence. No HTTP/GPU/runtime imports or mutations.
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
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import zipfile

HERE=Path(__file__).resolve().parents[1]
GPU_HELPER=HERE.parent/'scenarios/scripts/summarize_scenarios.py'
_spec=importlib.util.spec_from_file_location('cache_analysis_gpu_helper',GPU_HELPER)
gpu_helper=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(gpu_helper)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def mean(xs):return sum(xs)/len(xs) if xs else None
def dist(xs):
    xs=sorted(x for x in xs if isinstance(x,(int,float)) and math.isfinite(x))
    def p(q):
        if not xs:return None
        x=(len(xs)-1)*q;i=math.floor(x);j=math.ceil(x)
        return xs[i]+(xs[j]-xs[i])*(x-i)
    return dict(n=len(xs),mean=mean(xs),p50=p(.5),p95=p(.95),p99=p(.99),max=max(xs) if xs else None)


def metrics(path):
    data={}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        key,value=line.rsplit(None,1)
        value=float(value)
        if math.isfinite(value):data[key]=value
    return data


def identity(bank,tokenizer):
    f=bank['phase']
    context={k:f.get(k) for k in ('kind','arrival','duration_s','rate','concurrency')}
    if f['kind']=='agent':context.update({k:f.get(k) for k in ('max_turns','max_tools','max_output_tokens')})
    fields=('index','planned_offset_s','seed','profile','prompts','outputs','tool_waits_s') if f['kind']=='replay' else ('index','planned_offset_s','seed','task_id')
    rows=[{k:s.get(k) for k in fields} for s in bank['sessions']]
    return digest(dict(tokenizer_sha256=tokenizer,arrival=context,sessions=rows))


def engine_metrics(deltas):
    engines=defaultdict(dict)
    for key,value in deltas.items():
        m=re.search(r'(?:\{|,)engine="([^"]+)"',key)
        engine=m[1] if m else 'unlabelled'
        name=key.split('{',1)[0]
        engines[engine][name]=engines[engine].get(name,0.)+value
    result={}
    for engine,totals in sorted(engines.items()):
        q=totals.get('vllm:prefix_cache_queries_total');h=totals.get('vllm:prefix_cache_hits_total')
        timings={}
        for t in ('queue','prefill','decode','inference'):
            n=totals.get('vllm:request_'+t+'_time_seconds_count');s=totals.get('vllm:request_'+t+'_time_seconds_sum')
            timings[t]=dict(count=n,sum_s=s,mean_s=s/n if s is not None and n else None)
        result[engine]=dict(totals=totals,cache_fraction=h/q if h is not None and q else None,server_timings=timings)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    status = load_document(run / 'status.json')
    experiment = load_document(run / 'experiment.json')
    report = load_document(run / 'sessions.json')
    command = load_document(run / 'command.json')
    config = load_document(run / 'config.json')
    server_command = load_document(run / 'server-command.json')
    prepared = load_document(run / 'sessions-prepared.json')
    assert status.get('end_utc') and report['status'] in ('COMPLETED', 'FAILED'), 'Terminal files required'
    success = status['exit_code'] == 0 and (not status['timed_out']) and (experiment['status'] == 'PASS') and (report['status'] == 'COMPLETED')
    files = [run / n for n in ('status.json', 'experiment.json', 'sessions.json', 'command.json', 'server-command.json', 'config.json', 'sessions-prepared.json', 'source-snapshot.zip', 'gpu.csv')]
    inputs = {str(path): sha(path) for path in files}
    phases = []
    checks = []

    def check(label, value):
        assert value, label
        checks.append(label)
    check('supervisor status source hashes match launch', status['source_sha256'] == command['source_sha256'])
    with zipfile.ZipFile(run / 'source-snapshot.zip') as archive:
        check('snapshot member set matches source hash manifest', len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(command['source_sha256']))
        check('every snapshotted source matches recorded hash', all((hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in command['source_sha256'].items())))
    client_member = 'scenarios/scripts/agent_session_client.py'
    check('actual runtime client matches source snapshot', report['client_sha256'] == command['source_sha256'][client_member])
    for name, value in report.get('helper_sha256', {}).items():
        member = 'scripts/benchmark_client.py' if name == 'benchmark_client' else 'scenarios/scripts/' + name + '.py'
        check('actual runtime helper ' + name, command['source_sha256'][member] == value)
    check('prepared bank matches actual runtime digest', digest(prepared) == report['prepared_sha256'])
    check('tokenizer identity matches prepared bank', prepared['tokenizer_sha256'] == report['tokenizer_sha256'])
    apc_enabled = '--no-enable-prefix-caching' not in server_command['command']
    check('APC configuration explicit', any((a in server_command['command'] for a in ('--enable-prefix-caching', '--no-enable-prefix-caching'))))
    gpu_ledger = {}
    devices, gpu_warnings = gpu_helper.load_gpu(run / 'gpu.csv', timezone.utc, gpu_ledger)
    inputs[str(GPU_HELPER.resolve())] = sha(GPU_HELPER)
    banks = {p['phase']['id']: p for p in prepared['phases']}
    for phase in report['phases']:
        f = phase['phase']
        raw = run / Path(phase['raw_dir']).name
        originals = [json.loads(line) for line in (raw / 'sessions.jsonl').read_text().splitlines()]
        check(f['id'] + ' raw/session summary identity', sorted(originals, key=lambda s: s['index']) == phase['sessions'])
        bank = banks[f['id']]
        check(f['id'] + ' actual phase matches prepared', bank['phase'] == f)
        if f['kind'] == 'replay':
            for s in originals:
                spec = bank['sessions'][s['index']]
                check(f['id'] + f' session{s['index']} exact prepared prompts', all((r['payload']['prompt'] == spec['prompts'][r['turn']] for r in s['requests'])))
        check(f['id'] + ' routing headers follow declared treatment', all(({k.lower(): v for k, v in (r.get('headers') or {}).items() if k.lower() == 'x-data-parallel-rank'} == ({'x-data-parallel-rank': str(s['index'] % 4)} if f['dp_routing'] == 'sticky4' else {}) for s in originals for r in s['requests'])))
        phase_files = [raw / n for n in ('sessions.jsonl', 'before.prom', 'after.prom', 'reset.json')]
        inputs.update({str(path): sha(path) for path in phase_files})
        reset = load_document(raw / 'reset.json')
        check(f['id'] + ' actual idle-reset success', reset['http_status'] == 200 and reset['response']['success'] is True)
        requests = [dict(r, profile=s.get('profile'), task_id=s.get('task_id')) for s in originals for r in s['requests']]
        ok = [r for r in requests if r['ok']]
        known = [r for r in requests if isinstance(r.get('usage'), dict) and r.get('cached_tokens') is not None]
        prompt = sum((r['usage']['prompt_tokens'] for r in known))
        cached = sum((r['cached_tokens'] for r in known))
        generated = sum((r['usage']['completion_tokens'] for r in known))
        before = metrics(raw / 'before.prom')
        after = metrics(raw / 'after.prom')
        interesting = ('prefix_cache_queries_total', 'prefix_cache_hits_total', 'prompt_tokens_total', 'generation_tokens_total', 'request_prefill_kv_computed_tokens_sum', 'num_preemptions_total', 'request_num_preemptions_sum')

        def chosen(key):
            name = key.split('{', 1)[0]
            return name.startswith('vllm:') and (name[5:] in interesting or (name.startswith('vllm:spec_decode_') and name.endswith('_total')) or any((name == 'vllm:request_' + t + '_time_seconds_' + suffix for t in ('queue', 'prefill', 'decode', 'inference') for suffix in ('count', 'sum'))))
        deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in before.keys() | after.keys() if chosen(k)}
        check(f['id'] + ' nonnegative selected counter deltas', all((v >= -1e-07 for v in deltas.values())))

        def total(name):
            values = [v for k, v in deltas.items() if k.split('{', 1)[0] == 'vllm:' + name]
            return sum(values) if values else None
        timings = {}
        for t in ('queue', 'prefill', 'decode', 'inference'):
            count = total('request_' + t + '_time_seconds_count')
            value = total('request_' + t + '_time_seconds_sum')
            timings[t] = dict(count=count, sum_s=value, mean_s=value / count if count else None)
        if f['kind'] == 'replay' and phase['status'] == 'COMPLETED':
            check(f['id'] + ' complete usage', len(known) == len(requests) == len(ok))
            for name, expected in [('prefix_cache_queries_total', prompt if apc_enabled else 0), ('prefix_cache_hits_total', cached), ('prompt_tokens_total', prompt), ('generation_tokens_total', generated)]:
                check(f['id'] + ' exact ' + name, total(name) == expected)
            check(f['id'] + ' exact logical computed token count', total('request_prefill_kv_computed_tokens_sum') == prompt - cached)
            check(f['id'] + ' all server timing counts match requests', all((x['count'] == len(requests) for x in timings.values())))
        groups = defaultdict(list)
        for r in requests:
            groups[r['profile'] or r['task_id'], r['turn']].append(r)
        grouped = []
        for (label, turn), rows in sorted(groups.items()):
            valid = [r for r in rows if r['ok'] and r.get('cached_tokens') is not None]
            grouped.append(dict(label=label, turn=turn, requests=len(rows), successful=len(valid), input_tokens=[r['usage']['prompt_tokens'] for r in valid], cached_tokens=[r['cached_tokens'] for r in valid], computed_prompt_tokens=[r['usage']['prompt_tokens'] - r['cached_tokens'] for r in valid], output_tokens=[r['usage']['completion_tokens'] for r in valid], ttft_text_s=dist([r.get('ttft_s') for r in valid]), first_output_s=dist([r.get('first_output_s') for r in valid]), e2e_s=dist([r.get('e2e_s') for r in valid]), tpot_output_s=dist([r.get('tpot_output_s') for r in valid])))
        sessions = []
        for s in originals:
            listed = sum((s.get(k, 0.0) for k in ('model_wall_s', 'tool_wall_s', 'grading_wall_s')))
            sessions.append(dict(index=s['index'], label=s.get('profile') or s.get('task_id'), protocol_ok=s['protocol_ok'], task_success=s['task_success'], terminal_reason=s['terminal_reason'], calls=len(s['requests']), elapsed_s=s['elapsed_s'], model_wall_s=s['model_wall_s'], tool_wall_s=s['tool_wall_s'], grading_wall_s=s['grading_wall_s'], other_session_wall_s=s['elapsed_s'] - listed, request_errors=[dict(turn=r['turn'], error=r.get('error'), cached_usage_available=r.get('cached_tokens') is not None) for r in s['requests'] if not r['ok']]))
        gpu = gpu_helper.integrate_gpu(devices, phase['start_unix'], phase['end_unix'], [0, 1, 2, 3], 2.5)
        accepted = total('spec_decode_num_accepted_tokens_total')
        drafted = total('spec_decode_num_draft_tokens_total')
        drafts = total('spec_decode_num_drafts_total')
        inclusion = success and f['role'] in ('measure', 'screen', 'confirm')
        peaks = {}
        for name in ('num_requests_waiting', 'num_requests_running', 'kv_cache_usage_perc'):
            values = []
            for m in phase['metric_samples']:
                if m.get('error'):
                    continue
                engine_values = [v for k, v in m.get('values', {}).items() if k.split('{', 1)[0] == 'vllm:' + name]
                if engine_values:
                    values.append(max(engine_values) if name == 'kv_cache_usage_perc' else sum(engine_values))
            peaks[name] = max(values) if values else None
        phases.append(dict(id=f['id'], kind=f['kind'], role=f['role'], cache_scope=f['cache_scope'], dp_routing=f['dp_routing'], arrival=f['arrival'], concurrency=f.get('concurrency'), phase_status=phase['status'], whole_experiment_healthy=success, included_as_measurement=inclusion, workload_sha256=identity(bank, report['tokenizer_sha256']), workload_scope='Exact replay tokens/output/think waits and cohort configuration' if f['kind'] == 'replay' else 'Task assignment/budget/arrival matched, not generated histories', seed=f['seed'], apc_enabled=apc_enabled, wall_s=phase['total_wall_s'], sessions=len(originals), requests=len(requests), successful_requests=len(ok), known_cached_usage_requests=len(known), known_prompt_tokens=prompt, known_cached_tokens=cached, known_output_tokens=generated, logical_computed_prompt_tokens=prompt - cached, cache_fraction=cached / prompt if prompt and len(known) == len(requests) else None, session_latency_s=dist([s['scheduled_latency_s'] for s in originals if s['protocol_ok']]), request_ttft_text_s=dist([r.get('ttft_s') for r in ok]), request_first_output_s=dist([r.get('first_output_s') for r in ok]), request_tpot_output_s=dist([r.get('tpot_output_s') for r in ok]), request_e2e_s=dist([r.get('e2e_s') for r in ok]), output_tokens_s=generated / phase['total_wall_s'] if len(known) == len(requests) else None, successful_tasks=sum((s['task_success'] is True for s in originals)) if f['kind'] == 'agent' else None, gpu=gpu, output_tokens_per_joule=generated / gpu['energy_j'] if len(known) == len(requests) and gpu['energy_j'] else None, speculative=dict(accepted_tokens=accepted, draft_tokens=drafted, drafts=drafts, accepted_fraction=accepted / drafted if accepted is not None and drafted else None, mean_accepted_plus_one=1 + accepted / drafts if accepted is not None and drafts else None), raw_counter_deltas=deltas, server_timings=timings, observed_gauge_peaks=peaks, per_engine=engine_metrics(deltas), gauge_aggregation='Queue/running sum engines; KV is highest single-engine fraction, never sum of percentages.', by_profile_or_task_and_turn=grouped, per_session=sessions))
    check('all terminal input hashes unchanged', all((sha(Path(path)) == value for path, value in inputs.items())))
    result = dict(status='PASS_TERMINAL_CACHE_ANALYSIS' if success else 'PARTIAL_EVIDENCE_FROM_FAILED_EXPERIMENT', run=str(run), created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)), git_commit=command['git_commit'], runtime_source_sha256=command['source_sha256'], config=config, tokenizer_sha256=report['tokenizer_sha256'], gpu_analysis_warnings=gpu_warnings, whole_experiment_exit=status['exit_code'], whole_experiment_status=report['status'], checks=checks, input_sha256=inputs, phases=phases, limitations=['No new model requests. Completed phases of a failed experiment are mechanism evidence, not a healthy capacity baseline.', 'Cache fractions are token-weighted and only defined when every request has cached usage.', 'Server histogram deltas are phase/request sums; they do not identify per-request GPU time or unique device busy time.', 'Small pilot samples have descriptive quantiles only; warmup remains excluded from performance comparisons.', 'GPU energy uses recorded board power, boundary-clipped linear interpolation, no extrapolation, gaps >2.5s skipped; tokens/J only with full four-GPU coverage.', 'Per-engine counters describe aggregate load/locality; explicit routing header is request intent, not an independently traced actual route.', 'Growing-history cold versus later turns are different inputs; APC speedup requires a matched control.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), phases=len(phases), output=str(args.output)), indent=2))


if __name__=='__main__':main()
