#!/usr/bin/env python3
"""Read completed Stage B logs; map visible JIT/capture events to phase windows.

No GPU, network, model imports or service actions. Output is exclusive-create.
Second-resolution log timestamps are intervals, not exact event times. Does not
claim warning_once is an exhaustive JIT trace.
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
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sysconfig

HERE = Path(__file__).resolve().parents[1]
RESULTS = path_at(HERE.parent, 'scenarios/results')
DEFAULT_RUNS = ['context-tp4-screen', 'context-tp2pp2-screen', 'context-dspark3-screen',
                'context-dp4ep-screen', 'context-tp4-th2048-screen']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def timestamp(line, year):
    full = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})', line)
    if full:
        dt = datetime.strptime(full[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        return dt.timestamp()+int(full[2])/1000, .001
    short = re.search(r'\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b', line)
    if short:
        dt = datetime.strptime(f'{year}-{short[1]}', '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        return dt.timestamp(), 1.
    return None


def category(line):
    if 'resource_tracker:' in line and 'UserWarning:' in line:
        return 'resource_tracker_warning'
    if '[shutdown]' in line:
        return 'shutdown_warning' if 'WARNING' in line else 'shutdown_event'
    if any(x in line for x in ('Starting profiler...', 'Stopping profiler...',
                               'profiler_start', 'profiler_stop',
                               'Torch profiling enabled.', 'Shutting down profiler')):
        return 'profiler_lifecycle'
    if any(x in line for x in ('Application shutdown complete.', 'Shutting down',
                               'Finished server process')):
        return 'shutdown_event'
    if 'during inference:' in line and ('JIT' in line or 'autotun' in line.lower()):
        return 'runtime_jit_warning'
    if 'Graph capturing finished' in line or 'CUDA graph capture' in line and 'finished' in line:
        return 'graph_capture_finished'
    if 'Kernel JIT monitor activated' in line:
        return 'jit_monitor_activated'
    if 'No FlashInfer SM120' in line:
        return 'autotune_fallback_warning'
    if '[Autotuner]' in line or 'autotune cache loaded' in line:
        return 'autotune_event'
    if 'Capturing CUDA graphs' in line:
        return 'graph_capture_progress'
    if 'Warming up' in line or 'Warmup finished' in line or 'JIT warmup' in line:
        return 'kernel_warmup'
    if re.search(r'\b(?:WARNING|ERROR)\b', line) and re.search(r'compil|captur|autotun', line, re.I):
        return 'other_compile_capture_warning'
    if re.search(r'\bERROR\b|Traceback \(most recent call last\)', line):
        return 'other_error_or_traceback'
    if re.search(r'\bWARNING\b|UserWarning:', line):
        return 'other_warning'
    return None


def phase_mapping(t, width, phases):
    end = t+width
    hits = [p for p in phases if t < p['end_unix'] and end > p['start_unix']]
    formal_start = min(p['start_unix'] for p in phases if p['role'] != 'warmup')
    if end <= phases[0]['start_unix']:
        scope = 'STARTUP_BEFORE_CLIENT'
    elif end <= formal_start:
        scope = 'BEFORE_FORMAL_WARMUP_OR_BOUNDARY'
    elif hits:
        scope = 'FORMAL_OR_BOUNDARY' if any(p['role'] != 'warmup' for p in hits) else 'WARMUP'
    elif t >= phases[-1]['end_unix']:
        scope = 'AFTER_ALL_PHASES'
    else:
        scope = 'BETWEEN_PHASES'
    exact = [p for p in hits if p['start_unix'] <= t and end <= p['end_unix']]
    return dict(scope=scope, phase_candidates=[p['id'] for p in hits],
                unique_containing_phase=exact[0]['id'] if len(exact) == 1 else None,
                min_gap_before_formal_s=formal_start-end)


def flag(args, name, default=None):
    return args[args.index(name)+1] if name in args else default


def parse_startup_args(text):
    """Read literals and the known Pydantic repr without executing any expression."""
    node = ast.parse(text, mode='eval').body
    if not isinstance(node, ast.Dict):
        raise ValueError('Expected startup argument dictionary')
    result = {}
    for key, value in zip(node.keys, node.values):
        name = ast.literal_eval(key)
        if (name == 'profiler_config' and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name) and value.func.id == 'ProfilerConfig'
                and not value.args and all(k.arg is not None for k in value.keywords)):
            result[name] = dict(__repr_type__='ProfilerConfig',
                               **{k.arg: ast.literal_eval(k.value) for k in value.keywords})
        else:
            result[name] = ast.literal_eval(value)
    return result


def parse_records(raw, year):
    records = []
    for number, line in enumerate(raw.decode().split('\n'), 1):
        for piece in line.split('\r'):
            clean = re.sub(r'\x1b\[[0-9;]*m', '', piece)
            records.append(dict(line=number, text=clean, time=timestamp(clean, year)))
    return records


def collect_events(records, phases, source):
    events = []
    for pos, record in enumerate(records):
        kind = category(record['text'])
        if kind is None:
            continue
        event = dict(source=source, line=record['line'], category=kind, text=record['text'])
        if record['time']:
            t, width = record['time']
            event.update(timestamp_unix=t, timestamp_resolution_s=width,
                         timestamp_utc=datetime.fromtimestamp(t, timezone.utc).isoformat(),
                         mapping=phase_mapping(t, width, phases))
        else:
            before = next((r for r in reversed(records[:pos]) if r['time']), None)
            after = next((r for r in records[pos+1:] if r['time']), None)
            event['timestamp_unix'] = None
            event['neighbor_log_bracket'] = dict(
                before=None if before is None else dict(line=before['line'], unix=before['time'][0]),
                after=None if after is None else dict(line=after['line'], unix=after['time'][0]+after['time'][1]),
                caution='File-order brackets are descriptive; multiprocess stdout can reorder')
        if kind == 'resource_tracker_warning':
            match = re.search(r'There appear to be (\d+) leaked (\w+) objects', record['text'])
            event['reported_unregistered_resources'] = (
                dict(count=int(match[1]), type=match[2]) if match else None)
        events.append(event)
    return events


def profile_controls(capacity, phases):
    profiling = capacity.get('profiling')
    if profiling is None:
        assert not any(p['role'] == 'profile' for p in phases)
        return None
    assert profiling['excluded_from_performance'] is True
    sessions = []
    for session in profiling['sessions']:
        phase = next(p for p in phases if p['id'] == session['phase_id'])
        assert phase['role'] == 'profile'
        start, stop = session['controls']
        assert [start['path'], stop['path']] == ['/start_profile', '/stop_profile']
        assert all(c['status'] == 'PASS' and c['http_status'] == 200 for c in (start, stop))
        assert start['started_unix'] <= start['finished_unix'] <= phase['start_unix']
        assert phase['end_unix'] <= stop['started_unix'] <= stop['finished_unix']
        sessions.append(dict(phase_id=phase['id'], controls=session['controls'],
            client_phase_bracketed=True,
            stop_control_elapsed_s=stop['finished_unix']-stop['started_unix'],
            caution='Stop call includes trace finalization/export; its elapsed time is not inference latency.'))
    assert len(sessions) == sum(p['role'] == 'profile' for p in phases)
    return dict(excluded_from_performance=True, scope=profiling.get('scope'), sessions=sessions)


def review(name):
    if Path(name).name != name:
        raise ValueError('Only simple result directory names are accepted')
    path = RESULTS/name
    names = ['status.json', 'experiment.json', 'capacity.json', 'config.json',
             'server-command.json', 'command.json', 'server.log', 'process.log']
    # Check terminal supervisor state before reading a log that may still grow.
    status = json.loads((path/'status.json').read_bytes())
    if status.get('exit_code') != 0 or status.get('timed_out') or not status.get('end_utc'):
        raise ValueError(f'{name}: expected completed, successful supervisor')
    data = {f: (path/f).read_bytes() for f in names}
    parsed = {f: json.loads(v) for f, v in data.items() if f.endswith('.json')}
    if parsed['experiment.json'].get('status') != 'PASS' or parsed['capacity.json'].get('status') != 'PASSED':
        raise ValueError(f'{name}: incomplete launcher/client evidence')
    capacity, config = parsed['capacity.json'], parsed['config.json']
    args = config['args']
    command = parsed['server-command.json']['command']
    assert command[-len(args):] == args
    phases = [dict(id=r['phase']['id'], role=r['phase']['role'],
                   start_unix=r['start_unix'], end_unix=r['end_unix']) for r in capacity['runs']]
    controls = profile_controls(capacity, phases)
    year = datetime.fromisoformat(status['start_utc']).year
    records = parse_records(data['server.log'], year)
    startup = [r for r in records if 'non-default args:' in r['text']]
    assert startup
    startup_args = parse_startup_args(startup[0]['text'].split('non-default args:', 1)[1].strip())
    engine = [r for r in records if 'Initializing a V1 LLM engine' in r['text']]
    assert engine
    first_engine = engine[0]['text']
    effective = dict(
        chunked_prefill=dict(requested='--enable-chunked-prefill' in args,
            startup=startup_args.get('enable_chunked_prefill'),
            engine_log='enable_chunked_prefill=True' in first_engine),
        budget=dict(requested=flag(args, '--max-num-batched-tokens'),
                    startup=startup_args.get('max_num_batched_tokens')),
        threshold=dict(requested=flag(args, '--long-prefill-token-threshold'),
                       startup_nondefault=startup_args.get('long_prefill_token_threshold'),
                       evidence='explicit command; default-zero omitted from non-default log'),
        prefill_schedule_interval=dict(requested=flag(args, '--prefill-schedule-interval'),
                       startup_nondefault=startup_args.get('prefill_schedule_interval'),
                       evidence='Explicit values checked against startup; omitted value requires source-default inference'),
        profiler_config=dict(requested=flag(args, '--profiler-config'),
                             startup_nondefault=startup_args.get('profiler_config')),
        runner_v2=any('Using V2 Model Runner' in r['text'] for r in records),
        kv_dtype=dict(requested=flag(args, '--kv-cache-dtype'),
                      engine_log='kv_cache_dtype=fp8_ds_mla' in first_engine),
        indexer_fp8=any('Using FP8 indexer cache' in r['text'] for r in records),
        prefix_caching_disabled='enable_prefix_caching=False' in first_engine,
        jit_monitor_mode_warn="jit_monitor_mode='warn'" in first_engine,
        jit_verbose_false='jit_monitor_verbose=False' in first_engine,
        async_explicit_enabled_log=any('using async scheduling' in r['text'] for r in records),
        async_disable_warning=any('Async scheduling is disabled' in r['text'] or 'Async scheduling will be disabled' in r['text'] for r in records),
        async_command_override=[a for a in args if 'async-scheduling' in a],
        async_evidence='DP has explicit enable side-effect log; others default/source inference, no resolved field in engine log',
        tensor_parallel_size=flag(args, '--tensor-parallel-size', '1'),
        pipeline_parallel_size=flag(args, '--pipeline-parallel-size', '1'),
        data_parallel_size=flag(args, '--data-parallel-size', '1'),
        expert_parallel='--enable-expert-parallel' in args,
        speculative_config=flag(args, '--speculative-config'),
        extra_environment=config.get('env', {}),
    )
    events = collect_events(records, phases, 'server.log')
    events.extend(collect_events(parse_records(data['process.log'], year), phases, 'process.log'))
    # Refuse to certify a moving input even when its status was stale.
    hashes = {f:sha(v) for f,v in data.items()}
    assert all(sha((path/f).read_bytes()) == digest for f,digest in hashes.items())
    return dict(experiment_id=name, evidence_hashes=hashes,
                terminal=dict(exit_code=status['exit_code'], timed_out=status['timed_out'],
                              start_utc=status['start_utc'], end_utc=status['end_utc']),
                phases=phases, effective=effective, profile_controls=controls, events=events,
                category_counts=dict(Counter(e['category'] for e in events)),
                visible_jit_warnings=[e for e in events if e['category']=='runtime_jit_warning'],
                formal_visible_jit_count=sum(e.get('mapping',{}).get('scope')=='FORMAL_OR_BOUNDARY'
                    for e in events if e['category']=='runtime_jit_warning'),
                first_formal_unix=min(p['start_unix'] for p in phases if p['role']!='warmup'),
                nonwarmup_scope='Profile diagnostics excluded from performance' if controls else 'Normal performance run',
                cleanup=dict(resource_tracker_warnings=[e for e in events if e['category']=='resource_tracker_warning'],
                             shutdown_events=[e for e in events if e['category'].startswith('shutdown_')],
                             caution='Unregistered resource warning triggers cleanup attempts; not proof of cleanup success or request failure. No post-exit OS resource inventory in this review.'),
                stable_input_hashes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--output', required=True, type=Path)
    opts = parser.parse_args()
    result = dict(schema_version=2, status='PASS_WITH_OBSERVABILITY_LIMITATIONS', reviewed_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__).read_bytes()), timestamp_assumption='Log wall times interpreted as UTC, matching supervisor/phase clock; whole seconds represented as [t,t+1). No independent per-worker clock calibration.', jit_limitations=['verbose=false warning_once deduplicates repeated names/configurations', 'warning_once default local scope filters non-local-first rank', 'post-compile timestamp is not compile start or duration', 'monitor covers registered runtimes, not every host/driver/JIT event'], excluded=['All runs not explicitly selected; no running logs read'], experiments=[review(name) for name in opts.runs], source_hashes={})
    dist = importlib.metadata.distribution('vllm')
    for name in ['utils/jit_monitor.py', 'logger.py', 'v1/worker/gpu_worker.py', 'model_executor/warmup/flashinfer_sparse_mla_warmup.py', 'config/observability.py', 'config/vllm.py', 'config/scheduler.py']:
        raw = Path(dist.locate_file('vllm/' + name)).read_bytes()
        rec = next((f for f in dist.files if str(f) == 'vllm/' + name))
        assert rec.hash.value == base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        result['source_hashes'][name] = dict(sha256=sha(raw), wheel_record_match=True)
    tracker = Path(sysconfig.get_path('stdlib')) / 'multiprocessing/resource_tracker.py'
    result['resource_tracker_source'] = dict(path=str(tracker), sha256=sha(tracker.read_bytes()), evidence='Read installed standard-library finally cleanup path; no wheel RECORD for stdlib')
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(opts.output, 'x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(opts.output), experiments=[dict(id=e['experiment_id'], counts=e['category_counts'], formal_visible_jit_count=e['formal_visible_jit_count']) for e in result['experiments']]), indent=2))


if __name__ == '__main__':
    main()
