#!/usr/bin/env python3
"""Offline terminal Stage C runtime/warmup audit; never opens GPU or HTTP.

Reuses Stage B log parsing read-only, but distinguishes pilot from measurement.
Outputs are exclusive-create. No service/source/config/evidence is modified.
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
import ast
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import sysconfig
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
RESULTS = path_at(ROOT, 'scenarios/results')
B_HELPER = ROOT / 'long-context/scripts/review_runtime.py'
_spec = importlib.util.spec_from_file_location('stage_b_runtime_parser', B_HELPER)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
MEASURE_ROLES = {'screen', 'measure', 'confirm'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def mapping(t, width, phases):
    end = t + width
    hits = [p for p in phases if t < p['end_unix'] and end > p['start_unix']]
    first_measure = min(p['start_unix'] for p in phases if p['role'] in MEASURE_ROLES)
    exact = [p for p in hits if p['start_unix'] <= t and end <= p['end_unix']]
    if end <= phases[0]['start_unix']:
        scope = 'STARTUP_BEFORE_CLIENT'
    elif any(p['role'] in MEASURE_ROLES for p in hits):
        scope = 'MEASUREMENT_OR_BOUNDARY'
    elif hits:
        scope = 'NONMEASUREMENT_OR_BOUNDARY'
    elif t >= phases[-1]['end_unix']:
        scope = 'AFTER_ALL_PHASES'
    else:
        scope = 'BETWEEN_PHASES'
    return dict(scope=scope, phase_candidates=[p['id'] for p in hits],
                role_candidates=sorted({p['role'] for p in hits}),
                unique_containing_phase=exact[0]['id'] if len(exact) == 1 else None,
                min_gap_before_measurement_s=first_measure-end)


def field(text, name):
    match = re.search(r"'" + re.escape(name) + r"': (\[[^\]]*\]|True|False|\d+)", text)
    return ast.literal_eval(match[1]) if match else None


def collect_events(records, phases, source):
    events = []
    for pos, record in enumerate(records):
        text = record['text']
        kind = old.category(text)
        if kind is None and 'JIT kernel warmup' in text:
            kind = 'kernel_jit_warmup'
        if kind is None and 'NCCL WARN' in text:
            kind = 'nccl_warning'
        if kind is None:
            continue
        event = dict(source=source, line=record['line'], category=kind, text=text)
        if record['time']:
            t, width = record['time']
            event.update(timestamp_unix=t, timestamp_resolution_s=width,
                         timestamp_utc=utc(t), mapping=mapping(t, width, phases))
        else:
            before = next((r for r in reversed(records[:pos]) if r['time']), None)
            after = next((r for r in records[pos+1:] if r['time']), None)
            event.update(timestamp_unix=None, neighbor_log_bracket=dict(
                before=None if before is None else dict(line=before['line'], unix=before['time'][0]),
                after=None if after is None else dict(line=after['line'], unix=after['time'][0]+after['time'][1]),
                caution='File-order brackets are descriptive; multiprocess stdout can reorder.'))
        if kind == 'resource_tracker_warning':
            match = re.search(r'There appear to be (\d+) leaked (\w+) objects', text)
            event['reported_unregistered_resources'] = dict(count=int(match[1]), type=match[2]) if match else None
        events.append(event)
    return events


def source_identity():
    dist = importlib.metadata.distribution('vllm')
    names = ['utils/jit_monitor.py', 'logger.py', 'v1/worker/gpu_worker.py',
             'v1/worker/gpu/model_runner.py', 'v1/worker/gpu/cudagraph_utils.py',
             'model_executor/warmup/flashinfer_sparse_mla_warmup.py',
             'config/observability.py', 'config/vllm.py', 'config/scheduler.py',
             'distributed/utils.py',
             'v1/engine/coordinator.py', 'v1/metrics/loggers.py',
             'model_executor/models/registry.py', 'model_executor/models/utils.py',
             'models/deepseek_v4/__init__.py', 'models/deepseek_v4/nvidia/model.py',
             'models/deepseek_v4/nvidia/dspark.py']
    result = {}
    for name in names:
        path = Path(dist.locate_file('vllm/' + name))
        raw = path.read_bytes()
        entry = next(f for f in dist.files if str(f) == 'vllm/' + name)
        actual = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        if entry.hash.mode != 'sha256' or entry.hash.value != actual:
            raise ValueError('Installed source differs from wheel RECORD: ' + name)
        result[name] = dict(path=str(path), sha256=sha(raw), wheel_record_match=True)
    tracker = Path(sysconfig.get_path('stdlib')) / 'multiprocessing/resource_tracker.py'
    return dict(vllm_version=dist.version, sources=result,
                resource_tracker=dict(path=str(tracker), sha256=sha(tracker.read_bytes())))


def review(name):
    if Path(name).name != name or not name.startswith('c'):
        raise ValueError('Expected a simple Stage C experiment ID')
    path = RESULTS / name
    # Read terminal status first; do not inspect a running server log.
    terminal = json.loads((path / 'status.json').read_bytes())
    if (type(terminal.get('exit_code')) is not int or terminal['exit_code'] != 0
            or terminal.get('timed_out') is not False or terminal.get('supervisor_error')
            or not terminal.get('end_utc')):
        raise ValueError(name + ': successful terminal supervisor required')
    names = ['status.json', 'experiment.json', 'sessions.json', 'config.json',
             'server-command.json', 'command.json', 'server.log', 'process.log',
             'session-plan.json', 'source-snapshot.zip']
    raw = {n: (path / n).read_bytes() for n in names}
    data = {n: json.loads(v) for n, v in raw.items() if n.endswith('.json')}
    checks = []

    def check(label, value):
        if not value:
            raise ValueError(name + ': ' + label)
        checks.append(label)

    report = data['sessions.json']
    config = data['config.json']
    launched = data['server-command.json']
    launch = data['command.json']
    args = config['args']
    command = launched['command']
    check('launcher and client healthy terminal states',
          data['experiment.json']['status'] == 'PASS' and report['status'] == 'COMPLETED'
          and not report.get('active_phase'))
    check('terminal snapshot remained identical at initial read', data['status.json'] == terminal)
    check('supervisor and command experiment IDs and argv agree',
          terminal['experiment_id'] == launch['experiment_id'] == name
          and terminal['command'] == launch['command'])
    check('supervisor and command working directory agree with project',
          terminal['cwd'] == launch['cwd'] == str(ROOT.parents[1]))
    check('saved and client-executed plans agree with experiment ID',
          report['plan'] == data['session-plan.json'] and report['plan']['experiment_id'] == name)
    check('executed phase definitions and terminal states match saved plan',
          [row['phase'] for row in report['phases']] == report['plan']['phases']
          and all(row['status'] == 'COMPLETED' for row in report['phases']))
    check('supervisor and command source hashes agree',
          terminal['source_sha256'] == launch['source_sha256'])
    with zipfile.ZipFile(path / 'source-snapshot.zip') as archive:
        check('complete snapshot set matches launch manifest',
              len(archive.namelist()) == len(set(archive.namelist()))
              and set(archive.namelist()) == set(launch['source_sha256']))
        check('every snapshot member SHA matches launch manifest', all(
            sha(archive.read(n)) == h for n, h in launch['source_sha256'].items()))
        inventory = json.loads(archive.read('model-inventory.json'))
        saved_config = json.loads(archive.read('scenarios/configs/' + name + '.json'))
        check('actual config matches executed snapshot', saved_config == config)
    check('actual client SHA matches executed snapshot', report['client_sha256'] ==
          launch['source_sha256']['scenarios/scripts/agent_session_client.py'])
    check('supervisor restricts GPUs to 0,1,2,3',
          terminal['environment']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3')
    check('server environment matches config', launched['extra_environment'] == config.get('env', {}))
    expected_tail = args + ([] if '--enable-prompt-tokens-details' in args
                            else ['--enable-prompt-tokens-details'])
    check('actual command preserves config args and prompt detail injection', command[9:] == expected_tail)
    check('server binds intended local model and loopback port',
          command[1:9] == ['serve', inventory['model_path'], '--served-model-name', 'dsv4',
                            '--host', '127.0.0.1', '--port', str(config['port'])])
    model_path = Path(inventory['model_path']) / 'config.json'
    model_raw = model_path.read_bytes()
    check('model config bytes match snapshotted model inventory',
          sha(model_raw) == inventory['source_fingerprints']['config.json']['sha256'])
    model = json.loads(model_raw)
    check('no alternate model or HF config override in actual command',
          not any(x.startswith(('--hf-overrides', '--config-format', '--config-file', '--model='))
                  for x in expected_tail) and '--model' not in expected_tail)

    phases = []
    for row in report['phases']:
        p = row['phase']
        phases.append(dict(id=p['id'], role=p['role'], kind=p['kind'],
                           concurrency=p.get('concurrency'), start_unix=row['start_unix'],
                           end_unix=row['end_unix'], start_utc=utc(row['start_unix']),
                           end_utc=utc(row['end_unix'])))
    check('phase windows finite, ordered and nonoverlapping',
          all(math.isfinite(p['start_unix']) and math.isfinite(p['end_unix'])
              and p['start_unix'] < p['end_unix'] for p in phases)
          and all(a['end_unix'] <= b['start_unix'] for a, b in zip(phases, phases[1:])))
    year = datetime.fromisoformat(terminal['start_utc']).year
    records = old.parse_records(raw['server.log'], year)
    startup_records = [r for r in records if 'non-default args:' in r['text']]
    check('single API startup argument record present', len(startup_records) == 1)
    startup = old.parse_startup_args(startup_records[0]['text'].split('non-default args:', 1)[1].strip())
    engines = [r for r in records if 'Initializing a V1 LLM engine' in r['text']]
    check('resolved engine configuration logged', bool(engines))
    engine = engines[0]['text']
    parallel = {}
    for field_name in ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'):
        expected = int(old.flag(args, '--' + field_name.replace('_', '-'), '1'))
        check(field_name + ' matches startup and all engine logs',
              startup.get(field_name, 1) == expected
              and all(re.search(r'\b' + field_name + '=' + str(expected) + r'\b', e['text']) for e in engines))
        parallel[field_name] = expected
    apc = '--enable-prefix-caching' in args
    check('APC explicitly configured and matches startup/engine',
          ('--no-enable-prefix-caching' in args) != apc
          and startup.get('enable_prefix_caching') == apc
          and all('enable_prefix_caching=' + str(apc) in e['text'] for e in engines))
    for key, expected in [('reasoning_parser', 'deepseek_v4'), ('tool_call_parser', 'deepseek_v4'),
                          ('attention_backend', 'FLASHINFER_MLA_SPARSE_DSV4'), ('moe_backend', 'marlin'),
                          ('kv_cache_dtype', 'fp8_ds_mla')]:
        check(key + ' command/startup agree', old.flag(args, '--' + key.replace('_', '-')) == expected
              and startup.get(key) == expected)
    for key in ('max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'block_size'):
        check(key + ' command/startup agree',
              int(old.flag(args, '--' + key.replace('_', '-'))) == startup[key])
    check('GPU memory fraction command/startup agree',
          float(old.flag(args, '--gpu-memory-utilization')) == startup['gpu_memory_utilization'])
    threshold = int(old.flag(args, '--long-prefill-token-threshold', '0'))
    cadence = int(old.flag(args, '--prefill-schedule-interval', '1'))
    check('scheduler threshold/cadence explicit values agree with startup or source defaults',
          startup.get('long_prefill_token_threshold', 0) == threshold
          and startup.get('prefill_schedule_interval', 1) == cadence)
    check('prompt details / tools / chunked prefill enabled in startup',
          all(startup.get(k) is True for k in ('enable_prompt_tokens_details', 'enable_auto_tool_choice', 'enable_chunked_prefill')))
    check('chunked prefill and FP8 KV reflected in engine',
          all('enable_chunked_prefill=True' in e['text'] and 'kv_cache_dtype=fp8_ds_mla' in e['text'] for e in engines))
    ep = '--enable-expert-parallel' in args
    check('expert parallel request matches startup', startup.get('enable_expert_parallel', False) == ep)
    spec_arg = old.flag(args, '--speculative-config')
    speculative = json.loads(spec_arg) if spec_arg else None
    if speculative:
        check('speculative config matches startup and engine method/count',
              startup.get('speculative_config') == speculative
              and all("method='" + speculative['method'] + "'" in e['text']
                      and 'num_spec_tokens=' + str(speculative['num_speculative_tokens']) in e['text'] for e in engines))
    else:
        check('no speculative config in actual engine', all('speculative_config=None' in e['text'] for e in engines))
    check('actual V2 model runner log present', any('Using V2 Model Runner' in r['text'] for r in records))

    events = []
    for filename in ('server.log', 'process.log'):
        parsed = old.parse_records(raw[filename], year)
        events.extend(collect_events(parsed, phases, filename))
    warmups = []
    for row in report['phases']:
        if row['phase']['role'] != 'warmup':
            continue
        warmups.append(dict(phase=row['phase'], actual_peak_sessions=row['peak_active_sessions'],
            sessions=[dict(profile=s['profile'], requests=[
                dict(input_tokens=q['input_tokens'], output_tokens=q['output_tokens'],
                     cached_tokens=q['cached_tokens'], start_unix=q['dispatch_unix'], end_unix=q['end_unix'])
                for q in s['requests']]) for s in row['sessions']]))
    effective = dict(startup_args=startup, startup_line=startup_records[0]['line'],
        engine_config_records=[dict(line=r['line'], text=r['text']) for r in engines],
        parallel=parallel, expert_parallel=ep, apc_enabled=apc, speculative=speculative,
        long_prefill_token_threshold=threshold, prefill_schedule_interval=cadence,
        scheduler_defaults_caution='Missing nondefault fields use installed scheduler defaults 0 and 1; explicit nondefault values require startup agreement.',
        runner_v2=True, indexer_fp8=any('Using FP8 indexer cache' in r['text'] for r in records),
        jit_monitor_mode_warn=all("jit_monitor_mode='warn'" in e['text'] for e in engines),
        jit_verbose_false=all('jit_monitor_verbose=False' in e['text'] for e in engines),
        compilation_mode_none="'mode': <CompilationMode.NONE:" in engine,
        cudagraph_full_and_piecewise='CUDAGraphMode.FULL_AND_PIECEWISE' in engine,
        cudagraph_capture_sizes=field(engine, 'cudagraph_capture_sizes'),
        cudagraph_num_of_warmups=field(engine, 'cudagraph_num_of_warmups'),
        max_cudagraph_capture_size=field(engine, 'max_cudagraph_capture_size'),
        decoder_replay_cudagraph_capture_sizes=field(engine, 'decoder_replay_cudagraph_capture_sizes'),
        rank_assignment_records=[dict(line=r['line'], text=r['text']) for r in records if 'is assigned as DP rank' in r['text']],
        pp_layer_partition_records=[dict(line=r['line'], text=r['text']) for r in records if 'Hidden layers were unevenly partitioned:' in r['text']],
        async_enable_side_effect=any('using async scheduling' in r['text'] for r in records),
        async_disable_warnings=[dict(line=r['line'], text=r['text']) for r in records if 'Async scheduling is disabled' in r['text'] or 'Async scheduling will be disabled' in r['text']],
        async_evidence='Only explicit side-effect logs when present; otherwise source/default inference, no resolved async field.',
        model=dict(config_path=str(model_path), config_sha256=sha(model_raw),
                   architectures=model['architectures'], num_hidden_layers=model['num_hidden_layers'],
                   num_nextn_predict_layers=model['num_nextn_predict_layers'],
                   dspark_target_layer_ids=model['dspark_target_layer_ids'],
                   evidence='Snapshotted model inventory/config bytes and actual command/startup identity; layer count is metadata plus installed constructor evidence, not a per-worker layer execution trace.',
                   weight_identity_limit=inventory['identity_limit']))
    check('warmup observability and graph fields present', effective['jit_monitor_mode_warn']
          and effective['jit_verbose_false'] and bool(effective['cudagraph_capture_sizes']))
    jit = [e for e in events if e['category'] == 'runtime_jit_warning']
    first_measure = min(p['start_unix'] for p in phases if p['role'] in MEASURE_ROLES)
    hashes = {n: sha(v) for n, v in raw.items()}
    check('all input bytes unchanged after audit', all(sha((path / n).read_bytes()) == h for n, h in hashes.items())
          and sha(model_path.read_bytes()) == sha(model_raw))
    return dict(experiment_id=name, status='PASS_WITH_OBSERVABILITY_LIMITATIONS',
                evidence_sha256=hashes, git_commit=terminal['git_commit'], checks=checks,
                terminal={k: terminal[k] for k in ('start_utc', 'end_utc', 'exit_code', 'timed_out')},
                phases=phases, first_measurement_utc=utc(first_measure), effective=effective,
                workload_warmups=warmups, events=events,
                category_counts=dict(Counter(e['category'] for e in events)), visible_jit_warnings=jit,
                measurement_visible_jit_count=sum(e.get('mapping', {}).get('scope') == 'MEASUREMENT_OR_BOUNDARY' for e in jit),
                cleanup=dict(resource_tracker_warnings=[e for e in events if e['category'] == 'resource_tracker_warning'],
                             shutdown_events=[e for e in events if e['category'].startswith('shutdown_')],
                             limitation='Warnings trigger standard-library cleanup attempts; no post-exit object inventory and no proof of final cleanup success.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(schema_version=3, status='PASS_WITH_OBSERVABILITY_LIMITATIONS', created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__).read_bytes()), reused_parser=dict(path=str(B_HELPER), sha256=sha(B_HELPER.read_bytes())), installed_source_identity=source_identity(), timestamp_assumption='UTC consistent with supervisor/phase wall clock; seconds mapped to [t,t+1). No independent worker clock calibration; stdout may reorder.', limitations=['warning_once plus nonverbose kernel names deduplicate later shape specializations and filter non-local-first ranks.', 'JIT hook timestamp is post-compilation, not duration. Monitor is not exhaustive across every compiler/driver/runtime.', 'C1 external workload warmup and multi-size startup capture are distinct; neither certifies all mixed dynamic shapes warmed.', 'Pilots/warmups are excluded from measurement classification, not deleted from evidence.', 'Metadata layer count/source construction is not independent per-rank live model inspection.', 'Resource tracker and force-kill shutdown warnings retained; no inference failure or completed cleanup inferred from those warnings alone.'], experiments=[review(name) for name in args.runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output), experiments=[dict(id=e['experiment_id'], checks=len(e['checks']), jit=len(e['visible_jit_warnings']), measurement_jit=e['measurement_visible_jit_count'], categories=e['category_counts']) for e in result['experiments']]), indent=2))


if __name__ == '__main__':
    main()
