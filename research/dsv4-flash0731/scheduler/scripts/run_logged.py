#!/usr/bin/env python3
"""Bounded Stage D supervisor with immutable local/installed-source identity.

Reuses the research GPU lock and owned process-group pattern from Stage A/B/C.
Never modifies the installed environment or prior stages. Snapshot keys retain
research-relative paths; selected installed bytes use installed/vllm/ prefixes.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output, snapshot_inputs

import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
D_RUNTIME = ('run_logged.py', 'serve_study.py', 'study_client.py', 'research_scheduler.py')
D_OPTIONAL_RUNTIME = ('run_study_series.py',)
INSTALLED = (
    'config/scheduler.py', 'config/vllm.py', 'config/profiler.py',
    'engine/arg_utils.py', 'v1/engine/core.py', 'v1/core/sched/scheduler.py',
    'v1/core/sched/async_scheduler.py', 'v1/core/sched/interface.py',
    'v1/core/sched/request_queue.py', 'v1/core/kv_cache_manager.py',
    'v1/request.py', 'utils/import_utils.py',
    'entrypoints/openai/completion/serving.py',
    'entrypoints/openai/completion/protocol.py',
    'v1/engine/input_processor.py', 'v1/engine/async_llm.py',
)
DEPENDENCY_TREES = ('scripts', 'scenarios/scripts',
                    'long-context/scripts', 'sessions/scripts')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or any(p.is_symlink() for p in path.parents):
        raise ValueError('Regular source file without symlink parents required: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def stable_read(path):
    before = fingerprint(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        data = stream.read()
    if before != fingerprint(path):
        raise ValueError('Source changed while reading: ' + str(path))
    return data


def write_json(path, value, *, initial=False):
    data = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if initial:
        with open_output(path, 'x') as stream:
            stream.write(data)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with open_output(temporary, 'x') as stream:
            stream.write(data)
        temporary.replace(path)


def quality_fixture_inputs(research):
    """Exact fixture selection read and hashed by the frozen quality client."""
    directory = path_at(research, 'scenarios/workloads/agent')
    required = [directory / name for name in ('tasks.json', 'quality.json', 'replay_context.md')]
    if any((not p.is_file() for p in required)):
        raise ValueError('Missing required quality fixture metadata')
    tasks = json.loads(stable_read(directory / 'tasks.json'))['tasks']
    task_ids = {t['id'] for t in tasks}
    if any((not isinstance(t, str) or Path(t).name != t or t in ('.', '..') for t in task_ids)):
        raise ValueError('Invalid quality fixture task ID')
    paths = set(required)
    for task in task_ids:
        if not (directory / task).is_dir():
            raise ValueError('Missing quality fixture task directory: ' + task)
    paths.update((p for p in directory.rglob('*') if p.is_file() and (p.parent.name in task_ids or p.name in {'tasks.json', 'quality.json', 'replay_context.md'})))
    return paths


def source_inputs(phase):
    """Snapshot a documented superset of reused CPU helpers, not old raw trees."""
    research = phase.parent
    paths = set()
    for name in D_RUNTIME:
        path = phase / 'scripts' / name
        if not path.is_file():
            raise ValueError('Missing required Stage D runtime: ' + str(path))
        paths.add(path)
    paths.update(phase / 'scripts' / name for name in D_OPTIONAL_RUNTIME
                 if (phase / 'scripts' / name).is_file())
    for relative in ('configs', 'workloads'):
        directory = path_at(phase, relative)
        if not directory.is_dir():
            raise ValueError('Missing Stage D source directory: ' + str(directory))
        for path in directory.rglob('*'):
            if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                continue
            if path.is_file() or path.is_symlink():
                paths.add(path)
    for relative in DEPENDENCY_TREES:
        directory = research / relative
        if not directory.is_dir():
            raise ValueError('Missing reused dependency tree: ' + str(directory))
        paths.update(p for p in directory.rglob('*.py') if '__pycache__' not in p.parts)
    paths.update(quality_fixture_inputs(research))
    sources = {str(p.relative_to(research)): p for p in sorted(paths)}
    if research.resolve() == _layout_root:
        for path in snapshot_inputs():
            key = (str(path.relative_to(research)) if path.is_relative_to(research)
                   else 'project/' + str(path.relative_to(workspace_path())))
            sources[key] = path
        for name in ('environment.json', 'model-inventory.json'):
            sources['metadata/' + name] = workspace_path('results/setup/' + name)
        sources['metadata/scheduler-pair-blocks.json'] = workspace_path('results/setup/scheduler-pair-blocks.json')
    else:
        # Explicit synthetic test roots retain their self-contained fixture data.
        sources.update({name: research / name for name in ('environment.json', 'model-inventory.json')})
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError('Missing runtime snapshot input; run prepare first: ' + ', '.join(missing))
    return sources


def installed_inputs():
    """Read distribution metadata without importing vLLM, torch or CUDA."""
    distribution = importlib.metadata.distribution('vllm')
    site = Path(distribution.locate_file('')).absolute()
    all_files = [str(f) for f in distribution.files or ()]
    metadata_key = next(f for f in all_files if f.endswith('.dist-info/METADATA'))
    metadata_dir = Path(metadata_key).parent
    inputs = {'installed/vllm/' + name: site / 'vllm' / name for name in INSTALLED}
    inputs.update({'installed/' + str(metadata_dir / name): site / metadata_dir / name
                   for name in ('METADATA', 'RECORD', 'WHEEL')})
    for name in ('INSTALLER', 'direct_url.json'):
        path = site / metadata_dir / name
        if path.is_file():
            inputs['installed/' + str(metadata_dir / name)] = path
    versions = {}
    for name in ('vllm', 'torch', 'flashinfer-python', 'flashinfer-cubin', 'aiohttp', 'transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return inputs, dict(vllm_site=str(site), distribution_metadata=str(site / metadata_dir),
                        versions=versions, record_key='installed/' + str(metadata_dir / 'RECORD'))


def capture_sources(destination, phase, installed_provider=installed_inputs):
    sources = source_inputs(phase)
    installed, package = installed_provider()
    if set(sources) & set(installed):
        raise ValueError('Duplicate snapshot source keys')
    sources.update(installed)
    payloads = {key: stable_read(path) for key, path in sorted(sources.items())}
    rows = {key: dict(path=str(sources[key].absolute()), sha256=digest(data), bytes=len(data))
            for key, data in payloads.items()}
    wheel = {row[0]: row for row in csv.reader(io.StringIO(payloads[package['record_key']].decode()))}
    comparisons = {}
    for name in INSTALLED:
        key = 'installed/vllm/' + name
        row = wheel.get('vllm/' + name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payloads[key]).digest()).decode().rstrip('=')
        matches = bool(row and len(row) >= 3 and row[1] == 'sha256=' + encoded and row[2] == str(len(payloads[key])))
        comparisons[name] = dict(record_matches=matches, actual_sha256=rows[key]['sha256'], bytes=rows[key]['bytes'])
    identity = dict(schema_version=1, created_utc=utc(), python_executable=sys.executable,
                    python_realpath=str(Path(sys.executable).resolve()), python_version=sys.version,
                    package=package, installed_record_checks=comparisons, files=rows,
                    source_scope='Explicit D runtime entrypoints (not offline check/analyzer scripts), configs/workloads, frozen quality-client fixture closure, read-only superset of shared A/B/C Python helpers, environment/model inventory and selected installed vLLM sources/metadata.',
                    required_d_runtime=list(D_RUNTIME), optional_d_runtime=list(D_OPTIONAL_RUNTIME),
                    limits=['Selected installed sources are not a full wheel/binary/kernel snapshot.',
                            'A source snapshot does not by itself prove which custom class executed; runtime self-report must bind that separately.',
                            'Wheel RECORD agreement is not upstream authenticity or model-quality certification.'])
    write_json(destination / 'runtime-identity.json', identity, initial=True)
    if not all(row['record_matches'] for row in comparisons.values()):
        raise ValueError('Selected installed vLLM files differ from distribution RECORD; preserve diagnostic and inspect before running')
    with zipfile.ZipFile(destination / 'source-snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, data in payloads.items():
            archive.writestr(key, data)
    return identity


def check_sources(identity):
    return [key for key, row in identity['files'].items()
            if digest(stable_read(Path(row['path']))) != row['sha256']]


def validate_gpus(value):
    labels = value.split(',')
    if not labels or len(labels) != len(set(labels)) or not set(labels) <= {'0', '1', '2', '3'}:
        raise ValueError('Only unique assigned GPU IDs 0,1,2,3 are allowed')
    return value


def stop_group(process):
    """Signal only the session group created by this supervisor."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=10)


def run(args, *, phase=PHASE, installed_provider=installed_inputs):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or Path(args.name).name != args.name or args.name in {'.', '..'}:
        raise ValueError('Require a child command and simple unique experiment ID')
    validate_gpus(args.gpus)
    if args.timeout <= 0:
        raise ValueError('Timeout must be positive')
    lock_path = phase.parent / '.research-gpu.lock'
    with open_output(lock_path, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = subprocess.check_output(['nvidia-smi', '--id=' + args.gpus, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15)
        used = [int(line) for line in state.splitlines()]
        if len(used) != len(args.gpus.split(',')) or any((value > 512 for value in used)):
            raise RuntimeError('Selected GPUs are busy or inventory incomplete; no experiment started')
        destination = path_at(phase, 'results') / args.name
        destination.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus, VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', RESEARCH_RUN_DIR=str(destination))

        def git(*arguments):
            value = subprocess.run(['git', *arguments], cwd=phase, capture_output=True, text=True)
            return value.stdout.strip() if value.returncode == 0 else None
        record = dict(schema_version=1, experiment_id=args.name, objective_hypothesis=args.hypothesis, reference_id=args.reference, git_commit=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'), git_status=git('status', '--porcelain'), command=command, cwd=str(Path.cwd()), start_utc=utc(), timeout_seconds=args.timeout, gpu_idle_memory_mib=used, shared_lock=str(lock_path), environment={k: v for k, v in environment.items() if k.startswith(('CUDA_', 'VLLM_', 'NCCL_', 'FLASHINFER_', 'PYTHON')) or k in ('HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'RESEARCH_RUN_DIR', 'TOKENIZERS_PARALLELISM')})
        record['path_context'] = dict(project_root=str(workspace_path()), model_path=str(model_path()), run_dir=str(destination), research_root=str(phase.parent))
        start = time.monotonic()
        process, identity = (None, None)
        timed_out, interrupted = (False, None)
        error, source_changes = (None, [])
        previous_handlers = {}

        def interrupt(signum, frame):
            nonlocal interrupted
            interrupted = signum
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupt)
        try:
            identity = capture_sources(destination, phase, installed_provider)
            record.update(source_sha256={key: row['sha256'] for key, row in identity['files'].items()}, runtime_identity_sha256=digest((destination / 'runtime-identity.json').read_bytes()), source_snapshot_sha256=digest((destination / 'source-snapshot.zip').read_bytes()))
            write_json(destination / 'command.json', record, initial=True)
            source_changes = check_sources(identity)
            if source_changes:
                raise RuntimeError('Source changed before launch: ' + repr(source_changes))
            if interrupted:
                raise RuntimeError('Interrupted before launch')
            with open_output(destination / 'process.log', 'x') as output, open_output(destination / 'gpu.csv', 'x') as metrics:
                process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(destination / 'process-state.json', dict(pid=process.pid, process_group=process.pid, started_utc=utc()), initial=True)
                print(f'RUN {args.name}: pid={process.pid}, log={destination / 'process.log'}', flush=True)
                while process.poll() is None:
                    if interrupted:
                        break
                    if time.monotonic() - start > args.timeout:
                        timed_out = True
                        break
                    try:
                        sample = subprocess.run(['nvidia-smi', '--id=' + args.gpus, '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
                        metrics.write(sample.stdout if sample.returncode == 0 else '# nvidia-smi sampling failed: ' + str(sample.returncode) + '\n')
                    except subprocess.TimeoutExpired:
                        metrics.write('# nvidia-smi sampling timeout\n')
                    metrics.flush()
                    time.sleep(1)
        except BaseException as exc:
            error = repr(exc)
        finally:
            cleanup_error = None
            try:
                if process is not None:
                    stop_group(process)
                if identity is not None:
                    source_changes = check_sources(identity)
            except BaseException as exc:
                cleanup_error = repr(exc)
            record.update(exit_code=process.returncode if process else None, timed_out=timed_out, interrupted_signal=interrupted, supervisor_error=error, cleanup_error=cleanup_error, changed_source_members=source_changes, elapsed_seconds=time.monotonic() - start, end_utc=utc(), source_identity_unchanged=identity is not None and (not source_changes) and (not cleanup_error))
            code = 124 if timed_out else 128 + interrupted if interrupted else 1 if error or cleanup_error or source_changes else process.returncode if process else 1
            record['supervisor_exit_code'] = code
            if not (destination / 'command.json').exists():
                write_json(destination / 'command.json', {k: v for k, v in record.items() if k not in ('end_utc',)}, initial=True)
            write_json(destination / 'status.json', record, initial=True)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        print(json.dumps({k: record[k] for k in ('exit_code', 'supervisor_exit_code', 'timed_out', 'supervisor_error', 'changed_source_members', 'elapsed_seconds')}), flush=True)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--hypothesis', default='Stage D controlled profiling/optimization; see preregistered plan')
    parser.add_argument('--reference', default='none-exploratory-not-an-accepted-project-baseline')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
