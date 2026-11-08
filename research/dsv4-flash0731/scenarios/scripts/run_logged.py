"""Run one bounded research process; log its command, GPU usage, exit and timeout.

Only the new process group created here is signalled. No name-based/global kill.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output, snapshot_inputs
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
import fcntl


def source_inputs(root):
    """Map captured source bytes to stable research-relative archive names."""
    research = root.parent if True else root
    paths = set((root / 'scripts').glob('*.py'))
    paths.update(path_at(root, 'configs').glob('*.json'))
    paths.update(p for p in path_at(root, 'workloads').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    if True:
        paths.add(research / 'scripts/benchmark_client.py')
    sources = {str(p.relative_to(research)): p for p in sorted(paths)}
    if research.resolve() == _layout_root:
        for path in snapshot_inputs():
            key = (str(path.relative_to(research)) if path.is_relative_to(research)
                   else 'project/' + str(path.relative_to(workspace_path())))
            sources[key] = path
        for name in ('environment.json', 'model-inventory.json'):
            sources['metadata/' + name] = workspace_path('results/setup/' + name)
    else:
        sources.update({name: research / name for name in ('environment.json', 'model-inventory.json')})
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError('Missing runtime snapshot input; run prepare first: ' + ', '.join(missing))
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--hypothesis', default='Feasibility and performance screening; see configuration and plan.md')
    parser.add_argument('--reference', default='none-exploratory-not-an-accepted-project-baseline')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command and set(args.gpus.split(',')) <= {'0', '1', '2', '3'}
    root = Path(__file__).resolve().parents[1]
    lock = open_output(root.parent / '.research-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = subprocess.check_output(['nvidia-smi', f'--id={args.gpus}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any((int(line) > 512 for line in state.splitlines())):
        raise RuntimeError('Selected GPUs are busy; no experiment started')
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        raise ValueError('Experiment name must be a simple unique ID')
    destination = path_at(root, 'results') / args.name
    destination.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus, VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1', RESEARCH_RUN_DIR=str(destination))

    def git_read(*arguments):
        value = subprocess.run(['git', *arguments], capture_output=True, text=True)
        return value.stdout.strip() if value.returncode == 0 else None
    sources = source_inputs(root)
    record = {'experiment_id': args.name, 'objective_hypothesis': args.hypothesis, 'reference_id': args.reference, 'git_commit': git_read('rev-parse', 'HEAD'), 'git_branch': git_read('branch', '--show-current'), 'git_status': git_read('status', '--porcelain'), 'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}, 'command': command, 'cwd': str(Path.cwd()), 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'timeout_seconds': args.timeout, 'environment': {key: value for key, value in environment.items() if key.startswith(('CUDA_', 'VLLM_', 'NCCL_', 'FLASHINFER_')) or key in ('HF_HUB_OFFLINE', 'OMP_NUM_THREADS')}}
    record['path_context'] = dict(project_root=str(workspace_path()), model_path=str(model_path()), run_dir=str(destination), research_root=str(_layout_root))
    write_text(destination / 'command.json', json.dumps(record, indent=2) + '\n')
    with zipfile.ZipFile(destination / 'source-snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, source in sources.items():
            archive.write(source, key)
    start = time.monotonic()
    process = None
    timed_out = False
    error = None
    with open_output(destination / 'process.log', 'w') as output, open_output(destination / 'gpu.csv', 'w') as metrics:
        try:
            process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print(f'RUN {args.name}: pid={process.pid}, log={destination / 'process.log'}', flush=True)
            while process.poll() is None:
                if time.monotonic() - start > args.timeout:
                    timed_out = True
                    break
                try:
                    gpu = subprocess.run(['nvidia-smi', f'--id={args.gpus}', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
                    metrics.write(gpu.stdout)
                    metrics.flush()
                except subprocess.TimeoutExpired:
                    metrics.write('# nvidia-smi sampling timeout\n')
                    metrics.flush()
                time.sleep(1)
        except BaseException as exception:
            error = repr(exception)
            raise
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            record.update(exit_code=process.returncode if process else None, timed_out=timed_out, supervisor_error=error, elapsed_seconds=time.monotonic() - start, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_text(destination / 'status.json', json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('exit_code', 'timed_out', 'elapsed_seconds')}), flush=True)
    raise SystemExit(124 if timed_out else record['exit_code'])


if __name__ == "__main__":
    main()
