"""Run two local two-GPU replicas in the process group owned by run_logged.py.

Physical GPUs 0,1 and 2,3; ports 8260 and 8261. Startup is sequential.
Only direct child processes created here are signalled; run_logged.py performs
final group cleanup, including any surviving model workers.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

PORTS = (8260, 8261)
GPU_SUBSETS = ("0,1", "2,3")
MODEL_PATH = str(model_path())
MODEL_NAME = "dsv4"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with open_output(path, 'x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def update_record(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_config(config):
    arguments = config.get("args")
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError("Configuration args must be a list of strings.")
    values = {}
    for i, item in enumerate(arguments):
        if item.startswith("--"):
            name, equal, value = item.partition("=")
            if name in values:
                raise ValueError(f"Repeated option: {name}")
            values[name] = value if equal else (arguments[i + 1] if i + 1 < len(arguments) else None)
    topology = (values.get("--tensor-parallel-size", "1"),
                values.get("--pipeline-parallel-size", "1"))
    if topology not in (("2", "1"), ("1", "2")):
        raise ValueError("This launcher requires TP2/PP1 or TP1/PP2 per replica.")
    if values.get("--distributed-executor-backend") != "mp":
        raise ValueError("This launcher requires the local mp distributed executor.")
    for key in ("--data-parallel-size",):
        if key in values and values[key] != "1":
            raise ValueError(f"Independent replicas require {key} 1 or omission.")
    forbidden = {"--host", "--port", "--model", "--served-model-name", "--data-parallel-address",
                 "--data-parallel-rank", "--data-parallel-size-local", "--data-parallel-start-rank",
                 "--data-parallel-external-lb", "--data-parallel-hybrid-lb"}
    if forbidden.intersection(values):
        raise ValueError("Host, ports, model identity and replica topology are owned by this launcher.")
    if config.get("port", 8260) != 8260:
        raise ValueError("Base configuration port must be 8260; replica 1 uses 8261.")
    if not isinstance(config.get("env", {}), dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in config.get("env", {}).items()):
        raise ValueError("Configuration env must contain string keys and values.")
    if config.get("startup_timeout", 1200) <= 0 or config.get("client_timeout", 3600) <= 0:
        raise ValueError("Startup and client timeouts must be positive.")


def reserve_ports():
    """Reject existing listeners; hold both ports until each respective startup."""
    sockets = []
    try:
        for port in PORTS:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sockets.append(probe)
            probe.bind(("127.0.0.1", port))
        return sockets
    except BaseException:
        for probe in sockets:
            probe.close()
        raise


def http_get(port, path, timeout=2):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def wait_ready(server, port, startup_timeout, peers):
    begin = time.monotonic()
    last_error = None
    while time.monotonic() - begin < startup_timeout:
        for peer in [server, *peers]:
            if peer.poll() is not None:
                raise RuntimeError(f"Server pid={peer.pid} exited before both replicas were ready: {peer.returncode}")
        try:
            status, _ = http_get(port, "/health")
            if status == 200:
                status, payload = http_get(port, "/v1/models")
                models = json.loads(payload) if status == 200 else {}
                if any(item.get("id") == MODEL_NAME for item in models.get("data", [])):
                    if server.poll() is not None:
                        raise RuntimeError("Server exited during readiness probe.")
                    return time.monotonic() - begin
                last_error = "Health responded but served model dsv4 is absent."
        except (OSError, http.client.HTTPException, ValueError) as error:
            last_error = repr(error)
        time.sleep(2)
    raise TimeoutError(f"Replica on port {port} readiness timed out; last probe: {last_error}")


def stop_child(process, record):
    record["exit_before_cleanup"] = process.poll()
    if process.poll() is None:
        record["termination_requested"] = True
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            record["kill_requested"] = True
            process.kill()
            process.wait(timeout=10)
    record["exit_code"] = process.poll()
    record["cleanup_finished_at"] = now()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--suite', default='smoke,screen', help='Comma-separated smoke, screen and/or full.')
    parser.add_argument('--metrics-snapshots', action='store_true', help="Save each server's /metrics before and after clients.")
    parser.add_argument('--fresh-connections', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate both replica commands without starting services.')
    args = parser.parse_args()
    suites = args.suite.split(',')
    if len(set(suites)) != len(suites) or any((s not in {'smoke', 'screen', 'full'} for s in suites)):
        parser.error('Suites must be unique comma-separated smoke, screen and/or full values.')
    if args.validate_only:
        config = load_document(args.config.expanduser().resolve())
        validate_config(config)
        client_script = Path(__file__).with_name('benchmark_client.py')
        compile(client_script.read_text(), str(client_script), 'exec')
        vllm = Path(sys.executable).parent / 'vllm'
        replicas = [dict(gpus=gpus, command=[str(vllm), 'serve', MODEL_PATH,
                    '--served-model-name', MODEL_NAME, '--host', '127.0.0.1',
                    '--port', str(port), *config['args']]) for port, gpus in zip(PORTS, GPU_SUBSETS)]
        print(json.dumps(dict(replicas=replicas, suites=suites,
                              fresh_connections=args.fresh_connections,
                              metrics_snapshots=args.metrics_snapshots), indent=2))
        return 0
    if 'RESEARCH_RUN_DIR' not in os.environ:
        parser.error("Use run_logged.py, which owns the experiment's process group.")
    output = Path(os.environ['RESEARCH_RUN_DIR']).resolve()
    if not output.is_dir():
        parser.error('RESEARCH_RUN_DIR must already exist and belong to run_logged.py.')
    if set(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != {'0', '1', '2', '3'}:
        parser.error('The owning runner must authorize physical GPU indices 0,1,2,3.')
    result = dict(status='INITIALIZING', started_at=now(), suite=args.suite, launcher_pid=os.getpid(), process_group=os.getpgrp(), topology='two independent two-GPU replicas; TP/PP as in configuration; client round-robin routing', replicas=[], clients=[], preflight=[], metrics_snapshots=[])
    record_path = output / 'experiment.json'
    write_new(record_path, result)
    begin = time.monotonic()
    children = []
    logs = []
    reservations = []
    exit_code = 1
    interrupted_signal = None

    def persist():
        result['elapsed_seconds'] = time.monotonic() - begin
        update_record(record_path, result)

    def on_signal(signum, frame):
        nonlocal interrupted_signal
        interrupted_signal = signum
        raise KeyboardInterrupt(f'Received signal {signum}')
    old_term_handler = signal.signal(signal.SIGTERM, on_signal)
    old_int_handler = signal.signal(signal.SIGINT, on_signal)

    def run_child(command, environment, artifact, timeout, destination):
        entry = dict(command=command, status='STARTING', started_at=now(), extra_environment={'CUDA_VISIBLE_DEVICES': environment['CUDA_VISIBLE_DEVICES']})
        destination.append(entry)
        persist()
        log_path = output / artifact
        log = open_output(log_path, 'x', encoding='utf-8')
        logs.append(log)
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        entry.update(pid=process.pid, status='RUNNING', log=str(log_path))
        children.append((process, entry))
        persist()
        code = process.wait(timeout=timeout)
        entry.update(exit_code=code, status='PASS' if code == 0 else 'FAILED', finished_at=now())
        persist()
        if code != 0:
            raise RuntimeError(f'Client exited {code}; see {log_path}')

    def snapshot(stage):
        if not args.metrics_snapshots:
            return
        for index, port in enumerate(PORTS):
            entry = dict(replica=index, stage=stage, captured_at=now())
            try:
                status, body = http_get(port, '/metrics', timeout=5)
                entry['http_status'] = status
                if status != 200:
                    raise RuntimeError(f'Metrics HTTP {status}')
                path = output / f'replica-{index}' / f'metrics-{stage}.txt'
                with open_output(path, 'xb') as f:
                    f.write(body)
                entry.update(status='PASS', path=str(path))
            except Exception as error:
                entry.update(status='FAILED', error=repr(error))
            result['metrics_snapshots'].append(entry)
        persist()
    try:
        client_script = Path(__file__).with_name('benchmark_client.py')
        if not client_script.is_file():
            raise FileNotFoundError('Benchmark client must exist before launching models.')
        compile(client_script.read_text(), str(client_script), 'exec')
        config_path = args.config.expanduser().resolve()
        config = load_document(config_path)
        validate_config(config)
        result.update(configuration=config, configuration_path=str(config_path), configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        write_new(output / 'config.json', config)
        environment = os.environ.copy()
        environment.update(config.get('env', {}))
        client_environment = environment.copy()
        client_environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='')
        base_urls = ','.join((f'http://127.0.0.1:{port}' for port in PORTS))
        result['status'] = 'PREFLIGHT'
        persist()
        for suite in suites:
            command = [sys.executable, str(client_script), '--base-url', base_urls, '--model', MODEL_NAME, '--suite', suite, '--output', str(output / f'preflight-{suite}.json'), '--prepare-only']
            if args.fresh_connections:
                command += ['--fresh-connections']
            run_child(command, client_environment, f'preflight-{suite}.log', 120, result['preflight'])
        vllm = Path(sys.executable).parent / 'vllm'
        if not vllm.is_file():
            raise FileNotFoundError(f'No vllm entry point beside current Python: {vllm}')
        reservations = reserve_ports()
        result['port_checks'] = dict(ports=list(PORTS), status='RESERVED', checked_at=now())
        servers = []
        for index, (port, gpu_subset) in enumerate(zip(PORTS, GPU_SUBSETS)):
            directory = output / f'replica-{index}'
            directory.mkdir(exist_ok=False)
            server_environment = environment.copy()
            server_environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu_subset)
            command = [str(vllm), 'serve', MODEL_PATH, '--served-model-name', MODEL_NAME, '--host', '127.0.0.1', '--port', str(port), *config['args']]
            extra_environment = {**config.get('env', {}), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': gpu_subset}
            entry = dict(replica=index, port=port, gpus=gpu_subset, status='STARTING', command=command, pid=None, exit_code=None, extra_environment=extra_environment, started_at=now(), log=str(directory / 'server.log'))
            result['replicas'].append(entry)
            write_new(directory / 'server-command.json', dict(command=command, extra_environment=extra_environment))
            persist()
            log = open_output(directory / 'server.log', 'x', encoding='utf-8')
            logs.append(log)
            reservations[index].close()
            server = subprocess.Popen(command, env=server_environment, stdout=log, stderr=subprocess.STDOUT)
            children.append((server, entry))
            entry['pid'] = server.pid
            result['status'] = f'STARTING_REPLICA_{index}'
            persist()
            print(f'Starting replica={index} pid={server.pid} GPUs={gpu_subset} port={port}; {entry['log']}', flush=True)
            startup = wait_ready(server, port, config.get('startup_timeout', 1200), servers)
            entry.update(status='READY', startup_seconds=startup, ready_at=now())
            servers.append(server)
            write_new(directory / 'ready.json', entry)
            persist()
            print(f'Replica {index} READY after {startup:.2f}s', flush=True)
        result.update(status='READY', both_ready_at=now(), startup_seconds=time.monotonic() - begin)
        persist()
        snapshot('before')
        for suite in suites:
            if any((server.poll() is not None for server in servers)):
                raise RuntimeError('A replica exited before client execution.')
            command = [sys.executable, str(client_script), '--base-url', base_urls, '--model', MODEL_NAME, '--suite', suite, '--output', str(output / f'benchmark-{suite}.json')]
            if args.fresh_connections:
                command += ['--fresh-connections']
            run_child(command, client_environment, f'client-{suite}.log', config.get('client_timeout', 3600), result['clients'])
            if any((server.poll() is not None for server in servers)):
                raise RuntimeError('A replica exited during client execution.')
        snapshot('after')
        result['status'] = 'PASS'
        exit_code = 0
    except BaseException as error:
        result.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=repr(error))
        exit_code = 128 + (interrupted_signal or signal.SIGINT) if isinstance(error, KeyboardInterrupt) else 1
        print(f'Replica experiment {result['status']}: {error!r}', file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for probe in reservations:
            probe.close()
        for process, entry in reversed(children):
            try:
                stop_child(process, entry)
            except Exception as error:
                entry['cleanup_error'] = repr(error)
                result['status'] = 'FAILED_CLEANUP'
                exit_code = 1
        for log in logs:
            log.close()
        result.update(exit_code=exit_code, finished_at=now(), cleanup_scope='Only recorded direct children; owning run_logged.py cleans remaining process-group workers.')
        persist()
        for entry in result['replicas']:
            directory = output / f'replica-{entry['replica']}'
            if directory.is_dir():
                write_new(directory / 'exit.json', entry)
        signal.signal(signal.SIGTERM, old_term_handler)
        signal.signal(signal.SIGINT, old_int_handler)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
