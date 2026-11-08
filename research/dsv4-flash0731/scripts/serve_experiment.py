"""Launch a local vLLM experiment, wait for readiness, run a client and stop it.

Run under run_logged.py, which owns and cleans this experiment's process group.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--suite', default='smoke')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--chat-probe', action='store_true')
    parser.add_argument('--fresh-connections', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate inputs and print the launch command without starting a service.')
    args = parser.parse_args()
    client_script = Path(__file__).with_name('benchmark_client.py')
    assert client_script.is_file(), 'Benchmark client must exist before launching model'
    compile(client_script.read_text(), str(client_script), 'exec')
    config = load_document(Path(args.config))
    suites = args.suite.split(',')
    if len(set(suites)) != len(suites) or any(s not in {'smoke', 'screen', 'full', 'long'} for s in suites):
        parser.error('Suites must be unique comma-separated smoke, screen, full and/or long values.')
    if args.rounds is not None and not 1 <= args.rounds <= 10:
        parser.error('Rounds must be in 1..10.')
    if not isinstance(config.get('args'), list) or not all(isinstance(value, str) for value in config['args']):
        parser.error('Configuration args must be a list of strings.')
    command = [str(Path(sys.executable).parent / 'vllm'), 'serve', str(model_path()), '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', str(config.get('port', 8260))]
    command += config['args']
    if args.validate_only:
        print(json.dumps(dict(command=command, suites=suites, rounds=args.rounds,
                              fresh_connections=args.fresh_connections, chat_probe=args.chat_probe), indent=2))
        return 0
    assert 'RESEARCH_RUN_DIR' in os.environ, 'Use run_logged.py for process ownership'
    output = Path(os.environ['RESEARCH_RUN_DIR'])
    write_text(output / 'config.json', json.dumps(config, indent=2) + '\n')
    environment = os.environ.copy()
    environment.update(config.get('env', {}))
    write_text(output / 'server-command.json', json.dumps({'command': command, 'extra_environment': config.get('env', {})}, indent=2) + '\n')
    port = config.get('port', 8260)
    import socket
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    server = None
    begin = time.monotonic()
    result = {'configuration': config, 'suite': args.suite, 'status': 'STARTING'}
    try:
        with open_output(output / 'server.log', 'w') as log:
            server = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            print(f'Server pid={server.pid}, port={port}; see {output / 'server.log'}', flush=True)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited before readiness: {server.returncode}')
                if time.monotonic() - begin > config.get('startup_timeout', 1200):
                    raise TimeoutError('Server readiness timed out')
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    response.read()
                    ready = response.status == 200
                    connection.close()
                    if ready:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(2)
            result.update(status='READY', startup_seconds=time.monotonic() - begin)
            write_text(output / 'experiment.json', json.dumps(result, indent=2) + '\n')
            print(f'READY after {result['startup_seconds']:.2f}s', flush=True)

            def snapshot_metrics(label):
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
                try:
                    connection.request('GET', '/metrics')
                    response = connection.getresponse()
                    text = response.read().decode()
                    write_text(output / f'metrics-{label}.txt', text)
                finally:
                    connection.close()
            if args.chat_probe:
                subprocess.run([sys.executable, str(Path(__file__).with_name('chat_probe.py')), '--base-url', f'http://127.0.0.1:{port}', '--model', 'dsv4', '--output', str(output / 'chat-probe.json')], env=environment, check=True)
            for suite in suites:
                snapshot_metrics(f'before-{suite}')
                client = [sys.executable, str(client_script), '--base-url', f'http://127.0.0.1:{port}', '--model', 'dsv4', '--suite', suite, '--output', str(output / f'benchmark-{suite}.json')]
                if args.rounds is not None:
                    client += ['--rounds', str(args.rounds)]
                if args.fresh_connections:
                    client += ['--fresh-connections']
                subprocess.run(client, env=environment, check=True)
                snapshot_metrics(f'after-{suite}')
            result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAILED', error=repr(error))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - begin
        write_text(output / 'experiment.json', json.dumps(result, indent=2) + '\n')
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
