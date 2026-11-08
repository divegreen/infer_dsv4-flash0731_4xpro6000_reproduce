"""Start one owned local server, run explicit scenario tasks, then stop it."""

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
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--suites', default='decode')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--agent-modes', default='')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate inputs and print the launch command without starting a service.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_document(args.config)
    if args.validate_only:
        out = Path(os.environ.get('RESEARCH_RUN_DIR', str(path_at(root, 'results') / config['name'])))
    else:
        out = Path(os.environ['RESEARCH_RUN_DIR'])
    port = config.get('port', 8260)
    suites = [s for s in args.suites.split(',') if s]
    agent_modes = [s for s in args.agent_modes.split(',') if s]
    assert all((s in ['decode', 'context', 'arrival', 'cache', 'profile'] for s in suites))
    assert all((s in ['replay', 'closed-loop', 'quality'] for s in agent_modes))
    assert 'profile' not in suites or args.profile
    for name in ['scenario_client.py'] + (['agent_benchmark.py'] if agent_modes else []):
        p = root / 'scripts' / name
        compile(p.read_text(), str(p), 'exec')
    command = [str(Path(sys.executable).parent / 'vllm'), 'serve', str(model_path()), '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', str(port), *config['args']]
    if agent_modes:
        command += ['--enable-prompt-tokens-details']
    if args.profile:
        command += ['--profiler-config', json.dumps(dict(profiler='torch', torch_profiler_dir=str(out / 'profiles'), torch_profiler_activities=['CPU', 'CUDA'], torch_profiler_with_stack=False, torch_profiler_record_shapes=False, ignore_frontend=True, max_iterations=20))]
    if args.validate_only:
        print(json.dumps(dict(command=command, suites=suites, agent_modes=agent_modes,
                              rounds=args.rounds, profile=args.profile), indent=2))
        return 0
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    env = os.environ.copy()
    env.update(config.get('env', {}))
    env['VLLM_SERVER_DEV_MODE'] = '1'
    write_text(out / 'config.json', json.dumps(config, indent=2) + '\n')
    write_text(out / 'server-command.json', json.dumps(dict(command=command, extra_environment={**config.get('env', {}), 'VLLM_SERVER_DEV_MODE': '1'}), indent=2) + '\n')
    result = dict(status='STARTING', tasks=suites, agent_modes=agent_modes, profile=args.profile, clients=[])
    start = time.monotonic()
    server = None

    def save():
        write_text(out / 'experiment.json', json.dumps(result, indent=2) + '\n')
    save()
    try:
        with open_output(out / 'server.log', 'x') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while time.monotonic() - start < config.get('startup_timeout', 1200):
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode}')
                try:
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                    c.request('GET', '/health')
                    r = c.getresponse()
                    r.read()
                    c.close()
                    if r.status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('readiness timeout')
            result.update(status='READY', startup_seconds=time.monotonic() - start)
            save()
            print('READY', result['startup_seconds'], flush=True)
            for suite in suites:
                cmd = [sys.executable, str(root / 'scripts/scenario_client.py'), '--base-url', f'http://127.0.0.1:{port}', '--suite', suite, '--rounds', str(1 if suite == 'profile' else args.rounds), '--output', str(out / f'scenario-{suite}.json')]
                entry = dict(task=suite, command=cmd, status='RUNNING')
                result['clients'].append(entry)
                save()
                code = subprocess.call(cmd, env=env)
                entry.update(status='PASS' if code == 0 else 'FAILED', exit_code=code)
                save()
                if code:
                    raise RuntimeError(f'Scenario {suite} failed with {code}')
            for mode in agent_modes:

                def agent_metrics(label):
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
                    try:
                        c.request('GET', '/metrics')
                        r = c.getresponse()
                        body = r.read().decode()
                        if r.status != 200:
                            raise RuntimeError('Agent metrics snapshot failed')
                        write_text(out / f'metrics-agent-{mode}-{label}.txt', body)
                    finally:
                        c.close()
                agent_metrics('before')
                cmd = [sys.executable, str(root / 'scripts/agent_benchmark.py'), '--base-url', f'http://127.0.0.1:{port}', '--mode', mode, '--rounds', str(args.rounds), '--output', str(out / f'agent-{mode}.json'), '--reset-prefix-cache', '--max-tokens', str({'replay': 256, 'closed-loop': 768, 'quality': 128}[mode])]
                entry = dict(task=mode, command=cmd, status='RUNNING')
                result['clients'].append(entry)
                save()
                code = subprocess.call(cmd, env=env)
                entry.update(status='PASS' if code == 0 else 'FAILED', exit_code=code)
                save()
                agent_metrics('after')
                if code:
                    raise RuntimeError(f'Agent client {mode} execution failed with {code}')
            result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        result['elapsed_seconds'] = time.monotonic() - start
        save()


if __name__=='__main__':main()
