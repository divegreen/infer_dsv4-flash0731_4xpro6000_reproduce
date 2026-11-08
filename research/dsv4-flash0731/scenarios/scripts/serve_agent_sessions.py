"""Run one plan against an owned localhost server under run_logged.py."""

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

from agent_session_client import validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_document(args.config)
    plan = validate_plan(load_document(args.plan))
    port = config.get('port', 8260)
    if plan['base_url'] != f'http://127.0.0.1:{port}':
        raise ValueError('Plan URL and owned localhost server port differ')
    if plan.get('model') != 'dsv4':
        raise ValueError('Plan model must match the served alias dsv4')
    if not config.get('args') or any((a in ('--host', '--port') or a.startswith(('--host=', '--port=')) for a in config['args'])):
        raise ValueError('Launcher must own server address')
    if config.get('env', {}).get('VLLM_SERVER_DEV_MODE') != '1':
        raise ValueError('Owned cache reset endpoint requires explicit local development mode')
    protected_env = {'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'RESEARCH_RUN_DIR', 'VLLM_NO_USAGE_STATS', 'HF_HUB_OFFLINE', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS', 'PYTHONUNBUFFERED'}
    if protected_env.intersection(config.get('env', {})):
        raise ValueError('Config must not override supervisor-owned GPU/output/runtime environment')
    if any((p.get('dp_routing') == 'sticky4' for p in plan['phases'])):
        at = config['args'].index('--data-parallel-size') if '--data-parallel-size' in config['args'] else -1
        if at < 0 or config['args'][at + 1] != '4':
            raise ValueError('sticky4 requires actual DP4 deployment')
    command = [str(Path(sys.executable).parent / 'vllm'), 'serve', str(model_path()), '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', str(port), *config['args']]
    if '--enable-prompt-tokens-details' not in config['args']:
        command.append('--enable-prompt-tokens-details')
    if args.validate_only:
        print(json.dumps(dict(command=command, phases=[p['id'] for p in plan['phases']]), indent=2))
        return 0
    out = Path(os.environ['RESEARCH_RUN_DIR'])
    if plan.get('experiment_id') != out.name:
        raise ValueError('Plan experiment ID does not match unique result directory')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    env = os.environ.copy()
    env.update(config.get('env', {}))
    for name, value in [('config.json', config), ('session-plan.json', plan), ('server-command.json', dict(command=command, extra_environment=config.get('env', {})))]:
        with open_output(out / name, 'x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
    start = time.monotonic()
    result = dict(status='STARTING', clients=[], plan=str(args.plan.resolve()))
    server = None

    def save():
        temp = out / 'experiment.json.tmp'
        write_text(temp, json.dumps(result, indent=2) + '\n')
        temp.replace(out / 'experiment.json')
    save()
    try:
        with open_output(out / 'server.log', 'x') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            result['server_pid'] = server.pid
            save()
            while time.monotonic() - start < config.get('startup_timeout', 1200):
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode}')
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                try:
                    conn.request('GET', '/health')
                    response = conn.getresponse()
                    response.read()
                    if response.status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    conn.close()
                time.sleep(2)
            else:
                raise TimeoutError('Readiness timeout')
            result.update(status='READY', startup_seconds=time.monotonic() - start)
            save()
            print('READY', result['startup_seconds'], flush=True)
            cmd = [sys.executable, str(root / 'scripts/agent_session_client.py'), '--plan', str(args.plan.resolve()), '--output', str(out / 'sessions.json')]
            entry = dict(task='agent-sessions', command=cmd, status='RUNNING')
            result['clients'].append(entry)
            save()
            code = subprocess.call(cmd, env=env)
            entry.update(exit_code=code, status='PASS' if code == 0 else 'FAILED')
            save()
            if code:
                raise RuntimeError(f'Agent session client exited {code}; preserve partial evidence')
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
