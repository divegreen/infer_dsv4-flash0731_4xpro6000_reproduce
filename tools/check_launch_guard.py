#!/usr/bin/env python3
"""CPU regression for launch reservation, failure evidence, and cache identity.

All GPU inventory and inference calls are mocked. Real flock checks use temporary
files only; this command never queries a physical GPU or starts inference.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('guard_entry', PROJECT / 'tools/reproduce.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/validation/launch-guard.json')
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'status': 'PASS' if condition else 'FAIL'})
        if not condition:
            raise AssertionError(name)

    entry = {'id': 'synthetic-run', 'runner': 'unused.py', 'child_argv': ['unused.py'],
             'timeout': 1, 'gpus': '0,1,2,3', 'reference': 'none',
             'executable_definition': True, 'depends_on': [], 'requires_success': []}
    inventory = '\n'.join(f'{i}, GPU-synthetic-{i}, 0, 0' for i in range(4))

    def query(command, **kwargs):
        assert command[0] == 'nvidia-smi' and kwargs['timeout'] == 15
        return inventory if '--query-gpu=index,uuid,memory.used,utilization.gpu' in command else ''

    with tempfile.TemporaryDirectory(prefix='launch guard cpu ') as temporary:
        temp = Path(temporary)
        locks = temp / 'locks'
        locks.mkdir()
        with mock.patch.object(cli.tempfile, 'gettempdir', return_value=str(locks)):
            with mock.patch.object(cli.subprocess, 'check_output', side_effect=query) as gpu:
                with mock.patch.object(cli, 'PROJECT', temp / 'copy one'):
                    with cli.gpu_reservation(entry['gpus']) as guard:
                        check('lock paths use runtime temporary directory',
                              all(Path(path).parent == locks for path in guard['locks']))
                        with mock.patch.object(cli, 'PROJECT', temp / 'copy two'):
                            try:
                                with cli.gpu_reservation(entry['gpus']):
                                    pass
                            except RuntimeError:
                                check('separate project copies contend on the same devices', gpu.call_count == 2)
                            else:
                                check('separate project copies contend on the same devices', False)
                        probe = ('import fcntl,sys; f=open(sys.argv[1],"a+"); '
                                 'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
                        child = subprocess.run([sys.executable, '-B', '-c', probe, guard['locks'][0]],
                                               capture_output=True, text=True)
                        check('mutex excludes an independent CPU process', child.returncode != 0 and 'BlockingIOError' in child.stderr)
                child = subprocess.run([sys.executable, '-B', '-c', probe, guard['locks'][0]],
                                       capture_output=True, text=True)
                check('mutex releases after successful reservation', child.returncode == 0)
                try:
                    with cli.gpu_reservation(entry['gpus']):
                        raise RuntimeError('synthetic child failure')
                except RuntimeError:
                    pass
                with cli.gpu_reservation(entry['gpus']):
                    check('mutex releases after exception', True)

            cases = {
                'memory-busy': inventory.replace('GPU-synthetic-0, 0, 0', 'GPU-synthetic-0, 513, 0'),
                'utilization-busy': inventory.replace('GPU-synthetic-0, 0, 0', 'GPU-synthetic-0, 0, 1'),
                'inventory-missing': '\n'.join(inventory.splitlines()[:3]),
                'inventory-duplicate': inventory.replace('GPU-synthetic-3', 'GPU-synthetic-2'),
                'metric-unavailable': inventory.replace('GPU-synthetic-0, 0, 0', 'GPU-synthetic-0, N/A, 0'),
                'compute-process': inventory,
                'query-missing': FileNotFoundError('synthetic nvidia-smi missing'),
                'query-timeout': subprocess.TimeoutExpired('synthetic inventory', 15),
            }
            for name, response in cases.items():
                root = temp / name
                results = root / 'results'

                def failed_query(command, **kwargs):
                    if isinstance(response, Exception):
                        raise response
                    if '--query-gpu=index,uuid,memory.used,utilization.gpu' in command:
                        return response
                    return 'GPU-synthetic-0, 1234' if name == 'compute-process' else ''

                with mock.patch.object(cli, 'PROJECT', root), mock.patch.object(cli, 'RESULTS', results), \
                     mock.patch.object(cli, 'catalog', return_value={'experiments': [entry]}), \
                     mock.patch.object(cli.subprocess, 'check_output', side_effect=failed_query), \
                     mock.patch.object(cli.subprocess, 'run') as inference, contextlib.redirect_stdout(io.StringIO()):
                    row = cli.execute_one(entry, SimpleNamespace(dry_run=False, overwrite=False), 'test-signature')
                    check(name + ': rejects without inference or raw directory',
                          row['status'] == 'FAILED_BEFORE_LAUNCH' and not (results / 'runs').exists() and not inference.called)
                    state = cli.read(results / 'stage-state/runs/synthetic-run.json')
                    log = results / 'stage-state/logs/synthetic-run.log'
                    check(name + ': retains failure state and log',
                          state == row and bool(row['launcher_error']) and row['launcher_error'] in log.read_text())
                with mock.patch.object(cli.subprocess, 'check_output', side_effect=query):
                    with cli.gpu_reservation(entry['gpus']):
                        check(name + ': releases reservations', True)

            root = temp / 'memory-busy'
            results = root / 'results'
            prior_log = (results / 'stage-state/logs/synthetic-run.log').read_bytes()
            prior_state = (results / 'stage-state/runs/synthetic-run.json').read_bytes()

            def inference(command, **kwargs):
                check('child disables generated Python bytecode', kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1')
                cli.save(results / 'runs/synthetic-run/status.json',
                         {'experiment_id': entry['id'], 'end_utc': 'synthetic-only', 'exit_code': 0})
                return SimpleNamespace(returncode=0)

            with mock.patch.object(cli, 'PROJECT', root), mock.patch.object(cli, 'RESULTS', results), \
                 mock.patch.object(cli, 'catalog', return_value={'experiments': [entry]}), \
                 mock.patch.object(cli.subprocess, 'check_output', side_effect=query), \
                 mock.patch.object(cli.subprocess, 'run', side_effect=inference) as launch, contextlib.redirect_stdout(io.StringIO()):
                try:
                    cli.execute_one(entry, SimpleNamespace(dry_run=False, overwrite=False), 'test-signature')
                except ValueError:
                    check('failed no-directory attempt requires explicit overwrite', not launch.called)
                else:
                    check('failed no-directory attempt requires explicit overwrite', False)
                row = cli.execute_one(entry, SimpleNamespace(dry_run=False, overwrite=True), 'test-signature')
                check('overwrite retries prelaunch failure successfully', row['status'] == 'TERMINAL' and launch.call_count == 1)
                check('overwrite preserves failed no-directory attempt bytes',
                      (results / 'preserved/launcher-logs/synthetic-run.log/attempt-1').read_bytes() == prior_log
                      and (results / 'preserved/run-state/synthetic-run.json/attempt-1').read_bytes() == prior_state
                      and not (results / 'preserved/runs').exists())
                with mock.patch.object(cli, 'gpu_reservation', side_effect=AssertionError('must not acquire GPU')):
                    reused = cli.execute_one(entry, SimpleNamespace(dry_run=False, overwrite=False), 'test-signature')
                    check('terminal reuse does not query or reserve GPUs', reused['status'] == 'REUSED_TERMINAL')

            root = temp / 'cpu-validation'
            with mock.patch.object(cli, 'PROJECT', root), mock.patch.object(cli, 'RESULTS', root / 'results'), \
                 mock.patch.object(cli, 'catalog', return_value={'experiments': [entry], 'stages': []}), \
                 mock.patch.object(cli, 'prepare'), \
                 mock.patch.object(cli, 'gpu_reservation', side_effect=AssertionError('GPU forbidden')) as reservation, \
                 mock.patch.object(cli.subprocess, 'check_output', side_effect=AssertionError('GPU forbidden')) as query_mock, \
                 mock.patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')) as child, \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = cli.run(SimpleNamespace(stage=None, experiment=entry['id'], dry_run=True))
                check('dry-run performs no GPU query, subprocess, or output creation',
                      rc == 0 and not reservation.called and not query_mock.called and not child.called and not root.exists())
                rc = cli.preflight(SimpleNamespace(experiment=entry['id']))
                check('preflight uses only validate-only child with GPUs hidden',
                      not rc and not reservation.called and not query_mock.called and child.call_count == 1
                      and child.call_args.args[0][-1] == '--validate-only'
                      and child.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
                      and not (root / 'results/runs').exists())

            model = temp / 'synthetic-model'
            model.mkdir()
            (model / 'config.json').write_text('{"synthetic":1}')
            with mock.patch.object(cli.layout, 'model_path', return_value=model), \
                 mock.patch.object(cli, 'python_binary', return_value=temp / 'missing-runtime/bin/python'):
                before = cli.runtime_fingerprint()
                for relative in ('__pycache__/model.pyc', '.cache/huggingface/model.json', '.git/index',
                                 '.pytest_cache/lastfailed', 'download.lock', 'weights.incomplete'):
                    path = model / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('generated')
                check('generated model cache files do not invalidate runtime identity', cli.runtime_fingerprint() == before)
                (model / 'config.json').write_text('{"synthetic":2}')
                check('native model input changes still invalidate runtime identity', cli.runtime_fingerprint() != before)

    cli.save(args.output, {'generated': True, 'status': 'PASS', 'checks': checks, 'count': len(checks),
                          'scope': 'CPU mocks and real temporary-file flock only; no physical GPU query or inference'})
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
