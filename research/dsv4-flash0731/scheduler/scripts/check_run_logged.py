#!/usr/bin/env python3
"""CPU-only fixtures for the Stage D supervisor; never starts vLLM or CUDA."""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('stage_d_run_logged', HERE / 'run_logged.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def put(path: Path, data: bytes=b'fixture\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, data)


def fake_installed(root: Path):
    names = list(RUNNER.INSTALLED)
    files = {}
    for name in names:
        path = root / 'site' / 'vllm' / name
        put(path, ('# fake installed ' + name + '\n').encode())
        files['installed/vllm/' + name] = path
    metadata_dir = root / 'site' / 'vllm-0.0.dist-info'
    put(metadata_dir / 'METADATA', b'Metadata-Version: 2.1\nName: vllm\nVersion: 0.0\n')
    put(metadata_dir / 'WHEEL', b'Wheel-Version: 1.0\n')
    rows = []
    for name, path in sorted(files.items()):
        raw = path.read_bytes()
        encoded = __import__('base64').urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        rows.append('vllm/' + name.removeprefix('installed/vllm/') + ',sha256=' + encoded + ',' + str(len(raw)))
    record = metadata_dir / 'RECORD'
    put(record, ('\n'.join(rows) + '\n').encode())
    files['installed/vllm-0.0.dist-info/METADATA'] = metadata_dir / 'METADATA'
    files['installed/vllm-0.0.dist-info/WHEEL'] = metadata_dir / 'WHEEL'
    files['installed/vllm-0.0.dist-info/RECORD'] = record
    return files, dict(vllm_site=str(root / 'site'), distribution_metadata=str(metadata_dir),
                       versions={'vllm': '0.0'}, record_key='installed/vllm-0.0.dist-info/RECORD')


def fixture_phase(root: Path):
    phase = root / 'research' / 'model' / 'scheduler'
    for relative in ('configs', 'workloads'):
        put(phase / relative / 'fixture.txt')
    for name in RUNNER.D_RUNTIME:
        put(phase / 'scripts' / name, b'# MOCK runtime source\n')
    put(phase / 'scripts/offline-review.py', b'# MOCK not a runtime dependency\n')
    research = phase.parent
    for relative in RUNNER.DEPENDENCY_TREES:
        put(research / relative / 'dependency.py')
    fixtures = path_at(research, 'scenarios/workloads/agent')
    put(fixtures / 'tasks.json', b'{"tasks":[{"id":"mock_task"}]}\n')
    put(fixtures / 'quality.json', b'{"cases":[]}\n')
    put(path_at(fixtures, 'replay_context.md'), b'MOCK context\n')
    put(fixtures / 'mock_task/initial.py', b'# MOCK fixture\n')
    put(research / 'environment.json', b'{}\n')
    put(research / 'model-inventory.json', b'{}\n')
    return phase


def fake_nvidia_smi(bin_dir: Path):
    script = bin_dir / 'nvidia-smi'
    put(script, b'#!/bin/sh\ncase "$*" in *--query-gpu=memory.used*) printf "0\\n0\\n0\\n0\\n";; *) printf "2026-11-07, 0, GPU-0, 0, 0, 0, 30\\n2026-11-07, 1, GPU-1, 0, 0, 0, 30\\n2026-11-07, 2, GPU-2, 0, 0, 0, 30\\n2026-11-07, 3, GPU-3, 0, 0, 0, 30\\n";; esac\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def call(runner, phase, provider, name, child, timeout=3.):
    args = type('Args', (), dict(name=name, timeout=timeout, gpus='0,1,2,3',
                                 hypothesis='fixture', reference='fixture', command=['--', *child]))()
    return runner.run(args, phase=phase, installed_provider=provider)


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='stage-d-supervisor-') as temp:
        root = Path(temp)
        phase = fixture_phase(root)
        installed, package = fake_installed(root)
        provider = lambda: (installed, package)
        fake_nvidia_smi(root / 'bin')
        old_path = os.environ.get('PATH', '')
        os.environ['PATH'] = str(root / 'bin') + os.pathsep + old_path
        try:
            child = [sys.executable, '-c', 'import time; time.sleep(.3)']
            assert call(RUNNER, phase, provider, 'd-ok', child) == 0
            status = load_document(path_at(phase, 'results/d-ok/status.json'))
            assert status['supervisor_exit_code'] == 0 and status['source_identity_unchanged']
            assert path_at(phase, 'results/d-ok/source-snapshot.zip').is_file()
            checks.append('normal child exit and immutable source snapshot')
            identity = load_document(path_at(phase, 'results/d-ok/runtime-identity.json'))
            with zipfile.ZipFile(path_at(phase, 'results/d-ok/source-snapshot.zip')) as archive:
                assert set(archive.namelist()) == set(identity['files'])
                assert all((hashlib.sha256(archive.read(key)).hexdigest() == row['sha256'] and len(archive.read(key)) == row['bytes'] for key, row in identity['files'].items()))
                assert 'scheduler/scripts/offline-review.py' not in archive.namelist()
                assert all(('scheduler/scripts/' + name in archive.namelist() for name in RUNNER.D_RUNTIME))
            checks.append('all snapshot bytes match ledger; offline D tools excluded and required runtime included')
            try:
                call(RUNNER, phase, provider, 'd-ok', child)
            except FileExistsError:
                checks.append('duplicate experiment ID refuses overwrite')
            else:
                raise AssertionError('duplicate result overwritten')
            child_fail = [sys.executable, '-c', 'raise SystemExit(7)']
            assert call(RUNNER, phase, provider, 'd-failed', child_fail) == 7
            failed = load_document(path_at(phase, 'results/d-failed/status.json'))
            assert failed['exit_code'] == 7 and failed['supervisor_exit_code'] == 7
            checks.append('child failure retained with terminal status')
            child_timeout = [sys.executable, '-c', 'import time; time.sleep(5)']
            started = time.monotonic()
            assert call(RUNNER, phase, provider, 'd-timeout', child_timeout, timeout=0.2) == 124
            timed = load_document(path_at(phase, 'results/d-timeout/status.json'))
            assert timed['timed_out'] and timed['supervisor_exit_code'] == 124 and (time.monotonic() - started < 4)
            checks.append('timeout kills only owned process group')
            mutator = [sys.executable, '-c', f'import time; time.sleep(.15); open({str(phase / 'scripts/study_client.py')!r}, "ab").write(b"# changed\\n"); time.sleep(.2)']
            assert call(RUNNER, phase, provider, 'd-source-change', mutator) == 1
            changed = load_document(path_at(phase, 'results/d-source-change/status.json'))
            assert changed['changed_source_members'] and (not changed['source_identity_unchanged'])
            checks.append('source mutation is detected after child exit')
            lock_path = phase.parent / '.research-gpu.lock'
            with open_output(lock_path, 'a+') as lock:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    call(RUNNER, phase, provider, 'd-lock', child)
                except BlockingIOError:
                    checks.append('shared GPU lock refuses concurrent run')
                else:
                    raise AssertionError('lock was not enforced')
            try:
                RUNNER.validate_gpus('0,4')
            except ValueError:
                checks.append('GPU 4 is rejected')
            else:
                raise AssertionError('GPU 4 accepted')
            for value in ('0,0', '', '0,1,2,3,4'):
                try:
                    RUNNER.validate_gpus(value)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid GPU selection accepted')
            checks.append('empty and duplicate GPU selections refused')
            raw = (phase / 'scripts/research_scheduler.py').read_bytes()
            (phase / 'scripts/research_scheduler.py').unlink()
            try:
                RUNNER.source_inputs(phase)
            except ValueError as exc:
                assert 'Missing required' in str(exc)
                checks.append('missing required runtime refused before launch')
            else:
                raise AssertionError('missing runtime accepted')
            put(phase / 'scripts/research_scheduler.py', raw)
            installed_path = installed['installed/vllm/config/scheduler.py']
            put(installed_path, b'# intentionally modified MOCK installed source\n')
            assert call(RUNNER, phase, provider, 'd-record-drift', child) == 1
            drift = load_document(path_at(phase, 'results/d-record-drift/status.json'))
            assert drift['exit_code'] is None and drift['supervisor_exit_code'] == 1 and ('RECORD' in drift['supervisor_error'])
            assert not path_at(phase, 'results/d-record-drift/process-state.json').exists()
            assert path_at(phase, 'results/d-record-drift/runtime-identity.json').is_file()
            checks.append('wheel RECORD mismatch preserves diagnostic without starting child')
            assert len(list(path_at(phase, 'results').glob('*/source-snapshot.zip'))) == 4
        finally:
            os.environ['PATH'] = old_path
    output = dict(status='PASS', checks=checks, count=len(checks), scope='Temporary CPU fixtures only; no real GPU, HTTP, model or installed-environment mutation.')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
