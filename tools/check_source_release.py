#!/usr/bin/env python3
"""Check a source release using real packages, fresh Git metadata and CPU CLI calls.

All generated fixtures, packages and logs use a temporary directory. The optional
report is the only output written to --output. No GPU, model or network is used.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('source_release_entry', PROJECT/'tools/reproduce.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'results/validation/source-release.json')
    args = parser.parse_args()
    checks, commands = [], []

    def check(name, condition, detail=None):
        checks.append({'name': name, 'status': 'PASS' if condition else 'FAIL', 'detail': detail})
        if not condition:
            raise AssertionError(name + ': ' + str(detail))

    with tempfile.TemporaryDirectory(prefix='native source release ') as directory:
        temp = Path(directory)
        initial = temp/'initial.tar.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            cli.package(SimpleNamespace(output=str(initial)))
        manifest = json.loads(initial.with_suffix('.gz.manifest.json').read_text())
        with tarfile.open(initial) as tar:
            names = [str(Path(n).relative_to('portable-inference')) for n in tar.getnames()]
            check('no duplicate package members', len(names) == len(set(names)))
            check('manifest covers archive members', set(names) == set(manifest['members']))
            for member, name in zip(tar.getmembers(), names):
                check('native bytes ' + name,
                      hashlib.sha256(tar.extractfile(member).read()).hexdigest() == manifest['members'][name])
            tar.extractall(temp/'copy with spaces', filter='data')
        copy = temp/'copy with spaces/portable-inference'
        tool = copy/'tools/reproduce.py'
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                   DSV4_MODEL_PATH=str(temp/'absent model'), VLLM_PYTHON=str(temp/'absent python'))
        for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'):
            env.pop(key, None)

        def run(argv, *, cwd='/', expected=0):
            result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
            commands.append({'argv': [str(a) for a in argv], 'cwd': str(cwd), 'exit_code': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr})
            check('command ' + ' '.join(map(str, argv)), result.returncode == expected, result.stderr)
            return result

        cat = json.loads((copy/'research/dsv4-flash0731/experiments/catalog.json').read_text())
        for entry in cat['experiments']:
            for key in ('runner', 'config', 'plan'):
                if entry.get(key):
                    check('required input ' + entry['id'] + '/' + key, entry[key] in names)
        for path in copy.rglob('*.ipynb'):
            notebook = json.loads(path.read_text())
            check('notebook is an unexecuted template ' + path.name,
                  all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells']))
        run(['git', 'init', '-b', 'main', str(copy)])
        before = set(run(['git', 'ls-files', '--others', '--exclude-standard'], cwd=copy).stdout.splitlines())
        check('Git and package select the same native source', before == set(names), sorted(before.symmetric_difference(names)))
        run([sys.executable, '-B', str(tool), 'list'])
        check('list creates no results', not (copy/'results').exists())
        run([sys.executable, '-B', str(tool), 'validate'])
        run([sys.executable, '-B', str(tool), 'prepare'])
        run([sys.executable, '-B', str(tool), 'validate'])
        for entry in cat['experiments']:
            if entry['executable_definition']:
                run([sys.executable, '-B', str(tool), 'run', entry['id'], '--dry-run'])
        generated = list((copy/'research/dsv4-flash0731/experiments').glob('*/process*/*.json'))
        check('prepare actually generated stage views', len(generated) == len(cat['stages']) + len(cat['experiments']))
        # Independent contaminants reproduce the old publication bugs.
        artifacts = ['doc/validation.json', 'doc/validation/report.json',
                     'doc/history/extracted.json', 'archive/initial-project.tar.gz',
                     'research/dsv4-flash0731/doc/history/final-results.json',
                     'research/dsv4-flash0731/.research-gpu.lock', 'orders_history',
                     '.venv-vllm/lib/generated.py', '.venv-analysis/lib/generated.py',
                     '.cache/uv/state.json', 'tools/__pycache__/test.pyc',
                     'results/packages/old.tar.gz', 'tools/debug.log']
        for name in artifacts:
            path = copy/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"generated":true}\n')
        after = set(run(['git', 'ls-files', '--others', '--exclude-standard'], cwd=copy).stdout.splitlines())
        check('Git excludes prepared output and injected artifacts', before == after, sorted(after-before))
        rebuilt = temp/'after-prepare.tar.gz'
        run([sys.executable, '-B', str(tool), 'package', '--output', str(rebuilt)])
        after_manifest = json.loads(rebuilt.with_suffix('.gz.manifest.json').read_text())
        check('prepare and local outputs cannot alter source members or bytes', manifest['members'] == after_manifest['members'])
        with tarfile.open(rebuilt) as tar:
            for member in tar.getmembers():
                if member.name.endswith('.json'):
                    value = json.loads(tar.extractfile(member).read())
                    check('no generated JSON published ' + member.name,
                          not isinstance(value, dict) or value.get('generated') is not True)
        # Generated content in an unexpected native path must fail explicitly.
        unexpected = copy/'doc/unexpected-output.json'; unexpected.write_text('{"generated":true}')
        run([sys.executable, '-B', str(tool), 'package', '--output', str(temp/'reject-generated.tar.gz')], expected=2)
        unexpected.unlink()
        link = copy/'tools/external-link'; link.symlink_to(temp, target_is_directory=True)
        run([sys.executable, '-B', str(tool), 'package', '--output', str(temp/'reject-symlink.tar.gz')], expected=2)
        link.unlink()
        traversal = copy/'results/../tools/forbidden-package.tar.gz'
        run([sys.executable, '-B', str(tool), 'package', '--output', str(traversal)], expected=2)
        check('normalized output cannot enter source through parent components',
              not (copy/'tools/forbidden-package.tar.gz').exists())
        (copy/'results/source-link').symlink_to(copy/'tools', target_is_directory=True)
        run([sys.executable, '-B', str(tool), 'package', '--output',
             str(copy/'results/source-link/forbidden-package.tar.gz')], expected=2)
        check('output directory symlink cannot redirect a package into source',
              not (copy/'tools/forbidden-package.tar.gz').exists())
        native_parent = copy/'tools/uncreated-parent'
        run([sys.executable, '-B', str(tool), 'package', '--output',
             str(native_parent/'forbidden-package.tar.gz')], expected=2)
        check('rejected package path creates no directory', not native_parent.exists())
    report = {'generated': True, 'status': 'PASS', 'checks': checks, 'count': len(checks),
              'commands': commands, 'source_members': len(names),
              'scope': 'Real source packages, clean-tree CPU preparation and dry-run, Git candidate consistency; no inference'}
    cli.save(args.output, report)
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'source_members': len(names), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
