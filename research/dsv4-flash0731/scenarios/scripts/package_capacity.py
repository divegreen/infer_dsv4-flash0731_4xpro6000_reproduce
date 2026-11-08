#!/usr/bin/env python3
"""Immutable capacity-only evidence archive with independent member verification.

Contains selected a* capacity experiment directories, capacity tools/plans/docs,
analysis/figure/log artifacts, and environment metadata. Never includes other
scenario results, weights, environments, caches, or previous evidence archives.
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
import gzip
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile

import package_scenarios as core

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = re.compile(r'a[0-9][A-Za-z0-9._-]*')
PROJECT_FILES = ('AGENTS.md', 'progress.md', 'requirements-vllm.lock.txt', 'VLLM.md', 'activate-vllm.sh')
RESEARCH_FILES = ('environment.json', 'model-inventory.json', 'topology-raw.log',
                  'model-analysis.md', 'framework-capabilities.md', 'topology-analysis.md',
                  'roadmap.md', 'scripts/benchmark_client.py')


def terminal_results(stage, names=None):
    if names is None:
        names = [p.name for p in sorted(path_at(stage, 'results').iterdir()) if EXPERIMENT.fullmatch(p.name)]
    core.require(names and len(set(names)) == len(names), 'No capacity experiment IDs, or duplicate IDs')
    records = []
    for name in sorted(names):
        core.require(EXPERIMENT.fullmatch(name), f'Capacity ID must start a followed by a digit: {name}')
        directory = path_at(stage, 'results') / name
        core.require(directory.is_dir() and (not directory.is_symlink()), f'Invalid result directory: {name}')
        core.fingerprint(directory / 'command.json')
        core.fingerprint(directory / 'status.json')
        command = load_document(directory / 'command.json')
        status = load_document(directory / 'status.json')
        core.require(command.get('experiment_id') == name == status.get('experiment_id'), 'Result identity mismatch')
        core.require(any((Path(str(x)).name == 'serve_capacity.py' for x in command.get('command', []))), f'Not an owned capacity launcher result: {name}')
        core.require(bool(status.get('end_utc')), f'Nonterminal experiment: {name}')
        core.require(type(status.get('exit_code')) is int or status.get('timed_out') is True or bool(status.get('supervisor_error')), f'Missing terminal outcome: {name}')
        records.append(dict(id=name, exit_code=status.get('exit_code'), timed_out=status.get('timed_out'), supervisor_error=status.get('supervisor_error'), end_utc=status['end_utc']))
    return records


def collect(stage, experiments, summary=None, figure_dir=None):
    research = stage.parent
    project = research.parents[1]
    files, excluded = (set(), set())

    def visit(path):
        if core.excluded(path, stage):
            excluded.add(str(path.relative_to(project)))
            return
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            for child in sorted(path.iterdir()):
                visit(child)
        elif stat.S_ISREG(mode):
            core.require(path.resolve().is_relative_to(project.resolve()), 'Input leaves workspace')
            files.add(path)
        else:
            raise ValueError(f'Symlink/special file refused: {path}')
    for item in experiments:
        visit(path_at(stage, 'results') / item['id'])
        command = load_document(path_at(stage, 'results') / item['id'] / 'command.json')['command']
        for flag in ('--config', '--plan'):
            if flag in command:
                path = Path(command[command.index(flag) + 1])
                if path.is_absolute() and path.exists():
                    core.require(path.resolve().is_relative_to(stage.resolve()), 'Plan/config outside stage')
                    visit(path)
    for path in sorted((stage / 'scripts').glob('*capacity*.py')):
        visit(path)
    for name in ('run_logged.py', 'package_scenarios.py', 'summarize_scenarios.py'):
        path = stage / 'scripts' / name
        core.fingerprint(path)
        visit(path)
    if path_at(stage, 'workloads/capacity').exists():
        visit(path_at(stage, 'workloads/capacity'))
    for path in sorted(stage.iterdir()):
        if 'capacity' in path.name.lower() and path.suffix.lower() in ('.md', '.json'):
            visit(path)
    for directory, pattern in ((stage, '*capacity*.md'), (stage, '*capacity*.json'), (path_at(stage, 'logs'), '*capacity*'), (path_at(stage, 'summary'), '*capacity*')):
        if directory.is_dir():
            for path in sorted(directory.glob(pattern)):
                visit(path)
    if path_at(stage, 'logs').is_dir():
        for path in sorted(path_at(stage, 'logs').iterdir()):
            if EXPERIMENT.match(path.name):
                visit(path)
    if summary:
        core.require(summary.resolve().is_relative_to(stage.resolve()), 'Summary must be inside stage')
        visit(summary)
        companion = summary.with_suffix('.md')
        if companion.exists():
            visit(companion)
    if figure_dir:
        core.require(figure_dir.resolve().is_relative_to(path_at(stage, 'figures').resolve()), 'Figures must be under stage/figures')
        visit(figure_dir)
    elif path_at(stage, 'figures').is_dir():
        for path in sorted(path_at(stage, 'figures').glob('capacity*')):
            visit(path)
    for relative in PROJECT_FILES:
        path = project / relative
        core.fingerprint(path)
        visit(path)
    for relative in RESEARCH_FILES:
        path = research / relative
        if path.exists():
            visit(path)
    for required in ('environment.json', 'model-inventory.json', 'topology-raw.log'):
        core.require(research / required in files, f'Missing required environment/model/topology record: {required}')
    core.require(research / 'scripts/benchmark_client.py' in files, 'Missing shared benchmark helper')
    selected = {item['id'] for item in experiments}
    for path in files:
        if path.is_relative_to(path_at(stage, 'results')):
            core.require(path.relative_to(path_at(stage, 'results')).parts[0] in selected, 'Unselected result accidentally included')
    return (sorted(files), sorted(excluded), project)


def package(stage, name, names=None, summary=None, figure_dir=None):
    core.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', name), 'Use a unique simple evidence ID')
    stage = stage.resolve(strict=True)
    summary = Path(summary).expanduser().resolve(strict=True) if summary else None
    figure_dir = Path(figure_dir).expanduser().resolve(strict=True) if figure_dir else None
    with core.research_lock(stage):
        experiments = terminal_results(stage, names)
        output = path_at(stage, 'evidence')
        output.mkdir(parents=True, exist_ok=True)
        core.require(not output.is_symlink(), 'Evidence output cannot be a symlink')
        archive, manifest_path, sums, record_path = [output / (name + suffix) for suffix in ('.tar.gz', '.manifest.json', '.sha256', '.json')]
        core.require(not any((p.exists() or p.is_symlink() for p in (archive, manifest_path, sums, record_path))), 'Evidence ID exists; never overwrite a previous package')
        record = dict(status='IN PROGRESS', created_utc=core.utc(), evidence_id=name, scope='Capacity evidence integrity only; not a performance, SLO or model-quality PASS')
        core.new_json(record_path, record)
        try:
            sources, exclusions, project = collect(stage, experiments, summary, figure_dir)
            rows = [dict(path=str(p.relative_to(project)), sha256=core.sha(p), bytes=p.stat().st_size) for p in sources]
            core.require(all((core.safe_name(r['path']) for r in rows)), 'Unsafe archive member')
            manifest = dict(schema_version=1, created_utc=core.utc(), path_base='workspace root', experiments=experiments, files=rows, exclusions=exclusions, scope='Selected a* capacity result trees including failed results, raw JSONL, prompts, Prometheus, GPU/server logs, source snapshots; capacity tooling/plans/reports/figures and environment metadata.', selection_mode='explicit experiment selection' if names is not None else 'all matching capacity result directories', reference_closure_guaranteed=False, reference_caveat='Summary/docs/logs may refer to unselected experiments or prior scenario evidence. This archive contains only selected capacity raw result trees; old profile traces require their original evidence archive.', not_included='Other scenario result trees, weights, virtual environments, caches and previous archives')
            core.new_json(manifest_path, manifest)
            with open_output(sums, 'x') as stream:
                stream.write(core.sum_text(rows))
            with open_output(archive, 'xb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                    with open_output(tarfile, fileobj=compressed, mode='w|') as bundle:
                        for path, row in zip(sources, rows):
                            core.fingerprint(path)
                            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                                info = bundle.gettarinfo(str(path), arcname=row['path'], fileobj=stream)
                                core.require(info.isfile(), 'Source became nonregular')
                                bundle.addfile(info, stream)
            core.verify_tar(archive, rows)
            final_sources, final_exclusions, _ = collect(stage, experiments, summary, figure_dir)
            core.require(final_sources == sources and final_exclusions == exclusions, 'Membership changed while archiving')
            for path, row in zip(sources, rows):
                core.require(core.sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source changed while archiving')
            core.require(terminal_results(stage, [x['id'] for x in experiments]) == experiments, 'Supervisor outcomes changed while archiving')
            record.update(status='PASS', completed_utc=core.utc(), archive=archive.name, archive_sha256=core.sha(archive), archive_bytes=archive.stat().st_size, manifest=manifest_path.name, manifest_sha256=core.sha(manifest_path), sha256_file=sums.name, sha256_file_sha256=core.sha(sums), files=len(rows), original_bytes=sum((r['bytes'] for r in rows)), experiment_ids=[x['id'] for x in experiments])
            core.replace_record(record_path, record)
            core.verify(record_path)
            return record
        except BaseException as exc:
            record.update(status='FAILED', ended_utc=core.utc(), error=repr(exc))
            core.replace_record(record_path, record)
            raise


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix='capacity-package-fixture-') as temporary:
        project = Path(temporary)
        research = project / 'research/model'
        stage = research / 'scenarios-fixture'

        def put(path, data=b'fixture\n'):
            path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(path, data)
        for name in PROJECT_FILES:
            put(project / name)
        for name in ('environment.json', 'model-inventory.json', 'topology-raw.log'):
            put(research / name, b'{}\n')
        put(research / 'scripts/benchmark_client.py')
        for name in ('run_logged.py', 'package_scenarios.py', 'summarize_scenarios.py', 'capacity_client.py'):
            put(stage / 'scripts' / name)
        result = path_at(stage, 'results/a01-fixture-failed')
        put(result / 'command.json', core.json_bytes(dict(experiment_id=result.name, command=['python', 'serve_capacity.py'])))
        put(result / 'status.json', core.json_bytes(dict(experiment_id=result.name, exit_code=1, end_utc=core.utc())))
        for name in ('server.log', 'gpu.csv', 'capacity.json', 'capacity-x-requests.jsonl', 'capacity-x-metrics/before.prom', 'capacity-x-metrics/00000.prom', 'source-snapshot.zip'):
            put(result / name)
        put(path_at(stage, 'results/s01-old/DO_NOT_INCLUDE.txt'))
        put(stage / 'weights/model.safetensors')
        put(stage / '.venv/bin/python')
        put(path_at(stage, 'logs/capacity-audit.json'), b'{"status":"PASS"}\n')
        put(path_at(stage, 'workloads/capacity/series.json'), b'{}\n')
        put(stage / 'capacity-old-evidence.json', b'{}\n')
        put(path_at(stage, 'CAPACITY-REPRODUCE.md'), b'fixture reproduction instructions\n')
        put(path_at(stage, 'summary/capacity-summary.json'), b'{}\n')
        put(path_at(stage, 'figures/capacity-test/manifest.json'), b'{}\n')
        record = package(stage, 'capacity-fixture')
        record_path = path_at(stage, 'evidence/capacity-fixture.json')
        assert core.verify(record_path)['status'] == 'PASS'
        names = {r['path'] for r in load_document(path_at(stage, 'evidence') / record['manifest'])['files']}
        assert any((n.endswith('before.prom') for n in names))
        assert any((n.endswith('source-snapshot.zip') for n in names))
        assert any((n.endswith('capacity-old-evidence.json') for n in names))
        assert any((n.endswith('CAPACITY-REPRODUCE.md') for n in names))
        assert any((n.endswith('model-inventory.json') for n in names))
        assert any((n.endswith('topology-raw.log') for n in names))
        assert not any(('s01-old' in n or '/.venv/' in n or '/weights/' in n or ('/evidence/' in n) for n in names))
        checks += ['capacity failed experiment remains included', 'raw metrics/JSONL/source snapshot retained', 'old result trees/weights/environments/archives excluded', 'archive/member/sidecar hash verification']

        def rejects(label, call):
            try:
                call()
            except (ValueError, RuntimeError, OSError):
                checks.append(label)
            else:
                raise AssertionError('Not rejected: ' + label)
        rejects('duplicate evidence ID', lambda: package(stage, 'capacity-fixture'))
        relative = package(stage, 'relative-options', ['a01-fixture-failed'], Path(os.path.relpath(path_at(stage, 'summary/capacity-summary.json'))), Path(os.path.relpath(path_at(stage, 'figures/capacity-test'))))
        assert relative['status'] == 'PASS'
        checks += ['relative summary and figure paths normalized', 'capacity-stage JSON and model/topology metadata retained', 'uppercase CAPACITY-REPRODUCE.md included']
        (research / 'model-inventory.json').unlink()
        rejects('missing required model inventory', lambda: package(stage, 'missing-inventory', ['a01-fixture-failed']))
        put(research / 'model-inventory.json', b'{}\n')
        missing = path_at(stage, 'results/a02-unfinished')
        put(missing / 'command.json', core.json_bytes(dict(experiment_id=missing.name, command=['serve_capacity.py'])))
        rejects('unfinished capacity result', lambda: package(stage, 'unfinished'))
        with core.research_lock(stage):
            rejects('active shared research lock', lambda: package(stage, 'locked', ['a01-fixture-failed']))
        (result / 'unsafe-link').symlink_to(path_at(project, 'AGENTS.md'))
        rejects('symlink result input', lambda: package(stage, 'symlink', ['a01-fixture-failed']))
        with open_output(path_at(stage, 'evidence') / record['archive'], 'ab') as stream:
            stream.write(b'changed')
        rejects('archive corruption', lambda: core.verify(record_path))
    return dict(status='PASS', scope='Temporary CPU-only archive fixture; no actual capacity archive created', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--name')
    mode.add_argument('--verify', type=Path)
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--experiment', action='append', help='Select a capacity ID; repeat, otherwise include all a* capacity results')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--figure-dir', type=Path)
    parser.add_argument('--output', type=Path, help='New verification/self-test record, not archive location')
    args = parser.parse_args()
    if args.name and args.output:
        parser.error('--output applies only to --verify/--self-test')
    result = self_test() if args.self_test else core.verify(args.verify) if args.verify else package(
        args.root, args.name, args.experiment, args.summary, args.figure_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        core.new_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
