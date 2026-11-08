#!/usr/bin/env python3
"""Independent bounded CPU review of D packaging; all data are MOCK temp files."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import zipfile
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parents[1]
source=D/'analysis/package_phase_d.py'
spec=importlib.util.spec_from_file_location('independent_d_packaging_review',source)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode())

def run():
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append(dict(name=name, status='PASS'))
        except Exception as e:
            checks.append(dict(name=name, status='FAIL', error=repr(e)))

    def rejected(fn):
        try:
            fn()
        except (ValueError, RuntimeError, OSError, KeyError):
            return
        raise AssertionError('Invalid archive input accepted')
    with tempfile.TemporaryDirectory(prefix='d-archive-independent-MOCK-') as temporary:
        project = Path(temporary)
        research = project / 'research/model'
        phase = research / 'scheduler'
        for name in p.DEPENDENCY_TREES:
            write(research / name / 'MOCK.py', b'# source\n')
        for name in p.RESEARCH_REQUIRED:
            write(research / name, b'MOCK metadata\n')
        for name in p.PROJECT_REQUIRED:
            write(project / name, b'MOCK project\n')
        for name in p.SMALL_REFERENCE_CONFIGS:
            write(research / name, {})
        history = {}
        for name in p.REFERENCE_RECORDS:
            record = research / name
            archive = record.with_suffix('.tar.gz')
            write(archive, b'MOCK historical bytes ' + name.encode())
            write(record, dict(status='PASS', archive=archive.name, archive_sha256=p.sha(archive), archive_bytes=archive.stat().st_size))
            history[archive] = archive.read_bytes()
            history[record] = record.read_bytes()
        success = path_at(phase, 'results/d01-mock')
        failed = path_at(phase, 'results/d02-mock')
        early = path_at(phase, 'results/d03-mock')
        for folder, code in ((success, 0), (failed, 1), (early, 1)):
            launch = dict(experiment_id=folder.name, command=['MOCK'], cwd=str(project), git_commit='MOCK')
            if folder != early:
                member = 'scheduler/scripts/MOCK.py'
                data = b'# MOCK executed source\n'
                files = {member: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))}
                write(folder / 'runtime-identity.json', dict(files=files))
                with zipfile.ZipFile(folder / 'source-snapshot.zip', 'x') as z:
                    z.writestr(member, data)
                launch.update(source_sha256={member: files[member]['sha256']}, runtime_identity_sha256=p.sha(folder / 'runtime-identity.json'), source_snapshot_sha256=p.sha(folder / 'source-snapshot.zip'))
            write(folder / 'command.json', launch)
            write(folder / 'status.json', dict(launch, end_utc='MOCK terminal', supervisor_exit_code=code, exit_code=code))
        write(path_at(phase, 'report.md'), b'MOCK stage report\n')
        write(path_at(phase, 'results/d02-mock/failure.txt'), b'MOCK diagnostic preserved\n')
        write(path_at(research, 'scenarios/results/historical-raw/secret-marker.txt'), b'HISTORICAL_NOT_D_PAYLOAD')
        archive_record = [None]

        def roundtrip():
            result = p.package(phase, 'independent-mock')
            archive_record[0] = result
            manifest = load_document(path_at(phase, 'evidence') / result['manifest'])
            assert result['experiment_ids'] == ['d01-mock', 'd02-mock', 'd03-mock']
            assert manifest['experiments'][-1]['source_snapshot']['status'] == 'MISSING_IN_FAILED_RUN'
            with tarfile.open(path_at(phase, 'evidence') / result['archive'], 'r:gz') as archive:
                names = archive.getnames()
                assert any((name.endswith('d02-mock/failure.txt') for name in names))
                assert not any(('historical-raw' in name or name.endswith('.tar.gz') for name in names))
                target = next((name for name in names if name.endswith('d02-mock/status.json')))
                assert json.load(archive.extractfile(target))['supervisor_exit_code'] == 1
            assert all((path.read_bytes() == raw for path, raw in history.items()))
            assert p.verify(path_at(phase, 'evidence/independent-mock.json'))['status'] == 'PASS'
        check('own temporary archive preserves all success/failure IDs and immutable SHA-only historical references', roundtrip)

        def missing_success():
            target = success / 'source-snapshot.zip'
            raw = target.read_bytes()
            target.unlink()
            try:
                rejected(lambda: p.terminal_results(phase))
            finally:
                write_bytes(target, raw)
        check('successful source snapshot cannot be silently omitted', missing_success)

        def false_promote():
            target = early / 'status.json'
            raw = target.read_bytes()
            status = json.loads(raw)
            status['supervisor_exit_code'] = 0
            write(target, status)
            try:
                rejected(lambda: p.terminal_results(phase))
            finally:
                write_bytes(target, raw)
        check('missing-source prelaunch failure cannot be promoted to successful archive identity', false_promote)

        def oldraw():
            before = p.DEPENDENCY_TREES
            p.DEPENDENCY_TREES = (*before, 'scenarios/results')
            try:
                rejected(lambda: p.collect(phase, set()))
            finally:
                p.DEPENDENCY_TREES = before
        check('historical raw defense holds even if dependency selection is expanded', oldraw)
        check('archive path traversal rejected', lambda: rejected(lambda: p.package(phase, '../escape')))

        def duplicate():
            record = path_at(phase, 'evidence/independent-mock.json')
            raw = record.read_bytes()
            rejected(lambda: p.package(phase, 'independent-mock'))
            assert record.read_bytes() == raw
        check('duplicate archive identity leaves accepted record byte-identical', duplicate)

        def arbitrary_source():
            (phase / 'named-pipe').symlink_to('/tmp')
            rejected(lambda: p.collect(phase, set()))
            (phase / 'named-pipe').unlink()
        check('symlink source refused without following or extracting it', arbitrary_source)
    return dict(status='PASS' if all((x['status'] == 'PASS' for x in checks)) else 'FAIL', checks=checks, scope='Independent CPU-only synthetic archive controls; no real evidence archive, historical writes, GPU or network', source_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in (Path(__file__), source, Path(p.core.__file__))})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use unique output')
    result = run()
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), failures=[x for x in result['checks'] if x['status'] != 'PASS'])))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
