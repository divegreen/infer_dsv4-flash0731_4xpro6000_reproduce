"""CPU-only immutable scenario evidence archive and independent member verification.

No inference, GPU inspection, network, extraction, or experiment status rewriting.
Final packaging requires the shared research lock and terminal supervisor records.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {"__pycache__", ".git", ".cache", "venv", "weights", "models", "node_modules"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin"}
PARENT_REQUIRED = ("environment.json", "model-inventory.json", "topology-raw.log", "scripts/benchmark_client.py")
PARENT_OPTIONAL = ("model-analysis.md", "framework-capabilities.md", "topology-analysis.md", "report.md",
                   "DEPLOYMENT.md", "correctness-review.md", "roadmap.md", "plan.md", "experiment-registry.md",
                   "anomalies.md", "review-findings.md", "final-results-review.md")
PROJECT_REQUIRED = ("requirements-vllm.lock.txt", "VLLM.md", "activate-vllm.sh", "AGENTS.md", "progress.md")


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"Not a regular file (symlinks refused): {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def sha(path):
    before = fingerprint(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(before == fingerprint(path), f"Source changed while hashing: {path}")
    return digest


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def new_json(path, value):
    with open_output(path, 'xb') as stream:
        stream.write(json_bytes(value))


def replace_record(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    new_json(temp, value)
    temp.replace(path)


def excluded(path, stage):
    return path == path_at(stage, 'evidence') or path.name in EXCLUDED_DIRS or path.name.startswith('.venv') or (path.suffix.lower() in EXCLUDED_SUFFIXES)


def collect(stage):
    research = stage.parent
    project = research.parents[1]
    files, skipped = ([], [])

    def visit(path):
        if excluded(path, stage):
            skipped.append(str(path.relative_to(project)))
            return
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            for child in sorted(path.iterdir()):
                visit(child)
        elif stat.S_ISREG(mode):
            files.append(path)
        else:
            raise ValueError(f'Symlink/special file refused: {path}')
    visit(stage)
    for relative in PARENT_REQUIRED:
        path = research / relative
        fingerprint(path)
        visit(path)
    for relative in PARENT_OPTIONAL:
        path = research / relative
        if path.exists() or path.is_symlink():
            visit(path)
    for relative in PROJECT_REQUIRED:
        path = project / relative
        fingerprint(path)
        visit(path)
    install = path_at(project, 'logs/vllm-install')
    if install.exists():
        visit(install)
    return (sorted(set(files)), sorted(set(skipped)), project)


def terminal_results(stage):
    folder = path_at(stage, 'results')
    require(folder.is_dir(), 'results directory missing')
    rows = []
    for result in sorted(folder.iterdir()):
        if not result.is_dir() or result.is_symlink():
            continue
        require((result / 'command.json').is_file(), f'Missing command record: {result.name}')
        require((result / 'status.json').is_file(), f'Unfinished experiment: {result.name}')
        command = load_document(result / 'command.json')
        status = load_document(result / 'status.json')
        require(command.get('experiment_id') == result.name == status.get('experiment_id'), f'Experiment identity mismatch: {result.name}')
        require(bool(status.get('end_utc')), f'Nonterminal supervisor: {result.name}')
        code = status.get('exit_code')
        require(type(code) is int or bool(status.get('supervisor_error')) or status.get('timed_out') is True, f'Missing terminal outcome: {result.name}')
        rows.append(dict(id=result.name, exit_code=code, timed_out=status.get('timed_out'), supervisor_error=status.get('supervisor_error'), end_utc=status['end_utc']))
    require(rows, 'No terminal experiments found')
    return rows


@contextmanager
def research_lock(stage):
    path = stage.parent / '.research-gpu.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 384)
    with os.fdopen(fd, 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Research runner active; wait for all series to finish before packaging') from exc
        yield


def safe_name(name):
    path = PurePosixPath(name)
    return (isinstance(name, str) and not path.is_absolute() and ".." not in path.parts
            and str(path) == name and "\\" not in name and "\n" not in name and "\r" not in name
            and bool(path.parts))


def verify_tar(archive, rows):
    require(rows and len({row["path"] for row in rows}) == len(rows), "Empty/duplicate manifest paths")
    expected = {row["path"]: row for row in rows}
    require(all(safe_name(name) for name in expected), "Unsafe manifest path")
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            require(member.name not in seen, f"Duplicate archive member: {member.name}")
            require(safe_name(member.name) and member.isfile(), f"Unsafe/nonregular member: {member.name}")
            require(member.name in expected, f"Unexpected archive member: {member.name}")
            row = expected[member.name]
            require(member.size == row["bytes"], f"Byte-count mismatch: {member.name}")
            with bundle.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            require(actual == row["sha256"], f"SHA256 mismatch: {member.name}")
            seen.add(member.name)
    require(seen == set(expected), "Archive is missing manifest members")
    return len(seen)


def sum_text(rows):
    return "".join(f'{row["sha256"]}  {row["path"]}\n' for row in rows)


def verify(record_path):
    record = load_document(record_path)
    require(record.get('status') == 'PASS', 'Package record is not PASS')
    paths = {}
    for key in ('archive', 'manifest', 'sha256_file'):
        name = record[key]
        require(safe_name(name) and Path(name).name == name, f'Invalid sidecar name: {key}')
        path = record_path.parent / name
        require(sha(path) == record[key + '_sha256'], f'Sidecar/archive SHA256 mismatch: {key}')
        paths[key] = path
    manifest = load_document(paths['manifest'])
    rows = manifest['files']
    require(paths['sha256_file'].read_text() == sum_text(rows), 'Text/JSON manifests differ')
    require(paths['archive'].stat().st_size == record['archive_bytes'], 'Archive byte-count differs')
    count = verify_tar(paths['archive'], rows)
    require(count == record['files'], 'Record file count differs')
    require(sum((row['bytes'] for row in rows)) == record['original_bytes'], 'Record source bytes differ')
    return dict(status='PASS', files=count, archive_sha256=record['archive_sha256'], scope='Archive/member integrity only; not model quality or experiment success')


def package(stage, name):
    require(bool(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', name)), 'Use a simple unique evidence ID')
    stage = stage.resolve(strict=True)
    with research_lock(stage):
        experiments = terminal_results(stage)
        output = path_at(stage, 'evidence')
        output.mkdir(parents=True, exist_ok=True)
        require(not output.is_symlink(), 'Evidence directory cannot be a symlink')
        archive = output / f'{name}.tar.gz'
        manifest_path = output / f'{name}.manifest.json'
        sums = output / f'{name}.sha256'
        record_path = output / f'{name}.json'
        require(not any((p.exists() or p.is_symlink() for p in (archive, manifest_path, sums, record_path))), 'Evidence ID already exists; use a new ID')
        record = dict(status='IN PROGRESS', created_utc=utc(), evidence_id=name)
        new_json(record_path, record)
        try:
            sources, skipped, project = collect(stage)
            rows = []
            for path in sources:
                name_in_tar = str(path.relative_to(project))
                require(safe_name(name_in_tar), f'Unsupported path: {name_in_tar}')
                rows.append(dict(path=name_in_tar, sha256=sha(path), bytes=path.stat().st_size))
            manifest = dict(schema_version=1, path_base='workspace root', created_utc=utc(), files=rows, experiments=experiments, exclusions=skipped, scope='Stage code/config/workloads/reports/validation logs/all raw results including CUDA traces; parent metadata/helper/docs and installation evidence. No weights/venvs/caches/previous archives.')
            new_json(manifest_path, manifest)
            with open_output(sums, 'x') as stream:
                stream.write(sum_text(rows))
            with open_output(archive, 'xb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped:
                    with open_output(tarfile, fileobj=zipped, mode='w|') as bundle:
                        for path, row in zip(sources, rows):
                            fingerprint(path)
                            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                                info = bundle.gettarinfo(str(path), arcname=row['path'], fileobj=stream)
                                require(info.isfile(), f'Source became nonregular: {path}')
                                bundle.addfile(info, stream)
            verify_tar(archive, rows)
            final_sources, final_skipped, _ = collect(stage)
            require(sources == final_sources and skipped == final_skipped, 'Source membership changed during packaging')
            for path, row in zip(sources, rows):
                require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], f'Source changed during packaging: {path}')
            require(terminal_results(stage) == experiments, 'Supervisor records changed during packaging')
            record.update(status='PASS', completed_utc=utc(), archive=archive.name, archive_sha256=sha(archive), manifest=manifest_path.name, manifest_sha256=sha(manifest_path), sha256_file=sums.name, sha256_file_sha256=sha(sums), files=len(rows), original_bytes=sum((row['bytes'] for row in rows)), archive_bytes=archive.stat().st_size, scope='Local integrity-verified evidence snapshot; failed experiments remain included. This is not a quality/performance PASS or a cryptographic provenance signature.')
            replace_record(record_path, record)
            verify(record_path)
            return record
        except BaseException as exc:
            record.update(status='FAILED', ended_utc=utc(), error=repr(exc))
            replace_record(record_path, record)
            raise


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix='scenario-package-test-') as temp:
        project = Path(temp)
        research = project / 'research/model'
        stage = research / 'scenarios-fixture'
        stage.mkdir(parents=True)

        def put(path, data=b'fixture\n'):
            path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(path, data)
        for relative in PARENT_REQUIRED:
            put(research / relative)
        for relative in PROJECT_REQUIRED:
            put(project / relative)
        put(stage / 'scripts/client.py')
        put(path_at(stage, 'logs/validation.log'))
        result = path_at(stage, 'results/fixture-failed')
        put(result / 'command.json', json_bytes(dict(experiment_id=result.name)))
        put(result / 'status.json', json_bytes(dict(experiment_id=result.name, end_utc=utc(), exit_code=1)))
        put(result / 'profiles/rank0.pt.trace.json.gz', gzip.compress(b'{"traceEvents":[]}'))
        with zipfile.ZipFile(result / 'source-snapshot.zip', 'x') as saved:
            saved.writestr('client.py', 'fixture')
        put(stage / '__pycache__/x.pyc')
        put(stage / '.venv/bin/python')
        put(stage / 'weights/model.safetensors')
        put(path_at(stage, 'evidence/old.tar.gz'))
        record = package(stage, 'fixture-good')
        record_path = path_at(stage, 'evidence/fixture-good.json')
        assert verify(record_path)['status'] == 'PASS'
        rows = load_document(record_path.parent / record['manifest'])['files']
        names = {row['path'] for row in rows}
        assert any((n.endswith('.pt.trace.json.gz') for n in names))
        assert any((n.endswith('source-snapshot.zip') for n in names))
        assert not any(('__pycache__' in n or '/.venv/' in n or '/evidence/' in n or ('/weights/' in n) for n in names))
        checks.append('Roundtrip including failed experiment, raw trace and ZIP; excludes cache/venv/weights/archive recursion')

        def rejects(label, function):
            try:
                function()
            except (ValueError, RuntimeError, OSError):
                checks.append(label)
            else:
                raise AssertionError('Not rejected: ' + label)
        rejects('Duplicate evidence ID', lambda: package(stage, 'fixture-good'))
        incomplete = path_at(stage, 'results/incomplete')
        put(incomplete / 'command.json', b'{}')
        rejects('Unfinished experiment', lambda: package(stage, 'unfinished'))
        (incomplete / 'command.json').unlink()
        incomplete.rmdir()
        link = stage / 'outside-link'
        link.symlink_to(path_at(project, 'AGENTS.md'))
        rejects('Symlink input', lambda: package(stage, 'symlink'))
        assert load_document(path_at(stage, 'evidence/symlink.json'))['status'] == 'FAILED'
        link.unlink()
        bad = [dict(row) for row in rows]
        bad[0]['sha256'] = '0' * 64
        rejects('Manifest member-hash mismatch', lambda: verify_tar(record_path.parent / record['archive'], bad))
        duplicate = project / 'duplicate.tar.gz'
        with tarfile.open(duplicate, 'w:gz') as bundle:
            for _ in range(2):
                item = tarfile.TarInfo('a.txt')
                item.size = 1
                bundle.addfile(item, io.BytesIO(b'a'))
        rejects('Duplicate tar member', lambda: verify_tar(duplicate, [dict(path='a.txt', bytes=1, sha256=hashlib.sha256(b'a').hexdigest())]))
        with research_lock(stage):
            rejects('Cooperating runner lock held', lambda: package(stage, 'locked'))
        original_verifier = globals()['verify_tar']

        def mutate_after_archive(archive, manifest_rows):
            value = original_verifier(archive, manifest_rows)
            put(stage / 'scripts/client.py', b'changed during packaging\n')
            return value
        globals()['verify_tar'] = mutate_after_archive
        try:
            rejects('Source mutation during packaging', lambda: package(stage, 'mutated'))
        finally:
            globals()['verify_tar'] = original_verifier
        assert load_document(path_at(stage, 'evidence/mutated.json'))['status'] == 'FAILED'
        with open_output(record_path.parent / record['archive'], 'ab') as stream:
            stream.write(b'changed')
        rejects('Archive checksum corruption', lambda: verify(record_path))
    return dict(status='PASS', scope='Temporary CPU fixtures only; no real evidence package, GPU or HTTP', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--name", help="New final evidence ID; run only after series and analysis finish")
    mode.add_argument("--verify", type=Path, metavar="RECORD_JSON")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT, help="Stage directory with workspace/research/stage layout")
    parser.add_argument("--output", type=Path, help="Optional NEW self-test or verification result JSON")
    args = parser.parse_args()
    if args.name and args.output:
        parser.error("--output applies only to --self-test or --verify")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    result = self_test() if args.self_test else verify(args.verify) if args.verify else package(args.root, args.name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        new_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
