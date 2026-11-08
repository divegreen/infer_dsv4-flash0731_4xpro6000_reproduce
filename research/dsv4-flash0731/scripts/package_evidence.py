"""Create and verify an immutable local evidence archive; no model/venv/cache files."""

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
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='Unique local evidence ID')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        parser.error('Use a simple evidence ID, not a path.')
    output = path_at(ROOT, 'evidence')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'{args.name}.tar.gz'
    manifest = output / f'{args.name}.sha256'
    record = output / f'{args.name}.json'
    if any((p.exists() for p in (archive, manifest, record))):
        raise FileExistsError('Evidence ID already exists; choose a new ID.')
    sources = []
    for folder in path_at(ROOT, 'results').iterdir():
        if folder.is_dir() and (folder / 'command.json').is_file():
            sources.extend((p for p in folder.rglob('*') if p.is_file()))
    for folder in (path_at(ROOT, 'logs'), path_at(PROJECT, 'logs/vllm-install')):
        sources.extend((p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix != '.pyc')))
    sources += [ROOT / name for name in ('environment.json', 'model-inventory.json', 'topology-raw.log')]
    sources += [PROJECT / name for name in ('requirements-vllm.lock.txt', 'VLLM.md', 'activate-vllm.sh')]
    sources = sorted(set(sources))
    rows = {str(p.relative_to(PROJECT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sources}
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in sources:
            bundle.add(path, arcname=str(path.relative_to(PROJECT)), recursive=False)
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        assert {m.name for m in members} == set(rows) and len(members) == len(rows)
        for member in members:
            assert member.isfile() and member.size == rows[member.name]['bytes']
            with bundle.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == rows[member.name]['sha256']
    with open_output(manifest, 'x') as stream:
        for name, info in rows.items():
            stream.write(f'{info['sha256']}  {name}\n')
    metadata = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), archive=archive.name, archive_sha256=sha(archive), manifest=manifest.name, manifest_sha256=sha(manifest), files=len(rows), original_bytes=sum((v['bytes'] for v in rows.values())), archive_bytes=archive.stat().st_size, scope='Raw per-experiment records, logs, runtime source snapshots, installation evidence, environment and model metadata. Model payloads, virtualenv and caches excluded.')
    with open_output(record, 'x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
