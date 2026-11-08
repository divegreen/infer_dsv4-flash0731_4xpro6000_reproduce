"""Immutable Stage C evidence with explicit experiment IDs and shared GPU lock.

CPU filesystem archive only. Keep failed experiments, wire streams, task states,
source snapshots, analysis inputs and offline dependencies; no model/venv/cache.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('stage_c_archive_core',ROOT.parent/'scenarios/scripts/package_scenarios.py')
core=importlib.util.module_from_spec(sp);sp.loader.exec_module(core)
SHARED=('agent_session_client.py','agent_session_tasks.py','serve_agent_sessions.py','run_agent_series.py',
        'run_logged.py','capacity_client.py','summarize_scenarios.py','package_scenarios.py','agent_benchmark.py')
RUNTIME=('agent_session_client.py','agent_session_tasks.py','serve_agent_sessions.py',
         'run_agent_series.py','run_logged.py','capacity_client.py')
CROSS_STAGE_REQUIRED=('long-context/scripts/review_runtime.py',)
OFFLINE_REFERENCE_REQUIRED=(
    'configs/context-tp4-screen.json','configs/context-tp2pp2-screen.json',
    'configs/context-dspark3-screen.json','configs/context-dp4ep-screen.json',
    'workloads/agent/tasks.json','workloads/agent/replay_context.md',
    'workloads/agent/chunk_tail/initial.py')


def plain(path):
    p=Path(os.path.abspath(Path(path).expanduser()))
    core.require(not any(x.is_symlink() for x in (p,*p.parents)),f'Symlink path refused: {p}')
    return p


def terminal(phase, names):
    core.require(names and len(set(names)) == len(names), 'Explicit unique experiment IDs required')
    research = phase.parent
    scenario = research / 'scenarios'
    rows = []
    for name in sorted(names):
        core.require(re.fullmatch('c[0-9]+[A-Za-z0-9._-]*', name), 'Only Stage C IDs allowed')
        d = plain(path_at(scenario, 'results') / name)
        for n in ('command.json', 'status.json', 'source-snapshot.zip'):
            core.fingerprint(d / n)
        command = load_document(d / 'command.json')
        status = load_document(d / 'status.json')
        core.require(command.get('experiment_id') == status.get('experiment_id') == name, 'Supervisor ID mismatch')
        for key in ('command', 'cwd', 'source_sha256'):
            core.require(key in command and status.get(key) == command[key], f'Supervisor {key} mismatch')
        core.require(status.get('end_utc') and (type(status.get('exit_code')) is int or status.get('timed_out') is True or status.get('supervisor_error')), 'Nonterminal experiment')
        argv = command['command']
        core.require(len(argv) == 6 and argv[2] == '--config' and (argv[4] == '--plan'), 'Unexpected launch signature')
        script = plain(Path(argv[1]) if Path(argv[1]).is_absolute() else Path(command['cwd']) / argv[1])
        core.require(script == scenario / 'scripts/serve_agent_sessions.py' and re.fullmatch('python(?:[23](?:\\.\\d+)?)?', Path(argv[0]).name), 'Unexpected launcher')
        inputs = {}
        with zipfile.ZipFile(d / 'source-snapshot.zip') as z:
            expected = command['source_sha256']
            infos = z.infolist()
            core.require(len(infos) == len(set(z.namelist())) and set(z.namelist()) == set(expected), 'Source ZIP member set mismatch')
            required = {'scenarios/scripts/' + n for n in RUNTIME}
            required.update(('scripts/benchmark_client.py', 'environment.json', 'model-inventory.json', 'scenarios/workloads/sessions/tasks/catalog.json'))
            core.require(required <= set(expected), 'Launch snapshot lacks local runtime dependency')
            for i in infos:
                mode = i.external_attr >> 16
                core.require(core.safe_name(i.filename) and (not i.is_dir()) and (stat.S_IFMT(mode) in (0, stat.S_IFREG)), 'Unsafe snapshot member')
                with z.open(i) as f:
                    core.require(hashlib.file_digest(f, 'sha256').hexdigest() == expected[i.filename], 'Snapshot source SHA mismatch')
            for key, path in [('config', argv[3]), ('plan', argv[5])]:
                p = plain(Path(path) if Path(path).is_absolute() else Path(command['cwd']) / path)
                core.require(p.is_relative_to(scenario), 'Input outside scenario tree')
                member = str(p.relative_to(research))
                core.require(member in expected, 'Missing launch input')
                obj = json.loads(z.read(member))
                if key == 'plan':
                    core.require(obj.get('experiment_id') == name, 'Plan experiment ID differs')
                if key == 'config' and (d / 'config.json').exists():
                    core.require(load_document(d / 'config.json') == obj, 'Saved runtime config differs')
                current = core.sha(p) if p.exists() else None
                inputs[key] = dict(path=str(p), snapshot_member=member, launch_sha256=expected[member], current_workspace_sha256=current, current_matches_launch=current == expected[member])
        report = load_document(d / 'sessions.json') if (d / 'sessions.json').exists() else None
        if report is not None:
            prefix = 'scenarios/scripts/'
            core.require(report.get('client_sha256') == expected[prefix + 'agent_session_client.py'], 'Runtime client SHA differs from launch snapshot')
            helpers = report.get('helper_sha256', {})
            required_helpers = {'capacity_client', 'benchmark_client'}
            if any((p.get('kind') == 'agent' for p in report.get('plan', {}).get('phases', []))):
                required_helpers.add('agent_session_tasks')
            core.require(required_helpers <= set(helpers), 'Runtime helper identity missing')
            for helper, digest in helpers.items():
                member = 'scripts/benchmark_client.py' if helper == 'benchmark_client' else prefix + helper + '.py'
                core.require(member in expected and digest == expected[member], 'Runtime helper SHA differs from launch snapshot')
        rows.append(dict(id=name, exit_code=status.get('exit_code'), timed_out=status.get('timed_out'), supervisor_error=status.get('supervisor_error'), end_utc=status['end_utc'], git_commit=command.get('git_commit'), launch_inputs=inputs, source_snapshot_sha256=core.sha(d / 'source-snapshot.zip'), session_report_status=report.get('status') if report else None, runtime_report_identity_verified=report is not None, scope='Terminal outcome and byte/source identity; not request success, task quality or capacity acceptance.'))
    return rows


def collect(phase, experiments):
    research = phase.parent
    scenario = research / 'scenarios'
    project = research.parents[1]
    files = set()
    skips = set()

    def visit(path):
        p = plain(path)
        core.require(p.is_relative_to(project), 'Input outside project')
        mode = p.lstat().st_mode
        core.require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'Special file refused')
        archive = p.name.endswith(('.tar', '.tar.gz', '.tgz', '.tar.xz', '.7z')) or (p.suffix == '.zip' and p.name != 'source-snapshot.zip')
        if core.excluded(p, phase) or p.name in ('evidence', 'archives', 'cache', 'caches') or archive:
            skips.add(str(p.relative_to(project)))
        elif stat.S_ISDIR(mode):
            for x in sorted(p.iterdir()):
                visit(x)
        else:
            files.add(p)
    visit(phase)
    visit(path_at(scenario, 'workloads/sessions'))
    for e in experiments:
        visit(path_at(scenario, 'results') / e['id'])
        for obj in e['launch_inputs'].values():
            if Path(obj['path']).exists():
                visit(Path(obj['path']))
    for name in SHARED:
        visit(scenario / 'scripts' / name)
    for name in OFFLINE_REFERENCE_REQUIRED:
        visit(scenario / name)
    for name in CROSS_STAGE_REQUIRED:
        visit(research / name)
    for name in core.PARENT_REQUIRED:
        visit(research / name)
    for name in core.PROJECT_REQUIRED:
        visit(project / name)
    for name in ('roadmap.md', 'model-analysis.md', 'framework-capabilities.md', 'topology-analysis.md'):
        if (research / name).exists():
            visit(research / name)
    for p in [path_at(scenario, 'capacity-protocol.md'), path_at(scenario, 'capacity-report.md'), path_at(research, 'long-context/report.md'), path_at(research, 'long-context/kv-capacity-analysis.md')]:
        if p.exists():
            visit(p)
    selected = {e['id'] for e in experiments}
    for p in files:
        if p.is_relative_to(path_at(scenario, 'results')):
            core.require(p.relative_to(path_at(scenario, 'results')).parts[0] in selected, 'Unselected raw result')
    return (sorted(files), sorted(skips), project)


def package(phase, name, names):
    phase = plain(phase)
    core.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', name), 'Unique simple archive ID required')
    with core.research_lock(phase):
        experiments = terminal(phase, names)
        out = plain(path_at(phase, 'evidence'))
        out.mkdir(parents=True, exist_ok=True)
        archive, manifest_path, sums, record_path = [out / (name + ext) for ext in ('.tar.gz', '.manifest.json', '.sha256', '.json')]
        core.require(not any((p.exists() or p.is_symlink() for p in (archive, manifest_path, sums, record_path))), 'Archive ID exists')
        record = dict(status='IN PROGRESS', created_utc=core.utc(), evidence_id=name, scope='Stage C byte/source integrity; negative results retained; no performance or task-quality acceptance')
        core.new_json(record_path, record)
        try:
            files, excluded, project = collect(phase, experiments)
            rows = [dict(path=str(p.relative_to(project)), sha256=core.sha(p), bytes=p.stat().st_size) for p in files]
            core.require(rows and all((core.safe_name(r['path']) for r in rows)), 'Empty/unsafe file set')
            manifest = dict(schema_version=1, created_utc=core.utc(), path_base='workspace root', experiments=experiments, files=rows, exclusions=excluded, scope='Explicit Stage C raw results including wire/SSE/tool/session/graded workspaces and snapshots, methods, scripts, Notebook, figures, offline evidence and runtime dependency closure.', reference_closure_guaranteed=False, reference_caveat='A/B reference reports remain tied to their original immutable evidence packages; historical raw trees and packages are not duplicated.')
            selected = {e['id'] for e in experiments}
            manifest['unselected_stage_c_result_ids'] = sorted((p.name for p in path_at(phase.parent, 'scenarios/results').iterdir() if re.fullmatch('c[0-9]+[A-Za-z0-9._-]*', p.name) and p.name not in selected))
            manifest['selection_caveat'] = 'Explicit experiment selection; inspect omitted IDs before treating this package as a complete Stage C delivery.'
            manifest['external_runtime_prerequisites'] = 'Model weights/tokenizer and third-party environments are excluded; environment metadata, package versions and source hashes identify required external inputs. Local project runtime/helper/task sources are included.'
            core.new_json(manifest_path, manifest)
            with open_output(sums, 'x') as f:
                f.write(core.sum_text(rows))
            with open_output(archive, 'xb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as gz:
                    with open_output(tarfile, fileobj=gz, mode='w|') as tar:
                        for p, row in zip(files, rows):
                            plain(p)
                            core.fingerprint(p)
                            with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
                                info = tar.gettarinfo(str(p), arcname=row['path'], fileobj=f)
                                core.require(info.isfile(), 'Source changed type')
                                tar.addfile(info, f)
            core.verify_tar(archive, rows)
            endfiles, endskips, _ = collect(phase, experiments)
            core.require(endfiles == files and endskips == excluded, 'Membership changed during archive')
            for p, row in zip(files, rows):
                core.require(core.sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], 'File changed during archive')
            core.require(terminal(phase, names) == experiments, 'Runtime identity changed during archive')
            record.update(status='PASS', completed_utc=core.utc(), archive=archive.name, archive_sha256=core.sha(archive), archive_bytes=archive.stat().st_size, manifest=manifest_path.name, manifest_sha256=core.sha(manifest_path), sha256_file=sums.name, sha256_file_sha256=core.sha(sums), files=len(rows), original_bytes=sum((r['bytes'] for r in rows)), experiment_ids=[e['id'] for e in experiments])
            core.replace_record(record_path, record)
            core.verify(record_path)
            return record
        except BaseException as exc:
            record.update(status='FAILED', ended_utc=core.utc(), error=repr(exc))
            core.replace_record(record_path, record)
            raise


def selftest():
    checks = []
    with tempfile.TemporaryDirectory(prefix='phase-c-package-') as td:
        project = Path(td)
        research = project / 'research/model'
        phase = research / 'phase-c'
        scenario = research / 'scenarios'

        def put(p, v=b'fixture\n'):
            p.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(p, v)
        for name in core.PROJECT_REQUIRED:
            put(project / name)
        for name in core.PARENT_REQUIRED:
            put(research / name)
        for name in SHARED:
            put(scenario / 'scripts' / name)
        for name in OFFLINE_REFERENCE_REQUIRED:
            put(scenario / name)
        for name in CROSS_STAGE_REQUIRED:
            put(research / name)
        put(path_at(phase, 'report.md'))
        put(path_at(scenario, 'workloads/sessions/tasks/task/public/item.json'))
        put(path_at(scenario, 'workloads/sessions/tasks/catalog.json'), b'{}')
        name = 'c01-fixture-failed'
        config = path_at(scenario, 'configs') / f'{name}.json'
        plan = path_at(scenario, 'workloads/sessions') / f'{name}.json'
        put(config, core.json_bytes({'name': name}))
        put(plan, core.json_bytes({'experiment_id': name}))
        d = path_at(scenario, 'results') / name
        paths = [config, plan, *[scenario / 'scripts' / n for n in RUNTIME], research / 'scripts/benchmark_client.py', research / 'environment.json', research / 'model-inventory.json', path_at(scenario, 'workloads/sessions/tasks/catalog.json')]
        command = dict(experiment_id=name, cwd=str(project), git_commit='fixture', command=['python', str(scenario / 'scripts/serve_agent_sessions.py'), '--config', str(config), '--plan', str(plan)], source_sha256={str(p.relative_to(research)): core.sha(p) for p in paths})
        put(d / 'command.json', core.json_bytes(command))
        put(d / 'status.json', core.json_bytes(dict(command, exit_code=1, end_utc=core.utc(), timed_out=False)))
        with zipfile.ZipFile(d / 'source-snapshot.zip', 'x') as z:
            for p in paths:
                z.write(p, str(p.relative_to(research)))
        for n in ('sessions-raw/sessions.jsonl', 'sessions-raw/events.jsonl', 'sessions-raw/session-0000/task-state.json', 'sessions-raw/before.prom', 'server.log', 'gpu.csv'):
            put(d / n)
        put(phase / 'old.tar.gz')
        put(phase / 'weights/model.safetensors')
        put(path_at(scenario, 'results/a01-old/old.json'))
        result = package(phase, 'fixture', [name])
        manifest = load_document(path_at(phase, 'evidence') / result['manifest'])
        members = [x['path'] for x in manifest['files']]
        assert core.verify(path_at(phase, 'evidence/fixture.json'))['status'] == 'PASS'
        assert all((any((p.endswith(n) for p in members)) for n in ('task-state.json', 'events.jsonl', 'source-snapshot.zip', 'summarize_scenarios.py', 'package_scenarios.py')))
        assert not any(('a01-old' in p or '/weights/' in p or p.endswith('old.tar.gz') for p in members))
        checks += ['Full archive/member SHA roundtrip', 'Failed run and complete raw task/stream evidence retained', 'Dependency closure and historical/raw/archive exclusions']

        def reject(label, fn):
            try:
                fn()
            except (ValueError, RuntimeError, OSError):
                checks.append(label)
            else:
                raise AssertionError(label)
        reject('Existing archive preserved', lambda: package(phase, 'fixture', [name]))
        reject('Explicit IDs required', lambda: package(phase, 'empty', []))
        reject('Historical IDs forbidden', lambda: package(phase, 'old', ['a01-old']))
        with core.research_lock(phase):
            reject('Shared live GPU lock respected', lambda: package(phase, 'locked', [name]))
        (phase / 'link').symlink_to(path_at(project, 'AGENTS.md'))
        reject('Symlink inputs refused', lambda: package(phase, 'link', [name]))
        (phase / 'link').unlink()
        put(d / 'status.json', core.json_bytes(dict(experiment_id=name, exit_code=None)))
        reject('Running experiment refused', lambda: package(phase, 'running', [name]))
    return dict(status='PASS', checks=checks, scope='CPU synthetic archive fixture only')


def main():
    p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--experiment',action='append');p.add_argument('--verify',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:r=selftest()
    elif a.verify:r=core.verify(a.verify)
    else:
        core.require(a.name,'Provide archive name');r=package(ROOT,a.name,a.experiment)
    if a.output:core.new_json(a.output,r)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
