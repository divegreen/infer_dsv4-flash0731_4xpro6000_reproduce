#!/usr/bin/env python3
"""Independent CPU fixtures for Stage C archive identity and preservation.

Only temporary trees are packaged. No inference, GPU inspection or model calls.
The real c01/c02 records may be read to check compatibility, never archived here.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reviewed_stage_c_package',HERE/'package_sessions.py')
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)


def selftest():
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    def reject(label, fn, fragment):
        try:
            fn()
        except (ValueError, RuntimeError, OSError) as exc:
            check(label, fragment in str(exc))
        else:
            raise AssertionError(label)
    with tempfile.TemporaryDirectory(prefix='independent-phase-c-package-') as td:
        project = Path(td)
        research = project / 'research/model'
        phase = research / 'phase-c'
        scenario = research / 'scenarios'

        def put(p, value):
            p.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(p, value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode())
        for name in pack.core.PROJECT_REQUIRED:
            put(project / name, b'MOCK project prerequisite\n')
        for name in pack.core.PARENT_REQUIRED:
            put(research / name, b'MOCK research prerequisite\n')
        for name in pack.SHARED:
            put(scenario / 'scripts' / name, b'# MOCK runtime version one\n')
        for name in pack.OFFLINE_REFERENCE_REQUIRED:
            put(scenario / name, b'MOCK offline reference\n')
        for name in pack.CROSS_STAGE_REQUIRED:
            put(research / name, b'# MOCK frozen cross-stage parser\n')
        put(path_at(phase, 'report.md'), b'MOCK report\n')
        task = path_at(scenario, 'workloads/sessions/tasks/task/public/item.json')
        put(task, {'revision': 'old'})
        catalog = path_at(scenario, 'workloads/sessions/tasks/catalog.json')
        put(catalog, {'tasks': []})

        def run(name, code, report=True):
            d = path_at(scenario, 'results') / name
            config = path_at(scenario, 'configs') / f'{name}.json'
            plan = path_at(scenario, 'workloads/sessions') / f'{name}.json'
            put(config, {'name': name, 'revision': name})
            put(plan, {'experiment_id': name, 'phases': [{'kind': 'agent'}]})
            sources = [config, plan, catalog, task, *[scenario / 'scripts' / n for n in pack.RUNTIME], research / 'environment.json', research / 'model-inventory.json', research / 'scripts/benchmark_client.py']
            command = {'experiment_id': name, 'cwd': str(project), 'git_commit': 'MOCK', 'command': ['python', str(scenario / 'scripts/serve_agent_sessions.py'), '--config', str(config), '--plan', str(plan)], 'source_sha256': {str(p.relative_to(research)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
            put(d / 'command.json', command)
            put(d / 'status.json', dict(command, exit_code=code, end_utc='MOCK-terminal', timed_out=False))
            put(d / 'config.json', load_document(config))
            with zipfile.ZipFile(d / 'source-snapshot.zip', 'x') as z:
                for p in sources:
                    z.writestr(str(p.relative_to(research)), p.read_bytes())
            if report:
                prefixes = {'capacity_client': 'scenarios/scripts/capacity_client.py', 'benchmark_client': 'scripts/benchmark_client.py', 'agent_session_tasks': 'scenarios/scripts/agent_session_tasks.py'}
                put(d / 'sessions.json', {'status': 'FAILED' if code else 'COMPLETED', 'plan': load_document(plan), 'client_sha256': command['source_sha256']['scenarios/scripts/agent_session_client.py'], 'helper_sha256': {k: command['source_sha256'][v] for k, v in prefixes.items()}})
            for n in ('sessions-raw/events.jsonl', 'sessions-raw/requests.jsonl', 'sessions-raw/session-0000/task-state.json', 'sessions-raw/session-0000/public/edited.py', 'sessions-raw/before.prom', 'server.log', 'gpu.csv'):
                put(d / n, ('MOCK raw ' + n + '\n').encode())
            return (d, command, config, plan)
        old, oldcommand, oldconfig, oldplan = run('c01-old-failed', 1)
        put(scenario / 'scripts/agent_session_client.py', b'# MOCK runtime version two\n')
        put(task, {'revision': 'new'})
        new, newcommand, _, _ = run('c02-new-complete', 0)
        partial, _, _, _ = run('c03-pre-client-failed', 1, False)
        put(path_at(scenario, 'results/c04-omitted-failed/status.json'), {'exit_code': 1})
        put(oldconfig, {'revision': 'workspace drift'})
        oldplan.unlink()
        for p in [phase / 'old.tar.gz', phase / 'old.zip', phase / 'weights/model.safetensors', phase / '.venv/lib/site.py', phase / 'cache/item', path_at(scenario, 'results/a01-old/measurement.json'), path_at(research, 'long-context/evidence/old.tar.gz')]:
            put(p, b'EXCLUDED\n')
        put(path_at(research, 'long-context/report.md'), b'Immutable B reference\n')
        ids = [old.name, new.name, partial.name]
        result = pack.package(phase, 'independent-review', ids)
        record = path_at(phase, 'evidence/independent-review.json')
        manifest = load_document(record.parent / result['manifest'])
        check('Independent archive verification', pack.core.verify(record)['status'] == 'PASS')
        check('Failed and pre-client failure IDs retained', result['experiment_ids'] == ids and [e['exit_code'] for e in manifest['experiments']] == [1, 0, 1])
        check('Unselected Stage C failure explicitly disclosed', manifest['unselected_stage_c_result_ids'] == ['c04-omitted-failed'])
        oldrow = manifest['experiments'][0]
        check('Changed config and deleted current plan retain launch snapshots', not oldrow['launch_inputs']['config']['current_matches_launch'] and oldrow['launch_inputs']['plan']['current_workspace_sha256'] is None)
        check('Pre-client failure does not invent runtime identity verification', manifest['experiments'][2]['runtime_report_identity_verified'] is False)
        with tarfile.open(record.parent / result['archive'], 'r:gz') as archive:
            members = {m.name: archive.extractfile(m).read() for m in archive}
        prefix = 'research/model/scenarios/'
        check('Raw streams, task checkpoints, edits and GPU logs included', all((prefix + f'results/{old.name}/{n}' in members for n in ('sessions-raw/events.jsonl', 'sessions-raw/requests.jsonl', 'sessions-raw/session-0000/task-state.json', 'sessions-raw/session-0000/public/edited.py', 'gpu.csv'))))
        oldzip = zipfile.ZipFile(io.BytesIO(members[prefix + f'results/{old.name}/source-snapshot.zip']))
        newzip = zipfile.ZipFile(io.BytesIO(members[prefix + f'results/{new.name}/source-snapshot.zip']))
        client = 'scenarios/scripts/agent_session_client.py'
        fixture = 'scenarios/workloads/sessions/tasks/task/public/item.json'
        check('Old/new client and task snapshot bytes remain distinct', oldzip.read(client) != newzip.read(client) and oldzip.read(fixture) != newzip.read(fixture))
        check('Current runtime and fixture also retained', members[prefix + 'scripts/agent_session_client.py'] == newzip.read(client) and members[prefix + 'workloads/sessions/tasks/task/public/item.json'] == newzip.read(fixture))
        check('Models environments caches and nested archives excluded', not any(('/weights/' in p or '/.venv/' in p or '/cache/' in p or p.endswith(('old.tar.gz', 'old.zip')) for p in members)))
        check('A/B raw trees omitted but B report retained', not any(('/results/a01-old/' in p for p in members)) and 'research/model/long-context/report.md' in members)
        check('Local helper closure retained', all((prefix + 'scripts/' + n in members for n in pack.SHARED)) and 'research/model/scripts/benchmark_client.py' in members)
        check('Small frozen cross-stage parser retained without historical evidence tree', all(('research/model/' + n in members for n in pack.CROSS_STAGE_REQUIRED)))
        check('Legacy replay source and exact input dependencies retained', all((prefix + n in members for n in ('scripts/agent_benchmark.py', 'workloads/agent/tasks.json', 'workloads/agent/replay_context.md', 'workloads/agent/chunk_tail/initial.py'))))
        check('Matched plan generator frozen reference configurations retained', all((prefix + 'configs/' + n in members for n in ('context-tp4-screen.json', 'context-tp2pp2-screen.json', 'context-dspark3-screen.json', 'context-dp4ep-screen.json'))))
        legacy = path_at(scenario, 'workloads/agent/replay_context.md')
        legacy_raw = legacy.read_bytes()
        legacy.unlink()
        reject('Missing legacy input dependency refuses incomplete collection', lambda: pack.collect(phase, pack.terminal(phase, ids)), 'replay_context.md')
        put(legacy, legacy_raw)
        check('Model tokenizer and environment requirements explicit', bool(manifest['external_runtime_prerequisites']) and manifest['reference_closure_guaranteed'] is False)
        archive_bytes = (record.parent / result['archive']).read_bytes()
        reject('Duplicate evidence ID refuses overwrite', lambda: pack.package(phase, 'independent-review', ids), 'Archive ID exists')
        check('Rejected overwrite preserves archive bytes', (record.parent / result['archive']).read_bytes() == archive_bytes)
        reject('Duplicate experiment IDs refused', lambda: pack.terminal(phase, [old.name, old.name]), 'Explicit unique')
        with pack.core.research_lock(phase):
            reject('Shared research lock excludes live runner', lambda: pack.package(phase, 'locked', ids), 'Research runner active')
        status = load_document(new / 'status.json')
        bad = copy.deepcopy(status)
        bad['source_sha256'][client] = '0' * 64
        put(new / 'status.json', bad)
        reject('Supervisor snapshot mismatch refused', lambda: pack.terminal(phase, [new.name]), 'Supervisor source_sha256 mismatch')
        put(new / 'status.json', status)
        bad = copy.deepcopy(status)
        bad['command'][0] = 'other-python'
        put(new / 'status.json', bad)
        reject('Supervisor command mismatch refused', lambda: pack.terminal(phase, [new.name]), 'Supervisor command mismatch')
        put(new / 'status.json', status)
        report = load_document(new / 'sessions.json')
        bad = copy.deepcopy(report)
        bad['client_sha256'] = '0' * 64
        put(new / 'sessions.json', bad)
        reject('Actual client mismatch refused', lambda: pack.terminal(phase, [new.name]), 'Runtime client SHA')
        put(new / 'sessions.json', report)
        bad = copy.deepcopy(report)
        bad['helper_sha256']['agent_session_tasks'] = '0' * 64
        put(new / 'sessions.json', bad)
        reject('Actual task helper mismatch refused', lambda: pack.terminal(phase, [new.name]), 'Runtime helper SHA')
        put(new / 'sessions.json', report)
        original = (new / 'source-snapshot.zip').read_bytes()
        with zipfile.ZipFile(new / 'source-snapshot.zip', 'w') as z:
            for p in newzip.namelist():
                z.writestr(p, b'altered' if p == client else newzip.read(p))
        reject('Changed snapshot member refused', lambda: pack.terminal(phase, [new.name]), 'Snapshot source SHA')
        put(new / 'source-snapshot.zip', original)
        (phase / 'link').symlink_to(path_at(project, 'AGENTS.md'))
        reject('Symlink source refused', lambda: pack.package(phase, 'symlink', ids), 'Symlink path refused')
        (phase / 'link').unlink()
        check('Failed package record retained', load_document(path_at(phase, 'evidence/symlink.json'))['status'] == 'FAILED')
        verify = pack.core.verify_tar

        def mutate(archive, rows):
            result = verify(archive, rows)
            put(path_at(phase, 'report.md'), b'Changed during archive\n')
            return result
        pack.core.verify_tar = mutate
        try:
            reject('Source mutation during packaging detected', lambda: pack.package(phase, 'mutating', ids), 'File changed during archive')
        finally:
            pack.core.verify_tar = verify
        check('Failed mutable archive retained with FAILED status', load_document(path_at(phase, 'evidence/mutating.json'))['status'] == 'FAILED' and path_at(phase, 'evidence/mutating.tar.gz').is_file())
    return {'status': 'PASS', 'checks': checks, 'scope': 'Independent temporary CPU archive fixtures only; no actual research package created'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    result=selftest();result['implementation_selftest']=pack.selftest()
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),HERE/'package_sessions.py']}
    result['created_utc']=pack.core.utc()
    pack.core.new_json(args.output,result);print(json.dumps({'status':result['status'],'independent_checks':len(result['checks']),
        'implementation_checks':len(result['implementation_selftest']['checks']),'output':str(args.output)},indent=2))


if __name__=='__main__':main()
