#!/usr/bin/env python3
"""CPU-only synthetic D metadata/raw auditor success and rejection controls."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('checked_d_auditor',HERE/'audit_study.py')
mod=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mod)
base=mod.base


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode())


def fixture(root):
    path = base.full_fixture(root)
    directory = path.parent
    report = load_document(path)
    command = load_document(directory / 'command.json')
    config = load_document(directory / 'config.json')
    config['env'] = {'NCCL_DEBUG': 'WARN'}
    config['args'] += ['--scheduler-cls', 'research_scheduler.ObserveResearchScheduler']
    report.update(profile_harness_sha256=base.digest(b'# fake serving source'), profiling=dict(excluded_from_performance=False))
    report['runs'][0]['excluded_from_performance'] = False
    for row in report['runs'][0]['requests']:
        row['wire_request_id'] = 'd-' + row['request_id'].replace('/', '-')
        row['server_response_ids'] = ['cmpl-' + row['wire_request_id']]
    write(Path(report['runs'][0]['raw_requests_jsonl']), ('\n'.join((json.dumps(r) for r in report['runs'][0]['requests'])) + '\n').encode())
    with zipfile.ZipFile(directory / 'source-snapshot.zip') as archive:
        snapshot = {n: archive.read(n) for n in archive.namelist()}
    snapshot[root.name + '/scripts/study_client.py'] = snapshot.pop(root.name + '/scripts/capacity_client.py')
    snapshot[root.name + '/scripts/serve_study.py'] = b'# fake serving source'
    snapshot[root.name + '/scripts/run_logged.py'] = b'# fake supervisor source'
    snapshot[root.name + '/scripts/research_scheduler.py'] = b'# fake custom scheduler source'
    snapshot[root.name + '/configs/fixture.json'] = json.dumps(config).encode()
    async_key = 'installed/vllm/v1/core/sched/async_scheduler.py'
    snapshot[async_key] = b'# fake installed AsyncScheduler source'
    digest = base64.urlsafe_b64encode(hashlib.sha256(snapshot[async_key]).digest()).decode().rstrip('=')
    record_key = 'installed/vllm-0.0.dist-info/RECORD'
    snapshot[record_key] = f'vllm/v1/core/sched/async_scheduler.py,sha256={digest},{len(snapshot[async_key])}\n'.encode()
    files = {k: dict(path=str(root.parent / k), sha256=base.digest(v), bytes=len(v)) for k, v in snapshot.items()}
    identity = dict(files=files, package=dict(record_key=record_key))
    write(directory / 'runtime-identity.json', identity)
    with zipfile.ZipFile(directory / 'source-snapshot.zip', 'w') as archive:
        for key, data in snapshot.items():
            archive.writestr(key, data)
    command.update(command=['SYNTHETIC_NOT_EXECUTED', str(root / 'scripts/serve_study.py'), '--config', str(path_at(root, 'configs/fixture.json')), '--plan', report['plan_path']], cwd=str(root), source_sha256={k: v['sha256'] for k, v in files.items()}, runtime_identity_sha256=base.digest((directory / 'runtime-identity.json').read_bytes()), source_snapshot_sha256=base.digest((directory / 'source-snapshot.zip').read_bytes()))
    write(directory / 'command.json', command)
    write(directory / 'status.json', dict(command, exit_code=0, supervisor_exit_code=0, source_identity_unchanged=True, changed_source_members=[], cleanup_error=None, timed_out=False, supervisor_error=None, interrupted_signal=None, end_utc='MOCK terminal'))
    write(directory / 'experiment.json', dict(status='PASS', clients=[dict(exit_code=0)], profile=False, excluded_from_performance=False))
    write(directory / 'config.json', config)
    write(directory / 'server-command.json', dict(command=['SYNTHETIC_NOT_EXECUTED', 'serve', report['plan']['tokenizer_path'], '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', '8260', *config['args']], extra_environment=config['env'], owned_environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONPATH=str(root / 'scripts'), RESEARCH_D_TRACE_DIR='')))
    scheduler = dict(class_name='research_scheduler.ObserveResearchScheduler', async_scheduler=True, step_trace_enabled=False)
    for field, key in [('module', root.name + '/scripts/research_scheduler.py'), ('parent', async_key)]:
        scheduler[field + '_sha256'] = files[key]['sha256']
        scheduler[field + '_file'] = files[key]['path']
    write(directory / 'scheduler-identity-123.json', scheduler)
    write(path, report)
    return path


def run(root,path):
    audit=mod.Auditor(root)
    try:audit.report(path)
    except Exception as exc:audit.check(False,str(path),'Unexpected fixture exception: '+repr(exc))
    return audit.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    try:
        with tempfile.TemporaryDirectory(prefix='D-audit-MOCK-') as temp:
            root = Path(temp) / 'research/scheduler'
            path = fixture(root)
            directory = path.parent
            result = run(root, path)
            assert result['status'] == 'PASS', result['errors']
            checks.append('complete synthetic launch/raw/Prom/formula/source identity roundtrip')

            def reject(filename, mutate, label):
                target = directory / filename
                original = target.read_bytes()
                obj = json.loads(original)
                mutate(obj)
                write(target, obj)
                rejected = run(root, path)
                write(target, original)
                assert rejected['status'] == 'FAIL', rejected
                checks.append(label)
            reject('status.json', lambda d: d.update(supervisor_exit_code=1, source_identity_unchanged=False), 'child exit0 cannot mask failed supervisor/source integrity')
            reject('runtime-identity.json', lambda d: d['files'][next(iter(d['files']))].update(bytes=999), 'changed runtime identity hash rejected')
            reject('scheduler-identity-123.json', lambda d: d.update(module_sha256='0' * 64), 'actual custom scheduler SHA mismatch rejected')
            reject('server-command.json', lambda d: d['owned_environment'].update(PYTHONPATH='/tmp/untracked-modules'), 'wrong server import path rejected')
            reject('capacity.json', lambda d: d['runs'][0]['requests'][0].update(server_response_ids=['cmpl-other']), 'wrong actual SSE request identity rejected')
            reject('capacity.json', lambda d: d['runs'][0]['summary'].update(output_tokens_per_s=123456), 'inherited metric summary tamper rejected')
            report = load_document(path)
            command = load_document(directory / 'command.json')
            with zipfile.ZipFile(directory / 'source-snapshot.zip') as archive:
                snapshot = {n: archive.read(n) for n in archive.namelist()}
            key = root.name + '/workloads/capacity/fixture.json'
            raw_plan = json.loads(snapshot[key])
            next_phase = copy.deepcopy(raw_plan['phases'][0])
            next_phase['id'] = 'planned-unstarted'
            raw_plan['phases'].append(next_phase)
            snapshot[key] = json.dumps(raw_plan).encode()
            report['plan'] = base.normalize_plan(raw_plan)
            report['plan_sha256'] = base.digest(snapshot[key])
            report['status'] = 'FAILED'
            workload_path = Path(report['workload_file'])
            workload = load_document(workload_path)
            next_prepared = copy.deepcopy(workload[0])
            next_prepared['spec'] = report['plan']['phases'][1]
            workload.append(next_prepared)
            write(workload_path, workload)
            report['workload_sha256'] = base.digest(workload_path.read_bytes())
            report['prepared_requests']['planned-unstarted'] = 2
            identity = load_document(directory / 'runtime-identity.json')
            identity['files'][key].update(sha256=base.digest(snapshot[key]), bytes=len(snapshot[key]))
            write(directory / 'runtime-identity.json', identity)
            with zipfile.ZipFile(directory / 'source-snapshot.zip', 'w') as archive:
                for name, raw in snapshot.items():
                    archive.writestr(name, raw)
            command['source_sha256'][key] = base.digest(snapshot[key])
            command['runtime_identity_sha256'] = base.digest((directory / 'runtime-identity.json').read_bytes())
            command['source_snapshot_sha256'] = base.digest((directory / 'source-snapshot.zip').read_bytes())
            write(directory / 'command.json', command)
            status = load_document(directory / 'status.json')
            status.update(command, exit_code=1, supervisor_exit_code=1)
            write(directory / 'status.json', status)
            write(directory / 'capacity-plan.json', report['plan'])
            write(path, report)
            write(directory / 'experiment.json', dict(status='FAILED', clients=[dict(exit_code=1)], profile=False, excluded_from_performance=False))
            partial = run(root, path)
            assert partial['status'] == 'PASS', partial['errors']
            assert partial['reports'][0]['status'] == 'FAILED' and partial['runtime_reports'][0]['supervisor_success'] is False
            assert len(partial['reports'][0]['phases']) == 1
            checks.append('failed terminal executed prefix retains independent identity without inventing unstarted phase or capacity PASS')
        result = dict(status='PASS', checks=checks, scope='Synthetic CPU fixtures only; no model/GPU/HTTP or historical edits.')
    except BaseException as exc:
        import traceback
        result = dict(status='FAILED', checks=checks, error=repr(exc), traceback=traceback.format_exc())
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE / 'audit_study.py', mod.SOURCE)}
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), output=str(args.output))))
    return 0 if result['status'] == 'PASS' else 1


if __name__=='__main__':raise SystemExit(main())
