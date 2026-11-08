#!/usr/bin/env python3
"""CPU-only relocation/snapshot controls with explicitly synthetic metadata.

This exercises source capture and independent audit without loading a model,
querying a GPU, or executing any captured source from an experiment archive.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode())


def child_checks():
    """Executed in an isolated renamed project, with a different cwd."""
    required = {'project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py',
                'metadata/environment.json', 'metadata/model-inventory.json'}
    checks = []
    for topic, location in [('architecture', RESEARCH), ('scenarios', RESEARCH/'scenarios'),
                            ('scheduler', RESEARCH/'scheduler')]:
        module = load('portable_'+topic, location/'scripts/run_logged.py')
        sources = module.source_inputs(location)
        assert required <= sources.keys(), (topic, sources.keys())
        if topic == 'scheduler':assert 'metadata/scheduler-pair-blocks.json' in sources
        assert f'experiments/{topic}/configs/fixture.json' in sources
        assert all(p.is_file() and p.is_relative_to(RESEARCH.parents[1]) for p in sources.values())
        assert not any(key.startswith(topic+'/configs/') for key in sources)
        checks.append(topic+' binds physical inputs, current setup, layout, catalog and CLI')
    fixture = load('portable_fixture', HERE/'check_run_logged.py')
    runner = fixture.RUNNER
    out = RESEARCH.parents[1]/'results/runs/mock-capture'
    out.mkdir(parents=True)
    installed, package = fixture.fake_installed(RESEARCH.parents[1]/'mock-site')
    identity = runner.capture_sources(out, RESEARCH/'scheduler', lambda: (installed, package))
    with zipfile.ZipFile(out/'source-snapshot.zip') as archive:
        assert set(archive.namelist()) == set(identity['files'])
        for key, record in identity['files'].items():
            assert hashlib.sha256(archive.read(key)).hexdigest() == record['sha256']
        raw = json.loads(archive.read('experiments/scheduler/workloads/fixture.json'))
        assert raw['tokenizer_path'] == '${MODEL_PATH}' and raw['seed'] == 20261106
    assert runner.check_sources(identity) == []
    with (RESEARCH/'project_layout.py').open('a') as stream:
        stream.write('\n# SYNTHETIC source drift\n')
    assert runner.check_sources(identity) == ['project_layout.py']
    checks += ['captured bytes retain placeholders and original seed',
               'layout source drift is detected after capture']
    print(json.dumps({'status': 'PASS', 'checks': checks}))


def relocated_capture():
    with tempfile.TemporaryDirectory(prefix='portable-snapshot-') as tmp:
        project = Path(tmp)/'renamed project with spaces'
        research = project/'research/model'
        for name in ('project_layout.py', 'scripts/run_logged.py', 'scenarios/scripts/run_logged.py',
                     'scheduler/scripts/run_logged.py', 'scheduler/scripts/check_run_logged.py',
                     'scheduler/scripts/check_portable_snapshots.py'):
            put(research/name, (RESEARCH/name).read_bytes())
        for tree in ('scripts', 'scenarios/scripts', 'long-context/scripts', 'sessions/scripts'):
            put(research/tree/'dependency.py', b'# SYNTHETIC dependency\n')
        for name in ('serve_study.py', 'study_client.py', 'research_scheduler.py'):
            put(research/'scheduler/scripts'/name, b'# SYNTHETIC runtime source\n')
        put(research/'scripts/benchmark_client.py', b'# SYNTHETIC benchmark helper\n')
        for topic in ('architecture', 'scenarios', 'scheduler'):
            put(research/f'experiments/{topic}/configs/fixture.json', {'name': 'fixture'})
            put(research/f'experiments/{topic}/workloads/fixture.json',
                {'experiment_id': 'fixture', 'tokenizer_path': '${MODEL_PATH}', 'seed': 20261106})
        fixtures = research/'experiments/scenarios/workloads/agent'
        for name, value in [('tasks.json', {'tasks': [{'id': 'mock_task'}]}),
                            ('quality.json', {'cases': []}), ('replay_context.md', b'MOCK context\n'),
                            ('mock_task/initial.py', b'# MOCK initial fixture\n')]:
            put(fixtures/name, value)
        put(research/'experiments/catalog.json', {'synthetic_fixture': True})
        put(project/'tools/reproduce.py', b'# SYNTHETIC CLI, never executed\n')
        for name in ('environment.json', 'model-inventory.json'):
            put(project/'results/setup'/name, {'synthetic_fixture': True})
        put(project/'results/setup/scheduler-pair-blocks.json', {'synthetic_fixture': True, 'groups': [], 'input_sha256': {}})
        result = subprocess.run([sys.executable, str(research/'scheduler/scripts/check_portable_snapshots.py'),
                                 '--child'], cwd='/', text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return json.loads(result.stdout)['checks']


def centralized_audit():
    fixture = load('portable_audit_fixture', HERE/'check_audit_study.py')
    with tempfile.TemporaryDirectory(prefix='centralized-audit-') as tmp:
        root = Path(tmp)/'project with spaces/research/model/scheduler'
        path = fixture.fixture(root)
        directory = path.parent
        read = lambda name: json.loads((directory/name).read_text())
        launch, report, config = read('command.json'), read('capacity.json'), read('config.json')
        with zipfile.ZipFile(directory/'source-snapshot.zip') as archive:
            snapshot = {name: archive.read(name) for name in archive.namelist()}
        context = {'project_root': str(root.parents[2]), 'research_root': str(root.parent),
                   'model_path': '/MOCK relocated model', 'run_dir': str(directory)}
        config_key = 'experiments/scheduler/configs/fixture.json'
        plan_key = 'experiments/scheduler/workloads/capacity/fixture.json'
        raw_plan = json.loads(snapshot.pop('scheduler/workloads/capacity/fixture.json'))
        raw_plan['tokenizer_path'] = '${MODEL_PATH}'
        snapshot[plan_key] = json.dumps(raw_plan).encode()
        raw_config = json.loads(snapshot.pop('scheduler/configs/fixture.json'))
        raw_config['env']['MOCK_OUTPUT_PATH'] = '${RUN_DIR}/profiles'
        snapshot[config_key] = json.dumps(raw_config).encode()
        config['env']['MOCK_OUTPUT_PATH'] = context['run_dir']+'/profiles'
        for key in ('project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py',
                    'metadata/environment.json', 'metadata/model-inventory.json', 'metadata/scheduler-pair-blocks.json'):
            snapshot[key] = b'{"synthetic_fixture":true}\n'
        snapshot['metadata/scheduler-pair-blocks.json'] = json.dumps({
            'groups': [{'block':'MOCK portable pair','experiments':['fixture','MOCK-fair']}],
            'input_sha256': {key:fixture.base.digest(snapshot[key]) for key in (config_key,plan_key)}
        }).encode()
        launch['path_context'] = context
        launch['command'][launch['command'].index('--config')+1] = str(root.parent/config_key)
        launch['command'][launch['command'].index('--plan')+1] = str(root.parent/plan_key)
        report['plan_path'] = str(root.parent/plan_key)
        report['plan']['tokenizer_path'] = context['model_path']
        report['plan_sha256'] = fixture.base.digest(snapshot[plan_key])
        identity = read('runtime-identity.json')
        old_files = identity['files']
        identity['files'] = {key: dict(path=old_files.get(key, {}).get('path', str(root.parent/key)),
                                     sha256=fixture.base.digest(raw), bytes=len(raw))
                             for key, raw in snapshot.items()}
        put(directory/'runtime-identity.json', identity)
        with zipfile.ZipFile(directory/'source-snapshot.zip', 'w') as archive:
            for key, raw in snapshot.items():
                archive.writestr(key, raw)
        launch.update(source_sha256={key: value['sha256'] for key, value in identity['files'].items()},
                      runtime_identity_sha256=fixture.base.digest((directory/'runtime-identity.json').read_bytes()),
                      source_snapshot_sha256=fixture.base.digest((directory/'source-snapshot.zip').read_bytes()))
        status = read('status.json'); status.update(launch)
        server = read('server-command.json'); server['command'][2] = context['model_path']
        server['extra_environment'] = config['env']
        for name, value in [('command.json', launch), ('status.json', status), ('capacity.json', report),
                            ('capacity-plan.json', report['plan']), ('config.json', config), ('server-command.json', server)]:
            put(directory/name, value)
        result = fixture.run(root, path)
        assert result['status'] == 'PASS', result['errors']
        performance = load('portable_registered_performance', HERE/'analyze_performance.py')
        registration = performance.registered_block(directory,report,identity,{},root)
        assert registration['block'] == 'MOCK portable pair'
        assert registration['decision_path'].endswith('!metadata/scheduler-pair-blocks.json')
        bad_identity = json.loads(json.dumps(identity))
        bad_identity['files']['metadata/scheduler-pair-blocks.json']['sha256'] = '0'*64
        try:performance.registered_block(directory,report,bad_identity,{},root)
        except ValueError:pass
        else:raise AssertionError('Changed captured pair registration accepted')
        launch['path_context']['model_path'] = '/MOCK inconsistent model'
        put(directory/'command.json', launch)
        assert fixture.run(root, path)['status'] == 'FAIL'
    return ['centralized physical snapshot plus recorded model/output placeholders audit successfully',
            'normal pair registration is read from the launch snapshot without live decisions',
            'tampered pair registration SHA is rejected',
            'tampered captured path context is rejected']


def centralized_profile():
    fixture = load('portable_profile_fixture', HERE/'check_profiles.py')
    analyzer = fixture.a
    with tempfile.TemporaryDirectory(prefix='centralized-profile-') as tmp:
        directory = Path(tmp)/'profile with spaces'
        directory.mkdir()
        records, identity = fixture.source_fixture(directory)
        with zipfile.ZipFile(directory/'source-snapshot.zip') as archive:
            snapshot = {name: archive.read(name) for name in archive.namelist()}
        context = {'project_root': '/MOCK/project', 'model_path': '/MOCK/tokenizer',
                   'run_dir': str(directory), 'research_root': '/MOCK/project/research/model'}
        for category in ('configs','workloads'):
            old = 'scheduler/'+category+'/'+directory.name+'.json'
            new = 'experiments/'+old
            value = json.loads(snapshot.pop(old))
            if category == 'workloads':value['tokenizer_path'] = '${MODEL_PATH}'
            snapshot[new] = json.dumps(value).encode()
        for name in ('project_layout.py','experiments/catalog.json','project/tools/reproduce.py',
                     'metadata/environment.json','metadata/model-inventory.json'):
            snapshot[name] = b'{"synthetic_fixture":true}\n'
        identity['files'] = {key: dict(path='/MOCK/'+key,sha256=analyzer.sha(raw),bytes=len(raw))
                             for key,raw in snapshot.items()}
        put(directory/'runtime-identity.json',identity)
        with zipfile.ZipFile(directory/'source-snapshot.zip','w') as archive:
            for key,raw in snapshot.items():archive.writestr(key,raw)
        command = records['command.json']
        command.update(source_sha256={key:item['sha256'] for key,item in identity['files'].items()},
                       runtime_identity_sha256=analyzer.sha((directory/'runtime-identity.json').read_bytes()),
                       source_snapshot_sha256=analyzer.sha((directory/'source-snapshot.zip').read_bytes()),
                       path_context=context)
        records['status.json'].update(source_sha256=command['source_sha256'],path_context=context)
        assert analyzer.snapshot_identity(directory,records,{}) == identity
        records['capacity-plan.json']['seed'] += 1
        try:analyzer.snapshot_identity(directory,records,{})
        except ValueError:pass
        else:raise AssertionError('Changed workload seed accepted')
    return ['centralized profile snapshot expands recorded tokenizer only',
            'profile audit rejects changed workload seed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.child:
        child_checks(); return
    if not args.output:
        parser.error('--output is required')
    result = {'status': 'PASS', 'interpreter': sys.executable,
              'scope': 'Synthetic CPU source capture and audit only; no GPU, model, or installed vLLM execution'}
    try:
        result['checks'] = relocated_capture() + centralized_audit() + centralized_profile()
    except Exception as exc:
        import traceback
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
    put(args.output, result)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
