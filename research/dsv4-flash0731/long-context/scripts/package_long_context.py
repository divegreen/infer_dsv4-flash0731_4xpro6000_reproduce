#!/usr/bin/env python3
"""Immutable Stage B archive; explicit b* IDs, shared GPU lock, no inference.

Reuse Stage A's fingerprint/hash/tar verification primitives, without modifying
historical code or archives. Raw results live in the sibling scenario directory.
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
import math
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('phase_b_package_core', ROOT.parent/'scenarios/scripts/package_scenarios.py')
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)
EXPERIMENT = re.compile(r'b[0-9][A-Za-z0-9._-]*')
SHARED_SCRIPTS = ('capacity_client.py', 'capacity_audit.py', 'check_capacity_client.py',
                  'check_capacity_summary.py', 'run_logged.py', 'serve_capacity.py',
                  'run_capacity_series.py', 'summarize_capacity.py',
                  'summarize_scenarios.py', 'analyze_profiles.py', 'package_scenarios.py')
LAUNCHERS = {'serve_capacity.py', 'serve_long_context_profile.py'}
REFERENCE_DOCS = ('capacity-protocol.md', 'capacity-methods.md', 'capacity-report.md',
                  'capacity-bottlenecks-final.md', 'CAPACITY-REPRODUCE.md')


def plain_path(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    core.require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink path refused: '+str(path))
    return path


def owned_launcher(command, scenario):
    argv = command.get('command')
    core.require(isinstance(argv, list) and len(argv) == 6 and all(isinstance(x, str) for x in argv),
                 'Only an exact Python launcher with --config and --plan is allowed')
    core.require(re.fullmatch(r'python(?:[23](?:\.\d+)?)?', Path(argv[0]).name), 'Non-Python launcher refused')
    script = Path(argv[1])
    if not script.is_absolute():
        script = Path(command['cwd'])/script
    script = plain_path(script)
    core.require(script.name in LAUNCHERS and script == scenario/'scripts'/script.name,
                 'Launcher outside the two owned capacity/profile scripts')
    core.require(set(argv[2::2]) == {'--config', '--plan'}, 'Unexpected launcher flags')
    return script.name


def profile_options(directory):
    return dict(profiler='torch', torch_profiler_dir=str(directory/'profiles'),
                torch_profiler_activities=['CPU', 'CUDA'], torch_profiler_with_stack=False,
                torch_profiler_record_shapes=False, torch_profiler_with_memory=False,
                detailed_trace_annotation=False, ignore_frontend=True, delay_iterations=0,
                max_iterations=0, warmup_iterations=0, wait_iterations=0)


def profile_evidence(directory, status, inputs):
    """Successful profiles need complete controls; failed runs retain negatives."""
    with zipfile.ZipFile(directory / 'source-snapshot.zip') as snapshot:
        config = json.loads(snapshot.read(inputs['config']['snapshot_member']))
        plan = json.loads(snapshot.read(inputs['plan']['snapshot_member']))
    args = config.get('args', [])
    core.require(isinstance(args, list) and all((isinstance(a, str) for a in args)), 'Invalid profile config args')
    core.require(args.count('--profiler-config') == 1 and (not any((a.startswith(('--profiler-config=', '--profiler-config.')) for a in args))), 'Profile requires one explicit --profiler-config JSON argument')
    index = args.index('--profiler-config')
    core.require(index + 1 < len(args), 'Missing profiler config JSON')
    core.require(json.loads(args[index + 1]) == profile_options(directory), 'Profile options differ from bounded harness configuration')
    core.require(args.count('--enable-logging-iteration-details') == 1 and (not any((a in args for a in ('--disable-log-stats', '--no-enable-logging-iteration-details')))), 'Profile requires iteration logging and retained stats')
    core.require('--no-enable-prefix-caching' in args and '--enable-prefix-caching' not in args, 'Profile requires APC off')
    core.require(config.get('env', {}).get('VLLM_SERVER_DEV_MODE') == '1', 'Profile dev-mode config missing')
    phases = plan.get('phases', [])
    core.require(2 <= len(phases) <= 9 and phases[-1].get('role') == 'profile' and all((p.get('role') == 'warmup' for p in phases[:-1])), 'Profile launch plan must contain warmup phases followed by exactly one profile phase')
    profile_id = phases[-1]['id']
    issues = []

    def read_optional(name):
        path = directory / name
        if not path.exists():
            issues.append('Missing ' + name)
            return {}
        core.fingerprint(path)
        try:
            value = load_document(path)
            if not isinstance(value, dict):
                raise ValueError('Expected JSON object')
            return value
        except (ValueError, UnicodeError) as exc:
            issues.append(name + ': ' + str(exc))
            return {}
    report, experiment = (read_optional('capacity.json'), read_optional('experiment.json'))

    def check(condition, text):
        if not condition:
            issues.append(text)
    check(status.get('exit_code') == 0 and status.get('timed_out') is False and (not status.get('supervisor_error')), 'Supervisor did not finish exit0 without timeout/error')
    check(experiment.get('status') == 'PASS' and experiment.get('profile') is True and (experiment.get('excluded_from_performance') is True), 'Experiment is not a successful excluded profile')
    check(any((c.get('task') == 'capacity-profile' and c.get('status') == 'PASS' and (c.get('exit_code') == 0) for c in experiment.get('clients', []))), 'Missing successful capacity-profile client')
    check(report.get('schema_version') == 1 and report.get('status') == 'PASSED' and (not report.get('active_phase')), 'Profile report is not terminal PASSED capacity schema 1')
    observed_phases = report.get('plan', {}).get('phases', [])
    check([(p.get('id'), p.get('role')) for p in observed_phases] == [(p.get('id'), p.get('role')) for p in phases], 'Profile report phase identities/roles differ from launch')
    runs = report.get('runs', [])
    check([(r.get('phase', {}).get('id'), r.get('phase', {}).get('role')) for r in runs] == [(p.get('id'), p.get('role')) for p in phases], 'Profile run sequence/roles are incomplete or different')
    check(all((r.get('excluded_from_performance') is True for r in runs)), 'Profile runs must be excluded from performance')
    profiling = report.get('profiling', {})
    check(profiling.get('excluded_from_performance') is True, 'Missing profile performance-exclusion flag')
    sessions = profiling.get('sessions', [])
    check(len(sessions) == 1 and sessions[0].get('phase_id') == profile_id and (sessions[0].get('prefix') == profile_id), 'Missing or inconsistent single profile session')
    controls = sessions[0].get('controls', []) if len(sessions) == 1 else []
    check(len(controls) == 2 and [c.get('path') for c in controls] == ['/start_profile', '/stop_profile'], 'Profile requires exactly ordered start/stop controls')
    check(bool(controls) and all((c.get('status') == 'PASS' and c.get('http_status') == 200 and (not c.get('error')) for c in controls)), 'Profile controls are not HTTP200/PASS')
    if len(controls) == 2:
        start, stop = controls
        check(start.get('payload') == dict(profile_prefix=profile_id, delay_iterations=0, max_iterations=0) and stop.get('payload') is None, 'Profile control payloads differ from bounded capture')
        times = [c.get(k) for c in controls for k in ('started_unix', 'finished_unix', 'started_monotonic', 'finished_monotonic')]
        valid_times = all((type(t) in (int, float) and math.isfinite(t) for t in times))
        check(valid_times, 'Profile control timestamps missing/nonfinite')
        if valid_times:
            check(all((c['started_unix'] <= c['finished_unix'] and c['started_monotonic'] <= c['finished_monotonic'] for c in controls)) and start['finished_monotonic'] <= stop['started_monotonic'], 'Profile control timestamp order invalid')
            observed = [r for r in runs if r.get('phase', {}).get('id') == profile_id]
            run = observed[0] if len(observed) == 1 else {}
            left, right = (run.get('start_unix'), run.get('end_unix'))
            check(type(left) in (int, float) and type(right) in (int, float) and (start['finished_unix'] <= left <= right <= stop['started_unix']), 'Profile start/stop does not completely enclose the measured phase')
    check(not any((s.get('stop_error') for s in sessions)), 'Profile stop_error retained')
    claims_success = status.get('exit_code') == 0 or experiment.get('status') == 'PASS'
    if claims_success:
        core.require(not issues, 'Successful profile lacks complete evidence: ' + '; '.join(issues))
    explicitly_failed = report.get('status') == 'FAILED' or experiment.get('status') == 'FAILED' or any((c.get('status') == 'FAILED' for c in controls))
    return dict(profile_evidence_status='VERIFIED' if not issues else 'FAILED' if explicitly_failed else 'INCOMPLETE', profile_phase_id=profile_id, excluded_from_performance=True, issues=issues)


def launch_inputs(directory, command, research):
    """Verify immutable launch sources, without executing or extracting the ZIP."""
    snapshot = directory / 'source-snapshot.zip'
    core.fingerprint(snapshot)
    expected = command.get('source_sha256')
    core.require(isinstance(expected, dict) and expected, 'Missing launch source hash map')
    selected = {}
    with zipfile.ZipFile(snapshot) as saved:
        infos = saved.infolist()
        core.require(len(infos) == len({item.filename for item in infos}), 'Duplicate source snapshot member')
        core.require({item.filename for item in infos} == set(expected), 'Snapshot/source hash map membership mismatch')
        for item in infos:
            mode = item.external_attr >> 16
            core.require(core.safe_name(item.filename) and (not item.is_dir()) and (stat.S_IFMT(mode) in (0, stat.S_IFREG)), 'Unsafe source snapshot member')
            with saved.open(item) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            core.require(actual == expected[item.filename], 'Snapshot source hash mismatch: ' + item.filename)
        argv = command['command']
        for flag in ('--config', '--plan'):
            core.require(argv.count(flag) == 1, 'Launcher must record exactly one ' + flag)
            at = argv.index(flag)
            core.require(at + 1 < len(argv), 'Missing launcher argument for ' + flag)
            path = Path(argv[at + 1])
            if not path.is_absolute():
                path = Path(command['cwd']) / path
            path = plain_path(path)
            core.require(path.is_relative_to(research / 'scenarios'), 'Launch input outside scenario directory')
            member = str(path.relative_to(research))
            core.require(member in expected, 'Launch input absent from source snapshot: ' + member)
            body = saved.read(member)
            parsed = json.loads(body)
            if flag == '--config' and (directory / 'config.json').exists():
                core.fingerprint(directory / 'config.json')
                core.require(parsed == load_document(directory / 'config.json'), 'Saved runtime config differs from launch config')
            if flag == '--plan':
                core.require(parsed.get('experiment_id') == directory.name, 'Launch plan identity mismatch')
            selected[flag[2:]] = dict(path=str(path), snapshot_member=member, launch_sha256=expected[member], current_workspace_sha256=core.sha(path) if path.exists() else None)
    for item in selected.values():
        item['current_matches_launch'] = item['launch_sha256'] == item['current_workspace_sha256']
    return selected


def terminal_results(phase, names):
    core.require(names and len(set(names)) == len(names), 'Supply explicit nonempty unique b experiment IDs')
    research = phase.parent
    scenario = research / 'scenarios'
    records = []
    for name in sorted(names):
        core.require(EXPERIMENT.fullmatch(name), 'Invalid Stage B experiment ID: ' + name)
        directory = plain_path(path_at(scenario, 'results') / name)
        core.require(directory.is_dir(), 'Missing result directory: ' + name)
        for member in ('command.json', 'status.json'):
            core.fingerprint(directory / member)
        command = load_document(directory / 'command.json')
        status = load_document(directory / 'status.json')
        core.require(command.get('experiment_id') == name == status.get('experiment_id'), 'Supervisor/command ID mismatch')
        launcher = owned_launcher(command, scenario)
        core.require(bool(status.get('end_utc')), 'Nonterminal supervisor: ' + name)
        core.require(type(status.get('exit_code')) is int or status.get('timed_out') is True or bool(status.get('supervisor_error')), 'Missing terminal supervisor outcome: ' + name)
        inputs = launch_inputs(directory, command, research)
        profile = profile_evidence(directory, status, inputs) if launcher == 'serve_long_context_profile.py' else None
        records.append(dict(id=name, exit_code=status.get('exit_code'), timed_out=status.get('timed_out'), supervisor_error=status.get('supervisor_error'), end_utc=status['end_utc'], git_commit=command.get('git_commit'), launch_inputs=inputs, launcher=launcher, profile_evidence=profile, source_snapshot_sha256=core.sha(directory / 'source-snapshot.zip')))
    return records


def collect(phase, experiments):
    research, scenario = (phase.parent, phase.parent / 'scenarios')
    project = research.parents[1]
    files, excluded = (set(), set())

    def visit(path):
        path = plain_path(path)
        core.require(path.is_relative_to(project), 'Input outside workspace: ' + str(path))
        mode = path.lstat().st_mode
        core.require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'Nonregular input refused: ' + str(path))
        archive = path.name.endswith(('.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz', '.tbz2', '.txz', '.7z'))
        archive = archive or (path.suffix.lower() == '.zip' and path.name != 'source-snapshot.zip')
        if core.excluded(path, phase) or path.name in ('evidence', 'archives', 'cache', 'caches') or archive:
            excluded.add(str(path.relative_to(project)))
        elif stat.S_ISDIR(mode):
            for child in sorted(path.iterdir()):
                visit(child)
        else:
            files.add(path)
    visit(phase)
    for experiment in experiments:
        visit(path_at(scenario, 'results') / experiment['id'])
        for item in experiment['launch_inputs'].values():
            if Path(item['path']).exists():
                visit(Path(item['path']))
    for name in SHARED_SCRIPTS:
        core.fingerprint(scenario / 'scripts' / name)
        visit(scenario / 'scripts' / name)
    if any((item['launcher'] == 'serve_long_context_profile.py' for item in experiments)):
        core.fingerprint(scenario / 'scripts/serve_long_context_profile.py')
        visit(scenario / 'scripts/serve_long_context_profile.py')
    if path_at(scenario, 'workloads/long-context').exists():
        visit(path_at(scenario, 'workloads/long-context'))
    for name in REFERENCE_DOCS:
        if (scenario / name).exists():
            visit(scenario / name)
    for name in core.PARENT_REQUIRED:
        core.fingerprint(research / name)
        visit(research / name)
    for name in ('model-analysis.md', 'framework-capabilities.md', 'topology-analysis.md', 'roadmap.md'):
        if (research / name).exists():
            visit(research / name)
    for name in core.PROJECT_REQUIRED:
        core.fingerprint(project / name)
        visit(project / name)
    selected = {row['id'] for row in experiments}
    for path in files:
        if path.is_relative_to(path_at(scenario, 'results')):
            core.require(path.relative_to(path_at(scenario, 'results')).parts[0] in selected, 'Unselected raw result included')
    return (sorted(files), sorted(excluded), project)


def package(phase, name, names):
    core.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', name), 'Use a unique simple evidence ID')
    phase = plain_path(phase)
    core.require(phase.is_dir(), 'Stage B directory missing')
    with core.research_lock(phase):
        experiments = terminal_results(phase, names)
        output = plain_path(path_at(phase, 'evidence'))
        output.mkdir(parents=True, exist_ok=True)
        archive, manifest_path, sums, record_path = [output / (name + suffix) for suffix in ('.tar.gz', '.manifest.json', '.sha256', '.json')]
        core.require(not any((p.exists() or p.is_symlink() for p in (archive, manifest_path, sums, record_path))), 'Evidence ID exists; previous archive must remain unchanged')
        record = dict(status='IN PROGRESS', created_utc=core.utc(), evidence_id=name, scope='Stage B archive integrity only; failed runs remain included; not performance or quality acceptance')
        core.new_json(record_path, record)
        try:
            sources, exclusions, project = collect(phase, experiments)
            rows = [dict(path=str(p.relative_to(project)), sha256=core.sha(p), bytes=p.stat().st_size) for p in sources]
            core.require(rows and all((core.safe_name(row['path']) for row in rows)), 'Empty or unsafe archive members')
            manifest = dict(schema_version=1, created_utc=core.utc(), path_base='workspace root', experiments=experiments, files=rows, exclusions=exclusions, scope='Explicit b result trees with raw metrics/requests/GPU/server logs/source ZIP; Stage B methods/analysis/figures/CPU logs; exact launch inputs and shared runtime/environment references.', selection_mode='explicit b experiment IDs only', reference_closure_guaranteed=False, reference_caveat='Historical Stage A/scenario reports may refer to raw evidence in their immutable original packages. Those old raw trees and packages are not duplicated here. Current workspace config/plan drift is recorded; per-experiment ZIP is authoritative.', not_included='Unselected raw result trees, model weights, virtual environments, caches and previous archives')
            core.new_json(manifest_path, manifest)
            with open_output(sums, 'x') as stream:
                stream.write(core.sum_text(rows))
            with open_output(archive, 'xb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                    with open_output(tarfile, fileobj=compressed, mode='w|') as bundle:
                        for path, row in zip(sources, rows):
                            plain_path(path)
                            core.fingerprint(path)
                            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                                info = bundle.gettarinfo(str(path), arcname=row['path'], fileobj=stream)
                                core.require(info.isfile(), 'Source became nonregular')
                                bundle.addfile(info, stream)
            core.verify_tar(archive, rows)
            final_sources, final_exclusions, _ = collect(phase, experiments)
            core.require(final_sources == sources and final_exclusions == exclusions, 'Archive membership changed during packaging')
            for path, row in zip(sources, rows):
                core.require(core.sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source changed during packaging')
            core.require(terminal_results(phase, names) == experiments, 'Supervisor/launch inputs changed during packaging')
            record.update(status='PASS', completed_utc=core.utc(), archive=archive.name, archive_sha256=core.sha(archive), archive_bytes=archive.stat().st_size, manifest=manifest_path.name, manifest_sha256=core.sha(manifest_path), sha256_file=sums.name, sha256_file_sha256=core.sha(sums), files=len(rows), original_bytes=sum((row['bytes'] for row in rows)), experiment_ids=[row['id'] for row in experiments])
            core.replace_record(record_path, record)
            core.verify(record_path)
            return record
        except BaseException as exc:
            record.update(status='FAILED', ended_utc=core.utc(), error=repr(exc))
            core.replace_record(record_path, record)
            raise


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix='phase-b-package-fixture-') as temp:
        project = Path(temp)
        research = project / 'research/model'
        scenario, phase = (research / 'scenarios', research / 'phase-b-fixture')

        def put(path, data=b'fixture\n'):
            path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(path, data)
        for name in core.PROJECT_REQUIRED:
            put(project / name)
        for name in core.PARENT_REQUIRED:
            put(research / name)
        for name in SHARED_SCRIPTS:
            put(scenario / 'scripts' / name)
        for name in ('protocol.md', 'scripts/package_long_context.py', 'summary/final.json', 'figures/final.png', 'logs/cpu-validation.json'):
            put(phase / name)
        name = 'b01-fixture-failed'
        directory = path_at(scenario, 'results') / name
        config, plan = (path_at(scenario, 'configs') / f'{name}.json', path_at(scenario, 'workloads/long-context') / f'{name}.json')
        put(config, core.json_bytes({'name': 'fixture'}))
        put(plan, core.json_bytes({'experiment_id': name}))
        snapshot_sources = (config, plan, scenario / 'scripts/capacity_client.py')
        command = dict(experiment_id=name, cwd=str(project), git_commit='fixture', command=['python', str(scenario / 'scripts/serve_capacity.py'), '--config', str(config), '--plan', str(plan)], source_sha256={str(p.relative_to(research)): core.sha(p) for p in snapshot_sources})
        put(directory / 'command.json', core.json_bytes(command))
        put(directory / 'status.json', core.json_bytes(dict(experiment_id=name, exit_code=1, end_utc=core.utc())))
        put(directory / 'config.json', config.read_bytes())
        with zipfile.ZipFile(directory / 'source-snapshot.zip', 'x') as snapshot:
            for path in snapshot_sources:
                snapshot.write(path, str(path.relative_to(research)))
        for name_ in ('server.log', 'gpu.csv', 'capacity.json', 'capacity-plan.json', 'capacity-workload.json', 'requests.jsonl', 'metrics/before.prom', 'metrics/00000.prom', 'profiles/rank0.pt.trace.json.gz'):
            put(directory / name_)
        for name_ in ('weights/model.safetensors', '.venv/bin/python', '.cache/test', 'old.tar.gz', 'old.zip', 'evidence/old.tar.gz'):
            put(phase / name_)
        put(path_at(scenario, 'results/a01-history/DO_NOT_INCLUDE'))
        put(path_at(scenario, 'results/b99-unselected/DO_NOT_INCLUDE'))
        record = package(phase, 'test-roundtrip', [name])
        record_path = path_at(phase, 'evidence/test-roundtrip.json')
        assert core.verify(record_path)['status'] == 'PASS'
        manifest = load_document(record_path.parent / record['manifest'])
        included = {row['path'] for row in manifest['files']}
        assert all((any((p.endswith(n) for p in included)) for n in ('source-snapshot.zip', 'requests.jsonl', 'before.prom', 'rank0.pt.trace.json.gz', 'final.png', 'cpu-validation.json', 'environment.json', 'analyze_profiles.py', 'package_scenarios.py')))
        assert not any(('/a01-history/' in p or '/b99-unselected/' in p or '/.venv/' in p or ('/.cache/' in p) or ('/weights/' in p) or ('/evidence/' in p) or p.endswith(('old.zip', 'old.tar.gz')) for p in included))
        checks += ['CPU roundtrip retains failed b run and all raw/trace/source snapshot evidence', 'Stage B method/analysis/figure/log and shared runtime/environment included', 'Shared historical analyze_profiles.py present at the new analyzer import path', 'Only explicit raw experiments; no historical results/weights/venv/cache/old archives', 'Full archive and every member reread against SHA256/size manifest']

        def rejects(label, function):
            try:
                function()
            except (ValueError, RuntimeError, OSError, zipfile.BadZipFile):
                checks.append(label)
            else:
                raise AssertionError('Expected rejection: ' + label)
        rejects('No implicit experiment discovery', lambda: package(phase, 'no-ids', None))
        rejects('Duplicate requested experiment IDs', lambda: package(phase, 'duplicates', [name, name]))
        rejects('Stage A ID cannot be selected', lambda: package(phase, 'old-id', ['a01-history']))
        rejects('Duplicate archive ID preserves old package', lambda: package(phase, 'test-roundtrip', [name]))
        with core.research_lock(phase):
            rejects('Active shared GPU research lock prevents packaging', lambda: package(phase, 'locked', [name]))
        saved_status = (directory / 'status.json').read_bytes()
        put(directory / 'status.json', core.json_bytes(dict(experiment_id=name, exit_code=None)))
        rejects('Nonterminal supervisor prevents packaging', lambda: package(phase, 'running', [name]))
        put(directory / 'status.json', saved_status)
        link = phase / 'unsafe-link'
        link.symlink_to(path_at(project, 'AGENTS.md'))
        rejects('Symlink input refused', lambda: package(phase, 'linked', [name]))
        link.unlink()
        alias = project / 'phase-alias'
        alias.symlink_to(phase, target_is_directory=True)
        rejects('Symlink stage root refused before resolution', lambda: package(alias, 'root-link', [name]))
        put(config, core.json_bytes({'name': 'changed-after-experiment'}))
        drift = package(phase, 'explicit-launch-drift', [name])
        drift_manifest = load_document(path_at(phase, 'evidence') / drift['manifest'])
        assert not drift_manifest['experiments'][0]['launch_inputs']['config']['current_matches_launch']
        checks.append('Current input drift is explicit; immutable exact launch ZIP retained')
        put(directory / 'config.json', b'{"name":"wrong-runtime-config"}')
        rejects('Saved runtime config must equal immutable launch config', lambda: package(phase, 'wrong-config', [name]))
        put(directory / 'config.json', core.json_bytes({'name': 'fixture'}))
        altered = dict(command, source_sha256={k: '0' * 64 for k in command['source_sha256']})
        put(directory / 'command.json', core.json_bytes(altered))
        rejects('Source ZIP members checked against launch hash map', lambda: package(phase, 'bad-snapshot', [name]))
        put(directory / 'command.json', core.json_bytes(command))
        shell_command = dict(command, command=['bash', '-c', 'serve_capacity.py', '--config', str(config), str(plan)])
        put(directory / 'command.json', core.json_bytes(shell_command))
        rejects('Arbitrary shell command is never a launcher', lambda: package(phase, 'shell-command', [name]))
        unknown = dict(command, command=['python', str(scenario / 'scripts/other.py'), '--config', str(config), '--plan', str(plan)])
        put(directory / 'command.json', core.json_bytes(unknown))
        rejects('Third launcher remains outside exact whitelist', lambda: package(phase, 'third-launcher', [name]))
        put(directory / 'command.json', core.json_bytes(command))

        def profile_fixture(profile_name, iteration_logging=True, profiler_enabled=True):
            result = path_at(scenario, 'results') / profile_name
            profile_config = path_at(scenario, 'configs') / f'{profile_name}.json'
            profile_plan = path_at(scenario, 'workloads/long-context') / f'{profile_name}.json'
            profile_script = scenario / 'scripts/serve_long_context_profile.py'
            put(profile_script)
            options = ['--no-enable-prefix-caching']
            if profiler_enabled:
                options += ['--profiler-config', json.dumps(profile_options(result))]
            if iteration_logging:
                options.append('--enable-logging-iteration-details')
            put(profile_config, core.json_bytes(dict(name='fixture-profile', args=options, env={'VLLM_SERVER_DEV_MODE': '1'})))
            phases = [dict(id='warmup', role='warmup'), dict(id='profile', role='profile')]
            plan_body = dict(experiment_id=profile_name, phases=phases)
            put(profile_plan, core.json_bytes(plan_body))
            sources = (profile_config, profile_plan, profile_script)
            cmd = dict(experiment_id=profile_name, cwd=str(project), command=['python', str(profile_script), '--config', str(profile_config), '--plan', str(profile_plan)], source_sha256={str(p.relative_to(research)): core.sha(p) for p in sources})
            put(result / 'command.json', core.json_bytes(cmd))
            put(result / 'status.json', core.json_bytes(dict(experiment_id=profile_name, exit_code=0, timed_out=False, end_utc=core.utc())))
            put(result / 'config.json', profile_config.read_bytes())
            with zipfile.ZipFile(result / 'source-snapshot.zip', 'x') as saved:
                for source in sources:
                    saved.write(source, str(source.relative_to(research)))
            controls = [dict(path='/start_profile', payload=dict(profile_prefix='profile', delay_iterations=0, max_iterations=0), status='PASS', http_status=200, started_unix=8.0, finished_unix=9.0, started_monotonic=8.0, finished_monotonic=9.0), dict(path='/stop_profile', payload=None, status='PASS', http_status=200, started_unix=13.0, finished_unix=14.0, started_monotonic=13.0, finished_monotonic=14.0)]
            report = dict(schema_version=1, status='PASSED', plan=plan_body, runs=[dict(phase=p, excluded_from_performance=True, start_unix=10.0, end_unix=12.0) for p in phases], profiling=dict(excluded_from_performance=True, sessions=[dict(phase_id='profile', prefix='profile', controls=controls)]))
            put(result / 'capacity.json', core.json_bytes(report))
            experiment = dict(status='PASS', profile=True, excluded_from_performance=True, clients=[dict(task='capacity-profile', status='PASS', exit_code=0)])
            put(result / 'experiment.json', core.json_bytes(experiment))
            put(result / 'profiles/rank0.pt.trace.json.gz')
            return (result, report, experiment)
        profiledir, profile_report, profile_experiment = profile_fixture('b02-profile')
        profiled = package(phase, 'profile-success', [profiledir.name])
        pm = load_document(path_at(phase, 'evidence') / profiled['manifest'])
        assert pm['experiments'][0]['profile_evidence']['profile_evidence_status'] == 'VERIFIED'
        checks.append('Owned profile launcher with exact config/roles and enclosing HTTP200 controls archives as VERIFIED')
        profile_report['profiling']['sessions'][0]['controls'][1]['started_unix'] = 11.0
        put(profiledir / 'capacity.json', core.json_bytes(profile_report))
        rejects('Claimed successful profile must have complete temporal enclosure', lambda: package(phase, 'profile-short-control', [profiledir.name]))
        profile_report['profiling']['sessions'][0]['controls'][1]['started_unix'] = 13.0
        profile_report['runs'][-1]['phase']['role'] = 'measure'
        put(profiledir / 'capacity.json', core.json_bytes(profile_report))
        rejects('Profile run cannot masquerade as normal performance measurement', lambda: package(phase, 'profile-measure-role', [profiledir.name]))
        profile_report['runs'][-1]['phase']['role'] = 'profile'
        control = profile_report['profiling']['sessions'][0]['controls'][1]
        control.update(status='FAILED', http_status=500, error={'type': 'HTTPError'})
        put(profiledir / 'capacity.json', core.json_bytes(profile_report))
        rejects('Claimed profile success cannot hide an HTTP500 control', lambda: package(phase, 'profile-control-http500', [profiledir.name]))
        profile_report['status'] = 'FAILED'
        profile_experiment['status'] = 'FAILED'
        profile_experiment['clients'][0].update(status='FAILED', exit_code=1)
        put(profiledir / 'capacity.json', core.json_bytes(profile_report))
        put(profiledir / 'experiment.json', core.json_bytes(profile_experiment))
        put(profiledir / 'status.json', core.json_bytes(dict(experiment_id=profiledir.name, exit_code=1, timed_out=False, end_utc=core.utc())))
        failed_profile = package(phase, 'failed-profile-retained', [profiledir.name])
        fm = load_document(path_at(phase, 'evidence') / failed_profile['manifest'])
        assert fm['experiments'][0]['profile_evidence']['profile_evidence_status'] == 'FAILED'
        assert any(('HTTP200' in issue for issue in fm['experiments'][0]['profile_evidence']['issues']))
        checks.append('Terminal failed profile retains raw evidence with FAILED control diagnostics')
        (profiledir / 'capacity.json').unlink()
        (profiledir / 'experiment.json').unlink()
        incomplete_profile = package(phase, 'incomplete-profile-retained', [profiledir.name])
        im = load_document(path_at(phase, 'evidence') / incomplete_profile['manifest'])
        assert im['experiments'][0]['profile_evidence']['profile_evidence_status'] == 'INCOMPLETE'
        checks.append('Terminal pre-capture failure retained as INCOMPLETE with missing report/control reasons')
        missing_logging, _, _ = profile_fixture('b03-missing-logging', iteration_logging=False)
        rejects('Profile requires explicit iteration logging in immutable config', lambda: package(phase, 'profile-no-logging', [missing_logging.name]))
        missing_profiler, _, _ = profile_fixture('b04-missing-profiler', profiler_enabled=False)
        rejects('Profile requires explicit profiler JSON in immutable config', lambda: package(phase, 'profile-no-profiler', [missing_profiler.name]))
        with open_output(path_at(phase, 'evidence') / record['archive'], 'ab') as stream:
            stream.write(b'corrupt')
        rejects('Post-package archive corruption rejected', lambda: core.verify(record_path))
    return dict(status='PASS', scope='Temporary CPU-only fixtures; no real results packaged or GPU queried', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--name')
    modes.add_argument('--verify', type=Path)
    modes.add_argument('--self-test', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--experiment', action='append', help='Explicit b ID; repeat for every selected result')
    parser.add_argument('--output', type=Path, help='Unique self-test/verification JSON record')
    args = parser.parse_args()
    if args.name and (args.output or not args.experiment):
        parser.error('--name requires explicit --experiment IDs and does not take --output')
    if args.verify:
        record_path = plain_path(args.verify)
        core.fingerprint(record_path)
        result = core.verify(record_path)
    else:
        result = self_test() if args.self_test else package(args.root, args.name, args.experiment)
    if args.output:
        path = plain_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        core.new_json(path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
