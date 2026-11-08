#!/usr/bin/env python3
"""Read-only CPU evidence audit; never imports/runs a client or snapshot source.

Print JSON, or create --output NEW_JSON. --allow-incomplete permits exit 0 for
an explicitly INCOMPLETE audit while experiments run; it never changes that status.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import ast
import collections
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import random
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
_PATH_SPEC = importlib.util.spec_from_file_location('scenario_path_audit', Path(__file__).with_name('capacity_audit.py'))
_PATH_AUDIT = importlib.util.module_from_spec(_PATH_SPEC)
_PATH_SPEC.loader.exec_module(_PATH_AUDIT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ids_hash(ids):
    return digest(json.dumps(ids, separators=(",", ":")).encode())


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*fraction
    a, b = math.floor(index), math.ceil(index)
    return values[a]*(b-index)+values[b]*(index-a) if a != b else values[a]


def recompute_summary(rows, wall):
    good = [r for r in rows if r["ok"]]
    outputs = sum(r["usage"]["completion_tokens"] for r in good)
    slo = sum(r["ttft_s"]+r["dispatch_lag_s"] <= 1 and r["tpot_s"] <= .05
              and r["e2e_s"]+r["dispatch_lag_s"] <= 10 for r in good)
    result = dict(requests=len(rows), successful=len(good), failed=len(rows)-len(good),
                  wall_s=wall, output_tokens=outputs, output_tokens_per_s=outputs/wall,
                  requests_per_s=len(good)/wall, example_slo_pass=slo,
                  example_goodput_requests_per_s=slo/wall)
    for key in ("ttft_s", "e2e_s", "tpot_s", "dispatch_lag_s", "scheduled_ttft_s", "scheduled_e2e_s"):
        values = [r[key] for r in good]
        result[key] = dict(p50=percentile(values, .5), p95=percentile(values, .95),
                           max=max(values) if values else None)
    return result


def option(command, name, default=None):
    return command[command.index(name)+1] if name in command else default


class Auditor:

    def __init__(self, root):
        self.root = root.resolve()
        self.errors, self.warnings, self.incomplete = ([], [], [])
        self.sources, self.signatures, self.experiments = ({}, {}, [])
        self.totals = collections.Counter()
        self.read_stats = {}

    def check(self, condition, where, detail):
        if not condition:
            self.errors.append(dict(location=where, detail=detail))

    def equal(self, actual, expected, where):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                self.check(False, where, 'Expected object')
                return
            for key, value in expected.items():
                self.equal(actual.get(key), value, f'{where}.{key}')
        elif isinstance(expected, float):
            self.check(type(actual) in (float, int) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-09, abs_tol=1e-09), where, f'Expected {expected!r}, got {actual!r}')
        else:
            self.check(actual == expected, where, f'Expected {expected!r}, got {actual!r}')

    def read(self, path):
        raw = path.read_bytes()
        stat = path.stat()
        self.read_stats[path] = (stat.st_mtime_ns, stat.st_size)
        self.sources[str(path.resolve())] = digest(raw)
        return json.loads(raw)

    def pair(self, key, value, where):
        identity = digest(canonical(value).encode())
        if key in self.signatures:
            previous = self.signatures[key]
            self.check(identity == previous['sha256'], where, f'Workload identity differs from {previous['location']}')
            previous['observations'] += 1
        else:
            self.signatures[key] = dict(sha256=identity, location=where, observations=1)

    def scenario(self, path, command, snapshot, experiment_terminal):
        d = self.read(path)
        where = str(path.resolve())
        if 'suite' not in d:
            self.check(not experiment_terminal, where, 'Successful experiment contains uninitialized report')
            self.incomplete.append(where + ' initializing')
            return dict(file=path.name, status='INITIALIZING')
        suite = d['suite']
        status = d['status']
        if status in ('RUNNING', 'PREPARED', 'INITIALIZING'):
            self.incomplete.append(where)
            self.check(not experiment_terminal, where, 'Nonterminal report inside terminal successful experiment')
        self.check(status in ('PASSED', 'FAILED', 'RUNNING', 'PREPARED', 'INITIALIZING'), where, 'Unknown suite status')
        source_key = f'{self.root.name}/scripts/scenario_client.py'
        self.equal(d['client_sha256'], digest(snapshot[source_key]), where + '.client_sha256')
        self.check('scripts/benchmark_client.py' in snapshot, where, 'Missing imported historical helper snapshot')
        self.equal(d['protocol']['connections'], 'fresh_per_request', where + '.connections')
        self.equal(d['protocol']['retries'], 0, where + '.retries')
        self.equal(d['rounds'], int(option(command, '--rounds', '2')), where + '.rounds')
        self.equal(d['suite'], option(command, '--suite'), where + '.suite')
        self.equal(d['base_url'], option(command, '--base-url'), where + '.base_url')
        self.pair(f'tokenizer/{suite}', d['tokenizer_sha256'], where)
        cases = {c['name']: c for c in d['cases']}
        self.equal(len(cases), len(d['cases']), where + '.unique_cases')
        self.pair(f'cases/{suite}', d['cases'], where)
        for h, prompt in d['prompts'].items():
            ids = prompt['token_ids']
            self.check(all((type(i) is int and i >= 0 for i in ids)), where, f'Noninteger token IDs {h}')
            self.equal(len(ids), prompt['input_tokens'], where + '.prompt_length')
            self.equal(ids_hash(ids), h, where + '.prompt_hash')
            self.equal(prompt['sha256'], h, where + '.embedded_prompt_hash')
        self.totals['prompt_banks_verified'] += len(d['prompts'])
        phases = ['cold', 'exact_replay', 'fork_first', 'warm_serial', 'warm_parallel'] if suite == 'cache' else ['warmup', 'measure']
        expected_bank_keys = {f'{r}/{c}/{p}' for r in range(1, d['rounds'] + 1) for c in cases for p in phases}
        self.equal(sorted(d['banks']), sorted(expected_bank_keys), where + '.bank_keys')
        for key, bank in d['banks'].items():
            r, name, phase = key.split('/')
            case = cases[name]
            count = 1 if phase in ('cold', 'exact_replay', 'fork_first') else case['c'] if phase == 'warmup' and 'rate' not in case else 4 if phase == 'warmup' else case['n']
            self.equal(len(bank), count, where + f'.bank_count[{key}]')
            self.pair(f'bank/{suite}/{key}', bank, where)
            for i, item in enumerate(bank):
                self.check(item['prompt_sha256'] in d['prompts'], where, f'Unknown bank prompt {key}/{i}')
                self.equal(item['seed'], d['seed'] + i, where + '.bank_seed')
                if 'rate' not in case:
                    self.equal(d['prompts'][item['prompt_sha256']]['input_tokens'], case['p'], where + '.bank_input')
                    self.equal(item['output_tokens'], case['o'], where + '.bank_output')
                else:
                    choice = random.Random(d['seed'] + i).random()
                    p, o = (512, 128) if choice < 0.7 else (4096, 256) if choice < 0.9 else (16384, 128)
                    self.equal(d['prompts'][item['prompt_sha256']]['input_tokens'], p, where + '.mixed_input')
                    self.equal(item['output_tokens'], o, where + '.mixed_output')
        if suite == 'arrival':
            rate_cases = [name for name, case in cases.items() if 'rate' in case]
            for r in range(1, d['rounds'] + 1):
                for phase in phases:
                    for name in rate_cases[1:]:
                        self.equal(d['banks'][f'{r}/{name}/{phase}'], d['banks'][f'{r}/{rate_cases[0]}/{phase}'], where + '.matched_mixture_between_rates')
        if suite == 'cache':
            for r in range(1, d['rounds'] + 1):
                for name, case in cases.items():
                    prefix = f'{r}/{name}/'
                    self.equal(d['banks'][prefix + 'cold'], d['banks'][prefix + 'exact_replay'], where + '.exact_replay_bank')
                    ids = [d['prompts'][item['prompt_sha256']]['token_ids'] for phase in phases for item in d['banks'][prefix + phase]]
                    self.check(all((x[:case['prefix']] == ids[0][:case['prefix']] for x in ids)), where, 'Cache shared prefix differs')
        expected_runs = [(r, c['name']) for r in range(1, d['rounds'] + 1) for c in d['cases']]
        observed_runs = [(r['round'], r['case']) for r in d['runs']]
        self.equal(observed_runs, expected_runs[:len(observed_runs)], where + '.run_order')
        if status == 'PASSED':
            self.equal(observed_runs, expected_runs, where + '.complete_run_count')
        failures = requests = stages = 0
        for run in d['runs']:
            case = cases[run['case']]
            key = f'{run['round']}/{run['case']}'
            observed = [s['phase'] for s in run['stages']]
            self.equal(observed, phases[:len(observed)], where + f'.{key}.stage_order')
            if run['status'] == 'PASSED' or status == 'PASSED':
                self.equal(observed, phases, where + f'.{key}.complete_stages')
                self.equal(run['status'], 'PASSED', where + f'.{key}.run_status')
            if suite == 'cache' and run['stages']:
                reset = run.get('reset', {})
                self.equal(reset.get('status'), 200, where + '.reset_status')
                self.equal(json.loads(reset.get('body', '{}')).get('success'), True, where + '.reset_success')
            for stage in run['stages']:
                stages += 1
                phase = stage['phase']
                at = where + f'/{key}/{phase}'
                bank = d['banks'][key + '/' + phase]
                rows = stage['requests']
                self.equal(len(rows), len(bank), at + '.request_count')
                concurrency = 8 if phase == 'warm_parallel' else 1 if phase in ('cold', 'exact_replay', 'fork_first') else case['c']
                self.equal(stage['concurrency_limit'], concurrency, at + '.concurrency')
                self.equal(stage.get('offered_rate'), case.get('rate'), at + '.offered_rate')
                offsets = [0.0] * len(bank)
                if 'rate' in case and phase == 'measure':
                    rng = random.Random(d['seed'] + run['round'])
                    for i in range(1, len(bank)):
                        offsets[i] = offsets[i - 1] + rng.expovariate(case['rate'])
                self.equal(stage['planned_arrival_span_s'], offsets[-1], at + '.arrival_span')
                for i, (row, item) in enumerate(zip(rows, bank)):
                    requests += 1
                    self.equal(row['request_id'], key + f'/{phase}/{i}', at + '.request_id')
                    self.equal(row['prompt_sha256'], item['prompt_sha256'], at + '.request_prompt')
                    self.equal(row['seed'], item['seed'], at + '.request_seed')
                    self.equal(row['requested_output_tokens'], item['output_tokens'], at + '.request_output')
                    self.equal(row['input_tokens'], d['prompts'][item['prompt_sha256']]['input_tokens'], at + '.request_input')
                    self.equal(digest(row['text'].encode()), row['text_sha256'], at + '.text_hash')
                    self.equal(row['planned_offset_s'], offsets[i], at + '.planned_offset')
                    self.check(row['dispatch_lag_s'] >= 0, at, 'Negative dispatch lag')
                    if row['ok']:
                        self.equal(row['http_status'], 200, at + '.http_status')
                        self.equal(row['done_received'], True, at + '.done')
                        self.equal(row['error'], None, at + '.error')
                        self.check(bool(row['text']), at, 'Successful request has no full text')
                        usage = row['usage']
                        self.equal(usage['prompt_tokens'], row['input_tokens'], at + '.usage_input')
                        self.equal(usage['completion_tokens'], row['requested_output_tokens'], at + '.usage_output')
                        self.equal(usage['total_tokens'], usage['prompt_tokens'] + usage['completion_tokens'], at + '.usage_total')
                        self.check(0 <= row['ttft_s'] <= row['e2e_s'] <= row['elapsed_s'] + 0.001, at, 'Timing order violation')
                        self.equal(row['tpot_s'], (row['e2e_s'] - row['ttft_s']) / (usage['completion_tokens'] - 1), at + '.tpot')
                        self.equal(row['scheduled_ttft_s'], row['ttft_s'] + row['dispatch_lag_s'], at + '.scheduled_ttft')
                        self.equal(row['scheduled_e2e_s'], row['e2e_s'] + row['dispatch_lag_s'], at + '.scheduled_e2e')
                        events = row['text_event_seconds']
                        self.check(bool(events) and events == sorted(events), at, 'Text event order invalid')
                        if events:
                            self.equal(events[0], row['ttft_s'], at + '.first_text_event')
                            self.check(events[-1] <= row['e2e_s'], at, 'Text event later than DONE')
                    else:
                        failures += 1
                        self.check(bool(row.get('error')), at, 'Failure lacks recorded error')
                wall = stage['summary']['wall_s']
                self.check(wall > 0, at, 'Nonpositive batch wall time')
                self.equal(stage['summary'], recompute_summary(rows, wall), at + '.summary')
                for m in ('metrics_before', 'metrics_after'):
                    name = stage[m]
                    self.check(Path(name).name == name and (path.parent / name).is_file(), at, f'Missing or invalid {m}')
        if status == 'PASSED':
            self.equal(failures, 0, where + '.passed_has_no_failed_requests')
        self.totals.update(scenario_reports=1, stream_requests=requests, stream_failures=failures, stages=stages)
        return dict(file=path.name, suite=suite, status=status, requests=requests, failed=failures, phases_observed=stages, runs_observed=len(observed_runs), runs_expected=len(expected_runs), identity_scope='Prompt IDs/hash + seed + output budget + case/round/phase; timings not compared across configs')

    def experiment(self, directory):
        at = directory.name
        command = self.read(directory / 'command.json')
        if not (directory / 'status.json').exists():
            self.incomplete.append(at + '/status.json missing')
            status = None
        else:
            status = self.read(directory / 'status.json')
            for key in ('experiment_id', 'command', 'source_sha256', 'git_commit', 'environment'):
                self.equal(status.get(key), command.get(key), at + f'.status.{key}')
        self.equal(command['experiment_id'], at, at + '.experiment_id')
        self.check(set(command['environment']['CUDA_VISIBLE_DEVICES'].split(',')) <= {'0', '1', '2', '3'}, at, 'Unauthorized GPU identifier in record')
        archive = directory / 'source-snapshot.zip'
        self.sources[str(archive.resolve())] = digest(archive.read_bytes())
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            self.check(len(names) == len(set(names)), at, 'Duplicate snapshot members')
            self.equal(sorted(names), sorted(command['source_sha256']), at + '.snapshot_members')
            snapshot = {}
            for name in names:
                self.check(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts, at, 'Invalid snapshot member path')
                data = z.read(name)
                self.equal(digest(data), command['source_sha256'].get(name), at + '.snapshot_hash[' + name + ']')
                snapshot[name] = data
            self.totals['snapshot_members'] += len(names)
        config = self.read(directory / 'config.json')
        config_arg = option(command['command'], '--config')
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = Path(command['cwd']) / config_path
        key = _PATH_AUDIT.snapshot_source_key(config_path,self.root,'configs')
        self.check(key in snapshot,at,'Configuration missing from source snapshot')
        captured_config = _PATH_AUDIT.Auditor.snapshot_document(self,snapshot[key],key,command)
        self.equal(config, captured_config, at + '.config_vs_snapshot')
        self.check(canonical(config) == canonical(captured_config), at, 'Config has unexpected fields vs snapshot')
        _PATH_AUDIT.Auditor.path_sources(self,snapshot,command,status,at)
        serve = self.read(directory / 'server-command.json')
        interpreter = Path(command['command'][0])
        if not interpreter.is_absolute():
            interpreter = Path(command['cwd']) / interpreter
        self.equal(str(Path(serve['command'][0]).absolute()), str(interpreter.absolute().parent / 'vllm'), at + '.server_executable')
        prefix = [serve['command'][0], 'serve', command.get('path_context',{}).get('model_path',str(model_path())), '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', str(config.get('port', 8260)), *config['args']]
        self.equal(serve['command'][:len(prefix)], prefix, at + '.server_command')
        self.equal(serve['extra_environment'], {**config.get('env', {}), 'VLLM_SERVER_DEV_MODE': '1'}, at + '.server_environment')
        profiled = '--profile' in command['command']
        suffix = serve['command'][len(prefix):]
        launch_tree = ast.parse(snapshot[f'{self.root.name}/scripts/serve_scenarios.py'])
        has_details = any((isinstance(node, ast.If) and isinstance(node.test, ast.Name) and (node.test.id == 'agent_modes') and any((isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and (stmt.target.id == 'command') and isinstance(stmt.value, ast.List) and ([x.value for x in stmt.value.elts if isinstance(x, ast.Constant)] == ['--enable-prompt-tokens-details']) for stmt in node.body)) for node in ast.walk(launch_tree)))
        if option(command['command'], '--agent-modes', '') and has_details:
            self.equal(suffix[:1], ['--enable-prompt-tokens-details'], at + '.agent_prompt_details')
            suffix = suffix[1:]
        if profiled:
            self.check(len(suffix) == 2 and suffix[0] == '--profiler-config', at, 'Invalid profiler override')
        else:
            self.equal(suffix, [], at + '.unexpected_server_overrides')
        experiment = self.read(directory / 'experiment.json')
        expected_suites = [s for s in option(command['command'], '--suites', 'decode').split(',') if s]
        expected_modes = [s for s in option(command['command'], '--agent-modes', '').split(',') if s]
        self.equal(experiment['tasks'], expected_suites, at + '.tasks')
        self.equal(experiment['agent_modes'], expected_modes, at + '.agent_modes')
        success = status is not None and status.get('exit_code') == 0 and (not status.get('timed_out')) and (not status.get('supervisor_error'))
        if success:
            self.equal(experiment['status'], 'PASS', at + '.successful_supervisor_vs_server')
        expected_tasks = expected_suites + expected_modes
        clients = experiment['clients']
        self.equal([x['task'] for x in clients], expected_tasks[:len(clients)], at + '.client_order')
        if success:
            self.equal(len(clients), len(expected_tasks), at + '.complete_clients')
        reports = []
        listed = set()
        for c in clients:
            argv = c['command']
            report_path = Path(option(argv, '--output'))
            self.check(report_path.parent.resolve() == directory.resolve(), at, 'Client output outside experiment')
            listed.add(report_path.name)
            self.equal(option(argv, '--base-url'), f'http://127.0.0.1:{config.get('port', 8260)}', at + '.client_base_url')
            if not report_path.exists():
                if success or c['status'] == 'PASS':
                    self.check(False, at, 'Completed client has no report')
                else:
                    self.incomplete.append(str(report_path))
                continue
            if '--suite' in argv:
                reports.append(self.scenario(report_path, argv, snapshot, success))
            else:
                d = self.read(report_path)
                if 'mode' not in d:
                    self.check(not success, at, 'Successful experiment contains uninitialized Agent report')
                    self.incomplete.append(str(report_path) + ' initializing')
                    reports.append(dict(file=report_path.name, status='INITIALIZING'))
                    continue
                self.equal(d['client_sha256'], digest(snapshot[f'{self.root.name}/scripts/agent_benchmark.py']), at + '.agent_runtime_hash')
                for name, h in d['workload_sha256'].items():
                    keys = (f'experiments/{self.root.name}/workloads/agent/{name}',
                            f'{self.root.name}/workloads/agent/{name}')
                    key = next((key for key in keys if key in snapshot), None)
                    self.check(key is not None, at, 'Agent fixture missing from snapshot: '+name)
                    if key is not None:
                        self.equal(h, digest(snapshot[key]), at + '.agent_workload_hash')
                reports.append(dict(file=report_path.name, mode=d['mode'], status=d['status'], note='Provenance here; request/quality/agent summary audit in summarize_agent.py'))
            if c['status'] == 'PASS':
                self.equal(c.get('exit_code'), 0, at + '.client_exit')
                expected_report_status = 'PASSED' if '--suite' in argv else 'COMPLETED'
                self.equal(reports[-1]['status'], expected_report_status, at + '.client_status_vs_report')
        actual = {p.name for pattern in ('scenario-*.json', 'agent-*.json') for p in directory.glob(pattern)}
        self.equal(sorted(actual - listed), [], at + '.orphan_reports')
        self.totals['experiments'] += 1
        self.experiments.append(dict(id=at, supervisor_exit=status.get('exit_code') if status else None, timed_out=status.get('timed_out') if status else None, experiment_status=experiment['status'], reports=reports))

    def run(self):
        for directory in sorted(path_at(self.root, 'results').iterdir()):
            if not directory.is_dir():
                continue
            if not (directory / 'command.json').exists():
                self.warnings.append(dict(location=directory.name, detail='No command.json; not a started experiment'))
                continue
            try:
                launch = json.loads((directory / 'command.json').read_text())
                if not any(Path(arg).name == 'serve_scenarios.py' for arg in launch.get('command', [])):
                    continue
                self.experiment(directory)
            except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, zipfile.BadZipFile) as exc:
                self.errors.append(dict(location=directory.name, detail=f'Audit schema/read error: {type(exc).__name__}: {exc}'))
        for path, previous in self.read_stats.items():
            stat = path.stat()
            if previous != (stat.st_mtime_ns, stat.st_size):
                self.incomplete.append(str(path.resolve()) + ' changed during audit')
        return dict(status='FAIL' if self.errors else 'INCOMPLETE' if self.incomplete else 'PASS', scope='CPU read-only evidence consistency, not model accuracy, SLA or replayed GPU execution', audited_utc=datetime.now(timezone.utc).isoformat(), totals=dict(self.totals), errors=self.errors, warnings=self.warnings, incomplete=self.incomplete, experiments=self.experiments, workload_pairs=self.signatures, source_sha256=self.sources, limits=['Batch wall clock is taken from original monotonic measurement, not independently timed.', 'Equal provenance/runtime reported hashes do not constitute process attestation.', 'Recorded failures remain evidence; audit PASS does not turn them into successful experiments.'])


def self_test():
    """Explicit synthetic arithmetic/provenance negatives; never model results."""
    with tempfile.TemporaryDirectory(prefix='scenario-audit-offline-') as temp:
        root = Path(temp) / 'scenario-fixture'
        root.mkdir()
        ids = [1, 2]
        h = ids_hash(ids)
        item = dict(prompt_sha256=h, output_tokens=3, seed=20261105)
        row = dict(request_id='1/toy/warmup/0', prompt_sha256=h, input_tokens=2, requested_output_tokens=3, seed=20261105, ok=True, http_status=200, usage=dict(prompt_tokens=2, completion_tokens=3, total_tokens=5), ttft_s=0.1, e2e_s=0.5, tpot_s=0.2, error=None, text='x', text_sha256=digest(b'x'), text_event_seconds=[0.1, 0.3, 0.4], elapsed_s=0.51, done_received=True, planned_offset_s=0.0, dispatch_lag_s=0.0, scheduled_ttft_s=0.1, scheduled_e2e_s=0.5)
        stage = dict(phase='warmup', requests=[row], summary=recompute_summary([row], 0.6), concurrency_limit=1, planned_arrival_span_s=0.0, metrics_before='before.txt', metrics_after='after.txt')
        measure = copy.deepcopy(stage)
        measure['phase'] = 'measure'
        measure['requests'][0]['request_id'] = '1/toy/measure/0'
        source = b'# offline fixture, never executed\n'
        report = dict(suite='decode', status='PASSED', client_sha256=digest(source), protocol=dict(connections='fresh_per_request', retries=0), rounds=1, base_url='http://127.0.0.1:8260', tokenizer_sha256='offline', cases=[dict(name='toy', p=2, o=3, c=1, n=1)], seed=20261105, prompts={h: dict(token_ids=ids, input_tokens=2, sha256=h)}, banks={'1/toy/warmup': [item], '1/toy/measure': [item]}, runs=[dict(round=1, case='toy', status='PASSED', stages=[stage, measure])])
        path = root / 'scenario-decode.json'
        write_text(root / 'before.txt', '# offline\n')
        write_text(root / 'after.txt', '# offline\n')
        command = ['python', '--suite', 'decode', '--rounds', '1', '--base-url', report['base_url']]
        snapshots = {root.name + '/scripts/scenario_client.py': source, 'scripts/benchmark_client.py': b'offline'}

        def evaluate(d):
            write_text(path, json.dumps(d))
            audit = Auditor(root)
            audit.scenario(path, command, snapshots, True)
            return audit.errors
        assert not evaluate(report)
        labels = []
        for label in ('summary', 'prompt_hash', 'usage', 'runtime_hash', 'missing_phase'):
            bad = copy.deepcopy(report)
            if label == 'summary':
                bad['runs'][0]['stages'][0]['summary']['output_tokens_per_s'] += 1
            elif label == 'prompt_hash':
                bad['prompts'][h]['token_ids'][0] = 9
            elif label == 'usage':
                bad['runs'][0]['stages'][0]['requests'][0]['usage']['total_tokens'] = 99
            elif label == 'runtime_hash':
                bad['client_sha256'] = 'bad'
            else:
                bad['runs'][0]['stages'].pop()
            assert evaluate(bad), label
            labels.append(label)
    return dict(status='PASS', scope='CPU-only synthetic auditor self-test; no model/HTTP/GPU', corruption_cases_detected=labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    report = self_test() if args.self_test else Auditor(args.root).run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open_output(args.output, 'x') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(json.dumps({k: report[k] for k in ('status', 'totals', 'errors', 'incomplete', 'corruption_cases_detected') if k in report}, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get('errors') else 2 if report.get('incomplete') and (not args.allow_incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
