#!/usr/bin/env python3
"""CPU-only quality sidecar controls with the real frozen probe and mocked HTTP.

No model/GPU/network request. Never executes historical Agent tools or self-test.
Can validate an offline proposed harness/supervisor before runtime is unfrozen.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import contextlib
from copy import deepcopy
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D/'scripts'))

def load(name,path):
    loader=importlib.machinery.SourceFileLoader(name,str(path));spec=importlib.util.spec_from_loader(name,loader)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;loader.exec_module(module);return module

def run(harness_path, supervisor_path):
    h = load('quality_harness_checked', harness_path)
    r = load('quality_supervisor_checked', supervisor_path)
    legacy_path = D.parent / 'scenarios/scripts/agent_benchmark.py'
    q = load('quality_frozen_probe_checked', legacy_path)
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append(dict(name=name, status='PASS'))
        except Exception as exc:
            checks.append(dict(name=name, status='FAIL', error=repr(exc)))

    def reject(fn):
        try:
            fn()
        except (ValueError, RuntimeError, FileExistsError):
            return
        raise AssertionError('Invalid input/failed execution accepted')
    spec = dict(quality_probe=dict(rounds=2, max_tokens=128, timeout_s=180))
    plan = dict(base_url='http://127.0.0.1:8260', model='dsv4', phases=[dict(role='measure')])

    def no_probe():
        assert h.quality_command({}, plan, Path('/MOCK')) is None
    check('historical config without quality remains optional', no_probe)
    for key, value in [('rounds', 3), ('max_tokens', 256), ('timeout_s', 600), ('rounds', 2.0), ('unexpected', 1)]:

        def invalid(key=key, value=value):
            config = deepcopy(spec)
            config['quality_probe'][key] = value
            reject(lambda: h.quality_command(config, plan, Path('/MOCK')))
        check('reject unregistered quality spec ' + key + '=' + repr(value), invalid)
    check('profile and quality cannot be combined', lambda: reject(lambda: h.quality_command(spec, dict(plan, phases=[dict(role='profile')]), Path('/MOCK'))))
    with tempfile.TemporaryDirectory(prefix='d-quality-MOCK-') as temp:
        root = Path(temp)

        def exercise(protocol_error=False):
            out = root / ('protocol-failure' if protocol_error else 'wrong-answer')
            out.mkdir()
            capacity = dict(status='PASSED', runs=[dict(MOCK_performance=True)])
            raw = json.dumps(capacity).encode()
            write_bytes(out / 'capacity.json', raw)
            events = []
            payloads = []
            experiment = dict(status='READY', clients=[dict(task='capacity-study', exit_code=0)])

            def fake_http(url, path, payload, timeout):
                assert url == plan['base_url'] and path == '/v1/chat/completions' and (timeout == 180)
                payloads.append(payload)
                if protocol_error:
                    return dict(protocol_ok=False, e2e_s=0.01, attempts=1, error=dict(type='MOCK', message='synthetic protocol fault'))
                return dict(protocol_ok=True, e2e_s=0.01, attempts=1, response=dict(choices=[dict(message=dict(role='assistant', content='MOCK_WRONG_ANSWER'), finish_reason='stop')], usage=dict(prompt_tokens=12, completion_tokens=1, total_tokens=13)))
            q.http_post = fake_http

            def fake_child(command, **kwargs):
                assert command[1] == str(legacy_path) and kwargs['check'] is False and ('start_new_session' not in kwargs)
                old = sys.argv
                try:
                    sys.argv = command[1:]
                    with contextlib.redirect_stdout(kwargs['stdout']):
                        code = q.main()
                finally:
                    sys.argv = old
                return SimpleNamespace(returncode=code)
            call = lambda: h.execute_quality(spec, plan, out, experiment, lambda: events.append(deepcopy(experiment)), run_child=fake_child)
            if protocol_error:
                reject(call)
            else:
                call()
            assert len(payloads) == 16 and all((x['stream'] is False and x['max_tokens'] == 128 and ('ignore_eos' not in x) and ('tools' not in x) for x in payloads))
            assert (out / 'capacity.json').read_bytes() == raw and len(experiment['clients']) == 1
            report = load_document(out / 'quality.json')
            assert report['summary']['quality_passed'] == 0 and report['summary']['ttft_s'] is None and (report['summary']['tpot_s'] is None)
            side = experiment['quality_probe']
            assert side['excluded_from_performance'] and (not side['quality_scores_gate_performance'])
            assert events[0]['quality_probe']['status'] == 'RUNNING' and side['finished_utc']
            if protocol_error:
                assert side['status'] == 'FAILED' and side['exit_code'] == 1 and (report['status'] == 'COMPLETED_WITH_PROTOCOL_ERRORS')
            else:
                assert side['status'] == 'PASS' and side['exit_code'] == 0 and (report['status'] == 'COMPLETED')
                actual = {str(p.relative_to(q.FIXTURES)): hashlib.sha256(p.read_bytes()).hexdigest() for p in r.quality_fixture_inputs(D.parent)}
                assert actual == report['workload_sha256']
            return out
        check('16 real-probe mocked natural-EOS calls; all wrong answers accepted; fixture SHA closure and performance separation', lambda: exercise(False))
        check('16 real-probe mocked protocol faults retained and raised; performance bytes untouched', lambda: exercise(True))

        def unique():
            out = root / 'wrong-answer'
            raw = (out / 'quality.json').read_bytes()
            reject(lambda: h.execute_quality(spec, plan, out, {}, lambda: None, run_child=lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not run'))))
            assert (out / 'quality.json').read_bytes() == raw
        check('existing quality log/report never overwritten', unique)

        def fixture_set():
            temp_research = root / 'snapshot'
            directory = path_at(temp_research, 'scenarios/workloads/agent')
            for p in r.quality_fixture_inputs(D.parent):
                dest = directory / p.relative_to(q.FIXTURES)
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_bytes(dest, p.read_bytes())
            write_text(directory / 'self-test-results-UNRELATED.json', '{}')
            paths = r.quality_fixture_inputs(temp_research)
            assert len(paths) == 12 and (not any(('UNRELATED' in str(p) for p in paths)))
            (directory / 'quality.json').unlink()
            reject(lambda: r.quality_fixture_inputs(temp_research))
        check('fixture closure excludes historical self-test reports and requires quality metadata', fixture_set)
    return dict(status='PASS' if all((x['status'] == 'PASS' for x in checks)) else 'FAIL', checks=checks, scope='Real frozen quality client with injected CPU MOCK HTTP responses; no network/model/GPU, no general model quality claim', source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), harness_path, supervisor_path, legacy_path)})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--harness', type=Path, default=D / 'scripts/serve_study.py')
    p.add_argument('--supervisor', type=Path, default=D / 'scripts/run_logged.py')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use unique output')
    result = run(args.harness.resolve(), args.supervisor.resolve())
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), failures=[c for c in result['checks'] if c['status'] != 'PASS'])))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
