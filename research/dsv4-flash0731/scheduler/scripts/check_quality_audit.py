#!/usr/bin/env python3
"""CPU independent sidecar audit controls; real frozen client with mock HTTP."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D/'scripts'))
import audit_study as audit
import audit_quality as quality
import run_logged
import serve_study


def run():
    spec = importlib.util.spec_from_file_location('quality_audit_real_legacy', D.parent / 'scenarios/scripts/agent_benchmark.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    snapshots = {str(p.relative_to(D.parent)): p.read_bytes() for p in run_logged.quality_fixture_inputs(D.parent)}
    snapshots['scenarios/scripts/agent_benchmark.py'] = Path(client.__file__).read_bytes()
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append(dict(name=name, status='PASS'))
        except Exception as e:
            checks.append(dict(name=name, status='FAIL', error=repr(e)))
    with tempfile.TemporaryDirectory(prefix='d-quality-audit-MOCK-') as tmp:
        root = Path(tmp)

        def fixture(name, protocol_error=False):
            directory = root / name
            directory.mkdir()
            plan = dict(base_url='http://127.0.0.1:8260', model='dsv4', phases=[dict(role='measure')])
            cfg = dict(quality_probe=dict(rounds=2, max_tokens=128, timeout_s=180))
            cmd = serve_study.quality_command(cfg, plan, directory)

            def http(url, path, payload, timeout):
                response = dict(choices=[dict(message=dict(role='assistant', content='MOCK incorrect answer'), finish_reason='stop')], usage=dict(prompt_tokens=10, completion_tokens=3, total_tokens=13))
                return dict(request=deepcopy(payload), attempts=1, ttft_s=None, tpot_s=None, e2e_s=0.25, protocol_ok=not protocol_error, http_status=503 if protocol_error else 200, response_truncated=False, raw_response=json.dumps(response), response=response, error=dict(type='MOCK', message='protocol fault') if protocol_error else None)
            client.http_post = http
            old = sys.argv
            try:
                sys.argv = cmd[1:]
                with open_output(directory / 'quality.log', 'x') as log, contextlib.redirect_stdout(log):
                    code = client.main()
            finally:
                sys.argv = old
            side = dict(command=cmd, status='FAILED' if protocol_error else 'PASS', exit_code=code, report=str(directory / 'quality.json'), log=str(directory / 'quality.log'), report_sha256=quality.sha((directory / 'quality.json').read_bytes()), excluded_from_performance=True, quality_scores_gate_performance=False, started_utc='2026-11-07T01:00:01+00:00')
            performance = dict(plan=plan, finished_utc='2026-11-07T01:00:00+00:00', status='PASSED')
            return (directory, performance, cfg, dict(quality_probe=side), snapshots, not protocol_error)
        arguments = fixture('wrong-answer')

        def inspect(args):
            a = audit.Auditor(D)
            try:
                result = quality.audit(a, *args)
            except Exception as e:
                a.check(False, 'MOCK quality', repr(e))
                result = None
            return (result, a.errors)

        def accepted():
            result, errors = inspect(arguments)
            assert not errors, errors
            assert result['quality_passed'] == 0 and result['requests'] == 16 and (result['protocol_successes'] == 16)
        check('all incorrect natural-EOS answers audit consistently without gating performance', accepted)

        def mutation(change_report=None, change_args=None):
            args = list(arguments)
            args[1:] = deepcopy(args[1:])
            path = args[0] / 'quality.json'
            original = path.read_bytes()
            try:
                if change_report:
                    obj = json.loads(original)
                    change_report(obj)
                    write_text(path, json.dumps(obj))
                    args[3]['quality_probe']['report_sha256'] = quality.sha(path.read_bytes())
                if change_args:
                    change_args(args)
                _, errors = inspect(args)
                assert errors, 'Malformed sidecar accepted'
            finally:
                write_bytes(path, original)
        cases = [('request forces EOS ignore', lambda q: q['runs'][0]['http']['request'].update(ignore_eos=True)), ('score forgery', lambda q: q['runs'][0]['quality'].update(passed=True, content_matched=True)), ('summary forgery', lambda q: q['summary'].update(quality_passed=16)), ('message hash mismatch', lambda q: q['runs'][0].update(request_messages_sha256='bad')), ('case order mismatch', lambda q: q['runs'][0].update(case_id='other')), ('raw response diverges', lambda q: q['runs'][0]['http'].update(raw_response='{}')), ('retry hidden', lambda q: q['runs'][0]['http'].update(attempts=2)), ('incomplete yet successful', lambda q: q['runs'].pop()), ('fabricated TTFT', lambda q: q['summary'].update(ttft_s=0.001)), ('fixture hash mismatch', lambda q: q['workload_sha256'].update(quality_json='bad'))]
        for name, fn in cases:
            check('reject ' + name, lambda fn=fn: mutation(change_report=fn))
        check('reject quality overlapping completed performance', lambda: mutation(change_args=lambda a: a[3]['quality_probe'].update(started_utc='2026-11-07T00:59:59+00:00')))
        check('reject changed snapshot probe bytes', lambda: mutation(change_args=lambda a: a[4].update({'scenarios/scripts/agent_benchmark.py': b'MOCK changed'})))

        def failed():
            args = fixture('protocol-failed', True)
            result, errors = inspect(args)
            assert not errors, errors
            assert result['protocol_failures'] == 16 and result['execution_status'] == 'FAILED' and (args[1]['status'] == 'PASSED')
        check('failed quality protocol preserved distinctly from PASSED performance', failed)

        def notstarted():
            args = list(arguments)
            args[3] = {}
            args[-1] = False
            result, errors = inspect(args)
            assert not errors and result['status'] == 'NOT_STARTED'
        check('failed harness before quality does not invent requests', notstarted)
    return dict(status='PASS' if all((c['status'] == 'PASS' for c in checks)) else 'FAIL', checks=checks, scope='CPU only; real frozen probe payload/summary plus mocked HTTP, independent sidecar recomputation; no GPU/network/model', source_sha256={str(p): quality.sha(p.read_bytes()) for p in (Path(__file__), Path(audit.__file__), Path(quality.__file__), Path(client.__file__))})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use unique output')
    result = run()
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), failures=[c for c in result['checks'] if c['status'] != 'PASS'])))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
