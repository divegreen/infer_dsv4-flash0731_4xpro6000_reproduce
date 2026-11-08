#!/usr/bin/env python3
"""CPU-only mock validation of the bounded local Profile export timeout revision."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import asyncio
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import serve_study as h


def run():
    checks = []

    def check(name, call):
        try:
            call()
            checks.append(dict(name=name, status='PASS'))
        except Exception as exc:
            checks.append(dict(name=name, status='FAIL', error=repr(exc)))

    def eq(a, b):
        assert a == b, (a, b)

    def reject(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError('invalid configuration accepted')
    profile = dict(phases=[dict(role='profile')])
    normal = dict(phases=[dict(role='measure')])
    check('old profile default 120', lambda: eq(h.profile_control_timeout({}, profile), 120.0))
    check('old normal default 120', lambda: eq(h.profile_control_timeout({}, normal), 120.0))
    check('new bounded 600 accepted', lambda: eq(h.profile_control_timeout({'profile_control_timeout_s': 600}, profile), 600.0))
    for v in (0, -1, 601, True, '600', None, float('inf'), float('nan')):
        check('reject invalid ' + repr(v), lambda v=v: reject(lambda: h.profile_control_timeout({'profile_control_timeout_s': v}, profile)))
    check('reject explicit normal override', lambda: reject(lambda: h.profile_control_timeout({'profile_control_timeout_s': 600}, normal)))

    class Response:
        status = 200

        async def text(self):
            return ''

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:

        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    async def control_round(value):
        session = Session()
        capture = dict(controls=[])
        for path in ('/start_profile', '/stop_profile'):
            await h.control(session, 'http://127.0.0.1:8260', path, None, capture, lambda: None, value)
        eq([c[1]['timeout'].total for c in session.calls], [value, value])
        assert all((c[1]['allow_redirects'] is False for c in session.calls))
        eq([c['status'] for c in capture['controls']], ['PASS', 'PASS'])
    for value in (120.0, 600.0):
        check('mock aiohttp timeout propagation ' + str(value), lambda value=value: asyncio.run(control_round(value)))

    def source_contract():
        tree = ast.parse(Path(h.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == 'execute_plan')]
        eq(len(calls), 1)
        arg = next((k.value for k in calls[0].keywords if k.arg == 'control_timeout_s'))
        eq(ast.unparse(arg), 'profile_control_timeout(config, plan)')
    check('main dispatch uses validated configuration', source_contract)

    def original_config():
        for name in ('scheduler-tp4-observe-dense-profile', 'scheduler-tp4-fairshare-dense-profile'):
            cfg = load_document(path_at(ROOT, 'configs') / f'{name}.json')
            plan = h.validate_study_plan(load_document(path_at(ROOT, 'workloads') / f'{name}.json'))
            command = h.server_command(cfg, plan, path_at(ROOT, 'results') / name)
            eq(h.profile_control_timeout(cfg, plan), 120.0)
            changed = deepcopy(cfg)
            changed['profile_control_timeout_s'] = 600
            eq(h.server_command(changed, plan, path_at(ROOT, 'results') / name), command)
            eq(plan['timeout_s'], 180)
    check('old actual commands and request timeout unchanged', original_config)
    return dict(status='PASS' if all((c['status'] == 'PASS' for c in checks)) else 'FAIL', checks=checks, scope='CPU mock only; no live HTTP/GPU and no claim that 600 seconds suffices for actual export', source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(h.__file__))})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run()
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), failures=[x for x in result['checks'] if x['status'] != 'PASS'], source_sha256=result['source_sha256'])))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
