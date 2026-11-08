#!/usr/bin/env python3
"""Independent CPU launcher preflight. No server, GPU query or HTTP is run."""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent/'scenarios'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    launcher = ROOT / 'scripts/serve_agent_sessions.py'
    runner = ROOT / 'scripts/run_agent_series.py'
    supervisor = ROOT / 'scripts/run_logged.py'
    owned_sources = [launcher, runner, supervisor]
    before = {str(p): sha(p) for p in owned_sources}
    spec = importlib.util.spec_from_file_location('stagec_preflight_series', runner)
    series = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(series)
    pilot = path_at(ROOT, 'workloads/sessions/session-tp4-apc-pilot.json')
    pilot_config = path_at(ROOT, 'configs/session-tp4-apc-pilot.json')
    series_path = path_at(ROOT, 'workloads/sessions/series-pilot.json')
    plan = load_document(pilot)
    config = load_document(pilot_config)
    job = load_document(series_path)['jobs'][0]
    checks = []
    commands = []

    def check(label, value):
        assert value, label
        checks.append(label)

    def rejected(label, fn):
        try:
            fn()
        except (ValueError, RuntimeError, KeyError, FileNotFoundError):
            checks.append(label)
        else:
            raise AssertionError(label + ' unexpectedly accepted')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='stagec-preflight-') as temp:
        tmp = Path(temp)

        def call(label, argv, success):
            assert argv[0] == sys.executable
            assert argv[1] == str(launcher) and '--validate-only' in argv or (argv[1] == str(runner) and '--dry-run' in argv)
            proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
            commands.append(dict(label=label, command=argv, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
            check(label, (proc.returncode == 0) == success)
            return proc
        positive = call('actual pilot validate-only succeeds', [sys.executable, str(launcher), '--config', str(pilot_config), '--plan', str(pilot), '--validate-only'], True)
        rendered = json.loads(positive.stdout)
        cmd = rendered['command']
        check('rendered command owns loopback and configured port', cmd[cmd.index('--host') + 1] == '127.0.0.1' and cmd[cmd.index('--port') + 1] == '8260')
        check('actual command enables cached-token usage exactly once', cmd.count('--enable-prompt-tokens-details') == 1)
        check('all three actual pilot phases retained', rendered['phases'] == [p['id'] for p in plan['phases']])
        state = tmp / 'should-not-create-state.json'
        call('actual series dry-run succeeds', [sys.executable, str(runner), '--plan', str(series_path), '--state', str(state), '--dry-run'], True)
        check('dry-run never creates state', not state.exists())
        variants = []
        c = copy.deepcopy(config)
        c['args'] += ['--host', '0.0.0.0']
        variants.append(('reject host argument override', c, plan))
        c = copy.deepcopy(config)
        c['args'] += ['--port=8261']
        variants.append(('reject port argument override', c, plan))
        c = copy.deepcopy(config)
        c['port'] = 8261
        variants.append(('reject mismatched owned port', c, plan))
        c = copy.deepcopy(config)
        c['env'].pop('VLLM_SERVER_DEV_MODE')
        variants.append(('reject missing reset mode', c, plan))
        for name in ('CUDA_VISIBLE_DEVICES', 'RESEARCH_RUN_DIR', 'HF_HUB_OFFLINE'):
            c = copy.deepcopy(config)
            c['env'][name] = 'test-override'
            variants.append((f'reject protected {name}', c, plan))
        p = copy.deepcopy(plan)
        p['base_url'] = 'http://example.invalid:8260'
        variants.append(('reject external URL before I/O', config, p))
        p = copy.deepcopy(plan)
        p['model'] = 'other-model'
        variants.append(('reject served alias mismatch', config, p))
        p = copy.deepcopy(plan)
        for phase in p['phases']:
            phase['dp_routing'] = 'sticky4'
        variants.append(('reject sticky4 on TP4 deployment', config, p))
        for i, (label, c, p) in enumerate(variants):
            cf = tmp / f'config-{i}.json'
            pf = tmp / f'plan-{i}.json'
            write_text(cf, json.dumps(c))
            write_text(pf, json.dumps(p))
            call(label, [sys.executable, str(launcher), '--config', str(cf), '--plan', str(pf), '--validate-only'], False)
        c = copy.deepcopy(config)
        c['args'] += ['--enable-prompt-tokens-details']
        cf = tmp / 'details-explicit.json'
        write_text(cf, json.dumps(c))
        explicit = call('explicit cached-token usage flag remains valid', [sys.executable, str(launcher), '--config', str(cf), '--plan', str(pilot), '--validate-only'], True)
        check('explicit usage flag not duplicated', json.loads(explicit.stdout)['command'].count('--enable-prompt-tokens-details') == 1)
        for label, change in [('reject escaping config', dict(config='../outside.json')), ('reject escaping plan', dict(session_plan='../../outside.json')), ('reject nested experiment ID', dict(id='../outside')), ('reject plan/job identity mismatch', dict(id='another-id'))]:
            altered = dict(job, **change)
            rejected(label, lambda j=altered: series.command(j))
        repeated = tmp / 'repeated.json'
        write_text(repeated, json.dumps(dict(jobs=[job, job])))
        call('reject duplicate series IDs before state creation', [sys.executable, str(runner), '--plan', str(repeated), '--state', str(state), '--dry-run'], False)
        check('failed dry-run leaves state absent', not state.exists())
    # Calling the pure snapshot API avoids executing the supervisor's GPU,
    # locking or process lifecycle operations.
    snapshot_sources = series.supervisor.source_inputs(ROOT)
    snap = {key: sha(path) for key, path in snapshot_sources.items()}
    required = series.resume_sources(job)
    check('all resume dependencies included in inherited source snapshot', set(required) <= set(snap))
    series.verify_resume_sources(job, snap)
    checks.append('complete matching snapshot passes resume identity check')
    groups = ['scenarios/scripts/capacity_client.py', 'scripts/benchmark_client.py', 'scenarios/scripts/run_logged.py', 'scenarios/scripts/run_agent_series.py', 'scenarios/scripts/agent_session_tasks.py', 'metadata/environment.json', 'metadata/model-inventory.json', 'experiments/scenarios/workloads/sessions/tasks/catalog.json', 'project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py']
    for name in groups:
        altered = dict(snap)
        altered[name] = '0' * 64
        rejected('resume rejects changed ' + name, lambda d=altered: series.verify_resume_sources(job, d))
    fixture = next((k for k in required if '/tasks/' in k and k.endswith('/public/README.md')))
    altered = dict(snap)
    altered.pop(fixture)
    rejected('resume rejects newly added or missing snapshot fixture', lambda: series.verify_resume_sources(job, altered))
    altered = dict(snap)
    altered['experiments/scenarios/workloads/sessions/tasks/old-deleted-fixture.txt'] = '0' * 64
    rejected('resume rejects deleted fixture from older snapshot', lambda: series.verify_resume_sources(job, altered))
    altered = dict(snap)
    altered.pop('scripts/benchmark_client.py')
    rejected('resume rejects missing runtime helper hash', lambda: series.verify_resume_sources(job, altered))
    check('launcher/series/supervisor unchanged during checks', before == {str(p): sha(p) for p in owned_sources})
    check('no vllm or torch import', 'vllm' not in sys.modules and 'torch' not in sys.modules)
    result = dict(status='PASS_CPU_PREFLIGHT_GPU_NOT_RUN', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, commands=commands, source_sha256=before, actual_plan_sha256=sha(pilot), actual_config_sha256=sha(pilot_config), actual_series_sha256=sha(series_path), script_sha256=sha(Path(__file__)), resume_required_sources=required, snapshot_source_count=len(snap), snapshot_fixture_count=sum(('/workloads/sessions/tasks/' in k for k in snap)), limits=['No run_logged main, GPU query, port bind, HTTP, server start or live lock acquisition.', 'Snapshot construction and resume hash checks executed; actual process cleanup is static review and inherited A/B evidence only.', 'CLI validation does not prove vLLM accepts all model flags or that runtime tools/tasks produce correct results.', 'Existing-result skip still requires separate data audit; hash identity alone is not a successful inference claim.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), cli_commands=len(commands), resume_sources=len(required), snapshot_sources=len(snap), snapshot_task_files=result['snapshot_fixture_count'], output=str(args.output)), indent=2))


if __name__=='__main__':main()
