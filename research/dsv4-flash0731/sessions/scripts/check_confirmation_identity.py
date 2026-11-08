#!/usr/bin/env python3
"""Targeted CPU discovery/grouping controls for unknown started-window identity.

Temporary fixtures explicitly stub raw audit and stability to isolate grouping.
They are not model measurements or an independent proof of stability arithmetic.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

SOURCE = Path(__file__).with_name('audit_session_confirmations.py')
spec = importlib.util.spec_from_file_location('confirmation_identity_review', SOURCE)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def review():
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    phase = dict(id='probe', kind='replay', arrival='poisson', role='confirm', rate=0.2, duration_s=300, profiles=['coding', 'research', 'data'], cache_scope='session', dp_routing='none', seed=1)
    plan = dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64, model='dsv4', phases=[phase])
    config = dict(name='MOCK', args=['--tensor-parallel-size', '4'], env={'VLLM_SERVER_DEV_MODE': '1'})
    required = ['scenarios/scripts/agent_session_client.py', 'scenarios/scripts/capacity_client.py', 'scripts/benchmark_client.py', 'scenarios/scripts/serve_agent_sessions.py', 'scenarios/scripts/run_logged.py', 'environment.json', 'model-inventory.json']
    command = dict(source_sha256={k: 'a' * 64 for k in required}, environment={'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    report = dict(status='COMPLETED', plan=plan, tokenizer_sha256='b' * 64, transport='fresh_tcp_no_retry')
    key, _, missing = m.signature(config, phase, command, report)
    check('complete replay identity has no missing fields', not missing)
    for field in ('tokenizer_sha256', 'transport'):
        absent = {k: v for k, v in report.items() if k != field}
        _, _, missing = m.signature(config, phase, command, absent)
        check('missing ' + field + ' explicit', 'report.' + field in missing)
    for value in (None, {}):
        _, _, missing = m.signature(config, phase, {**command, 'environment': value}, report)
        check('missing environment ' + repr(value), 'command.environment' in missing)
    _, _, missing = m.signature(config, phase, {**command, 'environment': {'OTHER': 'mock'}}, report)
    check('GPU assignment identity required', 'command.environment.CUDA_VISIBLE_DEVICES' in missing)
    _, _, missing = m.signature({}, phase, command, report)
    check('missing config explicit', 'config' in missing)
    for field in ('timeout_s', 'session_timeout_s', 'max_active_sessions', 'model'):
        _, _, missing = m.signature(config, phase, command, {**report, 'plan': {k: v for k, v in plan.items() if k != field}})
        check('missing global ' + field, 'report.plan.' + field in missing)
    agent = {**phase, 'kind': 'agent', 'tasks': ['mock_task']}
    fixture = 'scenarios/workloads/sessions/tasks/'
    helper = 'scenarios/scripts/agent_session_tasks.py'
    catalog = {'tasks': [dict(id='mock_task', kind='coding', readable=['README.md', 'code.py'])]}
    task_keys = [fixture + 'catalog.json', helper, fixture + 'mock_task/public/README.md', fixture + 'mock_task/public/code.py', fixture + 'mock_task/private/grading_cases.json', fixture + 'mock_task/private/reference.json']
    agentcommand = {**command, 'source_sha256': {**command['source_sha256'], **{k: 'c' * 64 for k in task_keys}}}
    _, _, missing = m.signature(config, agent, agentcommand, report, catalog)
    check('complete selected task catalog identity accepted', not missing)
    for absent in task_keys:
        changed = {**agentcommand, 'source_sha256': {k: v for k, v in agentcommand['source_sha256'].items() if k != absent}}
        _, _, missing = m.signature(config, agent, changed, report, catalog)
        check('required task dependency cannot disappear ' + absent, absent in missing)
    _, _, missing = m.signature(config, agent, agentcommand, report, None)
    check('unverified catalog cannot establish completeness', 'launch_task_catalog' in missing)
    data_catalog = {'tasks': [dict(id='mock_task', kind='data', readable=['README.md', 'data.csv'])]}
    data_keys = [fixture + 'catalog.json', helper, fixture + 'mock_task/public/README.md', fixture + 'mock_task/public/data.csv', fixture + 'mock_task/private/answers.json']
    dc = {**command, 'source_sha256': {**command['source_sha256'], **{k: 'd' * 64 for k in data_keys}}}
    _, _, missing = m.signature(config, agent, dc, report, data_catalog)
    check('data objective answer identity accepted', not missing)
    del dc['source_sha256'][fixture + 'mock_task/private/answers.json']
    _, _, missing = m.signature(config, agent, dc, report, data_catalog)
    check('data objective answer cannot disappear', fixture + 'mock_task/private/answers.json' in missing)
    old_audit, old_stability = (m.independent.audit, m.independent.stability)
    m.independent.audit = lambda *a, **kw: {'scope': 'MOCK RAW AUDIT STUB'}
    m.independent.stability = lambda *a, **kw: {'classification': 'STABLE_IN_WINDOW', 'scope': 'MOCK STABILITY STUB'}
    try:
        with tempfile.TemporaryDirectory(prefix='confirmation-identity-controls-') as td:
            root = Path(td)

            def write_run(i, full=True, report_override=None):
                d = root / f'c9{i}-MOCK'
                d.mkdir(exist_ok=True)
                f = {**phase, 'seed': i}
                pl = {**plan, 'phases': [f]}

                def put(name, obj):
                    write_text(d / name, json.dumps(obj))
                put('session-plan.json', pl)
                put('command.json', command)
                put('config.json', {**config, 'name': d.name})
                (d / 'sessions-probe').mkdir(exist_ok=True)
                if full:
                    put('status.json', dict(end_utc='MOCK-TERMINAL', exit_code=0, timed_out=False))
                    put('experiment.json', {'status': 'PASS'})
                    put('sessions.json', {**report, 'plan': pl, 'phases': [dict(phase=f, sessions=[])], **(report_override or {})})
                return d
            write_run(1)
            write_run(2)
            healthy = m.scan(root, {})
            check('healthy two-seed group still confirms', healthy['confirmed_protocol_points'] == 1 and healthy['started_confirmation_windows'] == 2 and (not healthy['unresolved_started_identity']))
            third = write_run(3, False)
            incomplete = m.scan(root, {})
            check('incomplete raw directory remains discovered', incomplete['started_confirmation_windows'] == 3)
            check('unknown group cannot bypass healthy group', incomplete['confirmed_protocol_points'] == 0 and len(incomplete['unresolved_started_identity']) == 1)
            check('explicit conservative identity blocker', all((any(('Unresolved started confirmation identity' in b for b in g['confirmation']['blockers'])) for g in incomplete['groups'])))
            write_run(3, True, {'tokenizer_sha256': None})
            result = m.scan(root, {})
            check('missing tokenizer alone blocks confirmation', result['confirmed_protocol_points'] == 0)
            write_run(3, True, {'transport': None})
            result = m.scan(root, {})
            check('missing transport alone blocks confirmation', result['confirmed_protocol_points'] == 0)
            write_run(3, True)
            restored = m.scan(root, {})
            check('restored proven identity includes all three seeds', restored['confirmed_protocol_points'] == 1 and restored['groups'][0]['confirmation']['distinct_seeds'] == 3 and (restored['groups'][0]['confirmation']['started_windows'] == 3))
            write_run(3, True, {'tokenizer_sha256': 'e' * 64})
            separated = m.scan(root, {})
            check('known incompatible treatment stays visibly separate', separated['confirmed_protocol_points'] == 1 and len(separated['groups']) == 2 and (not separated['unresolved_started_identity']))
            for label, change in [('empty_required_hash', dict(source_sha256={**command['source_sha256'], required[0]: ''})), ('short_required_hash', dict(source_sha256={**command['source_sha256'], required[0]: 'aaa'})), ('nonhex_required_hash', dict(source_sha256={**command['source_sha256'], required[0]: 'g' * 64})), ('boolean_environment', dict(environment={'CUDA_VISIBLE_DEVICES': True}))]:
                write_run(3, True)
                write_text(third / 'command.json', json.dumps({**command, **change}))
                result = m.scan(root, {})
                check(label + ' cannot split away', result['confirmed_protocol_points'] == 0 and result['unresolved_started_identity'])
            write_run(3, True)
            write_text(third / 'config.json', json.dumps({'name': third.name}))
            result = m.scan(root, {})
            check('name-only configuration cannot split away', result['confirmed_protocol_points'] == 0 and result['unresolved_started_identity'])
            for label, change in [('bad_tokenizer', {'tokenizer_sha256': 'not-a-digest'}), ('boolean_transport', {'transport': True}), ('boolean_timeout', {'plan': {**plan, 'timeout_s': True}}), ('zero_session_timeout', {'plan': {**plan, 'session_timeout_s': 0}}), ('fractional_admission', {'plan': {**plan, 'max_active_sessions': 2.5}}), ('empty_model', {'plan': {**plan, 'model': ''}})]:
                write_run(3, True, change)
                result = m.scan(root, {})
                check(label + ' cannot split away', result['confirmed_protocol_points'] == 0 and result['unresolved_started_identity'])
            write_run(3, True, {'tokenizer_sha256': 'e' * 64})

            def reject_third(directory, ledger):
                if directory == third:
                    raise ValueError('MOCK terminal raw identity cannot be verified')
                return {'scope': 'MOCK RAW AUDIT STUB'}
            m.independent.audit = reject_third
            result = m.scan(root, {})
            check('valid-looking but unverified terminal identity cannot split away', result['confirmed_protocol_points'] == 0 and 'terminal_raw_identity_not_verified' in result['unresolved_started_identity'][0]['missing_identity_fields'])
    finally:
        m.independent.audit, m.independent.stability = (old_audit, old_stability)
    import zipfile
    resultdir = path_at(SOURCE.parents[2], 'scenarios/results/session-tp4-agent-open')
    realcommand = load_document(resultdir / 'command.json')
    realconfig = load_document(resultdir / 'config.json')
    realplan = load_document(resultdir / 'session-plan.json')
    with zipfile.ZipFile(resultdir / 'source-snapshot.zip') as z:
        catalog_raw = z.read(fixture + 'catalog.json')
    check('actual launch catalog hash matches', hashlib.sha256(catalog_raw).hexdigest() == realcommand['source_sha256'][fixture + 'catalog.json'])
    realcatalog = json.loads(catalog_raw)
    actual = load_document(path_at(SOURCE.parent.parent, 'logs/audit-session-tp4-agent-open-r1.json'))
    realreport = load_document(resultdir / 'sessions.json')
    identity_report = {key: realreport[key] for key in ('plan', 'tokenizer_sha256', 'transport')}
    check('actual c10 normalized plan unchanged', identity_report['plan'] == realplan)
    for f in realplan['phases']:
        if f['kind'] != 'agent':
            continue
        _, _, missing = m.signature(realconfig, f, realcommand, identity_report, realcatalog)
        check('actual c10 selected task source closure ' + f['id'], not missing)
    check('actual c10 remains a screen', all((f['role'] != 'confirm' for f in realplan['phases'])) and actual['status'] == 'PASS')
    return dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, reviewed_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Temporary CPU grouping fixtures with explicit raw/stability stubs and actual completed c10 source-closure metadata; no GPU/model/task execution.', limitations=['Synthetic stable labels isolate grouping only; frozen independent raw auditor retains stability arithmetic responsibility.', 'Any unresolved started identity conservatively blocks every confirmation in the current scan until resolved; this can withhold unrelated points but cannot certify a selected subset.', 'Actual c10 is screening evidence only; no real confirmation was performed by these checks.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = review()
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']))))


if __name__=='__main__':main()
