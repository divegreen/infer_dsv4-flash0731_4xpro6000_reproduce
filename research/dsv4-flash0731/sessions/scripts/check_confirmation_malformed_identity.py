#!/usr/bin/env python3
"""Independent CPU controls for malformed/unaudited confirmation identities.

All fixture stability and raw audit outcomes are explicitly stubbed. These
controls prove grouping behavior only, never model performance or stability.
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
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

SOURCE = Path(__file__).with_name('audit_session_confirmations.py')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location('independent_malformed_control', SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    original_source = SOURCE.read_bytes()
    names = ('scenarios/scripts/agent_session_client.py', 'scenarios/scripts/capacity_client.py', 'scripts/benchmark_client.py', 'scenarios/scripts/serve_agent_sessions.py', 'scenarios/scripts/run_logged.py', 'environment.json', 'model-inventory.json')
    phase = dict(id='window', kind='replay', arrival='poisson', role='confirm', rate=0.2, duration_s=300, profiles=['coding'], cache_scope='session', dp_routing='none')
    config = dict(name='MOCK_ONLY', args=['--tensor-parallel-size', '4'], env={'VLLM_SERVER_DEV_MODE': '1'})
    command = dict(source_sha256={k: 'a' * 64 for k in names}, environment={'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    plan = dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64, model='MOCK_ONLY')
    report = dict(status='COMPLETED', tokenizer_sha256='b' * 64, transport='fresh_tcp_no_retry')
    observations = []
    rejected = set()

    def audit(directory, ledger):
        if directory.name in rejected:
            raise ValueError('EXPLICIT CPU STUB: terminal identity not independently verified')
        return {'scope': 'EXPLICIT CPU RAW-AUDIT STUB'}
    m.independent.audit = audit
    m.independent.stability = lambda *a, **kw: dict(classification='STABLE_IN_WINDOW', scope='EXPLICIT CPU STABILITY STUB')
    with tempfile.TemporaryDirectory(prefix='independent-confirmation-malformed-') as td:
        root = Path(td)

        def write(i, change=None):
            d = root / f'c9{i}-MOCK_ONLY'
            d.mkdir(exist_ok=True)
            f = {**phase, 'seed': i}
            pl = {**plan, 'phases': [f]}
            docs = {'session-plan.json': pl, 'command.json': copy.deepcopy(command), 'config.json': {**copy.deepcopy(config), 'name': d.name}, 'sessions.json': {**copy.deepcopy(report), 'plan': pl, 'phases': [dict(phase=f, sessions=[])]}, 'status.json': dict(end_utc='MOCK_TERMINAL', exit_code=0, timed_out=False), 'experiment.json': dict(status='PASS')}
            if change:
                change(docs)
            for name, data in docs.items():
                write_text(d / name, json.dumps(data))
            (d / 'sessions-window').mkdir(exist_ok=True)
            return d

        def observe(label, expected_confirmed, expected_unresolved, expected_windows, expected_groups):
            result = m.scan(root, {})
            values = dict(confirmed=result['confirmed_protocol_points'], unresolved=len(result['unresolved_started_identity']), windows=result['started_confirmation_windows'], groups=len(result['groups']))
            expected = dict(confirmed=expected_confirmed, unresolved=expected_unresolved, windows=expected_windows, groups=expected_groups)
            if values != expected:
                raise AssertionError((label, values, expected))
            if expected_unresolved:
                assert all((any(('Unresolved started confirmation identity' in b for b in g['confirmation']['blockers'])) for g in result['groups']))
                assert all(('terminal_raw_identity_not_verified' in w['missing_identity_fields'] for w in result['unresolved_started_identity']))
            observations.append(dict(case=label, status='PASS', observed=values, unresolved=result['unresolved_started_identity'], confirmed_seeds=[g['confirmation']['seeds'] for g in result['groups'] if g['confirmation']['finite_protocol_stable_confirmed']]))
        write(1)
        write(2)
        observe('healthy_two_seed_control', 1, 0, 2, 1)
        third = write(3)
        rejected.add(third.name)

        def empty_hash(docs):
            docs['command.json']['source_sha256'][names[0]] = ''

        def name_only(docs):
            docs['config.json'] = {'name': third.name}

        def other_tokenizer(docs):
            docs['sessions.json']['tokenizer_sha256'] = 'c' * 64

        def other_runtime(docs):
            docs['command.json']['source_sha256'][names[0]] = 'd' * 64

        def missing_devmode(docs):
            docs['config.json']['env'] = {}
        for label, change in [('empty_required_source_hash_raw_rejected', empty_hash), ('config_name_only_raw_rejected', name_only), ('valid_looking_different_tokenizer_raw_rejected', other_tokenizer), ('valid_looking_different_runtime_raw_rejected', other_runtime), ('missing_config_devmode_raw_rejected', missing_devmode)]:
            write(3, change)
            observe(label, 0, 1, 3, 2)
        rejected.clear()
        write(3, other_tokenizer)
        observe('audited_known_different_identity_remains_separate', 1, 0, 3, 2)
        write(3)
        observe('audited_identical_identity_includes_all_three_seeds', 1, 0, 3, 1)
        assert observations[-1]['confirmed_seeds'] == [[1, 2, 3]]
    assert SOURCE.read_bytes() == original_source
    out = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashlib.sha256(original_source).hexdigest(), reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Independent temporary CPU grouping fixtures; raw audit and stability explicitly stubbed.', observations=observations, checks=len(observations), source_unchanged=True, supersedes_negative='confirmation-independent-malformed-identity-r1.json', limitations=['No live confirmation scan, model call, GPU work or raw evidence modification.', 'Synthetic stable labels are not model performance or finite-window stability evidence.'])
    with open_output(args.output, 'x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=out['status'], checks=out['checks'], source_sha256=out['source_sha256'])))

if __name__ == '__main__':
    main()
