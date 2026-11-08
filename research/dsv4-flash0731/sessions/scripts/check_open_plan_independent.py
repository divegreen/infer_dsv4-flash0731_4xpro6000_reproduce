#!/usr/bin/env python3
"""Independent CPU-only checks for publication/failure boundaries of open plans.

Creates mock configs/plans exclusively in TemporaryDirectory; no real plans,
services, model imports, tokenizer preparation, HTTP or GPU workloads.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
GENERATOR = HERE / 'scripts/build_open_session_plans.py'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original_sha = sha(GENERATOR)
    spec = importlib.util.spec_from_file_location('independent_open_plan_generator', GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    checks, observations = ([], [])

    def check(label, passed):
        if not passed:
            raise AssertionError(label)
        checks.append(label)
    with tempfile.TemporaryDirectory(prefix='stage-c-open-independent-MOCK-') as td:
        root = Path(td)

        def fixture(tag, kind='replay', timeout=5000):
            research = root / tag
            scenario, phase = (research / 'scenarios', research / 'sessions')
            path_at(scenario, 'configs').mkdir(parents=True)
            (phase / 'decisions').mkdir(parents=True)
            config = dict(name='MOCK', port=8260, args=['--tensor-parallel-size', '4'], env={'VLLM_SERVER_DEV_MODE': '1'})
            write_bytes(path_at(scenario, 'configs/mock.json'), gen.encode(config))
            write_bytes(research / 'mock-evidence.json', gen.encode({'kind': 'MOCK ONLY'}))
            decision = dict(schema_version=1, decision_id=tag, rationale='MOCK publication audit only', evidence_references=['mock-evidence.json'], service_protocol=dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64), warmup_seed=17, supervisor_timeout_s=timeout, selections=[dict(selection_id=kind, kind=kind, rates=[0.2], duration_s=180, seeds=[19], role='confirm', candidates=[dict(experiment_id='c08-mock-a', candidate='mock-a', config_ref='mock.json', dp_routing='none')])])
            path = phase / 'decisions/mock.json'
            write_bytes(path, gen.encode(decision))
            return (scenario, phase, path, decision)

        def rejects(fn):
            try:
                fn()
            except (ValueError, OSError):
                return True
            return False
        scenario, phase, path, d = fixture('late-candidate')
        d['selections'][0]['candidates'].append(dict(experiment_id='c09-mock-b', candidate='mock-b', config_ref='mock.json', dp_routing='sticky4'))
        write_bytes(path, gen.encode(d))
        check('invalid later candidate rejected', rejects(lambda: gen.generate(path, scenario, phase)))
        check('no first-candidate files published before all validation', not path_at(scenario, 'configs/c08-mock-a.json').exists() and (not path_at(scenario, 'workloads').exists()) and (not path_at(phase, 'logs').exists()))
        scenario, phase, path, d = fixture('write-failure')
        original_open = Path.open
        target = path_at(scenario, 'workloads/sessions/c08-mock-a.json')

        def failing_open(self, mode='r', *a, **kw):
            if self == target and mode == 'xb':
                raise OSError('MOCK injected publication failure')
            return original_open(self, mode, *a, **kw)
        with patch.object(Path, 'open', failing_open):
            check('injected second-write failure propagates', rejects(lambda: gen.generate(path, scenario, phase)))
        record_path = path_at(phase, 'logs/open-plan-generation-write-failure.json')
        record = load_document(record_path)
        check('partial publication recorded FAILED', record['status'] == 'FAILED' and 'MOCK injected' in record['error'] and (len(record['created_files']) == 1))
        config_path = path_at(scenario, 'configs/c08-mock-a.json')
        saved = config_path.read_bytes()
        check('first output preserved, later output absent', bool(saved) and (not target.exists()))
        check('failed generation ID cannot overwrite partial output', rejects(lambda: gen.generate(path, scenario, phase)))
        check('partial output and FAILED record preserved after retry', config_path.read_bytes() == saved and load_document(record_path) == record)
        scenario, phase, path, d = fixture('source-change')
        true_sources = gen.runtime_sources
        calls = [0]

        def changed_sources():
            calls[0] += 1
            source = true_sources()
            if calls[0] > 1:
                source['MOCK-CHANGED'] = 'mock'
            return source
        with patch.object(gen, 'runtime_sources', changed_sources):
            check('source identity change rejected before publish', rejects(lambda: gen.generate(path, scenario, phase)))
        check('source change creates no generation record or plan', not path_at(phase, 'logs').exists() and (not path_at(scenario, 'workloads').exists()))
        scenario, phase, path, d = fixture('timeout-boundary', kind='agent', timeout=181)
        record = gen.generate(path, scenario, phase)
        plan = load_document(path_at(scenario, 'workloads/sessions/c08-mock-a.json'))
        warm = plan['phases'][0]
        check('Agent warmup is six serial nonmeasurement tasks', warm['kind'] == 'agent' and warm['role'] == 'warmup' and (warm['sessions'] == 6) and (warm['concurrency'] == 1) and (warm['tasks'] == gen.TASKS))
        check('boundary timeout is recorded without claiming execution', record['status'] == 'GENERATED_NOT_EXECUTED' and record['cohorts'][0]['tokenizer_preflight']['status'] == 'NOT_PERFORMED')
        observations.append(dict(kind='TIME_BUDGET_LIMITATION', observed='181s supervisor budget is accepted for one 180s arrival horizon with six serial Agent warmups and 600s Session bound.', implication='Guard checks serial arrival horizons only; root must explicitly budget startup, warmup, drain, metrics and teardown. Acceptance is not proof that a run can finish within timeout.', disposition='No generator or runtime mutation; review the actual research decision timeout before launch.'))
        check('all generated outputs stay inside MOCK tree', all((Path(f['path']).is_relative_to(root) for f in record['created_files'])))
    check('frozen generator bytes unchanged', sha(GENERATOR) == original_sha)
    out = dict(status='PASS_WITH_DOCUMENTED_BUDGET_LIMITATION', created_utc=datetime.now(timezone.utc).isoformat(), generator_sha256=original_sha, check_script_sha256=sha(Path(__file__)), checks=checks, observations=observations, scope='CPU temporary MOCK plans only; no real experiment plans, tokenizer preflight, model execution, GPU or HTTP.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=out['status'], checks=len(checks), output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
