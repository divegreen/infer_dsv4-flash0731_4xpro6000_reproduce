#!/usr/bin/env python3
"""MOCK CPU controls for D normal-performance aggregation, no GPU/HTTP."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

a=module('checked_d_performance',HERE/'analyze_performance.py')
f=module('reused_d_mock_fixture',HERE/'check_audit_study.py')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use new revision')
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append(dict(name=name, status='PASS'))
        except Exception as exc:
            checks.append(dict(name=name, status='FAIL', error=repr(exc)))

    def expect(value, message='MOCK expectation failed'):
        if not value:
            raise AssertionError(message)
    with tempfile.TemporaryDirectory(prefix='D-normal-MOCK-') as tmp:
        root = Path(tmp) / 'research/scheduler'
        path = f.fixture(root)
        directory = path.parent
        identity = load_document(directory / 'runtime-identity.json')
        inputs = {key: value['sha256'] for key, value in identity['files'].items() if '/configs/' in key or '/workloads/' in key}
        decision = dict(groups=[dict(block='MOCK-trial', experiments=['fixture', 'MOCK-fair'])], input_sha256=inputs)
        f.write(root / 'decisions/normal-pair-blocks-r1.json', decision)
        experiment = load_document(directory / 'experiment.json')
        experiment['elapsed_seconds'] = 1.0
        f.write(directory / 'experiment.json', experiment)
        result = a.analyze([directory], root)
        check('MOCK full terminal raw/source recomputation', lambda: expect(result['status'] == 'PASS', repr(result['errors'])))
        if result['stages']:
            stage = result['stages'][0]
            check('MOCK successful normal phase ranked', lambda: expect(stage['ranking_eligible'] is True))
            check('MOCK no global capacity assertion', lambda: expect(stage['eligible_for_capacity_qualification'] is False and (not result['repeated_window_observations'])))
            check('MOCK single observation never paired', lambda: expect(result['paired_comparisons'][0]['status'] == 'SINGLE_OBSERVATION'))
            fair = deepcopy(stage)
            fair.update(policy='fairshare-v1', experiment='MOCK-fair')
            paired = a.compare([stage, fair])[0]
            check('MOCK exact paired deltas zero', lambda: expect(paired['status'] == 'PASS' and all((v['difference'] in (None, 0) for v in paired['deltas'].values()))))

            def changed(field, value):
                changed = deepcopy(fair)
                changed[field] = value
                return a.compare([stage, changed])[0]
            check('MOCK changed prompt identity rejected', lambda: expect(changed('arrival_identity', {'sha256': 'different'})['status'] == 'MISMATCH'))
            check('MOCK changed non-treatment config rejected', lambda: expect(changed('matched_conditions', {'changed': True})['status'] == 'MISMATCH'))
            check('MOCK failed member not ranked', lambda: expect(changed('ranking_eligible', False)['status'] == 'DESCRIPTIVE_ONLY'))
            other_pair = [deepcopy(stage), deepcopy(fair)]
            for value in other_pair:
                value.update(pair_block='MOCK-second', experiment=value['experiment'] + 'second')
            check('MOCK same-seed burst kept as two registered restart pairs', lambda: expect(len(a.compare([stage, fair, *other_pair])) == 2 and all((p['status'] == 'PASS' for p in a.compare([stage, fair, *other_pair])))))
            check('MOCK duplicate policy not silently selected', lambda: expect(a.compare([stage, fair, deepcopy(fair)])[0]['status'] == 'NOT VERIFIED'))
            different_seed = deepcopy(fair)
            different_seed['seed'] += 1
            check('MOCK different seeds not paired', lambda: expect(all((x['status'] == 'SINGLE_OBSERVATION' for x in a.compare([stage, different_seed])))))
            stable = deepcopy(stage)
            stable.update(mode='poisson', measurement_window_s=450.0, rate=0.5)
            stable['stability']['classification'] = 'STABLE_IN_WINDOW'
            stable2 = deepcopy(stable)
            stable2['seed'] += 1
            repeat = a.repeat_observations([stable, stable2])[0]
            check('MOCK repeated stable windows remain descriptive', lambda: expect(repeat['repeated_supplied_evidence'] and repeat['all_supplied_long_windows_stable'] and (repeat['capacity_certified'] is False)))
            stable2['stability']['classification'] = 'INCONCLUSIVE'
            check('MOCK inconclusive repeat not discarded', lambda: expect(a.repeat_observations([stable, stable2])[0]['all_supplied_long_windows_stable'] is False))
            changed_conditions = deepcopy(stable2)
            changed_conditions['matched_conditions'] = {'different': 'MOCK runtime'}
            check('MOCK different deployment repetitions remain separate', lambda: expect(len(a.repeat_observations([stable, changed_conditions])) == 2))
            check('MOCK Markdown complete render', lambda: expect('正常性能配对复算' in a.markdown(result) and stage['experiment'] in a.markdown(result)))

        def mutate_file(name, mutate):
            target = directory / name
            original = target.read_bytes()
            value = json.loads(original)
            mutate(value)
            f.write(target, value)
            report = a.analyze([directory], root)
            f.write(target, original)
            return report
        check('MOCK Profile rejected from normal', lambda: expect(mutate_file('experiment.json', lambda d: d.update(profile=True))['status'] == 'NOT VERIFIED'))
        check('MOCK raw request tamper rejected', lambda: expect(mutate_file('capacity.json', lambda d: d['runs'][0]['requests'][0].update(input_tokens=999))['status'] == 'NOT VERIFIED'))
        check('MOCK nonterminal rejected', lambda: expect(mutate_file('status.json', lambda d: d.update(end_utc=None))['status'] == 'NOT VERIFIED'))
        stepfile = directory / 'scheduler-steps-123.jsonl'
        write_text(stepfile, 'MOCK not real trace\n')
        check('MOCK step-traced run rejected', lambda: expect(a.analyze([directory], root)['status'] == 'NOT VERIFIED'))
        stepfile.unlink()
        originals = {name: (directory / name).read_bytes() for name in ('capacity.json', 'status.json', 'experiment.json')}
        report = json.loads(originals['capacity.json'])
        report['status'] = 'FAILED'
        f.write(path, report)
        status = json.loads(originals['status.json'])
        status.update(exit_code=1, supervisor_exit_code=1)
        f.write(directory / 'status.json', status)
        experiment = json.loads(originals['experiment.json'])
        experiment.update(status='FAILED', clients=[dict(exit_code=1)])
        f.write(directory / 'experiment.json', experiment)
        failed = a.analyze([directory], root)
        check('MOCK intact failed execution kept descriptively', lambda: expect(failed['status'] == 'PASS' and failed['stages'] and (not any((s['ranking_eligible'] for s in failed['stages']))), repr(failed['errors'])))
        for name, raw in originals.items():
            f.write(directory / name, raw)
        profileconfig = dict(name='MOCK', args=['--scheduler-cls', 'research_scheduler.ObserveResearchScheduler', '--profiler-config', '{}'])

        def badconfig():
            try:
                a.conditions(profileconfig, {'files': {'scripts/anything': {'sha256': 'MOCK'}}})
            except ValueError:
                return
            raise AssertionError('Profile configuration accepted')
        check('MOCK Profile flags rejected independent of report labels', badconfig)

    def different_models():
        config = {'name': 'MOCK', 'args': ['--scheduler-cls', 'research_scheduler.ObserveResearchScheduler']}
        identity = {'files': {'scripts/MOCK': {'sha256': 'same'}, 'model-inventory.json': {'sha256': 'modelA'}}}
        first = a.conditions(config, identity)[1]
        identity['files']['model-inventory.json']['sha256'] = 'modelB'
        second = a.conditions(config, identity)[1]
        expect(first != second, 'Different model inventory was silently matched')
    check('MOCK model inventory bound to conditions', different_models)
    check('CPU only imports', lambda: expect(all((name not in sys.modules for name in ('torch', 'vllm', 'aiohttp', 'http.client')))))
    result = dict(status='PASS' if all((c['status'] == 'PASS' for c in checks)) else 'FAIL', scope='MOCK CPU fixtures only, not measured performance or independent proof', utc=datetime.now(timezone.utc).isoformat(), checks=checks, count=len(checks), source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE / 'analyze_performance.py', HERE / 'check_audit_study.py', HERE / 'audit_study.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), failures=[c for c in checks if c['status'] != 'PASS'], output=str(args.output))))
    return 0 if result['status'] == 'PASS' else 1
if __name__=='__main__':raise SystemExit(main())
