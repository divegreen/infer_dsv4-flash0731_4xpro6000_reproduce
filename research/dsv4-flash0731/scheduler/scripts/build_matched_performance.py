#!/usr/bin/env python3
"""Generate Stage D's finite ABBA comparison; never overwrite experiment inputs.

Dry-run is the default. --write is allowed only after d01/d02 terminal success.
Runtime plans remain distinct from the immutable Stage B reference workloads.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
R = D.parent
S = R/'scenarios'
IDS = (
    ('scheduler-tp4-observe-matched-670', 'ObserveResearchScheduler', 0),
    ('scheduler-tp4-fairshare-matched-670', 'FairShareResearchScheduler', 0),
    ('scheduler-tp4-fairshare-matched-671', 'FairShareResearchScheduler', 1),
    ('scheduler-tp4-observe-matched-671', 'ObserveResearchScheduler', 1),
)

def load(path):
    return load_document(path)

def generate():
    confirm = load(path_at(S, 'workloads/long-context/context-tp4-th2048-cad8-confirm.json'))
    screen = load(path_at(S, 'workloads/long-context/context-tp4-th2048-cad8-screen.json'))
    profile = load(path_at(D, 'workloads/scheduler-tp4-observe-dense-profile.json'))
    config = load(path_at(D, 'configs/scheduler-tp4-observe-dense-profile.json'))
    outputs, jobs = ({}, [])
    for name, policy, seed_index in IDS:
        plan = deepcopy(confirm)
        plan.update(experiment_id=name, timeout_s=180, max_inflight=128)
        dense = deepcopy(profile['phases'][-1])
        dense.update(id='dense64-normal', role='interference', seed=202611750 + 2 * seed_index)
        dense.pop('max_inflight', None)
        burst = deepcopy(next((p for p in screen['phases'] if p['id'] == 'kv_residency-c8long')))
        warm = deepcopy(profile['phases'][:-1])
        for p in warm:
            p.pop('max_inflight', None)
        plan['phases'] = warm + [dense, deepcopy(confirm['phases'][5 + seed_index]), burst]
        cfg = deepcopy(config)
        cfg['name'] = name
        args = cfg['args']
        idx = args.index('--profiler-config')
        del args[idx:idx + 2]
        args.remove('--enable-logging-iteration-details')
        args[args.index('--scheduler-cls') + 1] = 'research_scheduler.' + policy
        cfg['quality_probe'] = dict(rounds=2, max_tokens=128, timeout_s=180)
        outputs[path_at(D, 'configs') / f'{name}.json'] = cfg
        outputs[path_at(D, 'workloads') / f'{name}.json'] = plan
        jobs.append(dict(id=name, config=f'{name}.json', capacity_plan=f'{name}.json', timeout=2100, reference='scheduler-tp4-observe-dense-profile', hypothesis='Unprofiled matched scheduler comparison with Poisson context mix, dense diagnostic and eight-long boundary; quality evaluated separately.'))
    outputs[path_at(D, 'workloads/series-matched-performance-r1.json')] = dict(jobs=jobs)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    outputs = generate()
    for i, j in ((0, 1), (2, 3)):
        a = deepcopy(outputs[path_at(D, 'workloads') / f'{IDS[i][0]}.json'])
        b = deepcopy(outputs[path_at(D, 'workloads') / f'{IDS[j][0]}.json'])
        a.pop('experiment_id')
        b.pop('experiment_id')
        assert a == b
        a = deepcopy(outputs[path_at(D, 'configs') / f'{IDS[i][0]}.json'])
        b = deepcopy(outputs[path_at(D, 'configs') / f'{IDS[j][0]}.json'])
        a.pop('name')
        b.pop('name')
        a['args'][a['args'].index('--scheduler-cls') + 1] = 'POLICY'
        b['args'][b['args'].index('--scheduler-cls') + 1] = 'POLICY'
        assert a == b
    if args.write:
        for name in ('scheduler-tp4-observe-dense-profile', 'scheduler-tp4-fairshare-dense-profile-retry'):
            status = load(path_at(D, 'results') / name / 'status.json')
            assert status['supervisor_exit_code'] == 0 and status['source_identity_unchanged']
            assert not status['timed_out'] and (not status.get('cleanup_error'))
        if any((p.exists() for p in outputs)):
            raise FileExistsError('Refuse to overwrite any generated input')
        for path, data in outputs.items():
            with open_output(path, 'x') as stream:
                json.dump(data, stream, indent=2)
                stream.write('\n')
    print(json.dumps(dict(status='WRITTEN_NOT_EXECUTED' if args.write else 'DRY_RUN', utc=datetime.now(timezone.utc).isoformat(), files=[str(p) for p in outputs], order=[x[0] for x in IDS], matched_plan_config_pairs=2, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))

if __name__=='__main__': main()
