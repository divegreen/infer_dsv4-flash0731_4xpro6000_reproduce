#!/usr/bin/env python3
"""CPU-only exact tokenizer/arrival identity check for the actual D ABBA inputs."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import study_client as client
from build_matched_performance import IDS

D=Path(__file__).resolve().parents[1]
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def signatures(prepared):
    return {p['spec']['id']:dict(identity=digest(dict(offsets=p['planned_offsets_s'],items=p['items'])),
            requests=len(p['items']),input_tokens=sum(i['input_tokens'] for i in p['items']),
            output_tokens=sum(i['requested_output_tokens'] for i in p['items'])) for p in prepared}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    reports = []
    cfgs = []
    hashes = {}
    checks = []
    for name, policy, seed in IDS:
        pp = path_at(D, 'workloads') / f'{name}.json'
        cp = path_at(D, 'configs') / f'{name}.json'
        plan = client.validate_plan(load_document(pp))
        cfg = load_document(cp)
        prepared, tok = client.prepare(plan)
        sig = signatures(prepared)
        assert all((len(prompt['token_ids']) == prompt['input_tokens'] for p in prepared for prompt in p['prompts'].values()))
        assert all((i['ignore_eos'] and i['requested_output_tokens'] == 128 for p in prepared for i in p['items']))
        assert not any((p['spec']['role'] == 'profile' for p in prepared))
        assert '--profiler-config' not in cfg['args'] and '--enable-logging-iteration-details' not in cfg['args']
        assert cfg['args'][cfg['args'].index('--scheduler-cls') + 1] == 'research_scheduler.' + policy
        assert cfg['quality_probe'] == dict(rounds=2, max_tokens=128, timeout_s=180)
        reports.append(dict(id=name, seed_index=seed, phases=sig, total_requests=sum((x['requests'] for x in sig.values()))))
        cfg.pop('name')
        cfg['args'][cfg['args'].index('--scheduler-cls') + 1] = 'POLICY'
        cfgs.append(cfg)
        for path in (pp, cp, tok, Path(client.__file__), Path(client.base.__file__)):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        del prepared
    for a, b in ((0, 1), (2, 3)):
        assert reports[a]['phases'] == reports[b]['phases']
        assert cfgs[a] == cfgs[b]
    historical_path = path_at(D.parent, 'scenarios/workloads/long-context/context-tp4-th2048-cad8-confirm.json')
    old = client.validate_plan(load_document(historical_path))
    prepared, _ = client.prepare(old)
    old_sigs = signatures(prepared)
    for report in reports:
        for key, value in report['phases'].items():
            if key.startswith('context_mix-'):
                assert value == old_sigs[key]
    hashes[str(historical_path)] = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    checks = ['actual tokenizer lengths match every item', '128 fixed outputs; all normal roles', 'two exact prompt/arrival pairs match', 'configs identical within pairs except scheduler', 'Poisson inputs/arrivals identical to B original450s two seeds', 'quality configuration identical', 'no profile/iteration logging']
    output = dict(status='PASS', cpu_only=True, checks=checks, runs=reports, source_sha256=hashes, limits='No model execution or performance claim; exact seed workloads are inherited, not selected by measured result.')
    with open_output(args.output, 'x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), request_counts=[r['total_requests'] for r in reports])))
if __name__=='__main__': main()
