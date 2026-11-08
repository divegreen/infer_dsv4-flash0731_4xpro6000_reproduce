#!/usr/bin/env python3
"""Compare offline cache summaries only within recorded workload identities."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ratio(a,b):return a/b if a is not None and b not in (None,0) else None


def point(experiment,phase):
    return dict(experiment=experiment,phase=phase['id'],kind=phase['kind'],role=phase['role'],
        seed=phase['seed'],concurrency=phase['concurrency'],apc=phase['apc_enabled'],
        cache_scope=phase['cache_scope'],dp_routing=phase['dp_routing'],
        workload_sha256=phase['workload_sha256'],sessions=phase['sessions'],requests=phase['requests'],
        successful_tasks=phase['successful_tasks'],wall_s=phase['wall_s'],
        input_tokens=phase['known_prompt_tokens'],cached_tokens=phase['known_cached_tokens'],
        computed_prompt_tokens=phase['logical_computed_prompt_tokens'],output_tokens=phase['known_output_tokens'],
        cache_fraction=phase['cache_fraction'],output_tokens_s=phase['output_tokens_s'],
        session_p50_s=phase['session_latency_s']['p50'],session_p95_s=phase['session_latency_s']['p95'],
        ttft_p50_s=phase['request_ttft_text_s']['p50'],ttft_p95_s=phase['request_ttft_text_s']['p95'],
        tpot_p50_s=phase['request_tpot_output_s']['p50'],tpot_p95_s=phase['request_tpot_output_s']['p95'],
        e2e_p95_s=phase['request_e2e_s']['p95'],server_timings=phase['server_timings'],
        energy_j=phase['gpu']['energy_j'],output_tokens_per_joule=phase['output_tokens_per_joule'],
        speculative=phase['speculative'],per_engine=phase['per_engine'],gauges=phase['observed_gauge_peaks'],
        workload_scope=phase['workload_scope'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    inputs = {str(path.resolve()): sha(path) for path in args.inputs}
    groups = defaultdict(list)
    excluded = []
    versions = {}
    checks = []
    for path in args.inputs:
        data = load_document(path)
        experiment = Path(data['run']).name
        if experiment in versions:
            raise ValueError('Duplicate experiment summaries')
        versions[experiment] = dict(commit=data['git_commit'], runtime=data['runtime_source_sha256'], tokenizer=data['tokenizer_sha256'])
        for phase in data['phases']:
            if phase['included_as_measurement']:
                groups[phase['workload_sha256']].append(point(experiment, phase))
            else:
                excluded.append(dict(experiment=experiment, phase=phase['id'], role=phase['role'], whole_experiment_healthy=phase['whole_experiment_healthy']))
    shared_runtime = ['scenarios/scripts/agent_session_client.py', 'scenarios/scripts/agent_session_tasks.py', 'scenarios/scripts/capacity_client.py', 'scripts/benchmark_client.py', 'environment.json', 'model-inventory.json']
    if all('metadata/environment.json' in v['runtime'] for v in versions.values()):
        shared_runtime = [name for name in shared_runtime if name not in ('environment.json', 'model-inventory.json')]
        shared_runtime += ['metadata/environment.json', 'metadata/model-inventory.json', 'project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py']
    for name in shared_runtime:
        identities = {v['runtime'].get(name) for v in versions.values()}
        if len(identities) != 1 or None in identities:
            raise ValueError('Unmatched executed runtime dependency ' + name)
        checks.append('same executed ' + name)
    if len({v['tokenizer'] for v in versions.values()}) != 1:
        raise ValueError('Tokenizer identity changed')
    checks.append('same actual tokenizer')
    fixture_maps = [{name: value for name, value in v['runtime'].items() if name.startswith(('scenarios/workloads/sessions/tasks/', 'experiments/scenarios/workloads/sessions/tasks/'))} for v in versions.values()]
    if not fixture_maps or not fixture_maps[0] or any((v != fixture_maps[0] for v in fixture_maps[1:])):
        raise ValueError('Matched task fixture set or bytes changed')
    checks.append('same complete task fixture set and hashes')
    matched = []
    for workload, points in sorted(groups.items()):
        if len(points) < 2:
            continue
        base = next((r for r in points if r['experiment'].startswith(('c03-', 'session-tp4-cache-screen')) and r['cache_scope'] == 'session'), None)
        if base is None:
            base = next((r for r in points if r['experiment'].startswith('c07-') and r['dp_routing'] == 'none'), points[0])
        comparisons = []
        for row in points:
            if row is base:
                continue
            if row['kind'] == 'replay':
                for name in ('sessions', 'requests', 'input_tokens', 'output_tokens'):
                    if row[name] != base[name]:
                        raise ValueError('Exact replay matched identity has differing ' + name)
            comparisons.append(dict(reference=dict(experiment=base['experiment'], phase=base['phase']), candidate=dict(experiment=row['experiment'], phase=row['phase']), wall_ratio_candidate_over_reference=ratio(row['wall_s'], base['wall_s']), throughput_ratio_candidate_over_reference=ratio(row['output_tokens_s'], base['output_tokens_s']), session_p95_ratio=ratio(row['session_p95_s'], base['session_p95_s']), ttft_p95_ratio=ratio(row['ttft_p95_s'], base['ttft_p95_s']), tpot_p95_ratio=ratio(row['tpot_p95_s'], base['tpot_p95_s']), energy_ratio=ratio(row['energy_j'], base['energy_j']), context='Matched exact replay' if row['kind'] == 'replay' else 'Matched task cohort only; actual call/token workload and quality may differ'))
        routing = []
        for sticky in [r for r in points if r['dp_routing'] == 'sticky4']:
            default = next((r for r in points if r['experiment'] == sticky['experiment'] and r['dp_routing'] == 'none' and (r['cache_scope'] == sticky['cache_scope'])), None)
            if default is None:
                continue
            routing.append(dict(experiment=sticky['experiment'], reference_phase=default['phase'], candidate_phase=sticky['phase'], comparison='sticky4 / default routing in the same deployment and workload', wall_ratio=ratio(sticky['wall_s'], default['wall_s']), throughput_ratio=ratio(sticky['output_tokens_s'], default['output_tokens_s']), energy_ratio=ratio(sticky['energy_j'], default['energy_j']), cache_fraction_default=default['cache_fraction'], cache_fraction_sticky=sticky['cache_fraction'], session_p95_ratio=ratio(sticky['session_p95_s'], default['session_p95_s']), default_tasks=default['successful_tasks'], sticky_tasks=sticky['successful_tasks'], caveat='Dynamic Agent actual work/quality may differ' if sticky['kind'] == 'agent' else 'Identical exact token replay'))
        matched.append(dict(workload_sha256=workload, kind=points[0]['kind'], seed=points[0]['seed'], concurrency=points[0]['concurrency'], points=points, comparisons=comparisons, within_deployment_routing_comparisons=routing))
    if any((sha(Path(path)) != value for path, value in inputs.items())):
        raise ValueError('Analysis input changed while reading')
    result = dict(status='PASS_MATCHED_OFFLINE_COMPARISON', created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)), inputs=inputs, versions=versions, checks=checks, matched_groups=matched, unpaired_points=[rows[0] for rows in groups.values() if len(rows) == 1], excluded_phases=excluded, limits=['Includes only healthy terminal experiment phases with screen/measure/confirm roles; no checkpoint or pilot ranks.', 'Two closed-loop seeds are screening repeats, not open-arrival capacity confirmation.', 'Ratios are descriptive within exact workload identity; multiple deployment differences preclude single-kernel causal attribution.', 'Dynamic Agent histories and task success vary; equal planned task cohorts do not mean equal actual model work.', 'Energy is sampled four-board energy with full window coverage only; no CPU/system energy.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], experiments=len(versions), matched_groups=len(matched), unpaired_points=len(result['unpaired_points']), excluded_phases=len(excluded), output=str(args.output)), indent=2))


if __name__=='__main__':main()
