#!/usr/bin/env python3
"""CPU-only Stage B confirmation review using frozen independent Stage A rules.

Uses no client, serving, CUDA or production summarizer imports. A PASS output
means evidence was independently recomputed, not that stability or SLO passed.
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
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent/'scenarios'
RULE_SOURCE = path_at(STAGE, 'logs/capacity-stability-independent-check.py')
SPEC = importlib.util.spec_from_file_location('frozen_independent_capacity_rules', RULE_SOURCE)
rules = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rules)
DEFAULT_RUNS = ('context-tp2pp2-confirm', 'context-tp4-th2048-cad8-confirm')


def read(path, ledger):
    path = path.resolve()
    raw = path.read_bytes()
    info = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    rules.require(str(path) not in ledger or ledger[str(path)] == info, 'Evidence changed while reading')
    ledger[str(path)] = info
    return json.loads(raw)


def conditions(run, result):
    bins = result['analysis_bins']
    epsilon = result['backlog_tolerance_requests']
    all_success = all(row.get('ok') and not row.get('censored') and not row.get('timed_out') for row in run['requests'])
    checks = dict(all_requests_successful=all_success,
                  sufficient_analysis_arrivals=result['analysis_arrivals'] >= 20,
                  waiting_samples_at_least_80_percent=result['server_sample_coverage_fraction'] >= .8,
                  dispatch_p95_at_most_100ms=result['dispatch_lag_p95_s'] is not None and result['dispatch_lag_p95_s'] <= .1,
                  finite_server_bin_means=all(rules.finite(b['server_mean_waiting']) for b in bins),
                  deficit_within_epsilon=result['completion_deficit_requests'] <= epsilon,
                  client_increase_within_epsilon=result['client_mean_outstanding_increase'] <= epsilon,
                  server_increase_within_epsilon=rules.finite(result['server_mean_waiting_increase']) and result['server_mean_waiting_increase'] <= epsilon,
                  client_bins_strictly_increasing=all(b['client_mean_outstanding'] > a['client_mean_outstanding'] for a, b in zip(bins, bins[1:])))
    prerequisites = all(checks[key] for key in ('all_requests_successful', 'sufficient_analysis_arrivals',
        'waiting_samples_at_least_80_percent', 'dispatch_p95_at_most_100ms', 'finite_server_bin_means'))
    if not prerequisites:
        reason = 'INCONCLUSIVE: one or more frozen completeness/coverage/dispatch prerequisites failed.'
    elif result['classification'] == 'STABLE_IN_WINDOW':
        reason = 'Completion deficit and first-to-last client/server queue increases are all within epsilon.'
    elif result['classification'] == 'GROWING_BACKLOG':
        reason = 'Deficit exceeds epsilon and all client bin means increase strictly, with first-to-last increase above epsilon.'
    else:
        reason = 'Evidence lies between frozen stability and sustained-growth rules; a deficit alone cannot prove backlog growth.'
    return dict(checks=checks, reason=reason)


def inspect(name, ledger):
    rules.require(Path(name).name == name, 'Expected simple experiment ID')
    directory = path_at(STAGE, 'results') / name
    report, status, experiment, config = [read(directory / filename, ledger) for filename in ('capacity.json', 'status.json', 'experiment.json', 'config.json')]
    audit = read(path_at(ROOT, 'logs') / f'audit-{name}-r1.json', ledger)
    rules.require(audit.get('status') == 'PASS' and (not audit.get('errors')) and (not audit.get('incomplete')), 'Independent integrity audit must pass')
    planned = [phase['id'] for phase in report['plan']['phases']]
    finished = [run['phase']['id'] for run in report['runs']]
    eligible = report.get('status') == 'PASSED' and (not report.get('active_phase')) and (planned == finished) and (status.get('exit_code') == 0) and status.get('end_utc') and (status.get('timed_out') is False) and (not status.get('supervisor_error')) and (experiment.get('status') == 'PASS') and all((client.get('exit_code') == 0 for client in experiment.get('clients', [])))
    rules.require(eligible, 'Expected fully terminal successful confirmation experiment')
    phases = []
    candidates = []
    for run in report['runs']:
        phase = run['phase']
        if phase.get('role') in ('warmup', 'profile'):
            continue
        rules.require(phase['mode'] == 'poisson' and phase['duration_s'] >= 180, 'Only long Poisson confirmation is in scope')
        stability = rules.stability(run, report['plan']['metrics_interval_s'])
        metrics = [rules.request_metrics(row) for row in run['requests']]
        tiers, subset_slo = rules.tier_metrics(metrics)
        candidates.append(dict(seed=phase['seed'], eligible=bool(eligible), classification=stability['classification'], p95_slo_pass=subset_slo))
        phases.append(dict(id=phase['id'], seed=phase['seed'], rate=phase['rate'], duration_s=phase['duration_s'], requests=len(run['requests']), stability=stability, rationale=conditions(run, stability), by_input_tier=tiers, eligible_subset_p95_slo_pass=subset_slo, whole_mix_joint_slo='UNDEFINED_GT16K'))
    rules.require(len(phases) >= 2 and len({p['rate'] for p in phases}) == 1, 'Expected repeated single-rate confirmation')
    combined = rules.confirmation(candidates, [])
    return dict(experiment=name, config_name=config['name'], terminal_eligible=bool(eligible), offered_rate_s=phases[0]['rate'], distinct_seeds=combined['distinct_confirmation_seeds'], phases=phases, finite_stable_confirmed_at_tested_rate=combined['finite_stable_confirmed'], finite_eligible_subset_p95_slo_confirmed=combined['finite_p95_slo_confirmed'], whole_mix_joint_slo='UNDEFINED_GT16K', confirmation_rule='Every started long window must be stable and at least two seeds must differ; do not select only passing seeds.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rules.require(not args.output.exists(), 'Output exists; use a new review ID')
    ledger = {}
    for source in (Path(__file__), RULE_SOURCE):
        raw = source.read_bytes()
        ledger[str(source.resolve())] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    experiments = [inspect(name, ledger) for name in args.runs]
    for path, info in ledger.items():
        rules.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'], 'Evidence changed during recomputation')
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), rule_version='finite-capacity-v1', scope='Independent terminal Poisson stability and applicable-tier P95 recomputation; no trace/Profile qualification', experiments=experiments, evidence_ledger=ledger, limitations=['PASS means successful independent recomputation, not that the tested rate is stable or SLO-compliant.', 'Only .5 req/s is tested here, so neither a positive nor a negative result establishes a maximum capacity boundary.', '32K/64K have no TTFT/E2E or joint Research SLO; eligible-subset P95 cannot certify whole-mix SLO capacity.', 'Embedded waiting metrics are used only after the separate integrity auditor matched them to raw Prometheus files.', 'Finite windows and two seeds do not establish infinite-horizon stability or production SLA.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', output=str(args.output), experiments=[dict(experiment=r['experiment'], finite_stable_confirmed=r['finite_stable_confirmed_at_tested_rate'], finite_eligible_subset_p95_confirmed=r['finite_eligible_subset_p95_slo_confirmed'], classifications=[p['stability']['classification'] for p in r['phases']]) for r in experiments]), indent=2))


if __name__ == '__main__':
    main()
