#!/usr/bin/env python3
"""CPU-only independent arithmetic review of frozen Stage B final results.

Imports no client or production analyzer. Raw timing measurements are inputs,
not newly measured latency. Output creation is exclusive and evidence is hashed.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = path_at(ROOT.parent, 'scenarios/results')
PROFILES = ('context-tp4-profile', 'context-tp4-th2048-profile', 'context-tp4-th2048-cad8-profile')
ledger = {}


def read(path, json_data=True):
    path = path.resolve()
    raw = path.read_bytes()
    entry = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    assert str(path) not in ledger or ledger[str(path)] == entry, str(path)
    ledger[str(path)] = entry
    return json.loads(raw) if json_data else raw


def close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (actual, expected)


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    result = dict(n=len(values), max=max(values) if values else None)
    for key, fraction in (('p50', .5), ('p95', .95), ('p99', .99)):
        position = (len(values)-1)*fraction
        lo, hi = math.floor(position), math.ceil(position)
        result[key] = values[lo]*(hi-position)+values[hi]*(position-lo) if values and hi != lo else values[lo] if values else None
    return result


def row_metrics(row, mode):
    assert row['ok'] and row['admitted'] and row['http_status'] == 200 and row['done_received']
    assert not any(row.get(k) for k in ('censored', 'timed_out', 'error'))
    usage = row['usage']
    assert usage['prompt_tokens'] == row['input_tokens']
    assert usage['completion_tokens'] == row['requested_output_tokens'] == 128
    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    assert row['finish_reason'] == 'length' and row['ignore_eos']
    assert hashlib.sha256(row['full_text'].encode()).hexdigest() == row['text_sha256']
    first, done = row['ttft_s'], row['e2e_s']
    assert 0 <= first <= done and row['text_event_seconds'][0] == first
    assert row['text_event_seconds'][-1] <= done
    lag = max(0., row['dispatch_offset_s']-row['planned_offset_s'])
    result = dict(ttft_s=first, e2e_s=done, dispatch_lag_s=lag,
                  scheduled_ttft_s=lag+first, scheduled_e2e_s=lag+done,
                  tpot_s=(done-first)/(usage['completion_tokens']-1),
                  ttft_ratio=None, e2e_ratio=None)
    for key in ('dispatch_lag_s', 'scheduled_ttft_s', 'scheduled_e2e_s', 'tpot_s'):
        close(result[key], row[key])
    limit = next((limit for bound, limit in ((1024, 1.), (4096, 3.), (16384, 8.)) if row['input_tokens'] <= bound), None)
    prefix = 'scheduled_' if mode in ('poisson', 'burst', 'trace') else ''
    ttft, e2e = result[prefix+'ttft_s'], result[prefix+'e2e_s']
    if limit is not None:
        result['ttft_ratio'], result['e2e_ratio'] = ttft/limit, e2e/(limit+.05*127)
    result['slo'] = None if limit is None else ttft <= limit and e2e <= limit+.05*127 and result['tpot_s'] <= .05
    result['legacy'] = ttft <= 1 and e2e <= 10 and result['tpot_s'] <= .05
    return result


def check_group(rows, measured, group, wall):
    assert group['requests'] == group['successful'] == len(rows)
    eligible = sum(r['slo'] is not None for r in measured)
    passed = sum(r['slo'] is True for r in measured)
    assert group['slo_eligible_requests'] == eligible and group['slo_passes'] == passed
    close(group['slo_joint_pass_rate'], passed/eligible if eligible else None)
    assert group['legacy_slo_passes'] == sum(r['legacy'] for r in measured)
    for key, actual in group['metrics'].items():
        expected = quantiles([r[key] for r in measured])
        for stat, value in expected.items():
            close(actual[stat], value)
    if 'cohort_slo_goodput_requests_s' in group:
        close(group['cohort_slo_goodput_requests_s'], passed/wall)
        close(group['cohort_requests_s'], len(rows)/wall)
        close(group['cohort_output_tokens_s'], len(rows)*128/wall)
        close(group['cohort_slo_goodput_output_tokens_s'], passed*128/wall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a unique output filename'
    read(Path(__file__), False)
    summary = read(path_at(ROOT, 'summary/final-r1.json'))
    assert summary['status'] == 'PASS' and (not summary['errors'])
    stages = {(s['experiment'], s['phase_id']): s for s in summary['stages']}
    assert len(stages) == len(summary['stages']) == 74
    normal = sorted({name for name, _ in stages})
    counts, confirmations, paired = ({}, [], {})
    for name in normal + list(PROFILES):
        directory = RESULTS / name
        report = read(directory / 'capacity.json')
        terminal = read(directory / 'status.json')
        experiment = read(directory / 'experiment.json')
        assert terminal.get('exit_code') == 0 and terminal.get('end_utc') and (terminal.get('timed_out') is False)
        assert not terminal.get('supervisor_error') and experiment['status'] == 'PASS'
        assert experiment['clients'] and all((c['exit_code'] == 0 for c in experiment['clients']))
        assert report['status'] == 'PASSED' and (not report.get('active_phase'))
        assert [p['id'] for p in report['plan']['phases']] == [r['phase']['id'] for r in report['runs']]
        count = Counter()
        for run in report['runs']:
            phase, rows = (run['phase'], run['requests'])
            streamed = [json.loads(line) for line in read(Path(run['raw_requests_jsonl']), False).splitlines() if line]
            assert sorted(streamed, key=lambda r: r['index']) == sorted(rows, key=lambda r: r['index'])
            assert sorted((r['index'] for r in rows)) == list(range(len(rows)))
            assert len(rows) == report['prepared_requests'][phase['id']]
            kind = phase.get('role', 'measure')
            count['phases'] += 1
            count['requests'] += len(rows)
            count[kind + '_phases'] += 1
            count[kind + '_requests'] += len(rows)
            measured = [row_metrics(row, phase['mode']) for row in rows]
            if name in PROFILES:
                assert kind in ('warmup', 'profile')
                continue
            stage = stages[name, phase['id']]
            wall = run['summary']['total_wall_s']
            assert abs(run['end_unix'] - run['start_unix'] - wall) < 0.005
            assert stage['included_as_measurement'] == (kind not in ('warmup', 'profile'))
            assert stage['offered_requests'] == len(rows) and stage['successful_output_tokens'] == len(rows) * 128
            check_group(rows, measured, stage['stats'], wall)
            for shape, group in stage['by_shape'].items():
                indices = [i for i, r in enumerate(rows) if r['shape_name'] == shape]
                check_group([rows[i] for i in indices], [measured[i] for i in indices], group, wall)
            short = [i for i, r in enumerate(rows) if r['input_tokens'] == 512]
            check_group([rows[i] for i in short], [measured[i] for i in short], stage['short512'], wall)
            close(stage['throughput']['cohort_requests_s'], len(rows) / wall)
            close(stage['throughput']['cohort_output_tokens_s'], len(rows) * 128 / wall)
            if kind == 'confirm':
                assert phase['duration_s'] == 450 and phase['rate'] == 0.5
                close(stage['drain_s'], max(0.0, wall - 450))
                identity = (report['tokenizer_sha256'], [(r['index'], r['planned_offset_s'], r['prompt_sha256'], r['input_tokens'], r['requested_output_tokens'], r['ignore_eos']) for r in sorted(rows, key=lambda r: r['index'])])
                if phase['seed'] in paired:
                    assert paired[phase['seed']] == identity
                else:
                    paired[phase['seed']] = identity
                confirmations.append(dict(experiment=name, phase_id=phase['id'], seed=phase['seed'], requests=len(rows), shape_counts=dict(Counter((str(r['input_tokens']) for r in rows))), short=stage['short512'], drain_s=stage['drain_s'], output_tokens_s=len(rows) * 128 / wall, stability=stage['stability']['classification'], in_flight_at_window_end=stage['in_flight_at_window_end']))
        counts[name] = dict(count)
    totals = {}
    for label, names in (('normal', normal), ('profile_runs', PROFILES)):
        total = Counter()
        for name in names:
            total.update(counts[name])
        totals[label] = dict(total)
    assert totals['normal']['phases'] == 74 and totals['normal']['requests'] == 3222
    assert totals['normal']['warmup_phases'] == 40 and totals['normal']['warmup_requests'] == 96
    assert totals['profile_runs']['phases'] == 21 and totals['profile_runs']['requests'] == 78
    assert totals['profile_runs']['profile_requests'] == 21 and totals['profile_runs']['warmup_requests'] == 57
    stability = read(path_at(ROOT, 'logs/stability-confirmations-independent-r1.json'))
    assert stability['status'] == 'PASS'
    for path, info in stability['evidence_ledger'].items():
        raw = read(Path(path), False)
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
    for experiment in stability['experiments']:
        assert not experiment['finite_stable_confirmed_at_tested_rate']
        assert not experiment['finite_eligible_subset_p95_slo_confirmed']
        for phase in experiment['phases']:
            stage = stages[experiment['experiment'], phase['id']]
            assert stage['stability']['classification'] == phase['stability']['classification']
            assert stage['stats']['p95_slo_pass'] == phase['eligible_subset_p95_slo_pass'] is False
    for path, info in ledger.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'], 'Input changed'
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), experiments=counts, totals=totals, confirmations=confirmations, evidence_ledger=ledger, checks=['11 complete terminal runs and 95 phases; all 3300 request protocol states/usage/text hashes', 'All 95 streamed JSONL files match embedded request arrays', '74 normal phases: independent P50/P95/P99/max, SLO, Goodput and cohort throughput arithmetic', 'Four confirmation windows have exact paired arrivals, prompts, shape counts and EOS policy', 'Existing independent stability review and all its evidence hashes remain current'], limitations=['Measured raw client timings and monotonic cohort wall remain inputs, not independent hardware measurements.', 'Stability is a separately authored independent recomputation reused after hash verification; not recomputed here.', 'The reviewer also authored offline analysis tools; this is arithmetic independence, not a third-party audit.', 'No maximum stable capacity or whole-mix SLO capacity certified at the one confirmation rate.', 'No new model quality or GPU experiment.'])
    with open_output(args.output, 'x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(dict(status='PASS', output=str(args.output), totals=totals, checked_files=len(ledger)), indent=2))


if __name__ == '__main__':
    main()
