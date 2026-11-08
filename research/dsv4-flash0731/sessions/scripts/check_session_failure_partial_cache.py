#!/usr/bin/env python3
"""Independent partial evidence review for failed c15; never upgrades run health."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = path_at(ROOT.parent, 'scenarios/results/session-dspark3-agent-confirm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks, ledger = ([], {})

    def read(path):
        raw = path.read_bytes()
        ledger[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    def load(path):
        return json.loads(read(path))

    def check(name, value):
        if not value:
            raise ValueError(name)
        checks.append(name)

    def equal(a, b):
        return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-07)
    artifact = load(path_at(ROOT, 'logs/c15-failure-review-r1.json'))
    cache = load(path_at(ROOT, 'logs/session-dspark3-agent-confirm-cache-confirm-r1.json'))
    report, status = (load(RUN / 'sessions.json'), load(RUN / 'status.json'))
    summary = load(path_at(ROOT, 'summary/session-dspark3-agent-confirm-terminal-r1.json'))
    stream = load(path_at(ROOT, 'logs/c15-independent-failure-stream-review-r1.json'))
    check('reuse independent failure-stream audit', stream['status'] == 'PASS_INDEPENDENT_FAILURE_STREAM_REVIEW')
    check('partial never healthy', artifact['status'] == 'FAILED_EXPERIMENT_PARTIAL_EVIDENCE_VERIFIED' and status['exit_code'] == 1 and (report['status'] == 'FAILED') and (cache['status'] == 'PARTIAL_EVIDENCE_FROM_FAILED_EXPERIMENT') and all((not phase['included_as_measurement'] for phase in cache['phases'])) and all((not stage['included_as_measurement'] for stage in summary['stages'])))
    check('calculation source preserved', hashlib.sha256(read(Path(artifact['calculation_record']))).hexdigest() == artifact['calculation_sha256'])
    for path, digest in artifact['input_sha256'].items():
        check('source identity ' + path, hashlib.sha256(read(Path(path))).hexdigest() == digest)
    check('same three recorded confirmation phases', len(artifact['phase_rows']) == 3 and [p['phase'] for p in artifact['phase_rows']] == [p['phase']['id'] for p in report['phases'][1:]] and (artifact['unexecuted_phases'] == ['agent-open-r02-s02']))
    gpu = defaultdict(list)
    for cells in csv.reader(io.StringIO(read(RUN / 'gpu.csv').decode())):
        if not cells:
            continue
        timestamp = datetime.strptime(cells[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        gpu[int(cells[1])].append((timestamp, float(cells[5])))
    check('authorized four GPU evidence', set(gpu) == {0, 1, 2, 3})
    energies = []
    for row in artifact['phase_rows']:
        name = row['phase']
        phase = next((p for p in report['phases'] if p['phase']['id'] == name))
        cached = next((p for p in cache['phases'] if p['id'] == name))
        requests = [request for session in phase['sessions'] for request in session['requests']]
        totals = dict(prompt_tokens=sum((r['usage']['prompt_tokens'] for r in requests)), cached_tokens=sum((r['usage']['prompt_tokens_details']['cached_tokens'] for r in requests)), output_tokens=sum((r['usage']['completion_tokens'] for r in requests)))
        check('all-attempt raw totals ' + name, row['all_attempt_usage_totals'] == totals and row['requests'] == len(requests) and (row['protocol_ok_requests'] == sum((r['ok'] for r in requests))))
        failed = [r for r in requests if not r['ok']]
        check('failed usage not canonical normalized counts ' + name, totals['prompt_tokens'] - cached['known_prompt_tokens'] == sum((r['usage']['prompt_tokens'] for r in failed)) and totals['cached_tokens'] - cached['known_cached_tokens'] == sum((r['usage']['prompt_tokens_details']['cached_tokens'] for r in failed)) and (totals['output_tokens'] - cached['known_output_tokens'] == sum((r['usage']['completion_tokens'] for r in failed))))
        check('diagnostic inclusion remains false ' + name, row['included_as_healthy_measurement'] is False and row['whole_experiment_status'] == 'FAILED')
        start, end = (phase['start_unix'], phase['end_unix'])
        energy = 0.0
        for device in cached['gpu']['devices']:
            samples = gpu[device['gpu']]
            integral = coverage = 0.0
            for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
                left, right = (max(start, t0), min(end, t1))
                if right <= left:
                    continue
                assert t1 > t0
                left_power = p0 + (p1 - p0) * (left - t0) / (t1 - t0)
                right_power = p0 + (p1 - p0) * (right - t0) / (t1 - t0)
                integral += (right - left) * (left_power + right_power) / 2
                coverage += right - left
            check('independent power integration ' + name + '/' + str(device['gpu']), equal(integral, device['measurements']['power_w']['integral']) and equal(coverage, end - start))
            energy += integral
        check('energy sum and power denominator ' + name, equal(energy, row['board_energy_j']) and equal(energy / (end - start), row['mean_board_power_sum_w']) and equal(row['gpu_window_s'], end - start) and (abs(end - start - row['wall_s']) < 1e-06))
        energies.append(dict(phase=name, energy_j=energy, window_s=end - start))
        for field, timing in row['server_timings'].items():
            check('overlapping request timing ' + name + '/' + field, timing['count'] == len(requests) and equal(timing['sum_s'] / timing['count'], timing['mean_s']))
        speculative = row['speculative']
        check('speculative definitions distinct ' + name, equal(speculative['accepted_fraction'], speculative['accepted_tokens'] / speculative['draft_tokens']) and equal(speculative['mean_accepted_plus_one'], 1 + speculative['accepted_tokens'] / speculative['drafts']))
    failure = artifact['failures'][0]
    check('failed normalized fields still null', all((v is None for v in failure['normalized_counts_left_unchanged'].values())) and failure['usage']['prompt_tokens'] == 2355 and (failure['usage']['completion_tokens'] == 4096) and (failure['usage']['prompt_tokens_details']['cached_tokens'] == 1024))
    lines = read(RUN / 'server.log').decode().split('\n')
    jit = artifact['runtime']['visible_jit_warnings']
    check('two visible JIT events preserved', len(jit) == 2)
    for event in jit:
        pieces = [re.sub('\\x1b\\[[0-9;]*m', '', part) for part in lines[event['line'] - 1].split('\r')]
        check('exact JIT LF source line ' + str(event['line']), event['text'] in pieces)
        timestamp = event['timestamp_unix']
        overlap = [p['phase']['id'] for p in report['phases'] if p['start_unix'] < timestamp + 1 and timestamp < p['end_unix']]
        check('second-resolution JIT overlap ' + str(event['line']), overlap == event['mapping']['phase_candidates'])
    check('formal TopK JIT not dropped', jit[1]['mapping']['unique_containing_phase'] == 'agent-open-r01-s01' and jit[1]['mapping']['scope'] == 'MEASUREMENT_OR_BOUNDARY' and ('ComputeGlobalTopk' in jit[1]['text']))
    for event in artifact['runtime']['cleanup']:
        if event.get('timestamp_unix') is not None:
            check('timestamped cleanup after all recorded windows ' + str(event['line']), event['timestamp_unix'] >= max((p['end_unix'] for p in report['phases'])) and event['mapping']['scope'] == 'AFTER_ALL_PHASES')
    check('all inputs unchanged', all((hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in ledger.items())))
    result = dict(status='PASS_INDEPENDENT_FAILED_PARTIAL_REVIEW', utc=datetime.now(timezone.utc).isoformat(), checks=checks, independent_board_energy=energies, evidence_ledger=ledger, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Failed-run diagnostic artifact; no healthy guard bypass. Four-board energy independently integrated from recorded CSV; prior failure-stream audit reused.', limitations=['c15 stays FAILED and every phase is excluded from healthy comparison/capacity.', 'Raw usage is used only for diagnostic all-attempt accounting; canonical normalized nulls remain null.', 'JIT occurrence retained without exclusive causal attribution; second-resolution warnings may be deduplicated.', 'Sampled board energy excludes CPU/tools and does not establish kernel or PCIe cause.', 'Initial reviewer used splitlines rather than LF-only source line numbering; failure preserved and only reviewer corrected.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), energy=energies)))


if __name__ == '__main__':
    main()
