"""CPU-only audit of saved experiment evidence; never contacts a model service.

Recomputes token totals/throughput from request records, checks workload identity,
source snapshots and GPU sampling scope. Does not establish semantic accuracy.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    failures, notes, runs = ([], [], [])
    identities, aggregate = ({}, {})
    counts = dict(experiments=0, benchmark_reports=0, requests=0, successful_requests=0, recorded_failed_requests=0, measured_requests=0, snapshots=0, snapshot_members=0, gpu_samples=0, prompt_hashes=0, client_source_mismatches=0)

    def check(condition, message):
        if not condition:
            failures.append(message)

    def equal_number(actual, expected, label):
        check(isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-08, abs_tol=1e-09), label)
    for folder in sorted(path_at(ROOT, 'results').iterdir()):
        if not folder.is_dir() or not (folder / 'command.json').is_file():
            continue
        counts['experiments'] += 1
        command = load_document(folder / 'command.json')
        status_path = folder / 'status.json'
        check(status_path.is_file(), f'{folder.name}: missing final status')
        status = load_document(status_path) if status_path.is_file() else {}
        runs.append(dict(experiment=folder.name, exit_code=status.get('exit_code'), timed_out=status.get('timed_out')))
        mask = command.get('environment', {}).get('CUDA_VISIBLE_DEVICES', '')
        check(bool(mask) and set(mask.split(',')) <= {'0', '1', '2', '3'}, f'{folder.name}: GPU mask {mask!r}')
        if folder.name.endswith(('-full-fresh', '-long-fresh')):
            check(set(mask.split(',')) == {'0', '1', '2', '3'}, f'{folder.name}: final four-GPU candidate used subset {mask!r}')
        gpu_file = folder / 'gpu.csv'
        if gpu_file.is_file():
            for line in gpu_file.read_text().splitlines():
                if line.startswith('#') or not line.strip():
                    continue
                fields = [s.strip() for s in line.split(',')]
                check(len(fields) >= 3 and fields[1] in {'0', '1', '2', '3'}, f'{folder.name}: unexpected GPU sample')
                counts['gpu_samples'] += 1
        archive = folder / 'source-snapshot.zip'
        if archive.is_file():
            counts['snapshots'] += 1
            with zipfile.ZipFile(archive) as z:
                for name, expected in command.get('source_sha256', {}).items():
                    check(name in z.namelist(), f'{folder.name}: snapshot missing {name}')
                    if name in z.namelist():
                        check(hashlib.sha256(z.read(name)).hexdigest() == expected, f'{folder.name}: snapshot hash mismatch {name}')
                        counts['snapshot_members'] += 1
        else:
            notes.append(f'{folder.name}: no per-run source archive (earlier run; consult Git/source hashes)')
        paths = sorted(folder.glob('benchmark*.json'))
        for path in paths:
            report = load_document(path)
            counts['benchmark_reports'] += 1
            client_hash = report.get('client', {}).get('sha256')
            recorded_hash = command.get('source_sha256', {}).get('scripts/benchmark_client.py')
            if client_hash and recorded_hash and (client_hash != recorded_hash):
                counts['client_source_mismatches'] += 1
                notes.append(f'{path.relative_to(ROOT)}: client self-reported SHA256 {client_hash} differs from runner-start {recorded_hash}; early mutable-source provenance limitation')
                check(not archive.is_file(), f'{folder.name}: archived client differs from reported running client')
            if folder.name.endswith(('-full-fresh', '-long-fresh')):
                check(bool(client_hash) and client_hash == recorded_hash, f'{folder.name}: final candidate runtime client hash missing/mismatched')
                check(report.get('client_transport', {}).get('connection_policy') == 'fresh_per_request', f'{folder.name}: final candidate transport is not fresh_per_request')
            if 'prompts' not in report or 'runs' not in report:
                notes.append(f'{path.relative_to(ROOT)}: older schema, not audited at request level')
                continue
            for sha, prompt in report['prompts'].items():
                actual = hashlib.sha256(json.dumps(prompt['token_ids'], separators=(',', ':')).encode()).hexdigest()
                check(actual == sha == prompt['sha256'], f'{folder.name}: prompt hash mismatch')
                check(len(prompt['token_ids']) == prompt['input_tokens'], f'{folder.name}: prompt length mismatch')
                counts['prompt_hashes'] += 1
            specs = {c['name']: c for c in report.get('case_specs', [])}
            run_keys = [(r['round'], r['case']) for r in report['runs']]
            check(len(run_keys) == len(set(run_keys)), f'{folder.name}: duplicate round/case')
            if report['status'] == 'PASSED':
                expected_keys = {(i, c) for i in range(1, report['rounds'] + 1) for c in specs}
                check(set(run_keys) == expected_keys, f'{folder.name}/{report['suite']}: incomplete PASSED suite')
            for run in report['runs']:
                for phase in ('warmup', 'measure'):
                    if phase not in run:
                        check(report['status'] != 'PASSED', f'{folder.name}: PASSED with missing {phase}')
                        continue
                    batch = run[phase]
                    requests = batch.get('requests', [])
                    summary = batch['summary']
                    successful = [r for r in requests if r['ok']]
                    all_successful = bool(requests) and len(successful) == len(requests)
                    check(summary['all_successful'] == all_successful, f'{folder.name}: all_successful flag')
                    check(summary['requests'] == len(requests), f'{folder.name}: summary request count')
                    expected_count = specs[run['case']]['requests' if phase == 'measure' else 'warmup_requests']
                    check(len(requests) == expected_count, f'{folder.name}: {phase} request count')
                    counts['requests'] += len(requests)
                    counts['successful_requests'] += len(successful)
                    counts['recorded_failed_requests'] += len(requests) - len(successful)
                    if phase == 'measure':
                        counts['measured_requests'] += len(requests)
                    check(summary['successful'] == len(successful) and summary['failed'] == len(requests) - len(successful), f'{folder.name}: request totals')
                    check(report['status'] != 'PASSED' or len(successful) == len(requests), f'{folder.name}: PASSED contains error')
                    tokens = sum((r['usage']['completion_tokens'] for r in successful))
                    check(summary['successful_output_tokens'] == tokens, f'{folder.name}: output sum')
                    check(summary['successful_input_tokens'] == sum((r['usage']['prompt_tokens'] for r in successful)), f'{folder.name}: input sum')
                    equal_number(summary['output_tokens_per_s'], tokens / summary['wall_s'], f'{folder.name}: throughput')
                    for r in successful:
                        p = report['prompts'][r['prompt_sha256']]
                        check(r['usage']['prompt_tokens'] == len(p['token_ids']), f'{folder.name}: actual prompt usage')
                        check(not r['ignore_eos'] or r['usage']['completion_tokens'] == r['requested_completion_tokens'], f'{folder.name}: actual completion usage')
                        check(r['http_status'] == 200 and r['done_received'] and (r['text_characters'] > 0), f'{folder.name}: success protocol')
                        check(0 <= r['ttft_s'] <= r['e2e_s'], f'{folder.name}: time ordering')
                        if r['usage']['completion_tokens'] > 1:
                            equal_number(r['tpot_s'], (r['e2e_s'] - r['ttft_s']) / (r['usage']['completion_tokens'] - 1), f'{folder.name}: TPOT formula')
                    if phase == 'measure':
                        key = (report['suite'], run['case'])
                        identity = [specs[run['case']]['concurrency'], [(r['prompt_sha256'], r['requested_completion_tokens'], r['ignore_eos'], r['seed']) for r in requests]]
                        check(identity == identities.setdefault(key, identity), f'{folder.name}: workload differs for {key}')
                        if all_successful and report['suite'] in {'screen', 'full', 'long'}:
                            item = aggregate.setdefault((folder.name, report['suite'], run['case']), [0, 0.0, 0, 0, report['status']])
                            item[0] += tokens
                            item[1] += summary['wall_s']
                            item[2] += 1
                            item[3] += len(requests)
    analysis = load_document(path_at(ROOT, 'results/analysis.json'))
    analysis_keys = []
    for suite in ('screen', 'full', 'long'):
        for row in analysis[suite]:
            key = (row['experiment'], suite, row['case'])
            analysis_keys.append(key)
            check(key in aggregate, f'Unexpected analysis row {key}')
            if key in aggregate:
                tokens, wall, rounds, requests, report_status = aggregate[key]
                equal_number(row['output_tokens_per_s'], tokens / wall, f'Analysis weighted throughput {key}')
                check(row['rounds'] == rounds, f'Analysis rounds {key}')
                check(row['requests'] == requests and row['report_status'] == report_status, f'Analysis request count/status {key}')
    check(len(analysis_keys) == len(set(analysis_keys)), 'Duplicate analysis keys')
    check(set(analysis_keys) == set(aggregate), 'Missing/extra analysis rows')
    final_file = ROOT / 'final-results.json'
    if final_file.is_file():
        final = load_document(final_file)
        final_keys = []
        by_key = {(r['experiment'], r['case']): r for suite in ('full', 'long') for r in analysis[suite] if r.get('connection_policy') == 'fresh_per_request' and r.get('report_status') == 'PASSED'}
        for row in final.get('metrics', []):
            key = (row['experiment'], row['case'])
            final_keys.append(key)
            check(key in by_key, f'Unexpected final-results metric {key}')
            if key in by_key:
                equal_number(row['output_tokens_per_s'], by_key[key]['output_tokens_per_s'], f'Final metrics throughput {key}')
                check(row['ttft_p50_ms_by_round'] == by_key[key]['ttft_p50_ms_by_round'], f'Final metrics TTFT {key}')
        check(len(final_keys) == len(set(final_keys)) and set(final_keys) == set(by_key), 'Missing/duplicate final-results metrics')
    else:
        failures.append('Missing final-results.json')
    result = dict(status='PASS' if not failures else 'FAIL', counts=counts, failures=failures, historical_limitations=notes, experiment_statuses=runs, scope='Saved evidence consistency only; not independent model accuracy, server reliability, full weight payload hashing, or physical bandwidth validation.')
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], counts=counts, failures=failures), ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
