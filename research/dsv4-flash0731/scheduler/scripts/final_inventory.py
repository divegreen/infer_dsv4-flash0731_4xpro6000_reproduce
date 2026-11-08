#!/usr/bin/env python3
"""Close the actual Stage D experiment inventory without filtering failed runs."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
EXPECTED={'scheduler-tp4-observe-dense-profile','scheduler-tp4-fairshare-dense-profile','scheduler-tp4-fairshare-dense-profile-retry','scheduler-tp4-observe-matched-670','scheduler-tp4-fairshare-matched-670','scheduler-tp4-fairshare-matched-671','scheduler-tp4-observe-matched-671'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(path_at(D, 'results').resolve()):
        raise ValueError('Use unique output outside raw results')
    folders = sorted(path_at(D, 'results').iterdir())
    assert {p.name for p in folders} == EXPECTED
    rows = []
    ledger = {}

    def load(p):
        ledger[str(p)] = sha(p)
        return load_document(p)
    for p in folders:
        status = load(p / 'status.json')
        experiment = load(p / 'experiment.json')
        capacity = load(p / 'capacity.json')
        launch = load(p / 'command.json')
        assert status['end_utc'] and status['source_identity_unchanged'] and (status['command'] == launch['command'])
        assert not status.get('cleanup_error') and (not status['timed_out'])
        phases = []
        for run in capacity['runs']:
            phase = run['phase']
            stream = p / f'capacity-{phase['id']}-requests.jsonl'
            ledger[str(stream)] = sha(stream)
            raw = [json.loads(x) for x in stream.read_text().splitlines()]
            assert sorted(raw, key=lambda x: x['index']) == sorted(run['requests'], key=lambda x: x['index'])
            phases.append(dict(id=phase['id'], role=phase['role'], requests=len(raw), successful=sum((x['ok'] for x in raw)), output_tokens=sum((x['usage']['completion_tokens'] for x in raw if x['ok']))))
        quality = load(p / 'quality.json') if (p / 'quality.json').exists() else None
        if quality:
            assert len(quality['runs']) == 16 and quality['status'] == 'COMPLETED'
        rows.append(dict(id=p.name, profile=experiment['profile'], git_commit=launch['git_commit'], supervisor_exit_code=status['supervisor_exit_code'], client_status=capacity['status'], phases=phases, quality_requests=len(quality['runs']) if quality else 0, quality_protocol_successes=sum((x['http']['protocol_ok'] for x in quality['runs'])) if quality else 0, quality_passed=sum((x['quality']['passed'] for x in quality['runs'])) if quality else 0))
    failed = [r['id'] for r in rows if r['supervisor_exit_code'] != 0]
    assert failed == ['scheduler-tp4-fairshare-dense-profile']
    normal = [r for r in rows if not r['profile']]
    profiles = [r for r in rows if r['profile']]
    counts = dict(started_experiments=len(rows), failed_experiments=len(failed), normal_experiments=len(normal), profile_attempts=len(profiles), all_capacity_requests=sum((p['requests'] for r in rows for p in r['phases'])), normal_capacity_requests=sum((p['requests'] for r in normal for p in r['phases'])), normal_formal_requests=sum((p['requests'] for r in normal for p in r['phases'] if p['role'] != 'warmup')), profile_all_requests=sum((p['requests'] for r in profiles for p in r['phases'])), quality_requests=sum((r['quality_requests'] for r in rows)), quality_protocol_successes=sum((r['quality_protocol_successes'] for r in rows)), quality_passed=sum((r['quality_passed'] for r in rows)))
    assert counts['normal_capacity_requests'] == 1156 and counts['normal_formal_requests'] == 1076 and (counts['profile_all_requests'] == 84) and (counts['quality_requests'] == 64)
    assert all((sha(Path(p)) == h for p, h in ledger.items()))
    report = dict(status='PASS', scope='All started D experiment and HTTP request inventory, including failed d02; no throughput aggregation across heterogeneous workloads.', counts=counts, experiments=rows, source_sha256=sha(Path(__file__)), input_sha256=ledger)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with open_output(a.output, 'x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(counts))
if __name__=='__main__':main()
