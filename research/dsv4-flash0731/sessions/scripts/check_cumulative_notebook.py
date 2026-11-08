#!/usr/bin/env python3
"""Execute only the new read-only cumulative Notebook cell against actual snapshots.

The expected summary is an in-memory union of existing, nonoverlapping audited
summaries. This does not claim a full Notebook/kernel run or a final dataset.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ledger = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)
    template = read(ROOT / 'notebooks/session-evidence.ipynb')
    code = next((''.join(c['source']) for c in template['cells'] if c['cell_type'] == 'code' and 'CUMULATIVE_RELATIVE = None' in ''.join(c['source'])))
    summaries = [read(path_at(ROOT, 'summary') / name) for name in ('screen-c03-c07-r1.json', 'open-screen-c08-c11-r1.json', 'session-tp4-replay-confirm-terminal-r1.json')]
    expected = dict(experiments=[e for s in summaries for e in s['experiments']], stages=[st for s in summaries for st in s['stages']])
    snapshot = read(path_at(ROOT, 'summary/cumulative-c03-c12-r2.json'))
    source = ROOT / 'scripts/summarize_session_workloads.py'
    ledger[str(source)] = dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size)

    def execute(source, report, include=False):
        scope = dict(PHASE=ROOT, json=json, hashlib=hashlib, Path=Path, report=report, INCLUDE_NON_MEASUREMENT=include, summary_path=path_at(ROOT, 'summary/screen-c03-c07-r1.json'), locate_original=lambda path: Path(path) if Path(path).is_file() else (_ for _ in ()).throw(FileNotFoundError(path)))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(source, 'cumulative-readonly-cell', 'exec'), scope)
        return (scope, stream.getvalue())
    checks = []
    scope, text = execute(code, None)
    assert scope['cumulative_report'] is None and scope['cumulative_counts_match'] is None
    checks.append('default no selected cumulative snapshot creates no numbers')
    actual = code.replace('CUMULATIVE_RELATIVE = None', "CUMULATIVE_RELATIVE = 'summary/cumulative-c03-c12-r2.json'")
    scope, text = execute(actual, expected)
    assert scope['cumulative_counts_match'] is True and scope['cumulative_report']['inventory'] == snapshot['inventory']
    checks.append('actual cumulative counts match all selected real source summaries')
    cohort_lines = [line for line in text.splitlines() if line.startswith('Cohort:')]
    assert len(cohort_lines) == sum((c['included_as_measurement'] for c in snapshot['cohorts']))
    assert all(('formal= True' in line for line in cohort_lines))
    checks.append('default display excludes pilot and failed cohort tables but retains complete experiment inventory')
    assert 'reaching/assigned=' in text and 'all-attempt cost=' in text and ('separate quality reference=' in text)
    checks.append('actual outputs expose turn denominator cost and separate reference')
    assert 'Measured Session concurrency by source phase:' in text
    checks.append('actual measured concurrency and preset distinction shown')
    for name in ('open-screen-c08-c11-r1.json', 'session-tp4-replay-confirm-terminal-r1.json'):
        scope['bind_cumulative_input'](path_at(ROOT, 'summary') / name)
    assert len(scope['cumulative_bound_files']) >= 17
    checks.append('all three real source summaries and selected score artifacts bind to cumulative input hashes')
    with tempfile.TemporaryDirectory(prefix='cumulative-notebook-binding-controls-') as td:
        candidate = Path(td) / 'cumulative-copy.json'
        copied = deepcopy(snapshot)
        key = next((k for k in copied['input_ledger'] if k.endswith('/summary/screen-c03-c07-r1.json')))
        copied['input_ledger'][key]['sha256'] = '0' * 64
        write_text(candidate, json.dumps(copied))
        modified = actual.replace("'summary/cumulative-c03-c12-r2.json'", repr(str(candidate)))
        try:
            execute(modified, expected)
        except AssertionError:
            checks.append('mismatched summary provenance rejected using temporary copy')
        else:
            raise AssertionError('Mismatched summary provenance accepted')
        copied = deepcopy(snapshot)
        copied['reference_inventory'][0]['available'] = False
        write_text(candidate, json.dumps(copied))
        try:
            execute(modified, expected)
        except AssertionError:
            checks.append('changed reference availability rejected using temporary copy')
        else:
            raise AssertionError('Changed reference selection accepted')
    old_source = actual.replace('cumulative-c03-c12-r2.json', 'cumulative-c03-c12-r1.json')
    try:
        execute(old_source, expected)
    except AssertionError:
        checks.append('old cumulative source revision rejected by current reviewed-source guard')
    else:
        raise AssertionError('Old unbound source snapshot accepted')
    _, expanded = execute(actual, expected, True)
    assert sum((line.startswith('Cohort:') for line in expanded.splitlines())) == len(snapshot['cohorts'])
    checks.append('explicit nonformal browsing preserves all cohorts')
    for label, changed in [('omitted_phase', {**expected, 'stages': expected['stages'][:-1]}), ('omitted_experiment', {**expected, 'experiments': expected['experiments'][:-1]})]:
        try:
            execute(actual, changed)
        except AssertionError:
            checks.append(label + ' rejected')
        else:
            raise AssertionError(label)
    bad = deepcopy(expected)
    bad['stages'][0]['summary']['sessions'] += 1
    try:
        execute(actual, bad)
    except AssertionError:
        checks.append('contradictory actual session denominator rejected')
    else:
        raise AssertionError('Contradictory denominator accepted')
    for name, info in ledger.items():
        raw = Path(name).read_bytes()
        assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256']
    out = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), checks=checks, input_ledger=ledger, inputs_unchanged=True, test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), actual_formal_cohorts=len(cohort_lines), actual_stdout_sha256=hashlib.sha256(text.encode()).hexdigest(), actual_stdout_bytes=len(text.encode()), scope='New cumulative code cell only with actual c03-c12 snapshot and in-memory union of its actual audit summaries; not full Notebook/kernel lifecycle or final Stage C execution.')
    with open_output(args.output, 'x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), actual_formal_cohorts=len(cohort_lines))))


if __name__=='__main__':main()
