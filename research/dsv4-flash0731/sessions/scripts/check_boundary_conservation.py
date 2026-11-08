#!/usr/bin/env python3
"""Independent c12 half-open event conservation, without changing stability."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def event_counts(pairs,left,right):
    events=[(a,1)for a,b in pairs]+[(b,-1)for a,b in pairs]
    qleft=sum(delta for t,delta in events if t<left)
    qright=sum(delta for t,delta in events if t<right)
    arrivals=sum(left<=a<right for a,b in pairs)
    completions=sum(left<=b<right for a,b in pairs)
    return arrivals,completions,qleft,qright


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = path_at(ROOT, 'logs/c12-boundary-conservation-r1.json')
    raw = source.read_bytes()
    record = json.loads(raw)
    checks = []
    windows = []

    def need(label, value):
        if not value:
            raise ValueError(label)
        checks.append(label)
    controls = [([(75, 300)], (1, 0, 0, 1)), ([(74, 75)], (0, 1, 1, 0)), ([(300, 301)], (0, 0, 0, 0)), ([(76, 76)], (1, 1, 0, 0)), ([(0, 75), (75, 300), (75, 76), (300, 301)], (2, 2, 1, 1))]
    for i, (pairs, expected) in enumerate(controls):
        result = event_counts(pairs, 75, 300)
        need('half-open boundary control ' + str(i), result == expected)
        A, C, ql, qr = result
        need('boundary conservation control ' + str(i), A - C == qr - ql)
    need('four unique registered windows', len(record['windows']) == 4 and len({r['phase'] for r in record['windows']}) == 4)
    for phase in record['windows']:
        name = phase['phase']
        matches = [(Path(p), sha) for p, sha in record['inputs_sha256'].items() if Path(p).parent.name == 'sessions-' + name]
        need('unique raw input ' + name, len(matches) == 1)
        path, expected_sha = matches[0]
        before = path.stat()
        h = hashlib.sha256()
        pairs = []
        with path.open('rb') as f:
            for line in f:
                h.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                x, y = (row['planned_offset_s'], row['end_offset_s'])
                need('complete finite row ' + name + '/' + str(row['index']), row['admitted'] is True and row['protocol_ok'] is True and (row['terminal_reason'] == 'replay_complete') and (type(x) in (int, float)) and (type(y) in (int, float)) and math.isfinite(x) and math.isfinite(y) and (0 <= x <= y))
                pairs.append((x, y))
        after = path.stat()
        need('original raw SHA and stable file ' + name, h.hexdigest() == expected_sha and (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns))
        A, C, ql, qr = event_counts(pairs, 75, 300)
        need('event-sweep agrees direct original counts ' + name, (A, C, ql, qr) == (phase['analysis_arrivals'], phase['analysis_completions'], phase['boundary_outstanding_start_left'], phase['boundary_outstanding_end_left']))
        need('direct occupancy agrees events ' + name, ql == sum((x < 75 <= y for x, y in pairs)) and qr == sum((x < 300 <= y for x, y in pairs)))
        need('four-way deficit equality ' + name, A - C == qr - ql == phase['deficit'] and phase['conservation_pass'] is True)
        windows.append(dict(phase=name, sessions=len(pairs), analysis_arrivals=A, analysis_completions=C, outstanding_start_left=ql, outstanding_end_left=qr, deficit=A - C, raw_sha256=h.hexdigest()))
    need('input calculation record unchanged', source.read_bytes() == raw)
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, windows=windows, calculation_record_sha256=hashlib.sha256(raw).hexdigest(), reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Independent streaming JSONL and endpoint event calculation; no runtime/classifier changes or GPU execution.', definition='[75,300) includes arrivals/completions at75 and excludes those at300; Q(t-) includes planned<t<=end.', limitations=['Boundary occupancy difference is not a difference of time-averaged occupancy bins.', 'Conservation is an algebra check, not a stability certification; c12 original classifications remain unchanged.'])
    with open_output(a.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), windows=len(windows))))


if __name__=='__main__':main()
