#!/usr/bin/env python3
"""CPU MOCK controls for log timestamp intervals, not model execution."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse,hashlib,importlib.util,json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PATH=Path(__file__).with_name('review_runtime.py')
spec=importlib.util.spec_from_file_location('checked_runtime_d',PATH);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Unique revision required')
    checks = []
    windows = [dict(id='warm', scope='WARMUP', start_unix=100.0, end_unix=102.0), dict(id='formal', scope='MEASUREMENT', start_unix=102.0, end_unix=104.0), dict(id='quality', scope='QUALITY', start_unix=104.25, end_unix=105.25)]

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    check('MOCK exact warmup', r.mapping(100.2, 0.001, windows)['scope'] == 'WARMUP')
    check('MOCK exact formal', r.mapping(102.0, 1.0, windows)['scope'] == 'MEASUREMENT')
    mixed = r.mapping(101.5, 1.0, windows)
    check('MOCK second-resolution warmup/formal boundary not forced', mixed['scope'] == 'MEASUREMENT_OR_BOUNDARY' and mixed['unique_containing_window'] is None)
    q = r.mapping(104.5, 1.0, windows)
    check('MOCK quality/shutdown boundary still definitely after formal', q['scope'] == 'QUALITY_OR_BOUNDARY' and q['definitely_after_performance'])
    check('MOCK exact quality', r.mapping(104.5, 0.001, windows)['scope'] == 'QUALITY')
    check('MOCK startup interval', r.mapping(98.0, 1.0, windows)['scope'] == 'STARTUP_BEFORE_CLIENT')
    check('MOCK real between-phase gap', r.mapping(104.1, 0.01, windows)['scope'] == 'BETWEEN_CLIENT_WINDOWS')
    check('MOCK after all clients', r.mapping(106.0, 1.0, windows)['scope'] == 'AFTER_ALL_CLIENT_WINDOWS')
    try:
        r.mapping(float('inf'), 1.0, windows)
    except ValueError:
        checks.append('MOCK nonfinite timestamp rejected')
    else:
        raise AssertionError('nonfinite accepted')
    stamp = r.b.timestamp('(Worker) WARNING 11-07 06:32:57 [jit_monitor] Triton JIT during inference: MOCK', 2026)
    check('actual log short timestamp has one-second uncertainty', stamp == (1794033177.0, 1.0))
    text = b'untimed header\nresource_tracker: UserWarning: There appear to be 2 leaked semaphore objects\n'
    events = r.events(text, 2026, windows, 'MOCK.log')
    e = events[0]
    check('MOCK untimed cleanup has no fabricated scope', e['timestamp_unix'] is None and e['mapping'] is None)
    check('MOCK reported resource count preserved not proved OS leak', e['reported_unregistered_resources'] == {'count': 2, 'type': 'semaphore'})
    epoch_windows = [dict(id='MOCK-warm', scope='WARMUP', start_unix=1794033176.0, end_unix=1794033179.0)]
    text = b'(Worker) WARNING 11-07 06:32:57 [jit_monitor.py] Triton kernel JIT compilation during inference: MOCK\n'
    e = r.events(text, 2026, epoch_windows, 'MOCK.log')[0]
    check('MOCK visible inference JIT in warmup is not formal JIT', e['category'] == 'runtime_jit_warning' and e['mapping']['scope'] == 'WARMUP')
    result = dict(status='PASS', scope='Synthetic CPU intervals only; no claims of exhaustive JIT coverage', checks=checks, count=len(checks), source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PATH, Path(__file__), r.B_PATH)})
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), output=str(args.output))))
if __name__=='__main__':main()
