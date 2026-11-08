#!/usr/bin/env python3
"""D-only terminal runtime/JIT log review with actual quality and phase windows.

Frozen B log parser is read-only. Second-resolution timestamps are intervals.
No GPU, HTTP, model imports, executable logs, or experimental-source mutation.
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
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
B_PATH=ROOT.parent/'long-context/scripts/review_runtime.py'
spec=importlib.util.spec_from_file_location('d_frozen_runtime_parser',B_PATH)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def need(value,message):
    if not value:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()

def mapping(t,width,windows):
    need(math.isfinite(t) and math.isfinite(width) and width>0,'Invalid timestamp interval')
    end=t+width;hits=[w for w in windows if t<w['end_unix'] and end>w['start_unix']]
    exact=[w for w in hits if w['start_unix']<=t and end<=w['end_unix']]
    roles={w['scope'] for w in hits}
    if len(exact)==1:scope=exact[0]['scope']
    elif 'MEASUREMENT' in roles:scope='MEASUREMENT_OR_BOUNDARY'
    elif 'QUALITY' in roles:scope='QUALITY_OR_BOUNDARY'
    elif roles:scope='WARMUP_OR_BOUNDARY'
    elif end<=windows[0]['start_unix']:scope='STARTUP_BEFORE_CLIENT'
    elif t>=windows[-1]['end_unix']:scope='AFTER_ALL_CLIENT_WINDOWS'
    else:scope='BETWEEN_CLIENT_WINDOWS'
    formal=[w for w in windows if w['scope']=='MEASUREMENT']
    return dict(scope=scope,window_candidates=[w['id'] for w in hits],role_candidates=sorted(roles),
        unique_containing_window=exact[0]['id'] if len(exact)==1 else None,
        definitely_after_performance=bool(formal) and t>=max(w['end_unix'] for w in formal),
        before_first_performance_margin_s=min(w['start_unix'] for w in formal)-end if formal else None)


def events(raw,year,windows,name):
    records=b.parse_records(raw,year);items=[]
    for pos,record in enumerate(records):
        text=record['text'];kind=b.category(text)
        if kind is None and 'JIT kernel warmup' in text:kind='kernel_jit_warmup'
        if kind is None and 'NCCL WARN' in text:kind='nccl_warning'
        if kind is None and re.search(r'\bINFO\b',text) and re.search(r'\bcompil',text,re.I):kind='compile_info'
        if kind is None:continue
        entry=dict(source=name,line=record['line'],category=kind,text=text)
        if record['time']:
            t,width=record['time'];entry.update(timestamp_unix=t,timestamp_resolution_s=width,
                timestamp_utc=datetime.fromtimestamp(t,timezone.utc).isoformat(),mapping=mapping(t,width,windows))
        else:
            before=next((x for x in reversed(records[:pos]) if x['time']),None)
            after=next((x for x in records[pos+1:] if x['time']),None)
            entry.update(timestamp_unix=None,mapping=None,neighbor_log_bracket=dict(
                before=None if before is None else dict(line=before['line'],unix=before['time'][0]),
                after=None if after is None else dict(line=after['line'],unix=after['time'][0]+after['time'][1]),
                caution='File-order only; multiprocess stdout may reorder; no forced time/phase attribution'))
        if kind=='resource_tracker_warning':
            match=re.search(r'There appear to be (\d+) leaked (\w+) objects',text)
            entry['reported_unregistered_resources']=dict(count=int(match[1]),type=match[2]) if match else None
        items.append(entry)
    return items


def review(directory, ledger):
    directory = directory.resolve()
    need(directory.parent == path_at(ROOT, 'results'), 'D-only result directory required')

    def read(name):
        path = directory / name
        raw = path.read_bytes()
        ledger[str(path)] = dict(sha256=sha(raw), bytes=len(raw))
        return raw
    status = json.loads(read('status.json'))
    need(status.get('end_utc') and type(status.get('supervisor_exit_code')) is int, 'Actual terminal supervisor required')
    report = json.loads(read('capacity.json'))
    experiment = json.loads(read('experiment.json'))
    config = json.loads(read('config.json'))
    command = json.loads(read('command.json'))
    server = json.loads(read('server-command.json'))
    need(status['source_sha256'] == command['source_sha256'], 'Terminal source inventory differs')
    need(server['command'][-len(config['args']):] == config['args'], 'Actual server argv differs from config')
    need(report['plan']['experiment_id'] == directory.name, 'Report experiment mismatch')
    need(report['status'] in ('PASSED', 'FAILED', 'CENSORED'), 'Nonterminal capacity report')
    windows = []
    for run in report['runs']:
        phase = run['phase']
        need(phase['role'] != 'profile', 'Profile not supported by normal runtime reviewer')
        windows.append(dict(id=phase['id'], role=phase['role'], scope='WARMUP' if phase['role'] == 'warmup' else 'MEASUREMENT', start_unix=run['start_unix'], end_unix=run['end_unix']))
    quality = experiment.get('quality_probe')
    if quality and quality.get('started_utc') and quality.get('finished_utc'):
        windows.append(dict(id='quality-sidecar', role='quality', scope='QUALITY', start_unix=datetime.fromisoformat(quality['started_utc']).timestamp(), end_unix=datetime.fromisoformat(quality['finished_utc']).timestamp()))
    need(windows and all((math.isfinite(w['start_unix']) and math.isfinite(w['end_unix']) and (w['start_unix'] < w['end_unix']) for w in windows)), 'Invalid actual client windows')
    need(all((x['end_unix'] <= y['start_unix'] for x, y in zip(windows, windows[1:]))), 'Overlapping/out-of-order client windows')
    year = datetime.fromisoformat(status['start_utc']).year
    collected = []
    for name in ('server.log', 'process.log', 'quality.log'):
        if (directory / name).exists():
            collected.extend(events(read(name), year, windows, name))
    jit = [e for e in collected if e['category'] == 'runtime_jit_warning']
    formal = [e for e in jit if e.get('mapping') and 'MEASUREMENT' in e['mapping']['role_candidates']]
    actual = []
    for path in directory.glob('scheduler-identity-*.json'):
        actual.append(json.loads(read(path.name)))
    success = status['supervisor_exit_code'] == 0 and status.get('exit_code') == 0 and (status.get('timed_out') is False) and (status.get('source_identity_unchanged') is True) and (not status.get('cleanup_error')) and (experiment['status'] == 'PASS') and (report['status'] == 'PASSED')
    return dict(experiment=directory.name, terminal_success=success, status='PASS', scope='Visible terminal log evidence only; no warning means no logged event, not absence of compilation', actual_client_windows=windows, actual_scheduler_identity=actual, events=collected, category_counts=dict(Counter((e['category'] for e in collected))), visible_jit_warnings=jit, formal_or_boundary_visible_jit_count=len(formal), quality_visible_jit_count=sum((bool(e.get('mapping')) and 'QUALITY' in e['mapping']['role_candidates'] for e in jit)), resource_tracker_warnings=[e for e in collected if e['category'] == 'resource_tracker_warning'], exit_events=[e for e in collected if e['category'].startswith('shutdown_') or e['category'] == 'resource_tracker_warning'], limitations=['JIT monitor may warn once per kernel/function/backend; this is not an exhaustive compilation trace.', 'Do not infer that every FairShare chunk/batch shape was warmed; formal logged JIT remains in the measurement, never removed.', 'Timestamp precision is an interval; quality/shutdown and warmup/formal boundaries are not assigned false exact times.', 'Untimed resource tracker warnings have file-order neighbors only; they do not establish leaked live OS objects or cleanup success.', 'Exit events after performance do not establish successful client semantics; protocol/raw audits remain separate.'])


def analyze(directories):
    ledger={str(p.resolve()):dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in (Path(__file__),B_PATH)}
    runs=[];errors=[]
    for directory in directories:
        try:runs.append(review(directory,ledger))
        except (ValueError,KeyError,OSError,TypeError) as exc:errors.append(dict(directory=str(directory),error=repr(exc)))
    for path,info in ledger.items():
        raw=Path(path).read_bytes()
        if sha(raw)!=info['sha256'] or len(raw)!=info['bytes']:errors.append(dict(path=path,error='Input changed during runtime review'))
    return dict(status='PASS' if runs and not errors else 'NOT VERIFIED',utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,python_executable=sys.executable,runs=runs,errors=errors,evidence_ledger=ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Use unique runtime-review output')
    report = analyze(args.result_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], runs=[dict(experiment=r['experiment'], formal_visible_jit=r['formal_or_boundary_visible_jit_count'], visible_jit=len(r['visible_jit_warnings']), resource_tracker_warnings=len(r['resource_tracker_warnings'])) for r in report['runs']], errors=report['errors'])))
    return 0 if report['status'] == 'PASS' else 1
if __name__=='__main__':raise SystemExit(main())
