#!/usr/bin/env python3
"""Independent started-ID closure around the frozen plan-based confirmation scan.

Run against actual results only after writers stop. No GPU, HTTP, model, grader,
or confirmation-rule implementation; missing plans fail instead of disappearing.
"""

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
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def need(value, message):
    if not value:
        raise ValueError(message)


def inventory(results, ledger):
    def load(path):
        data = path.read_bytes()
        ledger[str(path.resolve())] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        return json.loads(data)
    started, planned_only, confirms = [], [], []
    for directory in sorted(results.glob('c*')):
        if not directory.is_dir():
            continue
        raw_dirs = sorted(p.name for p in directory.glob('sessions-*') if p.is_dir())
        signals = [name for name in ('command.json', 'status.json', 'sessions.json',
                   'experiment.json', 'source-snapshot.zip') if (directory / name).exists()]
        if not signals and not raw_dirs:
            planned_only.append(directory.name)
            continue
        need(not directory.is_symlink(), 'Symlink result directory: ' + directory.name)
        need(re.fullmatch(r'c[0-9]+[A-Za-z0-9._-]*', directory.name), 'Unexpected started cID: ' + directory.name)
        for name in ('session-plan.json', 'command.json', 'status.json'):
            need((directory / name).is_file(), 'Started result lacks ' + name + ': ' + directory.name)
        plan = load(directory / 'session-plan.json')
        command = load(directory / 'command.json')
        status = load(directory / 'status.json')
        need(plan.get('experiment_id') == command.get('experiment_id') == status.get('experiment_id') == directory.name,
             'Launch/plan/status IDs differ: ' + directory.name)
        need(status.get('end_utc') and (type(status.get('exit_code')) is int or
             status.get('timed_out') is True or status.get('supervisor_error')), 'Nonterminal started result: ' + directory.name)
        phases = {p['id']: p for p in plan['phases']}
        need(len(phases) == len(plan['phases']), 'Duplicate planned phases: ' + directory.name)
        report = load(directory / 'sessions.json') if (directory / 'sessions.json').is_file() else {}
        active_label = report.get('active_phase')
        need(not active_label or (report.get('status') == 'FAILED' and active_label in phases),
             'Terminal result has unexplained active phase: ' + directory.name)
        observed = [p['phase']['id'] for p in report.get('phases', [])]
        need(len(set(observed)) == len(observed) and set(observed) <= phases.keys(),
             'Extra/duplicate reported phases: ' + directory.name)
        raw_phases = {name[len('sessions-'):] for name in raw_dirs}
        need(raw_phases <= phases.keys(), 'Unplanned raw phase directory: ' + directory.name)
        for phase_id in sorted(set(observed) | raw_phases):
            if phases[phase_id].get('role') == 'confirm':
                confirms.append((directory.name, phase_id))
        started.append(dict(experiment=directory.name, launch_signals=signals, raw_directories=raw_dirs,
                            exit_code=status.get('exit_code'), timed_out=status.get('timed_out'),
                            end_utc=status['end_utc'], planned_phases=sorted(phases),
                            planned_confirm_phases=sorted(p for p, value in phases.items() if value.get('role') == 'confirm'),
                            report_status=report.get('status'), terminal_failure_phase_label=active_label,
                            reported_phases=observed))
    return dict(started=started, planned_only=planned_only, started_confirms=confirms)


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix='started-closure-MOCK-') as tmp:
        root = Path(tmp)
        directory = root / 'cMOCK'
        directory.mkdir()
        write_text(directory / 'command.json', '{}')
        try:
            inventory(root, {})
        except ValueError as exc:
            need('Unexpected started cID' in str(exc), 'Unexpected invalid-ID rejection')
            checks.append('started malformed ID cannot silently disappear')
        else:
            raise ValueError('Malformed started ID accepted')
        directory.rename(root / 'c99-MOCK')
        directory = root / 'c99-MOCK'
        try:
            inventory(root, {})
        except ValueError as exc:
            need('session-plan.json' in str(exc), 'Expected missing-plan rejection')
            checks.append('command-only missing-plan result rejected, not skipped')
        else:
            raise ValueError('Missing plan accepted')
        plan = dict(experiment_id=directory.name, phases=[dict(id='one', role='confirm'), dict(id='two', role='confirm')])
        write_text(directory / 'session-plan.json', json.dumps(plan))
        write_text(directory / 'command.json', json.dumps(dict(experiment_id=directory.name)))
        write_text(directory / 'status.json', json.dumps(dict(experiment_id=directory.name, end_utc='MOCK', exit_code=1)))
        before = inventory(root, {})
        need(len(before['started']) == 1 and before['started_confirms'] == [] and (before['started'][0]['planned_confirm_phases'] == ['one', 'two']), 'Pre-phase failure missing')
        checks.append('launch failure before any window retained without fabricating arrivals')
        (directory / 'sessions-one').mkdir()
        (root / 'c98-plan-only').mkdir()
        write_text(root / 'c98-plan-only/session-plan.json', '{}')
        found = inventory(root, {})
        need(found['started_confirms'] == [('c99-MOCK', 'one')] and len(found['started']) == 1 and (found['started'][0]['exit_code'] == 1) and (found['planned_only'] == ['c98-plan-only']), 'MOCK inventory wrong')
        checks.extend(['raw-only phase is started without report', 'failed terminal result retained', 'unstarted plan not counted as executed'])
        (directory / 'sessions-extra').mkdir()
        try:
            inventory(root, {})
        except ValueError as exc:
            need('Unplanned raw' in str(exc), 'Expected extra raw rejection')
            checks.append('unknown raw phase cannot disappear from plan-based scan')
        else:
            raise ValueError('Extra raw accepted')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--results', type=Path, default=path_at(ROOT.parent, 'scenarios/results'))
    parser.add_argument('--confirmations', type=Path)
    parser.add_argument('--summaries', type=Path, nargs='+')
    parser.add_argument('--cumulative', type=Path, nargs='+')
    parser.add_argument('--archive-manifest', type=Path)
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a unique review output')
    ledger, checks = ({}, [])

    def load(path):
        data = path.read_bytes()
        ledger[str(path.resolve())] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        return json.loads(data)

    def check(name, value):
        need(value, name)
        checks.append(name)
    result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), self_test=args.self_test)
    if args.self_test:
        checks = self_test()
        result['scope'] = 'Temporary MOCK fixtures only; actual result tree not scanned.'
    else:
        need(args.confirmations and args.summaries and args.cumulative, 'Actual closure requires confirmations, summaries and cumulative evidence')
        found = inventory(args.results.resolve(), ledger)
        result['inventory'] = found
        started = {r['experiment'] for r in found['started']}
        confirmation = load(args.confirmations)
        check('confirmation aggregate PASS', confirmation['status'] == 'PASS')
        windows = [(w['experiment'], w['phase_id']) for group in confirmation['groups'] for w in group['windows']]
        check('every started confirm discovered exactly once', len(windows) == len(set(windows)) and set(windows) == set(found['started_confirms']) and (len(windows) == confirmation['started_confirmation_windows']))
        check('all started confirm experiment IDs examined', {e['experiment'] for e in confirmation['examined_runs']} == {e['experiment'] for e in found['started'] if e['planned_confirm_phases']})
        for collection, key, label in ((args.summaries, 'id', 'summary'), (args.cumulative, 'experiment', 'cumulative')):
            ids = []
            for path in collection:
                report = load(path)
                check(label + ' PASS ' + str(path), report['status'] == 'PASS')
                ids.extend((e[key] for e in report['experiments']))
            check(label + ' contains every started result including failures', len(ids) == len(set(ids)) and set(ids) == started)
        if args.archive_manifest:
            manifest = load(args.archive_manifest)
            ids = [e['id'] for e in manifest['experiments']]
            check('archive experiment closure', len(ids) == len(set(ids)) and set(ids) == started)
            check('archive omits no Stage C result directory', manifest['unselected_stage_c_result_ids'] == [])
            members = {r['path'] for r in manifest['files']}
            for name in started:
                for filename in ('command.json', 'status.json', 'session-plan.json', 'source-snapshot.zip'):
                    check('archive launch evidence ' + name + '/' + filename, 'research/dsv4-flash0731/scenarios/results/' + name + '/' + filename in members)
        check('input bytes and terminal membership unchanged', all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == i['sha256'] and Path(p).stat().st_size == i['bytes'] for p, i in ledger.items())) and inventory(args.results.resolve(), {}) == found)
        result['scope'] = 'Independent current-local started-ID/phase closure, not a capacity-rule recalculation or proof about removed/external experiments.'
    result.update(checks=checks, evidence_ledger=ledger, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), self_test=args.self_test)))


if __name__ == '__main__':
    main()
