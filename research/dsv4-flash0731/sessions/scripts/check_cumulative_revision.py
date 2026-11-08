#!/usr/bin/env python3
"""Independent r2 provenance/concurrency review; CPU only, original inputs read-only."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = path_at(ROOT.parent, 'scenarios/results')
SOURCE = ROOT / 'scripts/summarize_session_workloads.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks, ledger, regressions, peaks = ([], {}, [], [])

    def raw(path):
        path = path.resolve()
        data = path.read_bytes()
        ledger[str(path)] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        return data

    def load(path):
        return json.loads(raw(path))

    def check(name, value):
        if not value:
            raise ValueError(name)
        checks.append(name)

    def rejects(name, action):
        try:
            action()
        except ValueError as exc:
            regressions.append(dict(check=name, rejected=True, error=str(exc)))
            checks.append(name)
        else:
            raise ValueError('Expected rejection: ' + name)
    spec = importlib.util.spec_from_file_location('cumulative_revision_subject', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw(SOURCE)
    check('reviewed revision source', ledger[str(SOURCE)]['sha256'] == '234eaeff442809c264a200374d172fa5666c5298ac3011a72e72e5ab211ab531')
    number_review = load(path_at(ROOT, 'logs/cumulative-independent-numbers-r1.json'))
    check('reuse prior independent primitive numerical review', number_review['status'] == 'PASS_NUMERICAL_REVIEW' and len(number_review['checks']) == 5625)
    reports = []
    for prefix in ('cumulative-c03-c12', 'cumulative-pilots-c01-c02'):
        old = load(path_at(ROOT, 'summary') / (prefix + '-r1.json'))
        new = load(path_at(ROOT, 'summary') / (prefix + '-r2.json'))
        reports.append(new)
        check(prefix + ' reviewed source and PASS', new['status'] == 'PASS' and new['script_sha256'] == ledger[str(SOURCE)]['sha256'])
        check(prefix + ' inventories unchanged', old['inventory'] == new['inventory'] and old['experiments'] == new['experiments'])
        check(prefix + ' cohort identities unchanged', [c['cohort_sha256'] for c in old['cohorts']] == [c['cohort_sha256'] for c in new['cohorts']])
        for before, after in zip(old['cohorts'], new['cohorts']):
            cleaned = deepcopy(after)
            del cleaned['source_phase_concurrency']
            if prefix == 'cumulative-pilots-c01-c02':
                for value in (before, cleaned):
                    for group in [value['summary'], *value['by_task'].values(), *value['by_task_type'].values()]:
                        del group['quality']
            check('r1 numerical fields unchanged ' + after['cohort_sha256'], cleaned == before)
        for path, expected in new['input_ledger'].items():
            if Path(path).name in ('status.json', 'sessions.json', 'command.json', 'source-snapshot.zip'):
                raw(Path(path))
                check('actual terminal input hash ' + path, ledger[path] == expected)

    def independent_peak(rows, start_field):
        valid, missing = ([], 0)
        for row in rows:
            a, b = (row.get(start_field), row.get('end_offset_s'))
            if not all((type(x) in (int, float) and math.isfinite(x) for x in (a, b))) or b < a:
                missing += 1
            else:
                valid.append((a, b))
        lower = max((sum((a <= t < b for a, b in valid)) for t in {a for a, b in valid}), default=0)
        return dict(peak=None if missing else lower, known_interval_peak=lower, complete=not missing, intervals=len(valid), missing_intervals=missing)
    for report in reports:
        for cohort in report['cohorts']:
            experiment = cohort['experiment']
            data = load(RESULTS / experiment / 'sessions.json')
            runs = {r['phase']['id']: r for r in data['phases']}
            check('exact concurrency phase closure ' + cohort['cohort_sha256'], [p['phase'] for p in cohort['source_phase_concurrency']] == cohort['source_phases'])
            for item in cohort['source_phase_concurrency']:
                phase, evidence = (item['phase'], item['evidence'])
                run = runs[phase]
                rows = run['sessions']
                opened = run['phase']['arrival'] == 'poisson'
                active = independent_peak([r for r in rows if r.get('admitted') is True], 'start_offset_s')
                outstanding = independent_peak(rows, 'planned_offset_s') if opened else None
                context = experiment + '/' + phase
                check('independent admitted intervals ' + context, evidence['admitted_active_session_intervals'] == active)
                check('independent open outstanding ' + context, evidence['open_outstanding_session_intervals'] == outstanding)
                check('reported peak labeled independently ' + context, evidence['client_reported_peak_active_sessions'] == run['peak_active_sessions'] and evidence['client_reported_peak_is_measured'] is opened and ('preset' in evidence['client_reported_scope']) and ('Half-open' in evidence['interval_scope']))
                if not opened:
                    check('closed report is runtime preset ' + context, run['peak_active_sessions'] == min(run['phase']['concurrency'], run['planned_sessions']))
                peaks.append(dict(experiment=experiment, phase=phase, arrival=run['phase']['arrival'], active_peak=active['peak'], outstanding_peak=outstanding['peak'] if opened else None, reported_peak=run['peak_active_sessions']))
        for selected in report['reference_inventory']:
            path = Path(selected['path'])
            if not selected['available']:
                check('not-selected coverage remains explicit ' + str(path), not path.exists())
                continue
            score = load(path)
            experiment = selected['experiment']
            command = load(RESULTS / experiment / 'command.json')
            is_reference = 'fixture_source_sha256' in score
            filenames = ['status.json', 'sessions.json', 'command.json'] + (['source-snapshot.zip'] if is_reference else [])
            for name in filenames:
                original = str((RESULTS / experiment / name).resolve())
                matches = [info for p, info in score['input_ledger'].items() if p.endswith('/' + experiment + '/' + name)]
                check('independent score/raw binding ' + str(path) + '/' + name, matches == [report['input_ledger'][original]] == [ledger[original]])
            if is_reference:
                member = 'scenarios/workloads/sessions/tasks/catalog.json'
                with zipfile.ZipFile(io.BytesIO(raw(RESULTS / experiment / 'source-snapshot.zip'))) as archive:
                    catalog_raw = archive.read(member)
                    catalog = {r['id']: r for r in json.loads(catalog_raw)['tasks']}
                    check('catalog binds execution ' + experiment, hashlib.sha256(catalog_raw).hexdigest() == command['source_sha256'][member])
                    required = {member}
                    for row in score['sessions']:
                        check('eligible boolean result ' + experiment + '/' + str(row['session_index']), type(row['reference_eligible']) is bool and (type(row['reference_passed']) is bool if row['reference_eligible'] else row['reference_passed'] is None))
                        if row['reference_eligible']:
                            task = catalog[row['task_id']]
                            prefix = 'scenarios/workloads/sessions/tasks/' + task['id']
                            required.add(prefix + '/private/answers.json')
                            required.update((prefix + '/public/' + n for n in task['readable']))
                    check('reference fixture closed over eligible tasks ' + experiment, required <= score['fixture_source_sha256'].keys())
                    for filename, digest in score['fixture_source_sha256'].items():
                        check('reference fixture actual launch bytes ' + experiment + '/' + filename, digest == command['source_sha256'][filename] == hashlib.sha256(archive.read(filename)).hexdigest())
            else:
                check('objective helper bound to launch ' + experiment, score['grader_sha256'] == command['source_sha256']['scenarios/scripts/agent_session_tasks.py'])
    experiment = 'session-tp4-agent-open'
    score = load(path_at(ROOT, 'logs') / (experiment + '-json-reference-r1.json'))
    with tempfile.TemporaryDirectory(prefix='cumulative-independent-r2-') as tmp:
        key = next((p for p in score['input_ledger'] if p.endswith('/' + experiment + '/sessions.json')))
        score['input_ledger'][key]['sha256'] = '0' * 64
        write_text(Path(tmp) / (experiment + '-json-reference-r1.json'), json.dumps(score))
        rejects('original wrong sessions provenance regression now rejected', lambda: module.build([path_at(ROOT, 'summary') / (experiment + '-terminal-r1.json')], RESULTS, Path(tmp)))
    description = load(path_at(ROOT, 'summary/screen-c03-c07-workloads-r1.json'))
    experiment = 'session-tp4-cache-screen'
    description['experiments'] = [e for e in description['experiments'] if e['id'] == experiment]
    description['stages'] = [s for s in description['stages'] if s['experiment'] == experiment]
    score = load(path_at(ROOT, 'logs') / (experiment + '-json-reference-r1.json'))
    references = {(r['phase'], r['session_index']): deepcopy(r) for r in score['sessions']}
    catalogs = {experiment: {r['task_id']: r['kind'] for r in score['sessions']}}
    eligible = next((r for r in references.values() if r['reference_eligible']))
    eligible['reference_passed'] = None
    rejects('original eligible-without-score regression now rejected', lambda: module.cumulative(description, catalogs, {experiment: references}, {}))
    for label, intervals, expected in [('adjacent and zero length', [(0.0, 1.0), (1.0, 2.0), (1.0, 1.0)], 1), ('overlap equal endpoints', [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)], 2), ('all zero length', [(0.0, 0.0), (1.0, 1.0)], 0), ('empty', [], 0)]:
        check('exact half-open boundary control ' + label, module.interval_peak(intervals) == expected)
    control = dict(phase=dict(arrival='poisson'), peak_active_sessions=4, sessions=[dict(admitted=True, planned_offset_s=0.0, start_offset_s=1.0, end_offset_s=2.0), dict(admitted=True, planned_offset_s=0.0, start_offset_s=None, end_offset_s=3.0), dict(admitted=False, planned_offset_s=1.0, start_offset_s=None, end_offset_s=2.0)])
    projection = module.concurrency_projection(control)
    check('missing active interval is unknown not zero', projection['admitted_active_session_intervals'] == dict(peak=None, known_interval_peak=1, complete=False, intervals=1, missing_intervals=1))
    check('nonadmitted counts in open outstanding only', projection['open_outstanding_session_intervals']['peak'] == 3)
    check('all reviewed evidence unchanged', all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == i['sha256'] and Path(p).stat().st_size == i['bytes'] for p, i in ledger.items())))
    result = dict(status='PASS_INCREMENTAL_PROVENANCE_AND_CONCURRENCY_REVIEW', utc=datetime.now(timezone.utc).isoformat(), checks=checks, regressions=regressions, reviewed_source_sha256=ledger[str(SOURCE)]['sha256'], phase_concurrency=peaks, evidence_ledger=ledger, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitations=['No model, HTTP, GPU, task execution, or all-started capacity certification.', 'Original 5625 numerical checks reused by exact r1/r2 primary/performance equality; new reference binding and concurrent intervals checked independently.', 'Pilots remain nonformal; c02 r2 selects existing separate references via explicit map.', 'Intervals exclude wrapper/persistence tails; client reported open task peak is separate.', 'Original full raw/SSE/grade audits are reused; this is a selected terminal snapshot, not final Stage C delivery.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), phase_concurrency=len(peaks), regressions=regressions)))


if __name__ == '__main__':
    main()
