#!/usr/bin/env python3
"""Independent final Notebook/plot numeric and byte review; no rendering or GPU."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import base64
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ledger, checks = ({}, [])

    def read(path):
        raw = path.read_bytes()
        ledger[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return raw

    def load(path):
        return json.loads(read(path))

    def check(name, value):
        if not value:
            raise ValueError(name)
        checks.append(name)

    def equal(a, b):
        return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    report = load(path_at(ROOT, 'summary/final-r1.json'))
    stages = {(s['experiment'], s['phase']['id']): s for s in report['stages']}
    closed = load(path_at(ROOT, 'figures/closed-final-r1/manifest.json'))
    opened = load(path_at(ROOT, 'figures/open-final-r2/manifest.json'))
    old_opened = load(path_at(ROOT, 'figures/open-final-r1/manifest.json'))
    check('open visual repair changed no data or grouping', opened['groups'] == old_opened['groups'] and opened['excluded_open_phases'] == old_opened['excluded_open_phases'])
    modules = {}
    for key, filename, manifest in [('closed', 'plot_session_comparisons.py', closed), ('open', 'plot_open_sessions.py', opened)]:
        source = ROOT / 'scripts' / filename
        check('plot source bound ' + key, hashlib.sha256(read(source)).hexdigest() == manifest['script_sha256'])
        check('final summary bound ' + key, manifest['summary_sha256'] == ledger[str(path_at(ROOT, 'summary/final-r1.json'))]['sha256'])
        spec = importlib.util.spec_from_file_location('final_numeric_' + key, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[key] = module
    crows, cexcluded = modules['closed'].collect(report)
    cgroups = modules['closed'].groups(crows)
    orows, oexcluded, plans = modules['open'].collect(report)
    ogroups = modules['open'].groups(orows)
    check('closed manifest exact actual projections', json.loads(json.dumps(cgroups)) == closed['comparisons'] and cexcluded == closed['excluded'])
    check('open manifest exact actual projections', json.loads(json.dumps(ogroups)) == opened['groups'] and oexcluded == opened['excluded_open_phases'] and (plans == opened['audited_plan_inputs']))
    check('35 closed and 25 open accepted phases', len(crows) == 35 and len(orows) == 25 and (len(cgroups) == 7) and (len(ogroups) == 4))
    all_keys = [(r['experiment'], r['phase_id']) for r in crows + orows]
    check('plots exact normal accepted set and no failed ranking', len(all_keys) == len(set(all_keys)) == 60 and set(all_keys) == {k for k, s in stages.items() if s['included_as_measurement']} and (not any((k[0] in ('session-tp4-apc-pilot', 'session-dspark3-agent-confirm') for k in all_keys))))
    for kind, rows in [('closed', crows), ('open', orows)]:
        for row in rows:
            stage = stages[row['experiment'], row['phase_id']]
            stats = stage['summary']
            label = kind + '/' + row['experiment'] + '/' + row['phase_id']
            mapping = dict(sessions='sessions', protocol_completed='protocol_completed', successful_tasks='successful_tasks', protocol_sessions_s='completed_sessions_s', successful_tasks_s='successful_tasks_s', task_success_rate='task_success_rate', output_tokens_s='output_tokens_s', cache_fraction='cache_fraction')
            check('independent row metrics ' + label, all((row[k] == stats[v] for k, v in mapping.items())))
            check('independent cohort-rate denominator ' + label, equal(row['protocol_sessions_s'], stats['protocol_completed'] / stage['total_wall_s']) and equal(row['successful_tasks_s'], stats['successful_tasks'] / stage['total_wall_s'] if row['kind'] == 'agent' else None))
            if row['kind'] == 'agent':
                check('strict task fraction of all assigned ' + label, equal(row['task_success_rate'], stats['successful_tasks'] / stats['sessions']))
            else:
                check('replay no task grade ' + label, row['successful_tasks'] is row['successful_tasks_s'] is row['task_success_rate'] is None)
            visible_key, count_key = ('visible_latency_p95_s', 'visible_latency_samples') if kind == 'closed' else ('visible_p95_s', 'visible_latency_n')
            check('actual visible quantile and N ' + label, row[visible_key] == stats['user_visible_latency_s']['p95'] and row[count_key] == stats['user_visible_latency_s']['n'])
            if kind == 'open':
                check('window classification preserved ' + label, row['classification'] == stage['stability']['classification'] and row['completion_deficit'] == stage['stability']['completion_deficit'] and (row['outstanding_increase'] == stage['stability']['session_outstanding_increase']))
    check('c15 failed formal exclusions explicit', {(e['experiment'], e['phase_id']) for e in oexcluded} == {('session-dspark3-agent-confirm', phase) for phase in ('agent-open-r01-s01', 'agent-open-r01-s02', 'agent-open-r02-s01')})
    table = path_at(ROOT, 'figures/closed-final-r1') / closed['table']['file']
    blob = read(table)
    check('closed CSV hash', hashlib.sha256(blob).hexdigest() == closed['table']['sha256'])
    csvrows = list(csv.DictReader(io.StringIO(blob.decode())))
    byid = {(r['experiment'], r['phase_id']): r for r in crows}
    check('CSV row coverage', len(csvrows) == len(byid))
    for row in csvrows:
        actual = byid[row['experiment'], row['phase_id']]
        for field, value in actual.items():
            observed = row[field]
            check('CSV field ' + row['experiment'] + '/' + row['phase_id'] + '/' + field, observed == '' if value is None else json.loads(observed) == value if isinstance(value, (dict, list)) else observed == str(value))
    for groups, is_open in ((cgroups, False), (ogroups, True)):
        for group in groups:
            if is_open:
                pairs = {(r['condition'], r['rate'], r['seed']) for r in group['rows']}
                workload_cells = {(r['rate'], r['seed']) for r in group['rows']}
                expected = {(condition, rate, seed) for condition in group['conditions'] for rate, seed in workload_cells}
                missing = {(r['condition'], r['rate'], r['seed']) for r in group['missing_pairs']}
                complete = group['complete_matched_grid']
            else:
                pairs = {(r['condition'], r['seed']) for r in group['rows']}
                expected = {(condition, seed) for condition in group['conditions'] for seed in group['seeds']}
                missing = {(r['condition'], r['seed']) for r in group['missing_pairs']}
                complete = group['complete_paired_grid']
            check('no fabricated missing pairs ' + group['id'], len(pairs) == len(group['rows']) and missing == expected - pairs and (complete is (pairs == expected)))
    for directory, manifest in [('closed-final-r1', closed), ('open-final-r2', opened)]:
        for figure in manifest['figures']:
            raw = read(path_at(ROOT, 'figures') / directory / figure['file'])
            check('actual rendered image ' + directory + '/' + figure['file'], len(raw) == figure['bytes'] and hashlib.sha256(raw).hexdigest() == figure['sha256'])
    bookpath = ROOT / 'notebooks/session-evidence-final-r1-executed.ipynb'
    book = load(bookpath)
    execution = load(path_at(ROOT, 'logs/notebook-final-execution-r1.json'))
    template_path = ROOT / 'notebooks/session-evidence.ipynb'
    template = load(template_path)
    check('Notebook output exact execution', execution['status'] == 'PASS' and execution['error'] is None and (execution['output_sha256'] == ledger[str(bookpath)]['sha256']))
    for path, expected in execution['inputs'].items():
        read(Path(path))
        check('execution consumed input ' + path, ledger[path] == expected)
    code = [c for c in book['cells'] if c['cell_type'] == 'code']
    original_code = [c for c in template['cells'] if c['cell_type'] == 'code']
    check('eleven executed cells', len(code) == len(original_code) == len(execution['cells']) == 11)
    for number, (cell, original, entry) in enumerate(zip(code, original_code, execution['cells']), 1):
        actual = ''.join(cell['source'])
        expected = ''.join(original['source']).replace('CONFIRMATION_RELATIVE = None', "CONFIRMATION_RELATIVE = 'logs/session-confirmations-final-r1.json'")
        expected = expected.replace('CUMULATIVE_RELATIVE = None', "CUMULATIVE_RELATIVE = 'summary/cumulative-final-r1.json'")
        check('executed source and no errors ' + str(number), actual == expected and cell['execution_count'] == number and (entry['cell'] == number) and (entry['status'] == 'PASS') and (hashlib.sha256(actual.encode()).hexdigest() == entry['source_sha256']) and (not any((o['output_type'] == 'error' for o in cell['outputs']))))
    namespace, streams = ({}, [])
    for index in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21):
        cell = book['cells'][index]
        source = ''.join(cell['source'])
        if index == 11:
            source = source.split('\nif comparison_views:', 1)[0]
        if index == 17:
            source = source.split('\nif open_views:', 1)[0]
        capture = io.StringIO()
        with redirect_stdout(capture):
            exec(compile(source, str(bookpath) + ':read-only-cell' + str(index), 'exec'), namespace)
        stored = ''.join((''.join(output.get('text', [])) for output in cell['outputs'] if output['output_type'] == 'stream'))
        check('actual numeric stdout reproduced cell ' + str(index), capture.getvalue() == stored)
        streams.append(dict(cell_index=index, stdout_sha256=hashlib.sha256(stored.encode()).hexdigest(), bytes=len(stored.encode())))
    check('accepted versus failed inventory independent', len(namespace['stages']) == 60 and namespace['execution_inventory_rows'] == execution['execution_inventory_rows'] and (sum((r['all_executed_sessions'] for r in namespace['execution_inventory_rows'])) == 2564) and (sum((r['failed_run_executed_confirm_sessions'] for r in namespace['execution_inventory_rows'])) == 371) and (len(namespace['failed_run_rows']) == execution['failed_run_diagnostic_rows'] == 7))
    check('original unstarted phase visible', namespace['unstarted_phase_rows'] == execution['unstarted_phase_rows'] and len(namespace['unstarted_phase_rows']) == 1 and (namespace['unstarted_phase_rows'][0]['experiment'] == 'session-dspark3-agent-confirm'))
    check('actual final confirmation zero and blockers shown', namespace['confirmation']['confirmed_protocol_points'] == 0 and len(namespace['confirmation']['unresolved_started_identity']) == 3)
    check('cumulative counts and consumption bindings passed', namespace['cumulative_counts_match'] is True and namespace['cumulative_report']['inventory']['formal_agent_attempts'] == 908)
    cumulative = namespace['cumulative_report']
    for path in namespace['cumulative_bound_files']:
        read(Path(path))
        check('cumulative consumed file bound ' + path, ledger[path] == cumulative['input_ledger'][path])
    images = [output['data']['image/png'] for cell in code for output in cell['outputs'] if output['output_type'] == 'display_data' and 'image/png' in output.get('data', {})]
    check('eleven actual embedded images', len(images) == 11 and all((base64.b64decode(image).startswith(b'\x89PNG\r\n\x1a\n') for image in images)))
    check('all reviewed bytes unchanged', all((hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'] and Path(path).stat().st_size == info['bytes'] for path, info in ledger.items())))
    result = dict(status='PASS_INDEPENDENT_FINAL_NOTEBOOK_AND_FIGURES', utc=datetime.now(timezone.utc).isoformat(), checks=checks, numeric_cell_stdout=streams, static_figures=22, embedded_pngs=11, normal_phases=60, failed_executed_confirm_sessions=371, full_scan_confirmed_points=0, evidence_ledger=ledger, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitations=['No plot or GPU rerun; actual image layout uses separately recorded full visual inspections.', 'Sequential Notebook execution is verified as recorded, not a Jupyter kernel lifecycle test.', 'Inspected numerical/read-only cells replayed only against existing evidence; no raw/SSE/grader rerun.', 'Open r1 invisible-x presentation flaw preserved; r2 data/groups unchanged and actual image bytes checked.', 'Normal accepted plots exclude failed runs; executed failure rows and all-started blockers remain explicit separately.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), numeric_cells=len(streams), static_figures=22, embedded_pngs=11)))


if __name__ == '__main__':
    main()
