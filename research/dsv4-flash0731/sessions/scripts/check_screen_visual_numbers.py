#!/usr/bin/env python3
"""Independent numeric checks for frozen screen plots and executed Notebook.

No rendering, GPU/HTTP, model calls or original writes. Replays only inspected
read-only numeric Notebook cells; existing plot image bytes are hash-checked.
"""

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
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
from types import ModuleType

ROOT=Path(__file__).resolve().parents[1]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    checks = []
    ledger = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path.resolve())] = dict(sha256=sha(raw), bytes=len(raw))
        return raw

    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append(label)
    folder = path_at(ROOT, 'figures/screen-c03-c07-r2')
    manifest = json.loads(read(folder / 'manifest.json'))
    source = read(path_at(ROOT, 'logs/plot-session-screen-r2-source.py.txt'))
    check('frozen_plot_source_identity', sha(source) == manifest['script_sha256'])
    summary_raw = read(Path(manifest['summary']))
    report = json.loads(summary_raw)
    check('plot_summary_identity', sha(summary_raw) == manifest['summary_sha256'] and report['status'] == 'PASS')
    frozen = ModuleType('frozen_screen_plots')
    frozen.__file__ = str(ROOT / 'scripts/plot_session_comparisons.py')
    exec(compile(source, frozen.__file__, 'exec'), frozen.__dict__)
    rows, excluded = frozen.collect(report)
    views = frozen.groups(rows)
    check('manifest_exact_frozen_comparisons', json.loads(json.dumps(views)) == manifest['comparisons'])
    check('manifest_exact_exclusions', excluded == manifest['excluded'])
    check('seven_groups_35_normal_phases_6_exclusions', len(views) == 7 and len(rows) == 35 and (len(excluded) == 6))
    check('no_pilot_warmup_profile_formal', all((r['role'] in ('screen', 'measure', 'confirm') for r in rows)))
    check('162_real_task_attempts', sum((r['sessions'] for r in rows if r['kind'] == 'agent')) == 162)
    stages = {(s['experiment'], s['phase']['id']): s for s in report['stages']}
    metric_map = {'protocol_sessions_s': 'completed_sessions_s', 'successful_tasks_s': 'successful_tasks_s', 'task_success_rate': 'task_success_rate', 'output_tokens_s': 'output_tokens_s', 'cache_fraction': 'cache_fraction', 'actual_model_calls': 'model_calls', 'actual_tool_calls': 'tool_calls', 'successful_tasks': 'successful_tasks', 'protocol_completed': 'protocol_completed', 'sessions': 'sessions'}
    for row in rows:
        s = stages[row['experiment'], row['phase_id']]
        stats = s['summary']
        tag = row['experiment'] + '/' + row['phase_id']
        check(tag + ' exact_metrics', all((row[k] == stats[v] for k, v in metric_map.items())))
        check(tag + ' throughput_full_cohort_denominator', math.isclose(row['protocol_sessions_s'], stats['protocol_completed'] / s['total_wall_s'], rel_tol=1e-12))
        check(tag + ' visible_latency_sample_count', row['visible_latency_samples'] == stats['user_visible_latency_s']['n'] and row['visible_latency_p95_s'] == stats['user_visible_latency_s']['p95'])
        if row['kind'] == 'agent':
            check(tag + ' all_attempt_task_quality_denominator', math.isclose(row['task_success_rate'], stats['successful_tasks'] / stats['sessions'], rel_tol=1e-12) and math.isclose(row['successful_tasks_s'], stats['successful_tasks'] / s['total_wall_s'], rel_tol=1e-12))
        else:
            check(tag + ' no_replay_task_score', row['successful_tasks'] is None and row['successful_tasks_s'] is None and (row['task_success_rate'] is None))
    table_raw = read(folder / manifest['table']['file'])
    check('CSV_SHA', sha(table_raw) == manifest['table']['sha256'])
    csvrows = list(csv.DictReader(io.StringIO(table_raw.decode())))
    check('CSV_all_formal_rows', len(csvrows) == len(rows))
    byid = {(r['experiment'], r['phase_id']): r for r in rows}
    for r in csvrows:
        expected = byid[r['experiment'], r['phase_id']]
        for key, value in expected.items():
            actual = r[key]
            check('CSV ' + r['experiment'] + '/' + r['phase_id'] + '/' + key, actual == '' if value is None else json.loads(actual) == value if isinstance(value, (dict, list)) else actual == str(value))
    for view in views:
        pairs = {(r['condition'], r['seed']) for r in view['rows']}
        expected = {(c, s) for c in view['conditions'] for s in view['seeds']}
        check(view['id'] + ' no_fabricated_pairs', len(pairs) == len(view['rows']) and view['complete_paired_grid'] == (pairs == expected) and ({(m['condition'], m['seed']) for m in view['missing_pairs']} == expected - pairs))
        for seed in view['seeds']:
            selected = [r for r in view['rows'] if r['seed'] == seed]
            check(view['id'] + f' seed{seed} matched_input', len({r['arrival_workload_sha256'] for r in selected}) == 1 and len({r['sessions'] for r in selected}) == 1)
    for figure in manifest['figures']:
        raw = read(folder / figure['file'])
        check('figure byte identity ' + figure['file'], sha(raw) == figure['sha256'] and len(raw) == figure['bytes'])
    check('14_png_pdf_artifacts', len(manifest['figures']) == 14)
    bookpath = ROOT / 'notebooks/session-evidence-screen-c03-c07-executed-r1.ipynb'
    bookraw = read(bookpath)
    book = json.loads(bookraw)
    execution = json.loads(read(path_at(ROOT, 'logs/notebook-screen-c03-c07-execution-r1.json')))
    check('executed_notebook_identity', execution['status'] == 'PASS' and execution['output_sha256'] == sha(bookraw))
    code = [c for c in book['cells'] if c['cell_type'] == 'code']
    check('nine_executed_code_cells', len(code) == len(execution['cells']) == 9 and all((c['execution_count'] == i + 1 for i, c in enumerate(code))))
    for cell, entry in zip(code, execution['cells']):
        check('executed_cell_source_' + str(entry['cell']), sha(''.join(cell['source']).encode()) == entry['source_sha256'] and entry['status'] == 'PASS')
        check('no_notebook_error_' + str(entry['cell']), not any((o['output_type'] == 'error' for o in cell.get('outputs', []))))
    namespace = {}
    streams = []
    for index in (1, 3, 5, 7, 11, 13, 17):
        cell = book['cells'][index]
        source = ''.join(cell['source'])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(source, f'{bookpath}:cell{index}', 'exec'), namespace)
        stored = ''.join((''.join(o.get('text', [])) for o in cell['outputs'] if o['output_type'] == 'stream'))
        check(f'notebook_numeric_stdout_exact_cell{index}', buffer.getvalue() == stored)
        streams.append(dict(cell_index=index, stdout_sha256=sha(stored.encode()), bytes=len(stored.encode())))
    check('notebook_same_35_formal_phases', len(namespace['stages']) == 35 and {(s['experiment'], s['phase']['id']) for s in namespace['stages']} == set(byid))
    expected = io.StringIO()
    with redirect_stdout(expected):
        print('Formal matched groups:', len(views), '; excluded phases:', len(excluded))
        for view in views:
            print(view['id'], 'complete condition/seed grid:', view['complete_paired_grid'], 'missing:', view['missing_pairs'])
            for row in view['rows']:
                print(' ', row['experiment'], row['phase_id'], 'N=', row['sessions'], 'visible-latency N=', row['visible_latency_samples'], 'workload SHA=', row['arrival_workload_sha256'])
    plotcell = book['cells'][9]
    stored = ''.join((''.join(o.get('text', [])) for o in plotcell['outputs'] if o['output_type'] == 'stream'))
    check('notebook_group_numbers_exact', expected.getvalue() == stored)
    images = [o['data']['image/png'] for o in plotcell['outputs'] if o['output_type'] == 'display_data' and 'image/png' in o.get('data', {})]
    check('seven_actual_png_notebook_outputs', len(images) == 7 and all((base64.b64decode(v).startswith(b'\x89PNG\r\n\x1a\n') for v in images)))
    open_stdout = ''.join((''.join(o.get('text', [])) for o in book['cells'][15]['outputs'] if o['output_type'] == 'stream'))
    check('notebook_does_not_invent_open_capacity', 'Real eligible Poisson windows: 0' in open_stdout and 'no open performance plot generated' in open_stdout)
    check('all_checked_artifacts_unchanged', all((sha(Path(p).read_bytes()) == v['sha256'] for p, v in ledger.items())))
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='Independent numeric/byte review of existing frozen screen figures and executed Notebook; no plotting or GPU rerun', checks=checks, reviewer_sha256=sha(Path(__file__).read_bytes()), figures=len(manifest['figures']), comparisons=len(views), normal_phases=len(rows), normal_agent_attempts=162, notebook_numeric_stdout=streams, evidence_ledger=ledger, limitations=['Image byte identity and data pipeline checked; visual layout reuses the separate recorded seven-figure visual inspection.', 'Numeric Notebook cells use unchanged audited summary/descriptor inputs; this is not an independent rerun of GPU measurements or a Jupyter kernel lifecycle test.', 'The frozen r2 plot source is verified from its saved snapshot; current presentation wording can differ and requires a new artifact revision.'])
    with open_output(a.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), figures=len(manifest['figures']), normal_phases=len(rows), output=str(a.output)), indent=2))


if __name__=='__main__':main()
