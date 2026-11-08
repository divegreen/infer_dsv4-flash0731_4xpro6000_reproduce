#!/usr/bin/env python3
"""Execute authored evidence Notebook cells, preserving a unique output copy.

This is sequential Python cell execution, not a Jupyter kernel lifecycle test.
Notebook code reads existing evidence; the harness alone writes output artifacts.
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
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--workload', type=Path)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--cumulative', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'notebooks/session-evidence.ipynb'
    outputs = [args.output.resolve(), args.log.resolve()]
    raw_root = path_at(ROOT.parent, 'scenarios/results')
    if any((p.exists() or raw_root in p.parents for p in outputs)):
        raise ValueError('Use unique outputs outside raw results')
    paths = [source, ROOT / 'scripts/plot_session_comparisons.py', ROOT / 'scripts/plot_open_sessions.py', args.summary.resolve(), Path(__file__)]
    for p in (args.workload, args.confirmation, args.cumulative):
        if p:
            paths.append(p.resolve())
    if args.cumulative:
        paths.extend([ROOT / 'scripts/summarize_session_workloads.py', ROOT / 'scripts/describe_agent_workloads.py'])
    summary = load_document(args.summary)
    if summary.get('status') != 'PASS':
        raise ValueError('Actual evidence-analysis PASS required')
    paths.extend((path_at(ROOT.parent, 'scenarios/results') / e['id'] / 'session-plan.json' for e in summary['experiments']))
    reference_paths = [path_at(ROOT, 'logs') / (e['id'] + suffix) for e in summary['experiments'] for suffix in ('-json-reference-r1.json', '-objective-regrade-r1.json')]
    reference_existence = {str(p): p.is_file() for p in reference_paths}
    paths.extend((p for p in reference_paths if p.is_file()))
    before = {str(p.resolve()): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}
    nb = load_document(source)
    scope = {}
    cells = []
    counter = 0
    error = None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    replacements = {"SUMMARY_RELATIVE = 'summary/final-r1.json'": 'SUMMARY_RELATIVE = ' + repr(str(args.summary.resolve().relative_to(ROOT)))}
    if args.workload:
        replacements["WORKLOAD_RELATIVE = 'summary/workloads-final-r1.json'"] = 'WORKLOAD_RELATIVE = ' + repr(str(args.workload.resolve().relative_to(ROOT)))
    if args.confirmation:
        replacements['CONFIRMATION_RELATIVE = None'] = 'CONFIRMATION_RELATIVE = ' + repr(str(args.confirmation.resolve().relative_to(ROOT)))
    if args.cumulative:
        replacements['CUMULATIVE_RELATIVE = None'] = 'CUMULATIVE_RELATIVE = ' + repr(str(args.cumulative.resolve().relative_to(ROOT)))
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        counter += 1
        code = ''.join(cell['source'])
        for old, new in replacements.items():
            code = code.replace(old, new)
        cell['source'] = code.splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = counter
        stream = io.StringIO()
        images = []

        def show(*_args, **_kwargs):
            for number in plt.get_fignums():
                fig = plt.figure(number)
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
                images.append(base64.b64encode(buffer.getvalue()).decode())
                plt.close(fig)
        plt.show = show
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(code, str(source) + '#' + str(counter), 'exec'), scope)
        except Exception as exc:
            error = dict(cell=counter, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        stdout = stream.getvalue()
        if stdout:
            cell['outputs'].append(dict(output_type='stream', name='stdout', text=stdout.splitlines(keepends=True)))
        for data in images:
            cell['outputs'].append(dict(output_type='display_data', data={'image/png': data}, metadata={}))
        cells.append(dict(cell=counter, status='FAIL' if error else 'PASS', source_sha256=hashlib.sha256(code.encode()).hexdigest(), stdout_bytes=len(stdout.encode()), figures=len(images)))
        if error:
            cell['outputs'].append(dict(output_type='error', ename=error['type'], evalue=error['message'], traceback=error['traceback'].splitlines()))
            break
    unchanged = all((sha(Path(p)) == v['sha256'] and Path(p).stat().st_size == v['bytes'] for p, v in before.items()))
    unchanged = unchanged and reference_existence == {str(p): p.is_file() for p in reference_paths}
    workload_match = None
    if args.workload and (not error):
        description = scope.get('workload_report')
        eligible = {(s['experiment'], s['phase']['id']): s for s in scope['stages']}
        actual = {(s['experiment'], s['phase']['id']): s for s in description['stages']}
        workload_match = all((k in actual and all((actual[k]['summary'][field] == stage['summary'][field] for field in ('sessions', 'successful_tasks', 'model_calls', 'protocol_completed'))) for k, stage in eligible.items()))
    cumulative_match = scope.get('cumulative_counts_match') if args.cumulative else None
    status = 'PASS' if not error and unchanged and (workload_match is not False) and (not args.cumulative or cumulative_match is True) else 'FAIL'
    method = 'Sequential Python code-cell execution with captured stdout and actual Matplotlib PNG; not a Jupyter kernel lifecycle test.'
    nb['metadata']['execution_evidence'] = dict(method=method, source_notebook_sha256=before[str(source)]['sha256'], summary_sha256=sha(args.summary), status=status)
    for p in outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
    with open_output(outputs[0], 'x') as out:
        json.dump(nb, out, ensure_ascii=False, indent=1)
        out.write('\n')
    record = dict(status=status, utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, method=method, python=sys.version, python_executable=sys.executable, matplotlib=matplotlib.__version__, inputs=before, inputs_unchanged=unchanged, workload_summary_counts_match=workload_match, cumulative_summary_counts_match=cumulative_match, output=str(outputs[0]), output_sha256=sha(outputs[0]), cells=cells, error=error, normal_phases=len(scope.get('stages', [])), normal_agent_attempts=sum((s['summary']['sessions'] for s in scope.get('stages', []) if s['phase']['kind'] == 'agent')), closed_comparisons=len(scope.get('comparison_views', [])), open_comparisons=len(scope.get('open_views', [])), execution_inventory_rows=scope.get('execution_inventory_rows', []), failed_run_diagnostic_rows=len(scope.get('failed_run_rows', [])), unstarted_phase_rows=scope.get('unstarted_phase_rows', []), limitations=['Final future datasets require their own actual execution.', 'Full raw bytes are not rechecked by this harness; audit/descriptor integrity evidence remains separately linked.'])
    with open_output(outputs[1], 'x') as out:
        json.dump(record, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: record[k] for k in ('status', 'normal_phases', 'normal_agent_attempts', 'closed_comparisons', 'open_comparisons', 'inputs_unchanged', 'workload_summary_counts_match', 'cumulative_summary_counts_match', 'output', 'error')}, indent=2))
    raise SystemExit(0 if status == 'PASS' else 1)


if __name__=='__main__':main()
