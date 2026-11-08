#!/usr/bin/env python3
"""Execute authored D Python notebook cells in order; no Jupyter lifecycle claim."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse,base64,contextlib,hashlib,io,json,sys,traceback
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(a):
    if a.output.exists():
        raise FileExistsError('Use unique output paths')
    if any((path_at(D, 'results').resolve() in p.resolve().parents for p in (a.output, a.log))):
        raise ValueError('Do not write raw results')
    source = D / 'notebooks/optimization-evidence.ipynb'
    paths = [source, Path(__file__), D / 'scripts/plot_performance.py', path_at(D, 'summary/performance-final-r1.json'), path_at(D, 'summary/profile-matched-r1.json'), path_at(D, 'results/scheduler-tp4-fairshare-dense-profile/status.json'), path_at(D, 'figures/profile-matched-r1/request-allocations.png')]
    before = {str(p.resolve()): sha(p) for p in paths}
    nb = load_document(source)
    scope = {}
    records = []
    error = None
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        count = len(records) + 1
        cell['outputs'] = []
        cell['execution_count'] = count
        stream = io.StringIO()
        images = []
        code = ''.join(cell['source'])

        def show(*args, **kwargs):
            for index in plt.get_fignums():
                fig = plt.figure(index)
                b = io.BytesIO()
                fig.savefig(b, format='png', dpi=130, bbox_inches='tight')
                images.append(base64.b64encode(b.getvalue()).decode())
                plt.close(fig)
        plt.show = show
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(code, str(source) + '#' + str(count), 'exec'), scope)
        except Exception as exc:
            error = dict(cell=count, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if stream.getvalue():
            cell['outputs'].append(dict(output_type='stream', name='stdout', text=stream.getvalue().splitlines(keepends=True)))
        for image in images:
            cell['outputs'].append(dict(output_type='display_data', data={'image/png': image}, metadata={}))
        records.append(dict(cell=count, status='FAIL' if error else 'PASS', code_sha256=hashlib.sha256(code.encode()).hexdigest(), figures=len(images), stdout_bytes=len(stream.getvalue().encode())))
        if error:
            cell['outputs'].append(dict(output_type='error', ename=error['type'], evalue=error['message'], traceback=error['traceback'].splitlines()))
            break
    unchanged = all((sha(Path(p)) == v for p, v in before.items()))
    status = 'PASS' if not error and unchanged else 'FAIL'
    method = 'Sequential Python cell execution, captured stdout and actual Matplotlib figures; not a Jupyter kernel lifecycle test.'
    nb['metadata']['execution_evidence'] = dict(status=status, method=method, inputs=before)
    for p in (a.output, a.log):
        p.parent.mkdir(parents=True, exist_ok=True)
    with open_output(a.output, 'x') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    log = dict(status=status, method=method, command=sys.argv, python=sys.version, python_executable=sys.executable, matplotlib=matplotlib.__version__, inputs=before, inputs_unchanged=unchanged, cells=records, error=error, output=str(a.output), output_sha256=sha(a.output))
    write_text(a.log, json.dumps(log, indent=2) + '\n')
    print(json.dumps(dict(status=status, cells=len(records), figures=sum((r['figures'] for r in records)), error=error)))
    return 0 if status == 'PASS' else 1
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    a = ap.parse_args()
    if any((p.resolve().is_relative_to(path_at(D, 'results').resolve()) for p in (a.output, a.log))):
        raise ValueError('Never write notebook/log into raw results')
    if a.log.exists():
        raise FileExistsError('Use unique operation log')
    a.log.parent.mkdir(parents=True, exist_ok=True)
    with open_output(a.log, 'x') as f:
        json.dump(dict(status='IN PROGRESS'), f)
    try:
        return execute(a)
    except Exception as exc:
        write_text(a.log, json.dumps(dict(status='FAILED', command=sys.argv, error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()), output=str(a.output)), indent=2) + '\n')
        raise
if __name__=='__main__':raise SystemExit(main())
