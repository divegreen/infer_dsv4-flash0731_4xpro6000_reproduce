#!/usr/bin/env python3
"""One finite, CPU-only final derivation after the registered GPU queue completes."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
D=Path(__file__).resolve().parents[1];PROJECT=D.parents[2]
PYTHON=PROJECT/'.venv-vllm/bin/python'
IDS=['scheduler-tp4-observe-matched-670','scheduler-tp4-fairshare-matched-670','scheduler-tp4-fairshare-matched-671','scheduler-tp4-observe-matched-671']
STATE = path_at(D, 'logs/final-analysis-pipeline-r2-state.json')
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    state = dict(status='WAITING FOR REGISTERED QUEUE', pid=os.getpid(), started_utc=utc(), commands=[], source_sha256=sha(Path(__file__)))
    with open_output(STATE, 'x') as f:
        json.dump(state, f, indent=2)

    def save():
        temp = STATE.with_suffix('.json.tmp')
        write_text(temp, json.dumps(state, indent=2) + '\n')
        temp.replace(STATE)
    try:
        end = time.monotonic() + 1800
        while True:
            q = load_document(path_at(D, 'logs/series-matched-performance-r1-state.json'))
            if q['status'] == 'COMPLETED':
                break
            if q['status'] != 'IN PROGRESS':
                raise RuntimeError('GPU queue is not successful: ' + q['status'])
            if time.monotonic() > end:
                raise TimeoutError('Waiting for finite normal queue')
            time.sleep(10)
        assert [j['id'] for j in q['jobs']] == IDS and all((j['status'] == 'COMPLETED' for j in q['jobs']))
        for name in IDS:
            s = load_document(path_at(D, 'results') / name / 'status.json')
            assert s['supervisor_exit_code'] == 0 and s['source_identity_unchanged']
        uv = ['uv', 'run', '--offline', '--no-project', '--with', 'numpy==2.3.5', '--with', 'matplotlib==3.10.8', 'python']
        jobs = [('summary', [str(PYTHON), str(D / 'scripts/analyze_performance.py'), '--result-dirs', *[str(path_at(D, 'results') / n) for n in IDS], '--output-prefix', str(path_at(D, 'summary/performance-final-r1'))]), ('inventory', [str(PYTHON), str(D / 'scripts/final_inventory.py'), '--output', str(path_at(D, 'logs/final-inventory-r1.json'))]), ('tables', [str(PYTHON), str(D / 'scripts/build_delivery_tables.py'), '--summary', str(path_at(D, 'summary/performance-final-r1.json')), '--output-prefix', str(path_at(D, 'summary/report-tables-final-r1'))]), ('plots', uv + [str(D / 'scripts/plot_performance.py'), '--summary', str(path_at(D, 'summary/performance-final-r1.json')), '--output-dir', str(path_at(D, 'figures/performance-final-r1'))]), ('notebook', uv + [str(D / 'scripts/execute_notebook.py'), '--output', str(D / 'notebooks/optimization-evidence-executed-r1.ipynb'), '--log', str(path_at(D, 'logs/notebook-execution-r1.json'))])]
        env = os.environ.copy()
        env['UV_CACHE_DIR'] = str(PROJECT / '.cache/uv')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        for name, cmd in jobs:
            log = path_at(D, 'logs') / ('final-analysis-' + name + '-r1.log')
            entry = dict(name=name, command=cmd, log=str(log), started_utc=utc())
            state['commands'].append(entry)
            state.update(status='RUNNING', active=name)
            save()
            with open_output(log, 'x') as f:
                p = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)
            entry.update(exit_code=p.returncode, finished_utc=utc(), log_sha256=sha(log))
            save()
            if p.returncode:
                raise RuntimeError(name + ' failed; preserve logs and outputs, no automatic retry')
        state.update(status='COMPLETED', active=None)
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc))
        raise
    finally:
        state['updated_utc'] = utc()
        save()
if __name__=='__main__':main()
