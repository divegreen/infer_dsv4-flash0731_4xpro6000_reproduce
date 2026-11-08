"""Create immutable Stage C configs/plans; never overwrite old experiment IDs."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'scenarios'

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(path, 'x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def config(name, reference, apc):
    d = load_document(path_at(ROOT, 'configs') / reference)
    d['name'] = name
    args = d['args']
    args = [a for a in args if a not in ('--enable-prefix-caching', '--no-enable-prefix-caching')]
    args += ['--enable-prefix-caching' if apc else '--no-enable-prefix-caching']
    d['args'] = args
    d.setdefault('env', {})['VLLM_SERVER_DEV_MODE'] = '1'
    return d

def main():
    name = 'session-tp4-apc-pilot'
    c = config(name, 'context-tp4-screen.json', True)
    p = dict(schema_version=1, experiment_id=name, base_url='http://127.0.0.1:8260', model='dsv4', timeout_s=180, session_timeout_s=600, max_active_sessions=64, phases=[dict(id='warm-replay-shapes', kind='replay', arrival='closed', role='warmup', sessions=3, concurrency=1, profiles=['coding', 'research', 'data'], seed=202611601, cache_scope='session'), dict(id='pilot-replay-c2', kind='replay', arrival='closed', role='pilot', sessions=6, concurrency=2, profiles=['coding', 'research', 'data'], seed=202611602, cache_scope='session'), dict(id='pilot-agent-c1', kind='agent', arrival='closed', role='pilot', sessions=6, concurrency=1, tasks=['reservation_book', 'invoice_adjustments', 'release_gate', 'failover_policy', 'sales_net', 'service_latency'], seed=202611603, cache_scope='session', max_turns=12, max_tools=32, max_output_tokens=1024)])
    put(path_at(ROOT, 'configs') / f'{name}.json', c)
    put(path_at(ROOT, 'workloads/sessions') / f'{name}.json', p)
    series = dict(jobs=[dict(id=name, config=f'{name}.json', session_plan=f'phase-c/{name}.json', timeout=2400, reference='stage-A-and-B-frozen-evidence', hypothesis='Validate exact multi-round replay, streaming tool-only responses, actual APC hits and six independently graded agent tasks before matched capacity study.')])
    put(path_at(ROOT, 'workloads/sessions/series-pilot.json'), series)
    print(json.dumps(dict(status='CREATED_NOT_EXECUTED', id=name), indent=2))
if __name__=='__main__':main()
