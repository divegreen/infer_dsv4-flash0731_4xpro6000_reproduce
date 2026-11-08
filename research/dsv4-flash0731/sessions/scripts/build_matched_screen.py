"""Preregister finite matched Stage C screens; no service launch or overwrite."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
from copy import deepcopy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'scenarios'
TASKS=['reservation_book','invoice_adjustments','release_gate','failover_policy','sales_net','service_latency']
PROFILES=['coding','research','data']

def put(p, v):
    with open_output(p, 'x') as f:
        json.dump(v, f, indent=2)
        f.write('\n')

def replay(label,seed,n=24,c=8,scope='session',route='none',role='screen'):
    return dict(id=label,kind='replay',arrival='closed',role=role,sessions=n,concurrency=c,
                profiles=PROFILES,seed=seed,cache_scope=scope,dp_routing=route)

def agent(label,seed,n=12,c=6,scope='session',route='none',role='screen',tasks=None):
    return dict(id=label,kind='agent',arrival='closed',role=role,sessions=n,concurrency=c,
                tasks=tasks or TASKS,seed=seed,cache_scope=scope,dp_routing=route,
                max_turns=12,max_tools=32,max_output_tokens=4096)

def common_replay():
    return [replay('replay-c1-session',202611610,6,1)]+[
        replay(f'replay-c8-session-r{i+1}',202611611+i) for i in range(2)]

def common_agent(serial=True):
    return ([agent('agent-c1-session',202611620,6,1)] if serial else [])+[
        agent(f'agent-c6-session-r{i+1}',202611621+i) for i in range(2)]

def write(name, reference, apc, phases, hypothesis):
    cfg = load_document(path_at(ROOT, 'configs') / reference)
    cfg['name'] = name
    cfg['args'] = [a for a in cfg['args'] if a not in ('--enable-prefix-caching', '--no-enable-prefix-caching')]
    cfg['args'].append('--enable-prefix-caching' if apc else '--no-enable-prefix-caching')
    cfg.setdefault('env', {})['VLLM_SERVER_DEV_MODE'] = '1'
    plan = dict(schema_version=1, experiment_id=name, base_url='http://127.0.0.1:8260', model='dsv4', timeout_s=180, session_timeout_s=600, max_active_sessions=64, phases=phases)
    put(path_at(ROOT, 'configs') / f'{name}.json', cfg)
    put(path_at(ROOT, 'workloads/sessions') / f'{name}.json', plan)
    return dict(id=name, config=f'{name}.json', session_plan=f'phase-c/{name}.json', timeout=3600, reference='stage-A-and-B-frozen-evidence; c01/c02 pilot evidence', hypothesis=hypothesis)

def warm():return replay('warm-replay-shapes',202611609,3,1,role='warmup')

def main():
    phases = [warm(), agent('pilot-numeric-tool-schema', 202611608, 1, 1, role='pilot', tasks=['service_latency']), replay('replay-c1-session', 202611610, 6, 1)]
    for i, scopes in enumerate([['none', 'session', 'shared'], ['shared', 'session', 'none']]):
        phases += [replay(f'replay-c8-{scope}-r{i + 1}', 202611611 + i, scope=scope) for scope in scopes]
    phases += common_agent()
    jobs = [write('session-tp4-cache-screen', 'context-tp4-screen.json', True, phases, 'Same exact 4-turn tokens at C8 under none/session/shared namespaces, reversed treatment order across two seeds; C1/C6 real six-task control; independently fix numeric tool-schema interface before rankings.')]
    jobs.append(write('session-tp4-noapc-screen', 'context-tp4-screen.json', False, [warm()] + common_replay() + common_agent(False), 'Full APC-off deployment reference matched to TP4 on tokens/task plans; distinguishes no-hit namespace from no cache implementation.'))
    for name, ref in [('session-tp2pp2-agent-screen', 'context-tp2pp2-screen.json'), ('session-dspark3-agent-screen', 'context-dspark3-screen.json')]:
        jobs.append(write(name, ref, True, [warm()] + common_replay() + common_agent(), 'Matched replay C1/C8 and real agent C1/C6 with two high-concurrency seeds; keep task grades and dynamic workloads separate; no capacity claim from closed concurrency.'))
    phases = [warm()]
    for i, routes in enumerate([['none', 'sticky4'], ['sticky4', 'none']]):
        phases += [replay(f'replay-c8-{route}-r{i + 1}', 202611611 + i, route=route) for route in routes]
        phases += [agent(f'agent-c6-{route}-r{i + 1}', 202611621 + i, route=route) for route in routes]
    jobs.append(write('session-dp4ep-affinity-screen', 'context-dp4ep-screen.json', True, phases, 'Explicitly test default versus session rank affinity with identical replay/real-task plans; 24 replay sessions and12 agent sessions balance task types over4 ranks; same scopes and output budgets.'))
    put(path_at(ROOT, 'workloads/sessions/series-matched-screen.json'), dict(jobs=jobs))
    print(json.dumps(dict(status='CREATED_NOT_EXECUTED', jobs=[j['id'] for j in jobs]), indent=2))

if __name__=='__main__':main()
