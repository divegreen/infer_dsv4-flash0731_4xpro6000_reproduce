#!/usr/bin/env python3
"""Independent CPU preflight for the explicit agent-confirm-r1 decision.

Reconstruct arrival/task assignment, not dynamic Agent model outputs. No plan
generator import, task tool, grader, GPU, service launch or HTTP execution.
"""

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
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];RESEARCH=ROOT.parent;SCENARIO=RESEARCH/'scenarios'
sys.path.insert(0,str(SCENARIO/'scripts'))
import agent_session_client as client
TASKS=['reservation_book','invoice_adjustments','release_gate','failover_policy','sales_net','service_latency']


def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(obj):return sha(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def review():
    checks = []
    ledger = {}
    windows = []
    cohorts = {}
    selection_evidence = []

    def need(label, value):
        if not value:
            raise ValueError(label)
        checks.append(label)

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        entry = dict(sha256=sha(raw), bytes=len(raw))
        need('input immutable ' + str(path), str(path) not in ledger or ledger[str(path)] == entry)
        ledger[str(path)] = entry
        return raw

    def load(path):
        return json.loads(read(path))
    raw = read(ROOT / 'decisions/agent-confirm-r1.json')
    decision = json.loads(raw)
    generation = load(path_at(ROOT, 'logs/open-plan-generation-agent-confirm-r1.json'))
    provenance = generation['decision_provenance']
    need('explicit decision and raw provenance', decision['decision_id'] == 'agent-confirm-r1' and provenance['decision_sha256'] == sha(raw) and (read(path_at(SCENARIO, 'workloads/sessions/decision-agent-confirm-r1.json')) == raw))
    need('plan construction not execution', generation['status'] == 'GENERATED_NOT_EXECUTED')
    need('unchanged service protocol', decision['service_protocol'] == dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64) and decision['warmup_seed'] == 202611609 and (decision['supervisor_timeout_s'] == 3600))
    need('same frozen finite-session rule', sha(read(path_at(ROOT, 'finite-session-v1.md'))) == '9f48277f1e0cbfa75a0abb742f1a3bdbf6e7224d9b39324f0056881986faf68e')
    need('same frozen generator', provenance['generator_source_sha256'] == sha(read(ROOT / 'scripts/build_open_session_plans.py')) == '4e8a80cf2410f54b79f5d977d7a4238e13a7b95c91cd2151e1cd0fc78d49ea83')
    need('same frozen client', sha(read(SCENARIO / 'scripts/agent_session_client.py')) == '9b7889dade2b7b2499b898f887b6e33720bca630068f35759a106b8bf98b3183')
    for path, expected in provenance['runtime_source_sha256'].items():
        need('runtime/task source ' + path, sha(read(RESEARCH / path)) == expected)
    need('exact evidence list', [e['path'] for e in provenance['evidence_references']] == decision['evidence_references'])
    for evidence in provenance['evidence_references']:
        body = read(RESEARCH / evidence['path'])
        need('reference SHA ' + evidence['path'], sha(body) == evidence['sha256'] and len(body) == evidence['bytes'])
    expectations = {'session-tp4-agent-open': [(26, 20, 3.771, 'INCONCLUSIVE', 52.8, 37.599), (50, 34, 7.885, 'INCONCLUSIVE', 81.64, 55.388)], 'session-dspark3-agent-open': [(26, 23, 3.244, 'GROWING_BACKLOG', 22.118, 8.442), (50, 44, -0.653, 'INCONCLUSIVE', 36.717, 19.268)]}
    for name, expected in expectations.items():
        summary = load(path_at(ROOT, 'summary') / f'{name}-terminal-r1.json')
        audit = load(path_at(ROOT, 'logs') / f'audit-{name}-r1.json')
        need('healthy actual screen ' + name, summary['status'] == audit['status'] == 'PASS' and summary['experiments'][0]['terminal_success'] is True and (audit['experiments'][0]['runtime_terminal_success'] is True))
        byphase = {s['phase']['id']: s for s in summary['stages']}
        audited = {s['id']: s for s in audit['experiments'][0]['phases']}
        for index, label in enumerate(['agent-open-r02-s01', 'agent-open-r03-s01']):
            s = byphase[label]
            v = s['stability']
            a = audited[label]['stability']
            A, C, delta, trend, p95, drain = expected[index]
            need('rationale actual counters ' + name + '/' + label, (v['analysis_arrivals'], v['analysis_completions']) == (A, C) == (a['analysis_arrivals'], a['analysis_protocol_completions']) and round(v['session_outstanding_increase'], 3) == delta and (abs(v['session_outstanding_increase'] - a['session_mean_outstanding_increase']) < 1e-09) and (v['trend_verdict'] == a['trend_verdict'] == trend) and (v['classification'] == 'SCREEN_ONLY'))
            need('rationale quality and visible latency ' + name + '/' + label, s['summary']['successful_tasks'] == [12, 22][index] and s['summary']['sessions'] == [35, 65][index] and (abs(s['summary']['user_visible_latency_s']['p95'] - p95) < 0.0051) and (abs(s['drain_s'] - drain) < 0.00051))
            selection_evidence.append(dict(experiment=name, phase=label, arrivals=A, completions=C, outstanding_increase=v['session_outstanding_increase'], trend=trend, visible_latency_p95_s=s['summary']['user_visible_latency_s']['p95'], drain_s=s['drain_s']))
    runtime = load(path_at(ROOT, 'logs/session-dspark3-agent-open-runtime-open-screen-r1.json'))
    jit = [e for e in runtime['experiments'][0]['visible_jit_warnings'] if 'ComputeGlobalTopk' in e['text']]
    need('measurement JIT retained', len(jit) == 1 and jit[0]['mapping']['unique_containing_phase'] == 'agent-open-r01-s01' and (jit[0]['mapping']['scope'] == 'MEASUREMENT_OR_BOUNDARY') and ('TopK' in decision['rationale']))
    prior = load(ROOT / 'decisions/open-screen-r1.json')
    prior_seeds = {seed for s in prior['selections'] for seed in s['seeds']}
    queue = load(path_at(SCENARIO, 'workloads/sessions/series-agent-confirm-r1.json'))
    ids = ['session-tp4-agent-confirm', 'session-dspark3-agent-confirm']
    need('exact queue order', [j['id'] for j in queue['jobs']] == generation['experiment_ids'] == ids and queue['decision_provenance'] == provenance)
    generated = {c['experiment_id']: c for c in generation['cohorts']}
    total = measured = 0
    need('two explicitly separate selections', len(decision['selections']) == 2)
    for selection, expected_id, rates, job in zip(decision['selections'], ids, [[0.2, 0.3], [0.3, 0.6]], queue['jobs']):
        need('explicit selection ' + expected_id, selection['kind'] == 'agent' and selection['role'] == 'confirm' and (selection['rates'] == rates) and (selection['seeds'] == [202611642, 202611643]) and (selection['duration_s'] == 300) and (selection['cache_scope'] == 'session') and (len(selection['candidates']) == 1))
        need('independent seeds ' + expected_id, not set(selection['seeds']) & prior_seeds)
        candidate = selection['candidates'][0]
        name = candidate['experiment_id']
        g = generated[name]
        need('not started ' + name, name == expected_id and (not (path_at(SCENARIO, 'results') / name).exists()))
        config = load(path_at(SCENARIO, 'configs') / f'{name}.json')
        ref_raw = read(path_at(SCENARIO, 'configs') / candidate['config_ref'])
        need('config only name differs ' + name, config == {**json.loads(ref_raw), 'name': name})
        plan = load(path_at(SCENARIO, 'workloads/sessions') / f'{name}.json')
        meta = plan['decision_provenance']
        need('plan provenance ' + name, meta['decision_sha256'] == sha(raw) and meta['runtime_source_sha256'] == provenance['runtime_source_sha256'] and (meta['evidence_references'] == provenance['evidence_references']) and (meta['config_reference_sha256'] == g['config_reference_sha256'] == sha(ref_raw)))
        need('owned protocol ' + name, plan['base_url'] == 'http://127.0.0.1:8260' and config['port'] == 8260 and (plan['model'] == 'dsv4') and all((plan[k] == v for k, v in decision['service_protocol'].items())) and (client.validate_plan(plan) == plan))
        need('queue budget ' + name, job['config'] == name + '.json' and job['session_plan'] == 'phase-c/' + name + '.json' and (job['timeout'] == 3600))
        need('five phases ' + name, len(plan['phases']) == 5)
        for pi, f in enumerate(plan['phases']):
            label = name + '/' + f['id']
            need('same task budgets ' + label, f['kind'] == 'agent' and f['tasks'] == TASKS and ((f['max_turns'], f['max_tools'], f['max_output_tokens']) == (12, 32, 4096)) and (f['cache_scope'] == 'session') and (f['dp_routing'] == 'none'))
            if pi == 0:
                need('excluded six-task C1 warmup ' + label, f['arrival'] == 'closed' and f['role'] == 'warmup' and (f['sessions'] == 6) and (f['concurrency'] == 1) and (f['seed'] == 202611609))
                offsets = [None] * 6
            else:
                rate = rates[(pi - 1) // 2]
                seed = selection['seeds'][(pi - 1) % 2]
                need('explicit confirm timing ' + label, f['arrival'] == 'poisson' and f['role'] == 'confirm' and (f['rate'] == rate) and (f['seed'] == seed) and (f['duration_s'] == 300))
                rng = random.Random(seed)
                t = 0.0
                offsets = []
                while True:
                    t += rng.expovariate(rate)
                    if t >= 300:
                        break
                    offsets.append(t)
                schedule = [dict(index=i, planned_offset_s=t, seed=seed + i, work_item=TASKS[i % 6]) for i, t in enumerate(offsets)]
                window = g['windows'][pi - 1]
                A = sum((t >= 75 for t in offsets))
                need('independent arrivals and assignments ' + label, window['planned_sessions'] == len(offsets) and window['planned_analysis_arrivals'] == A >= 20 and (window['expected_analysis_arrivals'] == rate * 225) and (window['arrival_offsets_sha256'] == digest(offsets)) and (window['planned_schedule_sha256'] == digest(schedule)))
                windows.append(dict(experiment=name, phase=f['id'], rate=rate, seed=seed, planned_sessions=len(offsets), analysis_arrivals=A, tasks=dict(Counter((s['work_item'] for s in schedule)))))
                measured += len(offsets)
            bank = client.prepare({**plan, 'phases': [f]})
            rows = bank['phases'][0]['sessions']
            expected_rows = [dict(index=i, planned_offset_s=t, seed=f['seed'] + i, task_id=TASKS[i % 6]) for i, t in enumerate(offsets)]
            need('actual prepare yields exact planned tasks ' + label, rows == expected_rows)
            proof = dict(phase_id=f['id'], seed=f['seed'], planned_sessions=len(rows), prepared_input_sha256=digest(rows))
            need('real tokenizer/prepare metadata ' + label, bank['tokenizer_sha256'] == meta['tokenizer_preflight']['tokenizer_sha256'] and meta['tokenizer_preflight'] == g['tokenizer_preflight'] and (proof == meta['tokenizer_preflight']['phases'][pi]))
            need('session namespace semantics ' + label, client.salt_for(f, 0, 0) == client.salt_for(f, 0, 3) and client.salt_for(f, 0, 0) != client.salt_for(f, 1, 0))
            total += len(rows)
            if pi:
                cohorts[name, f['rate'], f['seed']] = (f, rows)
    shared = []
    for seed in [202611642, 202611643]:
        af, ar = cohorts[ids[0], 0.3, seed]
        bf, br = cohorts[ids[1], 0.3, seed]
        need('common rate matched across different phase labels ' + str(seed), ar == br and {k: v for k, v in af.items() if k != 'id'} == {k: v for k, v in bf.items() if k != 'id'})
        need('label is an isolated cache namespace ' + str(seed), af['id'] != bf['id'] and client.salt_for(af, 0, 0) != client.salt_for(bf, 0, 0))
        shared.append(dict(rate=0.3, seed=seed, reference_phase=af['id'], candidate_phase=bf['id'], planned_sessions=len(ar), assignment_sha256=digest(ar)))
    for created in generation['created_files']:
        body = read(Path(created['path']))
        need('created file identity ' + created['path'], sha(body) == created['sha256'] and len(body) == created['bytes'])
    need('all inputs unchanged', all((sha(Path(p).read_bytes()) == v['sha256'] for p, v in ledger.items())))
    return dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), reviewer_sha256=sha(Path(__file__).read_bytes()), scope='Independent CPU pre-execution decision/arrival/task/cache-namespace review; no GPU/model/tool/grader/HTTP execution.', checks=checks, experiments=ids, planned_sessions_including_warmup=total, planned_confirmation_sessions=measured, planned_windows=windows, common_rate_cross_selection_matches=shared, decision_numeric_evidence=selection_evidence, measurement_jit_evidence=jit, input_ledger=ledger, limitations=['Same task assignments and arrival offsets do not mean identical dynamic histories, model tokens, tool calls, grades or work.', 'Different phase IDs change isolated session cache-salt labels; scope/reuse relationships match and no shared cross-session cache is promised.', 'Only .30 Session/s is a shared cross-architecture confirmation point; .20 TP4 and .60 DSpark are architecture-specific probes.', 'All counts are planned, not completed; every started confirm window must remain in the final audit, including failures and inconclusive seeds.', 'Six-task C1 warmup does not prove all future shapes warmed; the c11 measurement JIT event is retained.', 'No new Session SLO, production SLA, mathematical maximum or quality-equivalence claim is introduced.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-prefix', type=Path, required=True)
    a = p.parse_args()
    outputs = [a.output_prefix.with_suffix(s) for s in ('.json', '.md')]
    if any((p.exists() or p.is_symlink() for p in outputs)):
        raise FileExistsError('Unique review revision required')
    result = review()
    with open_output(outputs[0], 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    lines = ['# 真实 Agent 长窗确认计划独立预检', '', f'PASS（CPU，{len(result['checks'])}项断言）；审查时c14/c15均未启动。', '', f'计划会话{result['planned_sessions_including_warmup']}，正式确认{result['planned_confirmation_sessions']}，其余为12次C1预热。', '', 'TP4 .20/.30、DSpark .30/.60，各300秒、seeds202611642/643。共同.30按rate+seed而非phase ID匹配，实际89/99次计划到达和任务分配相同。', '配置仅name变化、冻结任务/规则/运行源、180/600/64与12/32/4096预算已核对；没有动态Token相同的断言。', '', *['- ' + text for text in result['limitations']]]
    with open_output(outputs[1], 'x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), planned=result['planned_sessions_including_warmup'], confirm_plan=result['planned_confirmation_sessions'])))


if __name__=='__main__':main()
