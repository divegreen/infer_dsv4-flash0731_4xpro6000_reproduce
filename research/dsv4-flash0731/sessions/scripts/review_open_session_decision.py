#!/usr/bin/env python3
"""Read-only independent decision/plan/CPU-input review before GPU execution.

Does not import the plan generator, create plans, open HTTP or inspect GPUs.
Poisson schedules are independently reconstructed; actual token preparation
uses the frozen production client/tokenizer, with that shared dependency stated.
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

PHASE=Path(__file__).resolve().parents[1];RESEARCH=PHASE.parent;SCENARIO=RESEARCH/'scenarios'
sys.path.insert(0,str(SCENARIO/'scripts'))
import agent_session_client as client

TASKS=['reservation_book','invoice_adjustments','release_gate','failover_policy','sales_net','service_latency']
PROFILES=['coding','research','data']
GOLDEN={'coding':dict(inputs=[8192,12288,16384,20480],outputs=[128,256,128,256],waits=[.1,.2,.1]),
        'research':dict(inputs=[16384,24576,32768,40960],outputs=[128,128,128,512],waits=[.5,1.,.5]),
        'data':dict(inputs=[4096,6144,8192,10240],outputs=[128,256,128,256],waits=[.1,.5,.1])}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(obj):return sha(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    a = parser.parse_args()
    outputs = [a.output_prefix.with_suffix(s) for s in ('.json', '.md')]
    if any((p.exists() for p in outputs)):
        raise FileExistsError('Use a unique review revision')
    checks = []
    ledger = {}
    window_rows = []
    selection_evidence = []
    execution_observed = []
    planned_total = 0
    measured_total = 0

    def need(name, condition):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        info = dict(sha256=sha(raw), bytes=len(raw))
        need('unchanged repeated input ' + str(path), str(path) not in ledger or ledger[str(path)] == info)
        ledger[str(path)] = info
        return raw

    def load(path):
        return json.loads(read(path))
    decision_raw = read(a.decision)
    decision = json.loads(decision_raw)
    identity = decision['decision_id']
    generation = load(path_at(PHASE, 'logs') / f'open-plan-generation-{identity}.json')
    provenance = generation['decision_provenance']
    need('generated but not executed', generation['status'] == 'GENERATED_NOT_EXECUTED')
    need('decision raw identity', provenance['decision_sha256'] == sha(decision_raw))
    need('decision byte copy preserved', read(path_at(SCENARIO, 'workloads/sessions') / f'decision-{identity}.json') == decision_raw)
    need('frozen generator SHA', sha(read(PHASE / 'scripts/build_open_session_plans.py')) == provenance['generator_source_sha256'] == '4e8a80cf2410f54b79f5d977d7a4238e13a7b95c91cd2151e1cd0fc78d49ea83')
    for name, expected in provenance['runtime_source_sha256'].items():
        need('runtime/task source ' + name, sha(read(RESEARCH / name)) == expected)
    need('fixed replay token/profile specification', client.PROFILES == GOLDEN)
    need('unchanged global service protocol', decision['service_protocol'] == dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64))
    summaries = {}
    for expected in provenance['evidence_references']:
        raw = read(RESEARCH / expected['path'])
        need('decision reference identity ' + expected['path'], sha(raw) == expected['sha256'] and len(raw) == expected['bytes'])
        if expected['path'].endswith('.json'):
            evidence = json.loads(raw)
            expected_status = 'PASS_OFFLINE_CACHE_RESOURCE_SUMMARY' if expected['path'] == 'sessions/logs/cache-resources-c03-c06-r1.json' else 'PASS'
            need('decision evidence success enum ' + expected['path'], evidence.get('status') == expected_status)
            if 'experiments' in evidence and 'stages' in evidence:
                for e in evidence['experiments']:
                    need('healthy prior candidate ' + e['id'], e['terminal_success'] is True and type(e['supervisor_exit_code']) is int and (e['supervisor_exit_code'] == 0))
                    summaries[e['id']] = evidence
    selected_values = {'session-tp4-cache-screen': [('replay-c8-session-r1', 'completed_sessions_s'), ('replay-c8-session-r2', 'completed_sessions_s'), ('agent-c6-session-r1', 'successful_tasks_s'), ('agent-c6-session-r2', 'successful_tasks_s')], 'session-dspark3-agent-screen': [('replay-c8-session-r1', 'completed_sessions_s'), ('replay-c8-session-r2', 'completed_sessions_s'), ('agent-c6-session-r1', 'successful_tasks_s'), ('agent-c6-session-r2', 'successful_tasks_s')], 'session-dp4ep-affinity-screen': [('replay-c8-sticky4-r1', 'completed_sessions_s'), ('replay-c8-sticky4-r2', 'completed_sessions_s')]}
    for name, values in selected_values.items():
        byid = {s['phase']['id']: s for s in summaries[name]['stages']}
        for phase_id, metric in values:
            s = byid[phase_id]
            value = s['summary'][metric]
            need('rationale value ' + name + '/' + phase_id, f'{value:.5f}'.lstrip('0') in decision['rationale'] and s['included_as_measurement'] is True)
            selection_evidence.append(dict(experiment=name, phase=phase_id, metric=metric, value=value, sessions=s['summary']['sessions'], successful_tasks=s['summary']['successful_tasks']))
    queue = load(path_at(SCENARIO, 'workloads/sessions') / f'series-{identity}.json')
    need('queue provenance matches decision', queue['decision_provenance'] == provenance)
    expected_ids = [c['experiment_id'] for s in decision['selections'] for c in s['candidates']]
    need('queue exact ordering and experiment coverage', [j['id'] for j in queue['jobs']] == generation['experiment_ids'] == expected_ids)
    genmap = {c['experiment_id']: c for c in generation['cohorts']}
    jobs = {j['id']: j for j in queue['jobs']}
    for selection in decision['selections']:
        kind = selection['kind']
        reference_prepared = None
        reference_phases = None
        need('bounded screen not confirmation ' + kind, selection['role'] == 'screen' and selection['duration_s'] == 120 and (selection['rates'] == [0.15, 0.3, 0.6]) and (selection['seeds'] == [202611630]))
        for candidate in selection['candidates']:
            name = candidate['experiment_id']
            config = load(path_at(SCENARIO, 'configs') / f'{name}.json')
            ref = load(path_at(SCENARIO, 'configs') / candidate['config_ref'])
            need(name + ' config differs only in name', config == {**ref, 'name': name})
            plan = load(path_at(SCENARIO, 'workloads/sessions') / f'{name}.json')
            meta = plan['decision_provenance']
            cohort = genmap[name]
            need(name + ' metadata consistent', meta['decision_sha256'] == sha(decision_raw) and meta['evidence_references'] == provenance['evidence_references'] and (meta['runtime_source_sha256'] == provenance['runtime_source_sha256']))
            need(name + ' reference config SHA', meta['config_reference_sha256'] == sha(read(path_at(SCENARIO, 'configs') / candidate['config_ref'])) == cohort['config_reference_sha256'])
            need(name + ' loopback and protocol', plan['base_url'] == 'http://127.0.0.1:8260' and config['port'] == 8260 and (plan['model'] == 'dsv4') and all((plan[k] == v for k, v in decision['service_protocol'].items())))
            need(name + ' validated unchanged plan', client.validate_plan(plan) == plan)
            run = path_at(SCENARIO, 'results') / name
            if run.exists():
                launch = load(run / 'command.json')
                source = launch['source_sha256']
                need(name + ' concurrent root launch uses reviewed inputs', launch['experiment_id'] == name and launch['environment']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3' and (source['scenarios/configs/' + name + '.json'] == sha(read(path_at(SCENARIO, 'configs') / f'{name}.json'))) and (source['scenarios/workloads/sessions/' + name + '.json'] == sha(read(path_at(SCENARIO, 'workloads/sessions') / f'{name}.json'))))
                execution_observed.append(dict(experiment=name, status='ROOT_LAUNCH_ALREADY_STARTED', git_commit=launch['git_commit']))
            else:
                execution_observed.append(dict(experiment=name, status='NOT_STARTED_AT_REVIEW'))
            need(name + ' queue inputs/budget', jobs[name]['config'] == name + '.json' and jobs[name]['session_plan'] == 'phase-c/' + name + '.json' and (jobs[name]['timeout'] == 3600))
            phases = plan['phases']
            need(name + ' four planned phases', len(phases) == 4)
            need(name + ' matched candidate phases', reference_phases is None or reference_phases == phases)
            reference_phases = phases
            prepared = []
            for pi, f in enumerate(phases):
                need(name + '/' + f['id'] + ' fixed scope/routing/kind', f['kind'] == kind and f['cache_scope'] == 'session' and (f['dp_routing'] == 'none'))
                if kind == 'replay':
                    need(name + '/' + f['id'] + ' three fixed profiles', f['profiles'] == PROFILES)
                else:
                    need(name + '/' + f['id'] + ' six tasks and frozen budgets', f['tasks'] == TASKS and (f['max_turns'], f['max_tools'], f['max_output_tokens']) == (12, 32, 4096))
                if pi == 0:
                    need(name + ' controlled excluded warmup', f['role'] == 'warmup' and f['arrival'] == 'closed' and (f['concurrency'] == 1) and (f['sessions'] == (3 if kind == 'replay' else 6)) and (f['seed'] == decision['warmup_seed']))
                    offsets = [None] * f['sessions']
                else:
                    need(name + '/' + f['id'] + ' explicit open screen', f['role'] == 'screen' and f['arrival'] == 'poisson' and (f['duration_s'] == 120) and (f['seed'] == 202611630) and (f['rate'] == selection['rates'][pi - 1]))
                    rng = random.Random(f['seed'])
                    t = 0.0
                    offsets = []
                    while True:
                        t += rng.expovariate(f['rate'])
                        if t >= f['duration_s']:
                            break
                        offsets.append(t)
                    items = PROFILES if kind == 'replay' else TASKS
                    schedule = [dict(index=i, planned_offset_s=t, seed=f['seed'] + i, work_item=items[i % len(items)]) for i, t in enumerate(offsets)]
                    w = cohort['windows'][pi - 1]
                    need(name + '/' + f['id'] + ' independent Poisson reconstruction', len(offsets) == w['planned_sessions'] and digest(offsets) == w['arrival_offsets_sha256'] and (digest(schedule) == w['planned_schedule_sha256']))
                    need(name + '/' + f['id'] + ' planned analysis count', sum((t >= 30 for t in offsets)) == w['planned_analysis_arrivals'])
                    measured_total += len(offsets)
                    window_rows.append(dict(experiment=name, kind=kind, rate=f['rate'], planned_sessions=len(offsets), analysis_arrivals=w['planned_analysis_arrivals'], work_items=dict(Counter((r['work_item'] for r in schedule)))))
                bank = client.prepare({**plan, 'phases': [f]})
                rows = bank['phases'][0]['sessions']
                small = []
                need(name + '/' + f['id'] + ' actual CPU prepared arrivals', [r['planned_offset_s'] for r in rows] == offsets)
                for i, row in enumerate(rows):
                    need(name + '/' + f['id'] + f' assignment {i}', row.get('profile', row.get('task_id')) == (PROFILES if kind == 'replay' else TASKS)[i % (3 if kind == 'replay' else 6)])
                    if kind == 'replay':
                        spec = GOLDEN[row['profile']]
                        need(name + '/' + f['id'] + f' exact token contract {i}', [len(v) for v in row['prompts']] == spec['inputs'] and row['outputs'] == spec['outputs'] and (row['tool_waits_s'] == spec['waits']) and ([client.digest(v) for v in row['prompts']] == row['prompt_hashes']))
                    small.append({k: v for k, v in row.items() if k != 'prompts'})
                proof = dict(phase_id=f['id'], seed=f['seed'], planned_sessions=len(rows), prepared_input_sha256=digest(small))
                prepared.append(proof)
                planned_total += len(rows)
                need(name + '/' + f['id'] + ' actual tokenizer/prepared metadata', bank['tokenizer_sha256'] == meta['tokenizer_preflight']['tokenizer_sha256'] and proof == meta['tokenizer_preflight']['phases'][pi])
            need(name + ' cross-candidate actual token/task hashes', reference_prepared is None or reference_prepared == prepared)
            reference_prepared = prepared
    for info in generation['created_files']:
        need('generated artifact checksum ' + info['path'], sha(read(Path(info['path']))) == info['sha256'])
    need('all review inputs unchanged', all((sha(Path(p).read_bytes()) == v['sha256'] for p, v in ledger.items())))
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), decision_sha256=sha(decision_raw), scope='Independent CPU pre-execution review; no service/GPU/HTTP execution', reviewer_sha256=sha(Path(__file__).read_bytes()), checks=checks, experiment_ids=expected_ids, execution_observed=execution_observed, planned_sessions_including_warmup=planned_total, planned_screen_sessions=measured_total, planned_windows=window_rows, decision_numeric_evidence=selection_evidence, input_ledger=ledger, limitations=['Single-seed 120-second screens do not satisfy finite-session-v1 confirmation; no capacity certification.', 'Exact token preparation reuses frozen production client and tokenizer; Poisson counts and assigned workloads are independently reconstructed.', 'Agent dynamic histories, outputs, tools, task results and quality equivalence remain unexecuted/unverified.', 'The 3600-second supervisor budget is a finite operational budget, not a proven worst-case completion bound; timeout/censor/budget failures remain evidence.', 'Closed screening data justify an informative competitor, not a universal winner; future confirms require all started windows and at least two seeds.'])
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with open_output(outputs[0], 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    lines = ['# 开放 Session 筛选计划独立审查', '', f'状态：PASS（CPU；{len(checks)} 项断言）。主代理可在复核期间开始串行队列；已出现的启动command与被审查config/plan SHA核对一致，执行观测状态见JSON。本审查自身没有启动GPU或HTTP。', '', f'已核对 decision 原字节 SHA、全部引用依据、参考配置仅 name 变化、180/600/64 协议、localhost:8260、队列与运行源身份。总计划 {planned_total} sessions，其中正式 screen {measured_total}、warmup {planned_total - measured_total}。', '', '每个候选的 .15/.30/.60 session/s 均120秒、同seed；独立Poisson复算得到15/35/65到达，后75%为8/26/50。实际CPU tokenizer重建的全部回放输入token/hash、任务分配和warmup跨候选一致。', '', 'TP4/DSpark与DP亲和对照的选择依据数值逐项与冻结PASS摘要吻合；该有限筛选不表示已选出全局最优架构。', '', '限制：', '', *['- ' + v for v in result['limitations']]]
    with open_output(outputs[1], 'x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), planned_total=planned_total, planned_screen=measured_total, outputs=[str(p) for p in outputs]), indent=2))


if __name__=='__main__':main()
