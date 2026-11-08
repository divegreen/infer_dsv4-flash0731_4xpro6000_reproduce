#!/usr/bin/env python3
"""Independent CPU review of the evidence-selected replay-confirm-r1 plan.

Does not import the plan generator, start a service or send HTTP. Uses the
frozen runtime tokenizer for exact inputs and independently reconstructs arrivals.
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
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

PHASE = Path(__file__).resolve().parents[1]
RESEARCH = PHASE.parent
SCENARIO = RESEARCH/'scenarios'
sys.path.insert(0, str(SCENARIO/'scripts'))
import agent_session_client as client

PROFILES = ['coding', 'research', 'data']
GOLDEN = {
    'coding':dict(inputs=[8192,12288,16384,20480],outputs=[128,256,128,256],waits=[.1,.2,.1]),
    'research':dict(inputs=[16384,24576,32768,40960],outputs=[128,128,128,512],waits=[.5,1.,.5]),
    'data':dict(inputs=[4096,6144,8192,10240],outputs=[128,256,128,256],waits=[.1,.5,.1])}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def review():
    checks, ledger, windows, evidence_rows = ([], {}, [], [])

    def need(label, condition):
        if not condition:
            raise ValueError(label)
        checks.append(label)

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        info = dict(bytes=len(raw), sha256=sha(raw))
        need('input consistent ' + str(path), str(path) not in ledger or ledger[str(path)] == info)
        ledger[str(path)] = info
        return raw

    def load(path):
        return json.loads(read(path))
    raw = read(PHASE / 'decisions/replay-confirm-r1.json')
    decision = json.loads(raw)
    generation = load(path_at(PHASE, 'logs/open-plan-generation-replay-confirm-r1.json'))
    provenance = generation['decision_provenance']
    need('decision identity', decision['decision_id'] == 'replay-confirm-r1' and provenance['decision_sha256'] == sha(raw))
    need('decision raw copy', read(path_at(SCENARIO, 'workloads/sessions/decision-replay-confirm-r1.json')) == raw)
    need('generation is not execution', generation['status'] == 'GENERATED_NOT_EXECUTED')
    need('frozen generator source', sha(read(PHASE / 'scripts/build_open_session_plans.py')) == provenance['generator_source_sha256'] == '4e8a80cf2410f54b79f5d977d7a4238e13a7b95c91cd2151e1cd0fc78d49ea83')
    need('finite-session-v1 unchanged', sha(read(path_at(PHASE, 'finite-session-v1.md'))) == '9f48277f1e0cbfa75a0abb742f1a3bdbf6e7224d9b39324f0056881986faf68e')
    for name, expected in provenance['runtime_source_sha256'].items():
        need('frozen runtime ' + name, sha(read(RESEARCH / name)) == expected)
    need('known frozen client', sha(read(SCENARIO / 'scripts/agent_session_client.py')) == '9b7889dade2b7b2499b898f887b6e33720bca630068f35759a106b8bf98b3183')
    need('fixed replay specification', client.PROFILES == GOLDEN)
    need('unchanged timeouts and admission', decision['service_protocol'] == dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64))
    need('bounded supervisor and fixed warmup', decision['supervisor_timeout_s'] == 3600 and decision['warmup_seed'] == 202611609)
    need('exact reference list', [e['path'] for e in provenance['evidence_references']] == decision['evidence_references'])
    for expected in provenance['evidence_references']:
        body = read(RESEARCH / expected['path'])
        need('evidence identity ' + expected['path'], sha(body) == expected['sha256'] and len(body) == expected['bytes'])
    prior = load(PHASE / 'decisions/open-screen-r1.json')
    for run, revision, expected_counts, expected_increase in [('session-tp4-replay-open', 'r2', [(26, 23), (50, 13)], [4.328, 21.638]), ('session-dspark3-replay-open', 'r1', [(26, 21), (50, 24)], [4.754, 19.372])]:
        summary = load(path_at(PHASE, 'summary') / f'{run}-terminal-{revision}.json')
        audit = load(path_at(PHASE, 'logs') / f'audit-{run}-r1.json')
        need('terminal independent evidence ' + run, summary['status'] == audit['status'] == 'PASS' and summary['experiments'][0]['terminal_success'] is True and (audit['experiments'][0]['runtime_terminal_success'] is True))
        stages = {s['phase']['id']: s for s in summary['stages']}
        audited = {s['id']: s for s in audit['experiments'][0]['phases']}
        for i, label in enumerate(('replay-open-r02-s01', 'replay-open-r03-s01')):
            s = stages[label]
            a = audited[label]['stability']
            v = s['stability']
            need('rationale and independent counts ' + run + '/' + label, (v['analysis_arrivals'], v['analysis_completions']) == expected_counts[i] == (a['analysis_arrivals'], a['analysis_protocol_completions']) and round(v['session_outstanding_increase'], 3) == expected_increase[i] and (abs(v['session_outstanding_increase'] - a['session_mean_outstanding_increase']) < 1e-09))
            need('screen is not confirmation ' + run + '/' + label, s['included_as_measurement'] is True and s['phase']['role'] == 'screen' and (v['classification'] == 'SCREEN_ONLY') and (v['trend_verdict'] == a['trend_verdict'] == 'GROWING_BACKLOG'))
            evidence_rows.append(dict(experiment=run, phase=label, arrivals=v['analysis_arrivals'], completions=v['analysis_completions'], outstanding_increase=v['session_outstanding_increase']))
    need('one replay-only selection', len(decision['selections']) == 1)
    selection = decision['selections'][0]
    need('explicit two-rate two-seed confirmation', selection['kind'] == 'replay' and selection['role'] == 'confirm' and (selection['rates'] == [0.2, 0.3]) and (selection['seeds'] == [202611640, 202611641]) and (selection['duration_s'] == 300) and (selection['cache_scope'] == 'session'))
    need('independent seeds from screens', not set(selection['seeds']) & {seed for s in prior['selections'] for seed in s['seeds']})
    ids = ['session-tp4-replay-confirm', 'session-dspark3-replay-confirm']
    need('exact candidates', [c['experiment_id'] for c in selection['candidates']] == ids)
    queue = load(path_at(SCENARIO, 'workloads/sessions/series-replay-confirm-r1.json'))
    need('exact queue order and provenance', [j['id'] for j in queue['jobs']] == generation['experiment_ids'] == ids and queue['decision_provenance'] == provenance)
    cohorts = {c['experiment_id']: c for c in generation['cohorts']}
    reference_phases = reference_prepared = None
    total = measured = 0
    for candidate, job in zip(selection['candidates'], queue['jobs']):
        name = candidate['experiment_id']
        cohort = cohorts[name]
        need('not started at review ' + name, not (path_at(SCENARIO, 'results') / name).exists())
        config = load(path_at(SCENARIO, 'configs') / f'{name}.json')
        ref_raw = read(path_at(SCENARIO, 'configs') / candidate['config_ref'])
        need('reference config name-only change ' + name, config == {**json.loads(ref_raw), 'name': name})
        plan = load(path_at(SCENARIO, 'workloads/sessions') / f'{name}.json')
        meta = plan['decision_provenance']
        need('plan provenance ' + name, meta['decision_sha256'] == sha(raw) and meta['runtime_source_sha256'] == provenance['runtime_source_sha256'] and (meta['evidence_references'] == provenance['evidence_references']) and (meta['config_reference_sha256'] == cohort['config_reference_sha256'] == sha(ref_raw)))
        need('service identity ' + name, plan['model'] == 'dsv4' and plan['base_url'] == 'http://127.0.0.1:8260' and (config['port'] == 8260) and all((plan[k] == v for k, v in decision['service_protocol'].items())))
        need('queue budget ' + name, job['config'] == name + '.json' and job['session_plan'] == 'phase-c/' + name + '.json' and (job['timeout'] == 3600))
        need('no normalization drift ' + name, client.validate_plan(plan) == plan)
        phases = plan['phases']
        need('five matched phases ' + name, len(phases) == 5 and (reference_phases is None or phases == reference_phases))
        reference_phases = phases
        prepared = []
        for pi, phase in enumerate(phases):
            label = name + '/' + phase['id']
            need('fixed replay contract ' + label, phase['kind'] == 'replay' and phase['cache_scope'] == 'session' and (phase['dp_routing'] == 'none') and (phase['profiles'] == PROFILES))
            if pi == 0:
                need('excluded C1 warmup ' + label, phase['role'] == 'warmup' and phase['arrival'] == 'closed' and (phase['concurrency'] == 1) and (phase['sessions'] == 3) and (phase['seed'] == 202611609))
                offsets = [None] * 3
            else:
                rate, seed = [(0.2, 202611640), (0.2, 202611641), (0.3, 202611640), (0.3, 202611641)][pi - 1]
                need('explicit confirm window ' + label, phase['role'] == 'confirm' and phase['arrival'] == 'poisson' and (phase['duration_s'] == 300) and (phase['rate'] == rate) and (phase['seed'] == seed))
                rng = random.Random(seed)
                t = 0.0
                offsets = []
                while True:
                    t += rng.expovariate(rate)
                    if t >= 300:
                        break
                    offsets.append(t)
                schedule = [dict(index=i, planned_offset_s=t, seed=seed + i, work_item=PROFILES[i % 3]) for i, t in enumerate(offsets)]
                window = cohort['windows'][pi - 1]
                analysis_count = sum((t >= 75 for t in offsets))
                need('independent schedule ' + label, window['planned_sessions'] == len(offsets) and window['planned_analysis_arrivals'] == analysis_count >= 20 and (window['expected_analysis_arrivals'] == rate * 225) and (window['arrival_offsets_sha256'] == digest(offsets)) and (window['planned_schedule_sha256'] == digest(schedule)))
                windows.append(dict(experiment=name, phase=phase['id'], rate=rate, seed=seed, duration_s=300, planned_sessions=len(offsets), analysis_arrivals=analysis_count, profiles=dict(Counter((s['work_item'] for s in schedule)))))
                measured += len(offsets)
            bank = client.prepare({**plan, 'phases': [phase]})
            rows = bank['phases'][0]['sessions']
            need('actual CPU arrivals ' + label, [r['planned_offset_s'] for r in rows] == offsets)
            for i, row in enumerate(rows):
                spec = GOLDEN[PROFILES[i % 3]]
                need('exact token inputs ' + label + f'/{i}', row['profile'] == PROFILES[i % 3] and [len(v) for v in row['prompts']] == spec['inputs'] and (row['outputs'] == spec['outputs']) and (row['tool_waits_s'] == spec['waits']) and ([client.digest(v) for v in row['prompts']] == row['prompt_hashes']) and all((a == b[:len(a)] for a, b in zip(row['prompts'], row['prompts'][1:]))))
            proof = dict(phase_id=phase['id'], seed=phase['seed'], planned_sessions=len(rows), prepared_input_sha256=digest([{k: v for k, v in r.items() if k != 'prompts'} for r in rows]))
            need('full tokenizer preflight proof ' + label, meta['tokenizer_preflight']['status'] == 'CPU_PREPARED_ONLY' and meta['tokenizer_preflight'] == cohort['tokenizer_preflight'] and (bank['tokenizer_sha256'] == meta['tokenizer_preflight']['tokenizer_sha256']) and (proof == meta['tokenizer_preflight']['phases'][pi]))
            prepared.append(proof)
            total += len(rows)
        need('cross-candidate actual token identity ' + name, reference_prepared is None or prepared == reference_prepared)
        reference_prepared = prepared
    for created in generation['created_files']:
        body = read(Path(created['path']))
        need('created file hash ' + created['path'], sha(body) == created['sha256'] and len(body) == created['bytes'])
    need('all inputs unchanged during review', all((sha(Path(p).read_bytes()) == v['sha256'] for p, v in ledger.items())))
    return dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='Independent CPU confirmation-plan preflight; no GPU, service, HTTP or capacity execution.', reviewer_sha256=sha(Path(__file__).read_bytes()), checks=checks, experiments=ids, planned_sessions_including_warmup=total, planned_confirmation_sessions=measured, planned_windows=windows, decision_numeric_evidence=evidence_rows, input_ledger=ledger, limitations=['All counts are planned arrivals, not completed measurements or capacity confirmation.', 'Exact input reconstruction reuses the frozen runtime and real tokenizer; arrival schedules are independently implemented.', 'All started confirmation windows, including unhealthy/inconclusive/interrupted ones, must remain in the later audit.', 'At least two distinct seeds must all satisfy finite-session-v1; this preflight establishes none of those observed outcomes.', 'Session latency SLO is undefined; future finite-confirmed points are tested bounds, not mathematical maxima or production SLA.', 'GPU execution remains serialized after c08-c11; 3600 seconds is an operational guard, not a worst-case completion proof.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(s) for s in ('.json', '.md')]
    if any((p.exists() or p.is_symlink() for p in outputs)):
        raise FileExistsError('Unique review revision required')
    result = review()
    with open_output(outputs[0], 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    lines = ['# Replay 长窗确认计划独立预检', '', f'PASS（CPU，{len(result['checks'])} 项断言）；c12/c13 均未启动 GPU。', '', '固定 TP4/DSpark、.20/.30 Session/s、300 秒、seeds 202611640/641；', '独立重建每个候选的 52/55/90/87 次到达，后75% 39/38/73/65，全部满足计划样本下限。', f'总计划 {result['planned_sessions_including_warmup']} 会话，正式确认 {result['planned_confirmation_sessions']}、预热6。', '', '实际 token/prefix/hash、配置仅 name 变化、冻结运行源/finite-session-v1、来源数值和队列预算均已核对。', '这仅是执行前预检；所有已开始的窗口都必须进入最终独立容量审计。', '', *['- ' + v for v in result['limitations']]]
    with open_output(outputs[1], 'x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), planned=result['planned_sessions_including_warmup'], measured_plan=result['planned_confirmation_sessions'])))


if __name__ == '__main__': main()
