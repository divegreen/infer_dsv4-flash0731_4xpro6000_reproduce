#!/usr/bin/env python3
"""CPU-only independent actual-token preparation review of Stage C screens."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'scenarios'
sys.path.insert(0,str(OLD/'scripts'))
import agent_session_client as client


def need(value,message):
    if not value:raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a new review output ID')
    ledger = {}
    records = []
    groups = defaultdict(list)
    configs = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path.resolve())] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw
    series = json.loads(read(path_at(OLD, 'workloads/sessions/series-matched-screen.json')))
    read(Path(__file__))
    read(Path(client.__file__))
    read(OLD / 'scripts/agent_session_tasks.py')
    read(ROOT / 'scripts/build_matched_screen.py')
    need(len(series['jobs']) == 5, 'Expected five matched candidate deployments')
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='CPU plan/tokenizer/configuration review only; no GPU or model HTTP', plans=records)
    try:
        for job in series['jobs']:
            raw_plan = json.loads(read(path_at(OLD, 'workloads') / job['session_plan']))
            plan = client.validate_plan(raw_plan)
            config = json.loads(read(path_at(OLD, 'configs') / job['config']))
            configs[job['id']] = config
            need(plan['experiment_id'] == job['id'] == config['name'], 'Experiment/config identity')
            need(all((f['arrival'] == 'closed' and f['role'] in ('warmup', 'screen', 'pilot') for f in plan['phases'])), 'No open capacity phases authorized in screen')
            need(config['env']['VLLM_SERVER_DEV_MODE'] == '1', 'Cache reset environment')
            for name in ('--enable-auto-tool-choice', '--tool-call-parser', '--enable-chunked-prefill'):
                need(name in config['args'], 'Missing runtime flag ' + name)
            prepared = client.prepare(plan)
            phases = []
            for bank in prepared['phases']:
                f = bank['phase']
                rows = bank['sessions']
                need(len(rows) == f['sessions'], 'Session count mismatch')
                identity_fields = ('index', 'planned_offset_s', 'seed', 'profile', 'prompts', 'outputs', 'tool_waits_s') if f['kind'] == 'replay' else ('index', 'planned_offset_s', 'seed', 'task_id')
                identity = client.digest([{k: row.get(k) for k in identity_fields} for row in rows])
                cohort = {key: f.get(key) for key in ('kind', 'arrival', 'seed', 'sessions', 'concurrency', 'profiles', 'tasks')}
                if f['kind'] == 'agent':
                    need((f['max_turns'], f['max_tools'], f['max_output_tokens']) == (12, 32, 4096), 'Dynamic task budget changed')
                    cohort.update({key: f[key] for key in ('max_turns', 'max_tools', 'max_output_tokens')})
                else:
                    for row in rows:
                        shape = client.PROFILES[row['profile']]
                        need([len(v) for v in row['prompts']] == shape['inputs'], 'Replay exact lengths')
                        need(row['outputs'] == shape['outputs'] and row['tool_waits_s'] == shape['waits'], 'Replay budgets/waits')
                        need(all((b[:len(a)] == a for a, b in zip(row['prompts'], row['prompts'][1:]))), 'Exact nested history')
                        need(row['prompt_hashes'] == [client.digest(v) for v in row['prompts']], 'Prompt hash mismatch')
                groups[client.digest(cohort)].append(dict(experiment=job['id'], phase=f['id'], identity_sha256=identity, cohort=cohort))
                distribution = {str(rank): dict(Counter((rows[i].get('profile', rows[i].get('task_id')) for i in range(len(rows)) if i % 4 == rank))) for rank in range(4)}
                phases.append(dict(id=f['id'], kind=f['kind'], role=f['role'], seed=f['seed'], sessions=f['sessions'], concurrency=f['concurrency'], cache_scope=f['cache_scope'], routing=f['dp_routing'], identity_sha256=identity, rank_assignment=distribution if f['dp_routing'] == 'sticky4' else None))
            records.append(dict(experiment=job['id'], phases=phases, tokenizer_sha256=prepared['tokenizer_sha256'], planned_sessions=sum((len(b['sessions']) for b in prepared['phases']))))
            del prepared
        for rows in groups.values():
            need(len({r['identity_sha256'] for r in rows}) == 1, 'Same-cohort actual prepared workload differs')
        first = records[0]['phases']
        scopes = [[p['cache_scope'] for p in first if p['kind'] == 'replay' and p['concurrency'] == 8 and (p['seed'] == seed)] for seed in (202611611, 202611612)]
        need(scopes == [['none', 'session', 'shared'], ['shared', 'session', 'none']], 'TP4 cache treatment order not reversed')
        dp = records[-1]['phases']
        for seed, expected in [(202611611, ['none', 'sticky4']), (202611612, ['sticky4', 'none'])]:
            need([p['routing'] for p in dp if p['kind'] == 'replay' and p['seed'] == seed] == expected, 'DP route order mismatch')
        for seed, expected in [(202611621, ['none', 'sticky4']), (202611622, ['sticky4', 'none'])]:
            need([p['routing'] for p in dp if p['kind'] == 'agent' and p['seed'] == seed] == expected, 'DP dynamic route order mismatch')
        for p in dp:
            if p['routing'] != 'sticky4':
                continue
            if p['kind'] == 'replay':
                need(all((set(counts.values()) == {2} and len(counts) == 3 for counts in p['rank_assignment'].values())), 'Replay profile/rank imbalance')
            else:
                categories = {'reservation_book': 'coding', 'invoice_adjustments': 'coding', 'release_gate': 'research', 'failover_policy': 'research', 'sales_net': 'data', 'service_latency': 'data'}
                need(all((Counter({kind: sum((count for task, count in counts.items() if categories[task] == kind)) for kind in set(categories.values())}) == Counter(coding=1, research=1, data=1) for counts in p['rank_assignment'].values())), 'Agent category/rank imbalance')

        def normalize_config(c):
            return {k: v for k, v in c.items() if k != 'name'}
        on = normalize_config(configs[records[0]['experiment']])
        off = normalize_config(configs[records[1]['experiment']])
        on['args'] = [a for a in on['args'] if a not in ('--enable-prefix-caching', '--no-enable-prefix-caching')]
        off['args'] = [a for a in off['args'] if a not in ('--enable-prefix-caching', '--no-enable-prefix-caching')]
        need(on == off, 'TP4 APC on/off changes more than cache switch/name')
        need('--enable-prefix-caching' in configs[records[0]['experiment']]['args'] and '--no-enable-prefix-caching' in configs[records[1]['experiment']]['args'], 'APC treatment not actual config')
        result['matched_cohorts'] = [rows for rows in groups.values() if len(rows) > 1]
        result['checks'] = dict(real_tokenizer_preparation=True, cohort_identities=True, phase_role_boundaries=True, cache_order_reversed=True, dp_route_order_reversed=True, dp_profile_and_category_balance=True, tp4_apc_single_factor=True)
        result['limitations'] = ['Plans are not executed results; GPU/runtime/actual cache hit and dynamic task outcomes remain unverified.', 'Exact replay has matched inputs; dynamic cohorts match only tasks/order/budgets, not realized histories or output token counts.', 'DP12 agent sessions balance three task categories on each rank, but each specific task visits two of four ranks, not all four; fixed task/rank coupling remains.', 'C1 uses one seed and six sessions per included deployment; two seeds are only high-concurrency screens, not long open capacity confirmation.', 'TP4 APC-off omits Agent C1; full matched control covers replay C1/C8 and Agent C6, not every TP4 phase.', 'Reversed treatment order reduces drift bias but sequential architecture execution and different JIT warmth remain limitations.', 'The numeric-tool schema probe must be inspected before treating agent results as interface-clean; this plan audit cannot prove a not-yet-run probe succeeds.']
        for path, info in ledger.items():
            need(hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'], 'Plan/source changed during review')
    except Exception as exc:
        import traceback
        result.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    result['evidence_ledger'] = ledger
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], experiments=len(records), sessions=sum((r['planned_sessions'] for r in records)), matched_cohorts=len(result.get('matched_cohorts', [])), error=result.get('error')), indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__=='__main__':main()
