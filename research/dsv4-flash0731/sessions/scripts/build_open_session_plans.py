#!/usr/bin/env python3
"""Generate exclusive Stage C plans from an explicit research decision; CPU only.

Never selects candidates/rates or launches a service. --self-test writes plans
only in a temporary tree, using the unchanged production plan validator.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

PHASE=Path(__file__).resolve().parents[1]
SCENARIO=PHASE.parent/'scenarios'
sys.path.insert(0,str(SCENARIO/'scripts'))
import agent_session_client as client

PROFILES=['coding','research','data']
TASKS=['reservation_book','invoice_adjustments','release_gate','failover_policy','sales_net','service_latency']
GLOBAL_FIELDS={'timeout_s','session_timeout_s','max_active_sessions'}


def need(condition,message):
    if not condition:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def plain(path):
    path=Path(os.path.abspath(path))
    need(not any(p.is_symlink()for p in (path,*path.parents)),f'Symlink path refused: {path}')
    return path


def inside(root,relative):
    need(isinstance(relative,str)and relative and not Path(relative).is_absolute()and '..'not in Path(relative).parts,'Use a relative in-tree input path')
    p=plain(root/relative);need(p.is_relative_to(root),'Input outside root');return p


def fields(value,required,optional=()):
    need(isinstance(value,dict),'Expected an object')
    need(set(value)>=set(required)and set(value)<=set(required)|set(optional),'Missing or unknown decision fields: '+str(sorted(set(required)-set(value)))+'/'+str(sorted(set(value)-set(required)-set(optional))))


def label(value):return isinstance(value,str)and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',value)is not None
def integer(value,lo,hi):return type(value)is int and lo<=value<=hi
def number(value,lo,hi):return type(value)in(int,float)and math.isfinite(value)and lo<value<=hi


def runtime_sources():
    paths = [Path(__file__).resolve(), path_at(PHASE, 'finite-session-v1.md'), SCENARIO.parent / 'scripts/benchmark_client.py']
    paths += [SCENARIO / 'scripts' / n for n in ('agent_session_client.py', 'agent_session_tasks.py', 'capacity_client.py', 'serve_agent_sessions.py', 'run_logged.py', 'run_agent_series.py')]
    paths += [p for p in path_at(SCENARIO, 'workloads/sessions/tasks').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(SCENARIO.parent)): sha(plain(p).read_bytes()) for p in sorted(paths)}


def common(kind,seed,scope,route):
    f=dict(kind=kind,seed=seed,cache_scope=scope,dp_routing=route)
    if kind=='replay':f['profiles']=PROFILES.copy()
    else:f.update(tasks=TASKS.copy(),max_turns=12,max_tools=32,max_output_tokens=4096)
    return f


def tokenizer_preflight(plan):
    """Hash real prepared inputs one phase at a time; never persist token arrays."""
    phases=[];tokenizer_sha=None
    for phase in plan['phases']:
        one={**plan,'phases':[phase]};bank=client.prepare(one);tokenizer_sha=bank['tokenizer_sha256']
        rows=bank['phases'][0]['sessions'];small=[]
        for row in rows:
            if phase['kind']=='replay':
                expected=client.PROFILES[row['profile']]['inputs']
                need([len(p)for p in row['prompts']]==expected,'Actual tokenizer replay lengths differ')
                need([client.digest(p)for p in row['prompts']]==row['prompt_hashes'],'Actual replay hashes differ')
            small.append({k:v for k,v in row.items()if k!='prompts'})
        phases.append(dict(phase_id=phase['id'],seed=phase['seed'],planned_sessions=len(rows),prepared_input_sha256=digest(small)))
    return dict(status='CPU_PREPARED_ONLY',tokenizer_sha256=tokenizer_sha,phases=phases,
        caveat='Actual frozen tokenizer and exact replay input tokens; Agent initial task assignments only, not dynamic model output identity or execution.')


def validate_config(config,route):
    args=config.get('args');env=config.get('env',{})
    need(isinstance(args,list)and args and all(isinstance(a,str)for a in args),'Invalid reference config args')
    need(not any(a in ('--host','--port')or a.startswith(('--host=','--port='))for a in args),'Launcher owns host and port')
    need(isinstance(env,dict)and env.get('VLLM_SERVER_DEV_MODE')=='1','Reference config must enable owned cache reset endpoint')
    protected={'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','RESEARCH_RUN_DIR','VLLM_NO_USAGE_STATS','HF_HUB_OFFLINE','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','PYTHONUNBUFFERED'}
    need(not protected.intersection(env),'Config overrides supervisor-owned environment')
    need(integer(config.get('port',8260),1024,65535),'Invalid service port')
    if route=='sticky4':
        at=args.index('--data-parallel-size')if '--data-parallel-size'in args else -1
        need(at>=0 and at+1<len(args)and args[at+1]=='4','sticky4 requires a DP4 reference config')


def generate(decision_path, scenario=SCENARIO, phase=PHASE, verify_tokenizer=False):
    """Validate every input before publishing; leave claimed outputs on failure."""
    scenario = plain(scenario)
    phase = plain(phase)
    research = scenario.parent
    need(phase.parent == research, 'Phase and scenario must share research root')
    decision_path = plain(decision_path)
    need(decision_path.is_relative_to(research), 'Decision must be under research root')
    inputs = {}

    def read_input(path):
        raw = plain(path).read_bytes()
        need(path not in inputs or inputs[path] == raw, 'Input changed between reads')
        inputs[path] = raw
        return raw
    decision = json.loads(read_input(decision_path))
    fields(decision, ('schema_version', 'decision_id', 'rationale', 'evidence_references', 'service_protocol', 'warmup_seed', 'supervisor_timeout_s', 'selections'))
    need(decision['schema_version'] == 1 and label(decision['decision_id']), 'Invalid decision version/ID')
    need(isinstance(decision['rationale'], str) and decision['rationale'].strip(), 'Decision rationale required')
    protocol = decision['service_protocol']
    fields(protocol, GLOBAL_FIELDS)
    need(integer(decision['warmup_seed'], 0, 2 ** 63 - 257), 'Invalid warmup seed')
    need(number(decision['supervisor_timeout_s'], 0, 86400), 'Supervisor timeout must be finite and <= 86400 seconds')
    refs = decision['evidence_references']
    need(isinstance(refs, list) and refs and (len(set(refs)) == len(refs)), 'Unique evidence references required')
    evidence = []
    for name in refs:
        p = inside(research, name)
        read_input(p)
        evidence.append(dict(path=name, sha256=sha(inputs[p]), bytes=len(inputs[p])))
    selections = decision['selections']
    need(isinstance(selections, list) and selections, 'Explicit selections required')
    decision_id = decision['decision_id']
    decision_sha = sha(inputs[decision_path])
    source_hashes = runtime_sources()
    provenance = dict(decision_id=decision_id, decision_source=str(decision_path.relative_to(research)), decision_sha256=decision_sha, rationale=decision['rationale'], evidence_references=evidence, generator_source_sha256=sha(Path(__file__).read_bytes()), runtime_source_sha256=source_hashes, status='GENERATED_NOT_EXECUTED')
    outputs = {}
    jobs = []
    cohorts = []
    ids = set()
    selection_ids = set()
    for selection in selections:
        fields(selection, ('selection_id', 'kind', 'rates', 'duration_s', 'seeds', 'role', 'candidates'), ('cache_scope',))
        sid = selection['selection_id']
        need(label(sid) and sid not in selection_ids, 'Invalid/duplicate selection ID')
        selection_ids.add(sid)
        kind = selection['kind']
        need(kind in ('replay', 'agent'), 'Choose replay or agent separately')
        rates = selection['rates']
        seeds = selection['seeds']
        duration = selection['duration_s']
        role = selection['role']
        scope = selection.get('cache_scope', 'session')
        need(isinstance(rates, list) and rates and all((number(r, 0, 8) for r in rates)) and (len(set(rates)) == len(rates)), 'Unique bounded rates required')
        need(isinstance(seeds, list) and seeds and all((integer(s, 0, 2 ** 63 - 257) for s in seeds)) and (len(set(seeds)) == len(seeds)), 'Unique bounded seeds required')
        need(number(duration, 0, 1200) and role in ('screen', 'confirm'), 'Open plan must be bounded screen or confirm')
        need(scope in ('none', 'session', 'shared'), 'Invalid cache scope')
        if role == 'confirm':
            need(duration >= 180, 'Confirm duration must be at least 180 seconds')
            need(all((0.75 * duration * r >= 20 for r in rates)), 'Confirm expected analysis arrivals must be >= 20; do not assume actual count')
        need(decision['supervisor_timeout_s'] > duration * len(rates) * len(seeds), 'Supervisor budget must exceed serial arrival horizons and allow startup/warmup/drain')
        candidates = selection['candidates']
        need(isinstance(candidates, list) and candidates, 'Explicit candidate list required')
        schedule_by_phase = {}
        matched_prepared = None
        for candidate in candidates:
            fields(candidate, ('experiment_id', 'candidate', 'config_ref', 'dp_routing'))
            name = candidate['experiment_id']
            match = re.fullmatch('c([0-9]+)-[A-Za-z0-9][A-Za-z0-9._-]*', name) if isinstance(name, str) else None
            need(match is not None and int(match.group(1)) >= 8 and (name not in ids), 'Explicit unique c08+ experiment IDs required')
            ids.add(name)
            need(label(candidate['candidate']), 'Invalid candidate label')
            ref = candidate['config_ref']
            need(isinstance(ref, str) and Path(ref).name == ref and ref.endswith('.json'), 'Config reference must be a JSON basename')
            p = inside(path_at(scenario, 'configs'), ref)
            config = json.loads(read_input(p))
            config = deepcopy(config)
            config['name'] = name
            route = candidate['dp_routing']
            need(route in ('none', 'sticky4'), 'Explicit DP routing must be none or sticky4')
            validate_config(config, route)
            warm = dict(id='warm-' + kind, arrival='closed', role='warmup', sessions=3 if kind == 'replay' else 6, concurrency=1, **common(kind, decision['warmup_seed'], scope, route))
            phases = [warm]
            phase_metadata = []
            for ri, rate in enumerate(rates, 1):
                for si, seed in enumerate(seeds, 1):
                    f = dict(id=f'{kind}-open-r{ri:02d}-s{si:02d}', arrival='poisson', role=role, rate=rate, duration_s=duration, **common(kind, seed, scope, route))
                    phases.append(f)
                    offsets = client.arrivals(f)
                    items = PROFILES if kind == 'replay' else TASKS
                    rows = [dict(index=i, planned_offset_s=t, seed=seed + i, work_item=items[i % len(items)]) for i, t in enumerate(offsets)]
                    key = f['id']
                    identity = digest(rows)
                    need(key not in schedule_by_phase or schedule_by_phase[key] == identity, 'Candidates have different planned arrivals or work items')
                    schedule_by_phase[key] = identity
                    actual_analysis = sum((duration / 4 <= t < duration for t in offsets))
                    phase_metadata.append(dict(phase_id=key, rate=rate, seed=seed, expected_analysis_arrivals=0.75 * duration * rate, planned_sessions=len(offsets), planned_analysis_arrivals=actual_analysis, planned_analysis_arrivals_at_least20=actual_analysis >= 20, planned_schedule_sha256=identity, arrival_offsets_sha256=digest(offsets), sample_caveat='Expected/planned counts are not executed samples; all planned arrivals must later be audited. Never regenerate a seed to select a passing count.'))
            plan = client.validate_plan(dict(schema_version=1, experiment_id=name, base_url=f'http://127.0.0.1:{config.get('port', 8260)}', model='dsv4', **protocol, phases=phases))
            prepared = tokenizer_preflight(plan) if verify_tokenizer else dict(status='NOT_PERFORMED', caveat='Use --verify-tokenizer before GPU execution to check actual prepared inputs.')
            if verify_tokenizer:
                need(matched_prepared is None or prepared == matched_prepared, 'Candidates differ in actual prepared input tokens/tasks')
                matched_prepared = prepared
            plan['decision_provenance'] = {**provenance, 'selection_id': sid, 'candidate': candidate['candidate'], 'config_reference': ref, 'config_reference_sha256': sha(inputs[p]), 'planned_windows': phase_metadata, 'tokenizer_preflight': prepared, 'comparison_caveat': 'Replay token construction uses identical frozen code/seed/profile; dynamic Agent inputs/outputs are measured outcomes, not promised token-identical.'}
            config_path = plain(path_at(scenario, 'configs') / f'{name}.json')
            plan_path = plain(path_at(scenario, 'workloads/sessions') / f'{name}.json')
            outputs[config_path] = encode(config)
            outputs[plan_path] = encode(plan)
            need(not plain(path_at(scenario, 'results') / name).exists(), 'Experiment result ID already exists')
            jobs.append(dict(id=name, config=config_path.name, session_plan='phase-c/' + plan_path.name, timeout=decision['supervisor_timeout_s'], reference='decision ' + decision_id + ' sha256 ' + decision_sha + '; ' + ', '.join(refs), hypothesis=decision['rationale']))
            cohorts.append(dict(experiment_id=name, selection_id=sid, kind=kind, role=role, candidate=candidate['candidate'], config_reference=ref, config_reference_sha256=sha(inputs[p]), cache_scope=scope, dp_routing=route, windows=phase_metadata, tokenizer_preflight=prepared))
    queue_path = plain(path_at(scenario, 'workloads/sessions') / f'series-{decision_id}.json')
    decision_copy = plain(path_at(scenario, 'workloads/sessions') / f'decision-{decision_id}.json')
    outputs[queue_path] = encode(dict(jobs=jobs, decision_provenance=provenance))
    outputs[decision_copy] = inputs[decision_path]
    record_path = plain(path_at(phase, 'logs') / f'open-plan-generation-{decision_id}.json')
    need(not any((p.exists() or p.is_symlink() for p in (*outputs, record_path))), 'Output ID exists; nothing overwritten')
    need(all((plain(p).read_bytes() == raw for p, raw in inputs.items())), 'Decision/config/evidence changed during generation')
    need(runtime_sources() == source_hashes, 'Runtime/task sources changed during generation')
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='IN PROGRESS', started_utc=utc(), scope='CPU plan construction only; no GPU or HTTP execution', decision_provenance=provenance, cohorts=cohorts, queue=str(queue_path), planned_output_paths=[str(p) for p in outputs], created_files=[])
    with open_output(record_path, 'xb') as f:
        f.write(encode(record))
    try:
        for path, raw in outputs.items():
            plain(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open_output(path, 'xb') as f:
                f.write(raw)
            record['created_files'].append(dict(path=str(path), sha256=sha(raw), bytes=len(raw)))
        need(all((p.read_bytes() == raw for p, raw in outputs.items())), 'Output byte verification failed')
        need(all((plain(p).read_bytes() == raw for p, raw in inputs.items())), 'Source changed while writing plans')
        need(runtime_sources() == source_hashes, 'Runtime/task sources changed while writing plans')
        record.update(status='GENERATED_NOT_EXECUTED', finished_utc=utc(), experiment_ids=[j['id'] for j in jobs])
    except BaseException as exc:
        record.update(status='FAILED', finished_utc=utc(), error=repr(exc))
        raise
    finally:
        temp = record_path.with_suffix('.json.tmp')
        with open_output(temp, 'xb') as f:
            f.write(encode(record))
        temp.replace(record_path)
    return record


def selftest():
    checks = []

    def check(name, condition):
        need(condition, name)
        checks.append(name)

    def reject(name, fn):
        try:
            fn()
        except (ValueError, OSError):
            checks.append(name)
        else:
            raise AssertionError(name)
    with tempfile.TemporaryDirectory(prefix='open-session-generator-MOCK-') as td:
        research = Path(td)
        scenario = research / 'scenarios'
        phase = research / 'sessions'
        path_at(scenario, 'configs').mkdir(parents=True)
        (phase / 'decisions').mkdir(parents=True)
        config = dict(name='MOCK-reference', port=8260, args=['--tensor-parallel-size', '4'], env={'VLLM_SERVER_DEV_MODE': '1'})
        write_bytes(path_at(scenario, 'configs/mock.json'), encode(config))
        write_bytes(research / 'mock-summary.json', encode({'scope': 'MOCK CPU fixture'}))
        decision = dict(schema_version=1, decision_id='mock-open', rationale='MOCK validation only; not a candidate/rate recommendation', evidence_references=['mock-summary.json'], service_protocol=dict(timeout_s=180, session_timeout_s=600, max_active_sessions=64), warmup_seed=17, supervisor_timeout_s=5000, selections=[])
        for i, kind in enumerate(('replay', 'agent')):
            decision['selections'].append(dict(selection_id=kind, kind=kind, rates=[0.2], duration_s=180, seeds=[19, 23], role='confirm', candidates=[dict(experiment_id=f'c{8 + i * 2 + j:02d}-mock-{kind}-{j}', candidate=f'mock{j}', config_ref='mock.json', dp_routing='none') for j in range(2)]))
        path = phase / 'decisions/mock.json'
        write_bytes(path, encode(decision))
        result = generate(path, scenario, phase)
        check('temporary_generation_not_execution', result['status'] == 'GENERATED_NOT_EXECUTED')
        check('decision_byte_identity', result['decision_provenance']['decision_sha256'] == sha(path.read_bytes()))
        check('summary_reference_byte_identity', result['decision_provenance']['evidence_references'][0]['sha256'] == sha((research / 'mock-summary.json').read_bytes()))
        for i, kind in enumerate(('replay', 'agent')):
            candidates = result['cohorts'][2 * i:2 * i + 2]
            check(kind + '_same_arrivals_and_work_items', candidates[0]['windows'] == candidates[1]['windows'])
            plans = [load_document(path_at(scenario, 'workloads/sessions') / f'{c['experiment_id']}.json') for c in candidates]
            check(kind + '_validator_accepts', all((client.validate_plan(p) for p in plans)))
            check(kind + '_all_phases_separate_kind', all((f['kind'] == kind for p in plans for f in p['phases'])))
            check(kind + '_controlled_warmup', plans[0]['phases'][0] == plans[1]['phases'][0] and plans[0]['phases'][0]['sessions'] == (3 if kind == 'replay' else 6))
            check(kind + '_matched_phase_plan', plans[0]['phases'] == plans[1]['phases'])
            banks = []
            for p in plans:
                short = deepcopy(p)
                short['phases'] = short['phases'][:1]
                banks.append(client.prepare(short))
            check(kind + '_actual_prepared_inputs_match', banks[0] == banks[1])
            short = deepcopy(plans[0])
            short['phases'] = short['phases'][:1]
            prepared = tokenizer_preflight(short)
            check(kind + '_preflight_uses_actual_tokenizer', prepared['tokenizer_sha256'] == banks[0]['tokenizer_sha256'])
        created = {p: Path(p).read_bytes() for p in [f['path'] for f in result['created_files']]}
        reject('rerun_does_not_overwrite', lambda: generate(path, scenario, phase))
        check('all_original_output_bytes_preserved', all((Path(p).read_bytes() == raw for p, raw in created.items())))

        def invalid(change):
            value = deepcopy(decision)
            value['decision_id'] = 'invalid'
            for i, s in enumerate(value['selections']):
                for j, c in enumerate(s['candidates']):
                    c['experiment_id'] = f'c{20 + i * 2 + j}-invalid-{s['kind']}-{j}'
            change(value)
            write_bytes(path, encode(value))
            return generate(path, scenario, phase)
        reject('reject_historical_ids', lambda: invalid(lambda d: d['selections'][0]['candidates'][0].update(experiment_id='c07-old')))
        reject('reject_duplicate_ids', lambda: invalid(lambda d: d['selections'][0]['candidates'][1].update(experiment_id='c20-invalid-replay-0')))
        reject('reject_missing_route', lambda: invalid(lambda d: d['selections'][0]['candidates'][0].pop('dp_routing')))
        reject('reject_sticky_on_non_dp4', lambda: invalid(lambda d: d['selections'][0]['candidates'][0].update(dp_routing='sticky4')))
        reject('reject_short_confirm', lambda: invalid(lambda d: d['selections'][0].update(duration_s=179)))
        reject('reject_insufficient_expected_analysis', lambda: invalid(lambda d: d['selections'][0].update(rates=[0.01])))
        reject('reject_seed_duplicates', lambda: invalid(lambda d: d['selections'][0].update(seeds=[19, 19])))
        reject('reject_rate_duplicates', lambda: invalid(lambda d: d['selections'][0].update(rates=[0.2, 0.2])))
        reject('reject_unknown_decision_fields', lambda: invalid(lambda d: d.update(unexpected=True)))
        reject('reject_unbounded_global_protocol', lambda: invalid(lambda d: d['service_protocol'].update(max_active_sessions=999)))
        reject('reject_config_path_escape', lambda: invalid(lambda d: d['selections'][0]['candidates'][0].update(config_ref='../mock.json')))
        check('invalid_decision_creates_no_records', not path_at(phase, 'logs/open-plan-generation-invalid.json').exists())
        check('realized_counts_are_diagnostic', all(('planned_analysis_arrivals' in w and 'sample_caveat' in w for c in result['cohorts'] for w in c['windows'])))
        dp = {**config, 'args': ['--data-parallel-size', '4']}
        write_bytes(path_at(scenario, 'configs/mock-dp.json'), encode(dp))
        small = deepcopy(decision)
        small['decision_id'] = 'mock-tokenizer'
        small['selections'] = small['selections'][:1]
        small['selections'][0].update(role='screen', rates=[0.25], duration_s=8, seeds=[3])
        for j, c in enumerate(small['selections'][0]['candidates']):
            c.update(experiment_id=f'c{30 + j}-mock-preflight', config_ref='mock-dp.json', dp_routing='none' if j == 0 else 'sticky4')
        path = phase / 'decisions/mock-tokenizer.json'
        write_bytes(path, encode(small))
        tested = generate(path, scenario, phase, verify_tokenizer=True)
        prepared = [c['tokenizer_preflight'] for c in tested['cohorts']]
        check('integrated_exact_token_preflight_across_routes', prepared[0] == prepared[1] and prepared[0]['status'] == 'CPU_PREPARED_ONLY')
        check('preflight_checks_open_and_warmup_phases', len(prepared[0]['phases']) == 2 and prepared[0]['phases'][1]['planned_sessions'] > 0)
        tokenizer_sha = prepared[0]['tokenizer_sha256']
        low = deepcopy(small)
        low['decision_id'] = 'mock-low-count'
        low['selections'][0].update(role='confirm', rates=[20 / 135], duration_s=180, seeds=[0])
        for j, c in enumerate(low['selections'][0]['candidates']):
            c['experiment_id'] = f'c{40 + j}-mock-low-count'
        path = phase / 'decisions/mock-low-count.json'
        write_bytes(path, encode(low))
        tested = generate(path, scenario, phase)
        window = tested['cohorts'][0]['windows'][0]
        check('expected20_can_plan15_without_seed_replacement', window['expected_analysis_arrivals'] == 20 and window['planned_analysis_arrivals'] == 15 and (window['seed'] == 0))
        check('planned_count_shortfall_not_disguised', window['planned_analysis_arrivals_at_least20'] is False)
        low['decision_id'] = 'mock-existing-result'
        for j, c in enumerate(low['selections'][0]['candidates']):
            c['experiment_id'] = f'c{50 + j}-mock-existing-result'
        path_at(scenario, 'results').mkdir()
        path_at(scenario, 'results/c50-mock-existing-result').symlink_to(research / 'absent')
        path = phase / 'decisions/mock-existing-result.json'
        write_bytes(path, encode(low))
        reject('dangling_result_symlink_not_a_fresh_id', lambda: generate(path, scenario, phase))
    return dict(status='PASS', checks=checks, actual_tokenizer_sha256=tokenizer_sha, scope='Temporary MOCK plans and real CPU tokenizer only; no actual GPU plans, services or inference created')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    m = parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--decision', type=Path)
    m.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-tokenizer', action='store_true', help='CPU-prepare all planned phases and compare exact replay inputs/task assignments before writing')
    args = parser.parse_args()
    if args.self_test:
        need(args.output is not None and (not args.output.exists()), 'Self-test requires a new output log path')
        result = selftest()
        result.update(created_utc=utc(), generator_sha256=sha(Path(__file__).read_bytes()), client_sha256=sha(Path(client.__file__).read_bytes()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open_output(args.output, 'xb') as f:
            f.write(encode(result))
        print(json.dumps(dict(status=result['status'], checks=len(result['checks']), output=str(args.output)), indent=2))
    else:
        need(args.output is None, 'Generation record path is owned by decision ID; do not pass --output')
        result = generate(args.decision, verify_tokenizer=args.verify_tokenizer)
        print(json.dumps({k: result[k] for k in ('status', 'experiment_ids', 'queue')}, indent=2))


if __name__=='__main__':main()
