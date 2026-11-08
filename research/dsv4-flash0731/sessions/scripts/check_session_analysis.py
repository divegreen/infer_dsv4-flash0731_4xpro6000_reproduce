#!/usr/bin/env python3
"""Independent CPU evidence/metric boundary checks, never model results.

Actual localhost mock-client SSE/JSONL is reused; temporary supervisor/source
metadata is explicitly constructed for analysis API controls, not claimed as
an original GPU launch. No original fixture or experiment file is modified.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'scenarios'
SPEC=importlib.util.spec_from_file_location('session_analysis_under_review',ROOT/'scripts/analyze_sessions.py')
analysis=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(analysis)
checks=[]


def check(name,value,details=None):
    checks.append(dict(name=name,passed=bool(value),details=details))
    assert value,name


def reject(name,func):
    try:func()
    except (ValueError,KeyError,TypeError):check(name,True)
    else:check(name,False)


def write(path, value):
    write_text(path, json.dumps(value, indent=2) + '\n')


def prepared_run(directory, fixture, role=None, portable=False):
    directory.mkdir()
    report = load_document(fixture / 'successful-run.json')
    prepared = load_document(fixture / 'successful-run-prepared.json')
    if role is not None:
        for phase in report['plan']['phases']:
            phase['role'] = role
        for run in report['phases']:
            run['phase']['role'] = role
        for bank in prepared['phases']:
            bank['phase']['role'] = role
    for bank in prepared['phases']:
        for session in bank['sessions']:
            if 'prompts' in session:
                session['prompt_hashes'] = [analysis.hashed(p) for p in session['prompts']]
    report['prepared_sha256'] = analysis.hashed(prepared)
    for run in report['phases']:
        raw = directory / Path(run['raw_dir']).name
        shutil.copytree(fixture / Path(run['raw_dir']).name, raw)
        run['raw_dir'] = str(raw)

        def add_wire(request):
            request['wire_body'] = json.dumps(request['payload'], ensure_ascii=True, allow_nan=False)
            request['wire_body_sha256'] = hashlib.sha256(request['wire_body'].encode()).hexdigest()
        for session in run['sessions']:
            for request in session['requests']:
                add_wire(request)
        write_text(raw / 'sessions.jsonl', ''.join((json.dumps(s) + '\n' for s in run['sessions'])))
        events = [json.loads(line) for line in (raw / 'events.jsonl').read_text().splitlines()]
        by_request = {r['request_id']: r for s in run['sessions'] for r in s['requests']}
        for event in events:
            if event['event'] == 'model_request':
                canonical = by_request[event['request']['request_id']]
                event['request']['wire_body'] = canonical['wire_body']
                event['request']['wire_body_sha256'] = canonical['wire_body_sha256']
        write_text(raw / 'events.jsonl', ''.join((json.dumps(e) + '\n' for e in events)))
    plan = report['plan']
    config = {'name': 'OFFLINE_SIMULATION_NOT_MODEL', 'args': [], 'env': {}}
    sources = {}
    for name in ('agent_session_client', 'benchmark_client', 'capacity_client'):
        path = OLD.parent / 'scripts/benchmark_client.py' if name == 'benchmark_client' else OLD / 'scripts' / f'{name}.py'
        member = 'scripts/benchmark_client.py' if name == 'benchmark_client' else 'scenarios/scripts/' + name + '.py'
        sources[member] = path.read_bytes()
    sources['scenarios/configs/cpu-analysis.json'] = json.dumps(config).encode()
    sources['scenarios/workloads/sessions/cpu-analysis.json'] = json.dumps(plan).encode()
    report['client_sha256'] = hashlib.sha256(sources['scenarios/scripts/agent_session_client.py']).hexdigest()
    report['helper_sha256'] = {name: hashlib.sha256(sources['scripts/benchmark_client.py' if name == 'benchmark_client' else 'scenarios/scripts/' + name + '.py']).hexdigest() for name in ('capacity_client', 'benchmark_client')}
    with zipfile.ZipFile(directory / 'source-snapshot.zip', 'w') as archive:
        for name, body in sources.items():
            archive.writestr(name, body)
    command = dict(experiment_id=directory.name, git_commit='CPU_SIMULATION', source_sha256={name: hashlib.sha256(body).hexdigest() for name, body in sources.items()}, command=['python', 'serve_agent_sessions.py', '--config', 'cpu-analysis.json', '--plan', 'cpu-analysis.json'])
    if portable:
        # The local HTTP fixture never loads a tokenizer; this explicit path is
        # synthetic provenance for the relocation check, not model evidence.
        plan.setdefault('tokenizer_path','/MOCK/tokenizer')
        context=dict(project_root=str(directory.parent),model_path=plan['tokenizer_path'],
                     run_dir=str(directory),research_root=str(directory.parent/'research/model'))
        old_plan=sources.pop('scenarios/workloads/sessions/cpu-analysis.json')
        raw_plan=json.loads(old_plan);raw_plan['tokenizer_path']='${MODEL_PATH}'
        sources['experiments/scenarios/workloads/sessions/cpu-analysis.json']=json.dumps(raw_plan).encode()
        sources['experiments/scenarios/configs/cpu-analysis.json']=sources.pop('scenarios/configs/cpu-analysis.json')
        for member in ('project_layout.py','experiments/catalog.json','project/tools/reproduce.py',
                       'metadata/environment.json','metadata/model-inventory.json'):
            sources[member]=b'{"synthetic_fixture":true}\n'
        with zipfile.ZipFile(directory/'source-snapshot.zip','w') as archive:
            for name,body in sources.items():archive.writestr(name,body)
        command['source_sha256']={name:hashlib.sha256(body).hexdigest() for name,body in sources.items()}
        command['path_context']=context
        command['command'][3]=str(Path(context['research_root'])/'experiments/scenarios/configs/cpu-analysis.json')
        command['command'][5]=str(Path(context['research_root'])/'experiments/scenarios/workloads/sessions/cpu-analysis.json')
    write(directory / 'command.json', command)
    status=dict(experiment_id=directory.name, exit_code=0, end_utc='CPU_SIMULATION', timed_out=False)
    if portable:status['path_context']=context
    write(directory / 'status.json', status)
    write(directory / 'experiment.json', dict(status='PASS', clients=[dict(exit_code=0)]))
    write(directory / 'sessions.json', report)
    write(directory / 'sessions-prepared.json', prepared)
    write(directory / 'config.json', config)
    write(directory / 'session-plan.json', plan)
    return (report, prepared)


def stability_fixture():
    rows=[]
    for index,arrival in enumerate(range(0,240,2)):
        rows.append(dict(index=index,kind='replay',admitted=True,censored=False,protocol_ok=True,
            terminal_reason='replay_complete',task_success=None,planned_offset_s=float(arrival),
            end_offset_s=float(arrival)+1,dispatch_lag_s=.001,
            requests=[dict(ok=True,timed_out=False,cancelled=False)]))
    return dict(phase=dict(arrival='poisson',role='confirm'),status='COMPLETED',error=None,protection_triggered=False,
        measurement_window_s=240.,sessions=rows,metric_samples=[dict(offset_s=float(t),http_status=200,values={
            'vllm:num_requests_waiting{engine="0"}':0.,'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"}':99.}) for t in range(240)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path, default=path_at(ROOT, 'logs/agent-session-client-fixtures-r2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    try:
        with tempfile.TemporaryDirectory(prefix='phasec-analysis-cpu-') as temporary:
            directory = Path(temporary) / 'cpu-synthetic-analysis'
            report, prepared = prepared_run(directory, args.fixture_dir)
            initial = analysis.analyze([directory])
            check('Actual mock-client SSE/session/event/Prom evidence passes wrapped end-to-end analyzer', initial['status'] == 'PASS', initial.get('errors'))
            portable = Path(temporary) / 'portable session with spaces'
            prepared_run(portable,args.fixture_dir,portable=True)
            relocated = analysis.analyze([portable])
            check('Centralized physical inputs and recorded tokenizer placeholder preserve session analysis',
                  relocated['status']=='PASS',relocated.get('errors'))
            stage = initial['stages'][0]
            check('Replay never gets task score or task cost', stage['summary']['successful_tasks'] is None and stage['summary']['tokens_per_successful_task'] is None)
            check('Ordinary measure phase included with explicit flag', stage['included_as_measurement'])
            old_source = 'async def stream_request():\n    if not 0 <= ix < 4: raise ValueError()\n'
            new_source = 'MAX_TOOL_CALLS_PER_ROUND = 8\nasync def stream_request():\n    if not 0 <= ix < MAX_TOOL_CALLS_PER_ROUND: raise ValueError()\n'
            check('Executed legacy source retains four-tool bound', analysis.recorded_tool_limit(old_source) == 4)
            check('Executed new constant source retains eight-tool bound', analysis.recorded_tool_limit(new_source) == 8)
            reject('Unknown source bound fails closed rather than assuming current runtime', lambda: analysis.recorded_tool_limit('async def stream_request():\n    pass\n'))
            r = deepcopy(report['phases'][0]['sessions'][0]['requests'][0])
            r['usage']['prompt_tokens'] = True
            reject('Boolean token usage rejected from raw SSE disagreement', lambda: analysis.check_request(r))
            r = deepcopy(report['phases'][0]['sessions'][0]['requests'][0])
            r['first_output_s'] += 0.01
            reject('First output timing independently checked against raw SSE', lambda: analysis.check_request(r))
            r = deepcopy(report['phases'][0]['sessions'][0]['requests'][0])
            r['payload']['prompt'][0] += 1
            reject('Payload source hash protects token IDs', lambda: analysis.check_request(r))
            r = deepcopy(report['phases'][0]['sessions'][0]['requests'][0])
            r['tpot_s'] += 0.01
            reject('Average token interval recomputed from actual output count', lambda: analysis.check_request(r))
            r = deepcopy(report['phases'][0]['sessions'][0]['requests'][0])
            r['wire_body'] += ' '
            reject('Exact wire byte change is detected independently of semantic JSON', lambda: analysis.check_request(r))
            original_session = report['phases'][0]['sessions'][0]
            s = deepcopy(original_session)
            r = s['requests'][0]
            r['payload']['cache_salt'] = 'wrong-namespace'
            r['payload_sha256'] = analysis.hashed(r['payload'])
            r['wire_body'] = json.dumps(r['payload'])
            r['wire_body_sha256'] = hashlib.sha256(r['wire_body'].encode()).hexdigest()
            reject('Actual cache treatment checked separately from workload identity', lambda: analysis.check_session(s, prepared['phases'][0]['sessions'][0], prepared['phases'][0]['phase'], report['phases'][0]['total_wall_s']))
            s = deepcopy(original_session)
            s['requests'][0]['headers']['X-data-parallel-rank'] = '3'
            reject('Unplanned DP route rejected', lambda: analysis.check_session(s, prepared['phases'][0]['sessions'][0], prepared['phases'][0]['phase'], report['phases'][0]['total_wall_s']))
            original = (directory / 'source-snapshot.zip').read_bytes()
            with zipfile.ZipFile(directory / 'source-snapshot.zip', 'a') as z:
                z.writestr('unlisted.py', 'not in source map')
            bad = analysis.analyze([directory])
            check('Unlisted source ZIP member rejected', bad['status'] == 'FAIL')
            write_bytes(directory / 'source-snapshot.zip', original)
            raw = directory / Path(report['phases'][0]['raw_dir']).name
            events = (raw / 'events.jsonl').read_text()
            lines = events.splitlines()
            changed = [line for line in lines if json.loads(line)['event'] != 'model_request']
            write_text(raw / 'events.jsonl', '\n'.join(changed) + '\n')
            bad = analysis.analyze([directory])
            check('Missing raw model event rejected despite intact session JSONL', bad['status'] == 'FAIL')
            write_text(raw / 'events.jsonl', events)
            sample = report['phases'][0]['metric_samples'][0]
            prom = raw / Path(sample['raw_file']).name
            body = prom.read_text()
            write_text(prom, body + '\nvllm:num_requests_waiting{engine="unlisted"} 7\n')
            bad = analysis.analyze([directory])
            check('Mutated raw Prometheus rejected', bad['status'] == 'FAIL')
            write_text(prom, body)
            for role in ('warmup', 'profile', 'pilot'):
                role_directory = Path(temporary) / ('cpu-role-' + role)
                prepared_run(role_directory, args.fixture_dir, role=role)
                altered = analysis.analyze([role_directory])
                check(role + ' source/plan consistent input passes but is not included', altered['status'] == 'PASS' and (not altered['stages'][0]['included_as_measurement']))
                text = analysis.markdown(altered)
                prefix = '| ' + role_directory.name + ' / ' + stage['phase']['id'] + ' |'
                check(role + ' excluded from ranked table, retained in disclosure', prefix not in text and 'excluded from normal performance: ' + role in text)
            bank = deepcopy(prepared['phases'][0])
            base = analysis.workload_identity(bank, prepared['tokenizer_sha256'])['sha256']
            for key, value in (('cache_scope', 'none'), ('dp_routing', 'sticky4'), ('id', 'different-label'), ('role', 'pilot')):
                candidate = deepcopy(bank)
                candidate['phase'][key] = value
                check('Replay identity excludes treatment/label ' + key, analysis.workload_identity(candidate, prepared['tokenizer_sha256'])['sha256'] == base)
            for key in ('prompts', 'outputs', 'tool_waits_s'):
                candidate = deepcopy(bank)
                if key == 'prompts':
                    candidate['sessions'][0][key][0][0] += 1
                else:
                    candidate['sessions'][0][key][0] += 1
                check('Replay identity changes with actual ' + key, analysis.workload_identity(candidate, prepared['tokenizer_sha256'])['sha256'] != base)
            sample = deepcopy(report['phases'][0]['sessions'][0])
            sample.update(kind='agent', task_success=True, terminal_reason='natural_stop')
            failed = deepcopy(sample)
            failed.update(task_success=False, protocol_ok=False, terminal_reason='request_error')
            failed['requests'][0]['ok'] = False
            cost = analysis.summarize([sample, failed], 10.0)
            expected = sum((r['usage']['prompt_tokens'] for s in (sample, failed) for r in s['requests']))
            check('Cost per successful task includes failed-call known token usage', cost['tokens_per_successful_task']['prompt'] == expected and cost['all_calls_known_usage']['complete'])
            failed['requests'][0]['usage'] = None
            cost = analysis.summarize([sample, failed], 10.0)
            check('Unknown failed-call usage is explicit incomplete lower bound', not cost['tokens_per_successful_task']['token_usage_complete'])
            no_wins = deepcopy(sample)
            no_wins['task_success'] = False
            check('Zero successful tasks gives null cost, not zero', analysis.summarize([no_wins], 10.0)['tokens_per_successful_task']['prompt'] is None)
            zero_cache = deepcopy(sample)
            for req in zero_cache['requests']:
                req['cached_tokens'] = 0
            check('Known zero cached tokens reported as zero', analysis.summarize([zero_cache], 10.0)['cached_tokens'] == 0)
            zero_cache['requests'][0]['cached_tokens'] = None
            check('Partially known cached usage cannot yield aggregate hit fraction', analysis.summarize([zero_cache], 10.0)['cache_fraction'] is None)
            stable = stability_fixture()
            v = analysis.stability(stable)
            check('Analytic 1-second sessions at two-second arrivals stable', v['classification'] == 'STABLE_IN_WINDOW' and v['analysis_arrivals'] == 90 and (v['analysis_completions'] == 90) and (v['completion_deficit'] == 0))
            check('Exact waiting gauge excludes by_reason breakdown', all((b['server_mean_waiting'] == 0 for b in v['analysis_bins'])))
            growing = deepcopy(stable)
            for row in growing['sessions']:
                row['end_offset_s'] = 500.0
            v = analysis.stability(growing)
            check('Uncompleted arrivals give analytically growing backlog', v['classification'] == 'GROWING_BACKLOG' and v['analysis_completions'] == 0)
            quality = deepcopy(stable)
            for row in quality['sessions']:
                row.update(kind='agent', terminal_reason='natural_stop', task_success=False, grade={'success': False})
            check('Objective task failure may remain protocol stable', analysis.stability(quality)['classification'] == 'STABLE_IN_WINDOW')
            quality['sessions'][0]['grade'] = None
            check('Missing grade blocks stable certification', analysis.stability(quality)['classification'] == 'INCONCLUSIVE')
            for field, value in (('terminal_reason', 'turn_budget'), ('censored', True), ('admitted', False)):
                candidate = deepcopy(stable)
                candidate['sessions'][0][field] = value
                check('Unhealthy session blocks stability ' + field, analysis.stability(candidate)['classification'] == 'INCONCLUSIVE')
            candidate = deepcopy(stable)
            candidate['sessions'][0]['requests'][0]['timed_out'] = True
            check('Timed-out request cannot hide in complete session', analysis.stability(candidate)['classification'] == 'INCONCLUSIVE')
            candidate = deepcopy(stable)
            for row in candidate['sessions']:
                row['dispatch_lag_s'] = 0.101
            check('Dispatch lag gate retained', analysis.stability(candidate)['classification'] == 'INCONCLUSIVE')
            candidate = deepcopy(stable)
            candidate['sessions'] = candidate['sessions'][:10]
            check('Insufficient analysis arrivals inconclusive', analysis.stability(candidate)['classification'] == 'INCONCLUSIVE')
            candidate = deepcopy(stable)
            candidate['metric_samples'] = [deepcopy(stable['metric_samples'][60]) for _ in range(500)]
            check('Duplicate sample times cannot inflate monitoring coverage', analysis.stability(candidate)['classification'] == 'INCONCLUSIVE' and analysis.stability(candidate)['server_sample_coverage'] < 0.01)
            candidate = deepcopy(stable)
            for sample in candidate['metric_samples']:
                sample['http_status'] = 503
            check('HTTP-error gauges cannot satisfy observation coverage', analysis.stability(candidate)['classification'] == 'INCONCLUSIVE' and analysis.stability(candidate)['server_sample_coverage'] == 0)
            candidate = deepcopy(stable)
            candidate['phase']['role'] = 'pilot'
            check('Long pilot still not confirm capacity', analysis.stability(candidate)['classification'] == 'SCREEN_ONLY')
            candidate = deepcopy(stable)
            candidate['owned_run_terminal_success'] = False
            check('Failed supervisor prevents phase stable certification', analysis.stability(candidate)['classification'] == 'INCONCLUSIVE')
            early = deepcopy(stable)
            for row in early['sessions'][:30]:
                row['end_offset_s'] = 61.0
            v = analysis.stability(early)
            check('Prior-window sessions may complete giving C above A', v['analysis_completions'] > v['analysis_arrivals'] and v['classification'] == 'STABLE_IN_WINDOW')
        status = 'PASS'
    except BaseException as exc:
        status = 'FAIL'
        checks.append(dict(name='unexpected exception', passed=False, error=repr(exc)))
        raise
    finally:
        result = dict(status=locals().get('status', 'FAIL'), created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, analyzer_sha256=analysis.sha(ROOT / 'scripts/analyze_sessions.py'), checker_sha256=analysis.sha(Path(__file__)), original_mock_report_sha256=analysis.sha(args.fixture_dir / 'successful-run.json'), scope='OFFLINE_SIMULATION_NOT_MODEL; read-only actual mock stream data plus explicitly synthetic supervisor and analytic boundary controls.')
        with open_output(args.output, 'x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], checks=len(checks), output=str(args.output)), indent=2))


if __name__=='__main__':main()
