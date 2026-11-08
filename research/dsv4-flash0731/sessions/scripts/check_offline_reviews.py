#!/usr/bin/env python3
"""Independent CPU review controls; only temporary copies are perturbed.

No GPU, HTTP, model, task execution, runtime editing or raw evidence rewriting.
REVIEW_REQUIRED means the offline reviewer accepted a contradictory fixture,
not that the original GPU experiment has that contradiction.
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
import json
from pathlib import Path
import tempfile
from types import ModuleType

ROOT=Path(__file__).resolve().parents[1]


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2)+'\n').encode()


def load(name,filename):
    path=ROOT/'scripts'/filename;raw=path.read_bytes();module=ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module,dict(path=str(path),sha256=sha(raw),bytes=len(raw))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    runtime, runtime_source = load('independent_runtime_review', 'review_session_runtime.py')
    descriptor, descriptor_source = load('independent_workload_description', 'describe_agent_workloads.py')
    checks = []
    findings = []
    ledger = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path)] = dict(sha256=sha(raw), bytes=len(raw))
        return raw

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    phases = [dict(id='warm', role='warmup', start_unix=100.0, end_unix=110.0), dict(id='probe', role='pilot', start_unix=110.0, end_unix=120.0), dict(id='screen', role='screen', start_unix=120.25, end_unix=140.0)]
    for label, t, width, scope, exact in [('startup_halfopen_boundary', 99.0, 1.0, 'STARTUP_BEFORE_CLIENT', None), ('warmup_exact', 100.0, 1.0, 'NONMEASUREMENT_OR_BOUNDARY', 'warm'), ('pilot_is_not_measurement', 110.0, 1.0, 'NONMEASUREMENT_OR_BOUNDARY', 'probe'), ('pilot_second_before_measurement', 119.0, 1.0, 'NONMEASUREMENT_OR_BOUNDARY', 'probe'), ('second_resolution_crosses_measurement_start', 120.0, 1.0, 'MEASUREMENT_OR_BOUNDARY', None), ('exact_measurement_interval', 120.3, 0.001, 'MEASUREMENT_OR_BOUNDARY', 'screen'), ('halfopen_after_all', 140.0, 1.0, 'AFTER_ALL_PHASES', None), ('warmup_pilot_boundary_not_measurement', 109.5, 1.0, 'NONMEASUREMENT_OR_BOUNDARY', None), ('subsecond_gap_unassigned', 120.01, 0.1, 'BETWEEN_PHASES', None)]:
        actual = runtime.mapping(t, width, phases)
        check(label, actual['scope'] == scope and actual['unique_containing_phase'] == exact)
    real = []
    for name in ('session-tp4-cache-screen', 'session-tp4-noapc-screen', 'session-tp2pp2-agent-screen', 'session-dspark3-agent-screen'):
        d = runtime.RESULTS / name
        status = json.loads(read(d / 'status.json'))
        command = json.loads(read(d / 'command.json'))
        report = json.loads(read(d / 'sessions.json'))
        plan = json.loads(read(d / 'session-plan.json'))
        healthy = type(status.get('exit_code')) is int and status['exit_code'] == 0 and (status.get('timed_out') is False) and (not status.get('supervisor_error')) and bool(status.get('end_utc'))
        identity = status['experiment_id'] == command['experiment_id'] == name and status['command'] == command['command'] and (status['cwd'] == command['cwd']) and (status['source_sha256'] == command['source_sha256'])
        report_match = report['plan'] == plan and report['status'] == 'COMPLETED' and (not report.get('active_phase'))
        check(name + '_actual_terminal_identity', healthy and identity and report_match)
        real.append(dict(id=name, healthy_terminal=healthy, supervisor_identity=identity, report_plan_terminal=report_match))
    r1path = path_at(ROOT, 'logs/runtime-review-c03-c04-r1.json')
    r1 = json.loads(read(r1path))
    snapshot = read(path_at(ROOT, 'logs/runtime-review-c03-c04-r1-script.txt'))
    check('runtime_r1_script_identity_preserved', sha(snapshot) == r1['script_sha256'])
    for experiment in r1['experiments']:
        check(experiment['experiment_id'] + '_r1_original_input_hashes', all((sha(read(runtime.RESULTS / experiment['experiment_id'] / name)) == h for name, h in experiment['evidence_sha256'].items())))
        check(experiment['experiment_id'] + '_pilot_warning_not_counted_as_measurement', all(('pilot' not in e.get('mapping', {}).get('role_candidates', []) or 'screen' in e['mapping']['role_candidates'] or e['mapping']['scope'] != 'MEASUREMENT_OR_BOUNDARY' for e in experiment['visible_jit_warnings'])))
    with tempfile.TemporaryDirectory(prefix='independent-offline-review-MOCK-') as td:
        temp = Path(td)
        name = 'session-tp4-cache-screen'
        copy = temp / name
        copy.mkdir()
        original = {n: read(runtime.RESULTS / name / n) for n in ('status.json', 'experiment.json', 'sessions.json', 'config.json', 'server-command.json', 'command.json', 'server.log', 'process.log', 'session-plan.json', 'source-snapshot.zip')}
        for n, raw in original.items():
            write_bytes(copy / n, raw)
        saved = runtime.RESULTS
        runtime.RESULTS = temp
        try:
            check('unmodified_temporary_runtime_copy_passes', runtime.review(name)['status'] == 'PASS_WITH_OBSERVABILITY_LIMITATIONS')
            cases = [('boolean_supervisor_exit', 'status.json', lambda x: x.update(exit_code=False)), ('supervisor_error_present', 'status.json', lambda x: x.update(supervisor_error='MOCK control error')), ('supervisor_argv_mismatch', 'status.json', lambda x: x.update(command=['MOCK different launcher'])), ('supervisor_id_mismatch', 'status.json', lambda x: x.update(experiment_id='c99-MOCK-other')), ('supervisor_cwd_mismatch', 'status.json', lambda x: x.update(cwd='/MOCK/other')), ('completed_report_still_active', 'sessions.json', lambda x: x.update(active_phase='MOCK-unfinished')), ('runtime_saved_plan_identity_mismatch', 'session-plan.json', lambda x: x.update(experiment_id='c99-MOCK-other'))]
            for label, filename, mutate in cases:
                changed = json.loads(original[filename])
                mutate(changed)
                write_bytes(copy / filename, encode(changed))
                try:
                    runtime.review(name)
                except Exception as exc:
                    checks.append(label + '_rejected: ' + type(exc).__name__)
                else:
                    findings.append(dict(id=label, component='review_session_runtime', observed='Contradictory temporary fixture accepted', actual_experiment_contradiction=False, expected='Reject inconsistent terminal/provenance metadata'))
                finally:
                    write_bytes(copy / filename, original[filename])
        finally:
            runtime.RESULTS = saved
        summary_path = path_at(ROOT, 'summary/session-tp4-cache-screen-terminal-r1.json')
        summary = json.loads(read(summary_path))
        pilot = next((s for s in summary['stages'] if s['phase']['role'] == 'pilot'))
        probe = deepcopy(summary)
        probe['stages'] = [deepcopy(pilot)]
        probe_path = temp / 'MOCK-pilot-summary.json'
        write_bytes(probe_path, encode(probe))
        actual = descriptor.describe([probe_path], runtime.RESULTS)
        check('unmodified_pilot_description_excluded', actual['stages'][0]['included_as_measurement'] is False)
        probe['stages'][0]['included_as_measurement'] = True
        probe['stages'][0]['exclusion_reason'] = None
        write_bytes(probe_path, encode(probe))
        try:
            actual = descriptor.describe([probe_path], runtime.RESULTS)
        except Exception as exc:
            checks.append('contradictory_pilot_measurement_flag_rejected: ' + type(exc).__name__)
        else:
            check('controlled_summary_label_reaches_output', actual['stages'][0]['included_as_measurement'] is True)
            findings.append(dict(id='pilot_promoted_by_summary_flag', component='describe_agent_workloads', observed='Unchanged original pilot raw JSONL accepted in measurement-labelled output after changing temporary summary flag only', actual_experiment_contradiction=False, expected='Derive formal inclusion from raw phase role and successful terminal records, assert summary metadata agrees'))
    req = dict(ok=True, usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120, prompt_tokens_details=dict(cached_tokens=0)), cached_tokens=0, dispatch_offset_s=1.0, end_offset_s=3.0, turn=0)
    op = dict(name='execute_tool', ready_offset_s=3.0, worker_start_offset_s=3.5, worker_end_offset_s=4.5, end_offset_s=5.0, queue_s=0.5, worker_s=1.0)
    row = dict(index=0, kind='agent', task_id='MOCK', admitted=True, planned_offset_s=0.0, start_offset_s=1.0, end_offset_s=5.0, task_success=True, protocol_ok=True, terminal_reason='natural_stop', requests=[req], tools=[dict(wall_s=0.8)], host_operations=[op], grade={'wall_s': 0.0}, elapsed_s=4.0, scheduled_latency_s=5.0, user_visible_latency_s=3.0)
    timing = descriptor.timing_projection(row)
    check('independent_timing_partition_model_queue_worker_resume', timing['model_request_wall_s'] == 2 and timing['tool_worker_wall_s'] == 1 and (timing['host_queue_wall_s'] == 0.5) and (timing['host_resume_wall_s'] == 0.5) and (timing['residual_wall_s'] == 0))
    check('nested_tool_wall_not_added_twice', timing['tool_internal_wall_s'] == 0.8 and sum((timing[k] for k in descriptor.COMPONENTS)) == 4)
    missing = deepcopy(row)
    missing.update(index=1, task_success=False)
    missing['requests'][0].update(ok=False, usage=None, cached_tokens=None)
    group = descriptor.group_summary([descriptor.session_projection(row), descriptor.session_projection(missing)], 'agent', 5.0)
    check('independent_all_attempt_cost_lower_bound', group['all_attempt_cost']['total_tokens_per_successful_task'] == 120 and group['all_attempt_cost']['complete'] is False)
    check('known_zero_cache_distinct_from_missing', group['request_lengths']['cached_tokens']['n'] == 1 and group['request_lengths']['cached_tokens']['missing'] == 1 and (group['request_lengths']['cached_tokens']['p50'] == 0))
    check('original_evidence_unchanged', all((sha(Path(p).read_bytes()) == v['sha256'] for p, v in ledger.items())))
    result = dict(status='REVIEW_REQUIRED' if findings else 'PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='Independent CPU arithmetic/metadata controls and immutable real-input identity checks; no new GPU evidence', reviewed_sources=[runtime_source, descriptor_source], reviewer_sha256=sha(Path(__file__).read_bytes()), checks=checks, findings=findings, actual_terminal_identity=real, evidence_ledger=ledger, limitations=['Contradictory fixtures are temporary copies, not observed errors in c03-c06 actual evidence.', 'No full source/SSE/task re-audit performed; existing auditors retain that responsibility.', 'Startup config comparison covers selected fields; not every possible CLI/environment/default interaction.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(a.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), findings=[f['id'] for f in findings], output=str(a.output)), indent=2))
    raise SystemExit(2 if findings else 0)


if __name__=='__main__':main()
