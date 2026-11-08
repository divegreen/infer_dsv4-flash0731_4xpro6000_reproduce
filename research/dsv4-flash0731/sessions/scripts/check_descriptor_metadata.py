#!/usr/bin/env python3
"""Independent synthetic metadata controls plus existing real artifact checks.

No production runtime/grader imports; only temporary files are perturbed.
Existing independent arithmetic/reviewer r2 evidence is checked, not re-executed.
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
def encoded(value):return (json.dumps(value,sort_keys=True)+'\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = ROOT / 'scripts/describe_agent_workloads.py'
    source_raw = source.read_bytes()
    m = ModuleType('metadata_review')
    m.__file__ = str(source)
    exec(compile(source_raw, str(source), 'exec'), m.__dict__)
    checks = []

    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append(label)
    with tempfile.TemporaryDirectory(prefix='descriptor-metadata-MOCK-') as td:
        root = Path(td)
        results = path_at(root, 'results')
        d = results / 'c99-MOCK'
        d.mkdir(parents=True)
        phase = dict(id='one', kind='replay', role='screen', arrival='closed')
        plan = dict(experiment_id=d.name, phases=[phase])
        status = dict(experiment_id=d.name, exit_code=0, timed_out=False, supervisor_error=None, end_utc='MOCK')
        owned = dict(status='PASS')
        report = dict(status='COMPLETED', plan=plan, phases=[dict(phase=phase, status='COMPLETED', total_wall_s=1.0)])
        row = dict(index=0, kind='replay', profile='coding', task_success=None, protocol_ok=False, terminal_reason='censored', requests=[], tools=[], host_operations=[], admitted=False)
        stats = dict(sessions=1, successful_tasks=None, protocol_completed=0, model_calls=0, successful_model_calls=0, tool_calls=0, all_calls_known_usage=dict(prompt_tokens=0, output_tokens=0, known_calls=0, complete=True))
        baseline = {'status.json': status, 'experiment.json': owned, 'sessions.json': report, 'session-plan.json': plan, 'sessions-one/sessions.jsonl': row}
        baseline = json.loads(json.dumps(baseline))
        summary = dict(status='PASS', errors=[], experiments=[dict(id=d.name, terminal_success=True, supervisor_exit_code=0, execution_status='COMPLETED')], stages=[dict(experiment=d.name, phase=phase, terminal_success=True, included_as_measurement=True, exclusion_reason=None, total_wall_s=1.0, arrival_workload_sha256='MOCK', summary=stats)], input_ledger={})

        def write(objects, overview):
            overview = deepcopy(overview)
            overview['input_ledger'] = {}
            for name, value in objects.items():
                path = d / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = encoded(value)
                write_bytes(path, raw)
                overview['input_ledger'][str(path)] = dict(sha256=sha(raw), bytes=len(raw))
            path = root / 'MOCK-summary.json'
            write_bytes(path, encoded(overview))
            return path

        def invoke(objects=None, overview=None):
            return m.describe([write(objects or baseline, overview or summary)], results)
        result = invoke()
        check('synthetic_base_metadata_passes', result['status'] == 'PASS' and result['stages'][0]['included_as_measurement'] is True)

        def reject(label, mutate, fragment):
            objects = deepcopy(baseline)
            overview = deepcopy(summary)
            mutate(objects, overview)
            try:
                invoke(objects, overview)
            except ValueError as exc:
                check(label, fragment in str(exc))
            else:
                raise AssertionError(label + ' accepted')
        reject('reject_boolean_supervisor_exit', lambda b, s: b['status.json'].update(exit_code=False), 'Nonterminal/invalid')
        reject('reject_missing_terminal_time', lambda b, s: b['status.json'].update(end_utc=None), 'Nonterminal/invalid')
        reject('reject_failed_run_claimed_healthy', lambda b, s: b['status.json'].update(exit_code=1), 'terminal flag')
        reject('reject_supervisor_error_claimed_healthy', lambda b, s: b['status.json'].update(supervisor_error='MOCK error'), 'terminal flag')
        reject('reject_timedout_claimed_healthy', lambda b, s: b['status.json'].update(timed_out=True), 'terminal flag')
        reject('reject_active_report_claimed_complete', lambda b, s: b['sessions.json'].update(active_phase='one'), 'terminal flag')
        reject('reject_failed_owned_experiment', lambda b, s: b['experiment.json'].update(status='FAILED'), 'terminal flag')
        reject('reject_report_saved_plan_mismatch', lambda b, s: b['session-plan.json'].update(experiment_id='c100-MOCK'), 'original saved plan')
        reject('reject_executed_phase_plan_mismatch', lambda b, s: b['sessions.json']['phases'][0]['phase'].update(kind='agent'), 'Executed phase metadata')
        reject('reject_summary_phase_role_mismatch', lambda b, s: s['stages'][0]['phase'].update(role='pilot'), 'Summary phase metadata')
        reject('reject_summary_measurement_flag_mismatch', lambda b, s: s['stages'][0].update(included_as_measurement=False), 'formal inclusion')
        reject('reject_summary_cohort_wall_mismatch', lambda b, s: s['stages'][0].update(total_wall_s=2.0), 'cohort wall')
        reject('reject_raw_phase_failed_claimed_measurement', lambda b, s: b['sessions.json']['phases'][0].update(status='FAILED'), 'formal inclusion')
        reject('reject_summary_report_state_mismatch', lambda b, s: s['experiments'][0].update(execution_status='RUNNING'), 'exit/report state')
        objects = deepcopy(baseline)
        overview = deepcopy(summary)
        objects['status.json']['exit_code'] = 1
        overview['experiments'][0].update(terminal_success=False, supervisor_exit_code=1)
        overview['stages'][0].update(terminal_success=False, included_as_measurement=False, exclusion_reason='MOCK failed supervisor')
        result = invoke(objects, overview)
        check('failed_run_preserved_outside_measurement', result['stages'][0]['included_as_measurement'] is False)
        overview = deepcopy(summary)
        path = write(baseline, overview)
        wrong = load_document(path)
        wrong['input_ledger'][str(d / 'sessions.json')]['sha256'] = '0' * 64
        write_bytes(path, encoded(wrong))
        try:
            m.describe([path], results)
        except ValueError as exc:
            check('reject_raw_metadata_hash_drift', 'Changed audited input' in str(exc))
        else:
            raise AssertionError('hash drift accepted')
    prior = path_at(ROOT, 'logs/independent-offline-runtime-recheck-r2.json')
    r2 = load_document(prior)
    expected = next((s for s in r2['reviewed_sources'] if s['path'].endswith('/describe_agent_workloads.py')))
    check('existing_independent_r2_pass_and_same_descriptor', r2['status'] == 'PASS' and (not r2['findings']) and (expected['sha256'] == sha(source_raw)))
    actual_path = path_at(ROOT, 'summary/screen-c03-c07-workloads-r1.json')
    actual = load_document(actual_path)
    check('screen_artifact_same_reviewed_descriptor', actual['status'] == 'PASS' and actual['script_sha256'] == sha(source_raw))
    check('all_screen_experiments_preserved', len(actual['experiments']) == 5 and {e['id'] for e in actual['experiments']} == {'session-tp4-cache-screen', 'session-tp4-noapc-screen', 'session-tp2pp2-agent-screen', 'session-dspark3-agent-screen', 'session-dp4ep-affinity-screen'})
    check('actual_formal_roles_exclude_warmup_pilot', all((s['phase']['role'] in ('screen', 'measure', 'confirm') for s in actual['stages'] if s['included_as_measurement'])))
    check('artifact_raw_metadata_closure', all((any((k.endswith('/' + e['id'] + '/' + n) for k in actual['input_ledger'])) for e in actual['experiments'] for n in ('status.json', 'experiment.json', 'sessions.json', 'session-plan.json'))))
    check('actual_session_count_preserved', sum((s['summary']['sessions'] for s in actual['stages'])) == 586)
    check('descriptor_source_unchanged', source.read_bytes() == source_raw)
    output = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='Incremental independent CPU metadata controls; no GPU evidence or original data edits', checks=checks, descriptor_sha256=sha(source_raw), reviewer_sha256=sha(Path(__file__).read_bytes()), reused_evidence=dict(path=str(prior), sha256=sha(prior.read_bytes()), checks=len(r2['checks'])), actual_artifact=dict(path=str(actual_path), sha256=sha(actual_path.read_bytes()), experiments=len(actual['experiments']), phases=len(actual['stages']), sessions=586), limitations=['Real artifact source/role/closure counts checked; underlying raw byte integrity and arithmetic reuse existing independent r2 audit.', 'Descriptor remains downstream of an evidence-analysis PASS; it does not independently re-audit source ZIP, SSE or task grading.'])
    with open_output(a.output, 'x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), output=str(a.output)), indent=2))


if __name__=='__main__':main()
