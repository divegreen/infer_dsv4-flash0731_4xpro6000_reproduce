#!/usr/bin/env python3
"""CPU-only independent replay/no-grade and Agent grader-identity controls.

Uses temporary metadata and an Agent budget termination to test source checks
without executing any task, candidate code or objective grading function.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = ROOT / 'scripts/regrade_sessions.py'
    before = sha(source)
    m = ModuleType('regrade_review')
    m.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), 'exec'), m.__dict__)
    checks = []
    grader = m.TASK_SOURCE
    key = 'scenarios/scripts/agent_session_tasks.py'
    expected = sha(grader)

    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append(label)
    with tempfile.TemporaryDirectory(prefix='regrade-applicability-MOCK-') as td:
        root = Path(td)
        run = root / 'c99-MOCK'
        run.mkdir()
        counter = 0
        status = dict(exit_code=0, end_utc='MOCK-terminal')
        command = dict(source_sha256={key: expected})
        replay = dict(status='COMPLETED', helper_sha256={}, phases=[dict(phase=dict(id='one', kind='replay'), sessions=[dict(kind='replay', grade=None, task_success=None)])])

        def invoke(report, cmd=None, task_source=None):
            nonlocal counter
            counter += 1
            out = root / f'result-{counter}.json'
            for n, obj in [('status.json', status), ('sessions.json', report), ('command.json', command if cmd is None else cmd)]:
                write_text(run / n, json.dumps(obj) + '\n')
            m.TASK_SOURCE = grader if task_source is None else task_source
            saved = sys.argv
            sys.argv = [str(source), '--result-dir', str(run), '--output', str(out)]
            try:
                with redirect_stdout(io.StringIO()):
                    m.main()
            finally:
                sys.argv = saved
            return load_document(out)
        no_agent = invoke(replay, dict(source_sha256={}), root / 'absent-grader.py')
        check('replay_never_requires_or_reads_grader', no_agent['grading_applicability'] == 'NOT_APPLICABLE_REPLAY_ONLY' and no_agent['grader_sha256'] is None)
        check('replay_no_task_success_invented', no_agent['regraded'] == 0 and no_agent['sessions'] == [] and ('not task success' in no_agent['scope']))

        def reject(label, report, cmd=None, path=None):
            try:
                invoke(report, cmd, path)
            except ValueError:
                checks.append(label)
            else:
                raise AssertionError(label + ' accepted')
        for field, value in [('grade', {'success': True}), ('task_success', True), ('kind', 'agent')]:
            bad = deepcopy(replay)
            bad['phases'][0]['sessions'][0][field] = value
            reject('reject_replay_claim_' + field, bad)
        agent = dict(status='COMPLETED', helper_sha256={'agent_session_tasks': expected}, phases=[dict(phase=dict(id='one', kind='agent'), raw_dir='sessions-one', sessions=[dict(kind='agent', index=0, task_id='reservation_book', terminal_reason='turn_budget', task_success=False)])])
        positive = invoke(agent)
        check('actual_current_agent_source_identity_accepted', positive['grading_applicability'] == 'AGENT_TASKS' and positive['grader_sha256'] == expected)
        check('budget_failure_preserved_without_grading', positive['regraded'] == 0 and positive['sessions'][0]['classification'] == 'turn_budget' and (positive['sessions'][0]['regraded'] is False))
        absent = deepcopy(agent)
        absent['helper_sha256'] = {}
        reject('agent_missing_executed_helper_rejected', absent)
        wrong = deepcopy(agent)
        wrong['helper_sha256']['agent_session_tasks'] = '0' * 64
        reject('agent_executed_helper_mismatch_rejected', wrong)
        reject('agent_command_helper_missing_rejected', agent, dict(source_sha256={}))
        changed = root / 'changed-grader.py'
        write_text(changed, '# MOCK source drift\n')
        reject('agent_current_source_mismatch_rejected', agent, path=changed)
    real = path_at(ROOT.parent, 'scenarios/results')
    command03 = load_document(real / 'session-tp4-cache-screen/command.json')
    check('actual_c03_snapshot_helper_matches_current', command03['source_sha256'][key] == expected)
    regraded = load_document(path_at(ROOT, 'logs/session-tp4-replay-open-objective-regrade-r2.json'))
    check('real_c08_r2_explicit_not_applicable', regraded['grading_applicability'] == 'NOT_APPLICABLE_REPLAY_ONLY' and regraded['grader_sha256'] is None and (regraded['regraded'] == 0) and (regraded['sessions'] == []))
    check('real_c08_gpu_success_kept_separate', regraded['supervisor_exit_code'] == 0 and regraded['experiment_execution_status'] == 'COMPLETED')
    check('source_frozen_during_review', sha(source) == before and sha(grader) == expected)
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, reviewed_source_sha256=before, actual_task_source_sha256=expected, reviewer_sha256=sha(Path(__file__)), real_regrade=dict(path=str(path_at(ROOT, 'logs/session-tp4-replay-open-objective-regrade-r2.json')), sha256=sha(path_at(ROOT, 'logs/session-tp4-replay-open-objective-regrade-r2.json'))), scope='Independent CPU applicability/source-identity regression review, no task/grade/model/GPU/HTTP execution', limitations=['Positive Agent control imports the real matching helper but uses a budget terminal to avoid invoking context/grade/tool functions.', 'Natural-final objective reruns are covered by existing actual screen regrade evidence, not repeated here.', 'NOT_APPLICABLE indicates no objective scoring for replay; neither PASS nor zero regraded is an Agent quality result.'])
    with open_output(a.output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), output=str(a.output)), indent=2))


if __name__=='__main__':main()
