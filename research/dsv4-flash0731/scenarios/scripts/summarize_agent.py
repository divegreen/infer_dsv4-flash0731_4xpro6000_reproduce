#!/usr/bin/env python3
"""CPU-only independent checks/summary of saved Agent scenarios, no inference.

Pairs replay by the complete fixed request payload, retaining cache/config labels.
Non-streaming TTFT/TPOT remain unavailable. Quality failures are not removed.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import collections
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def distribution(values):
    values = sorted(values)
    def quantile(p):
        index = (len(values)-1)*p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(n=len(values), p50=quantile(.5) if values else None,
                p95=quantile(.95) if values else None,
                min=values[0] if values else None, max=values[-1] if values else None)


class Review:

    def __init__(self, root):
        self.root = root.resolve()
        self.errors, self.incomplete, self.reports, self.sources = ([], [], [], {})
        self.replay_groups = collections.defaultdict(list)

    def check(self, condition, location, detail):
        if not condition:
            self.errors.append(dict(location=location, detail=detail))

    def load(self, path):
        raw = path.read_bytes()
        self.sources[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def http(self, h, location):
        self.check(h.get('attempts') == 1, location, 'Unexpected HTTP retry count')
        self.check(h.get('ttft_s') is None and h.get('tpot_s') is None, location, 'Non-streaming TTFT/TPOT must be null')
        self.check(h.get('timing_scope') == 'non_streaming_full_response', location, 'Incorrect timing scope')
        request = h['request']
        self.check(request.get('stream') is False and request.get('temperature') == 0, location, 'Sampling/streaming differs')
        self.check(request.get('reasoning_effort') == 'none' and request.get('chat_template_kwargs', {}).get('enable_thinking') is False, location, 'Unexpected reasoning mode')
        self.check(type(h['e2e_s']) in (float, int) and h['e2e_s'] >= 0, location, 'Invalid E2E')
        if h['protocol_ok']:
            self.check(h['http_status'] == 200 and h['error'] is None, location, 'Successful protocol has HTTP error')
            self.check(not h.get('response_truncated') and (not h.get('response_incomplete')), location, 'Successful protocol is incomplete')
            raw = json.loads(h['raw_response'])
            self.check(raw == h['response'], location, 'Raw response differs from parsed response')
            if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list) or len(raw['choices']) != 1:
                self.check(False, location, 'Successful raw response does not contain one choice')
                return
            self.check(raw['choices'][0]['message'] == h['message'], location, 'Parsed message differs')
            self.check(raw['usage'] == h['usage'], location, 'Parsed usage differs')
            self.check(raw['choices'][0]['finish_reason'] == h['finish_reason'], location, 'Parsed finish reason differs')
            u = h['usage']
            self.check(all((type(u[k]) is int and u[k] >= 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))), location, 'Invalid actual usage')
            self.check(u['prompt_tokens'] + u['completion_tokens'] == u['total_tokens'], location, 'Usage total mismatch')
            cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
            self.check(h['cached_tokens'] == cached, location, 'Cached-token field mismatch')
            self.check(math.isclose(h['output_tokens_per_e2e_s'], u['completion_tokens'] / h['e2e_s'], rel_tol=1e-09), location, 'Token/E2E arithmetic mismatch')
        else:
            self.check(bool(h.get('error')), location, 'Protocol failure missing error')

    def file(self, path):
        d = self.load(path)
        where = str(path.relative_to(self.root))
        if 'mode' not in d:
            self.incomplete.append(where + ' initializing')
            return
        complete = d['status'] in ('COMPLETED', 'COMPLETED_WITH_PROTOCOL_ERRORS')
        if d['status'] == 'RUNNING':
            self.incomplete.append(where)
        directory = path.parent
        with zipfile.ZipFile(directory / 'source-snapshot.zip') as z:
            prefix = self.root.name + '/'
            code = z.read(prefix + 'scripts/agent_benchmark.py')
            self.check(hashlib.sha256(code).hexdigest() == d['client_sha256'], where, 'Runtime client hash differs from snapshot')
            task_data = json.loads(z.read(prefix + 'workloads/agent/tasks.json'))['tasks']
            quality_data = json.loads(z.read(prefix + 'workloads/agent/quality.json'))['cases']
            initial = {t['id']: z.read(prefix + f'workloads/agent/{t['id']}/initial.py').decode() for t in task_data}
        mode = d['mode']
        expected = [f'fixed-turn-{i}' for i in (1, 2, 3)] if mode == 'replay' else [t['id'] for t in task_data] if mode == 'closed-loop' else [q['id'] for q in quality_data]
        observed = [(r['round'], r.get('case_id', r.get('task_id'))) for r in d['runs']]
        planned = [(round_id, case) for round_id in range(1, d['rounds'] + 1) for case in expected]
        self.check(observed == planned[:len(observed)], where, 'Run order/identity differs from plan')
        if complete:
            self.check(observed == planned, where, 'Completed report missing cases/rounds')
        config = self.load(directory / 'config.json')
        all_http, calls, per_case = ([], [], [])
        quality_by_id = {q['id']: q for q in quality_data}
        for i, run in enumerate(d['runs']):
            loc = where + f'.runs[{i}]'
            if mode == 'closed-loop':
                http_rows = [t['http'] for t in run.get('turns', [])]
                for t in run.get('turns', []):
                    self.check(t['request_messages'] == t['http']['request']['messages'], loc, 'Turn history differs from sent payload')
                operations = run.get('tool_calls', [])
                calls.extend(operations)
                self.check(run['initial_source'] == initial[run['task_id']], loc, 'Initial fixture differs from snapshot')
                baseline = run['baseline']
                self.check(baseline.get('executed') and (not baseline.get('passed')), loc, 'Initial failing-test control absent')
                if 'grading' in run:
                    grading = run['grading']
                    self.check(run['task_tests_passed'] == grading['passed'], loc, 'Task test result differs from final grader')
                    if grading['passed']:
                        self.check(grading.get('executed') and grading['tests_run'] > 0 and (grading['failures'] == grading['errors'] == 0), loc, 'Invalid grader PASS')
                    success = bool(run['completed'] and run['protocol_ok'] and grading['passed'])
                    self.check(run['agent_success'] == success, loc, 'Agent-success definition mismatch')
                    metric = run['tool_metrics']
                    expect = dict(total=len(operations), valid=sum((c['valid'] for c in operations)), action_ok=sum((c['action_ok'] for c in operations)))
                    for k, v in expect.items():
                        self.check(metric[k] == v, loc, 'Tool count mismatch: ' + k)
                    rate = expect['action_ok'] / len(operations) if operations else None
                    self.check(metric['action_success_rate'] == rate, loc, 'Tool success-rate mismatch')
                    requested_test = any((c['call']['function']['name'] == 'run_tests' for c in operations))
                    self.check(run['model_requested_run_tests'] == requested_test, loc, 'Model-test-call flag mismatch')
                per_case.append(dict(round=run['round'], task_id=run['task_id'], task_tests_passed=run.get('task_tests_passed'), agent_success=run.get('agent_success'), model_requested_run_tests=run.get('model_requested_run_tests'), termination=run.get('termination'), tool_calls=len(operations), tool_action_errors=sum((not c['action_ok'] for c in operations)), wall_s=run.get('wall_s'), model_calls=len(http_rows)))
            else:
                http_rows = [run['http']]
                self.check(run['request_messages'] == run['http']['request']['messages'], loc, 'Stored input history differs from payload')
                self.check(run['request_messages_sha256'] == sha(canonical(run['request_messages'])), loc, 'Input-history hash mismatch')
                if mode == 'replay':
                    self.check(run['fixed_replay'] and (not run['model_output_fed_to_next_turn']), loc, 'Replay changed to closed-loop semantics')
                    identity = sha(canonical(run['http']['request']))
                    h = run['http']
                    self.replay_groups[identity].append(dict(experiment=directory.name, config=config.get('name'), source_file=where, round=run['round'], case_id=run['case_id'], protocol_ok=h['protocol_ok'], cache_label=d['configuration']['cache_label'], reset_requested=d['configuration']['reset_prefix_cache'], e2e_s=h['e2e_s'], cached_tokens=h.get('cached_tokens'), usage=h.get('usage'), finish_reason=h.get('finish_reason'), text_sha256=sha((h.get('message') or {}).get('content') or '')))
                else:
                    case = quality_by_id[run['case_id']]
                    h = run['http']
                    matched = False
                    if h['protocol_ok']:
                        text = h['message'].get('content') or ''
                        try:
                            matched = canonical(json.loads(text)) == canonical(case['expected']) if case['kind'] == 'json' else text.strip().casefold() == case['expected']
                        except ValueError:
                            pass
                    passed = matched and h.get('finish_reason') == 'stop'
                    self.check(run['quality']['passed'] == passed, loc, 'Quality grade differs from independent rule')
                    per_case.append(dict(round=run['round'], case_id=run['case_id'], quality_passed=passed, content_matched=matched, finish_reason=h.get('finish_reason')))
            for h in http_rows:
                self.http(h, loc)
            all_http.extend(http_rows)
        good = [h for h in all_http if h['protocol_ok']]
        recomputed = dict(requests=len(all_http), protocol_successes=len(good), protocol_failures=len(all_http) - len(good), prompt_tokens=sum((h['usage']['prompt_tokens'] for h in good)), completion_tokens=sum((h['usage']['completion_tokens'] for h in good)), request_e2e_sum_s=sum((h['e2e_s'] for h in all_http)), ttft_s=None, tpot_s=None, quality_passed=sum((r.get('quality', {}).get('passed', False) for r in d['runs'])), agent_successes=sum((r.get('agent_success', False) for r in d['runs'])), task_tests_passed=sum((r.get('task_tests_passed', False) for r in d['runs'])))
        if complete:
            for key, value in recomputed.items():
                self.check(d['summary'][key] == value, where, 'Client summary mismatch: ' + key)
            self.check((d['status'] == 'COMPLETED') == (recomputed['protocol_failures'] == 0), where, 'Protocol summary/status mismatch')
        for reset in d.get('reset_events', []):
            if reset.get('reset_success'):
                self.check(reset['http_status'] == 200 and reset.get('response', {}).get('success') is True, where, 'False successful reset claim')
        self.reports.append(dict(experiment=directory.name, file=where, mode=mode, status=d['status'], config=config.get('name'), client_sha256=d['client_sha256'], cache_label=d['configuration']['cache_label'], reset_events=d['reset_events'], runs_expected=len(planned), runs_observed=len(observed), summary=recomputed, request_e2e_s=distribution([h['e2e_s'] for h in all_http]), total_wall_s=d.get('wall_s'), tool_calls=len(calls), tool_schema_errors=sum((not c['valid'] for c in calls)), tool_action_errors=sum((not c['action_ok'] for c in calls)), tool_action_success_rate=sum((c['action_ok'] for c in calls)) / len(calls) if calls else None, agent_success_rate=recomputed['agent_successes'] / len(planned) if complete and mode == 'closed-loop' else None, task_tests_pass_rate=recomputed['task_tests_passed'] / len(planned) if complete and mode == 'closed-loop' else None, quality_pass_rate=recomputed['quality_passed'] / len(planned) if complete and mode == 'quality' else None, task_wall_s=distribution([c['wall_s'] for c in per_case if c.get('wall_s') is not None]), cases=per_case))

    def run(self):
        for path in sorted(path_at(self.root, 'results').glob('*/agent-*.json')):
            try:
                self.file(path)
            except (OSError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, zipfile.BadZipFile) as exc:
                self.errors.append(dict(location=str(path.relative_to(self.root)), detail=f'Read/schema error: {type(exc).__name__}: {exc}'))
        pairs = []
        all_replay_inputs = collections.defaultdict(dict)
        for identity, observations in self.replay_groups.items():
            by_experiment = collections.defaultdict(dict)
            for row in observations:
                key = (row['round'], row['case_id'])
                by_experiment[row['experiment']][key] = row
                all_replay_inputs[row['experiment']][key] = identity
            for a, b in itertools.combinations(sorted(by_experiment), 2):
                common = sorted(by_experiment[a].keys() & by_experiment[b].keys())
                pairs.append(dict(request_payload_sha256=identity, a=a, b=b, matched_pairs=[dict(round=k[0], case_id=k[1], a=by_experiment[a][k], b=by_experiment[b][k]) for k in common]))
        coverage = []
        for a, b in itertools.combinations(sorted(all_replay_inputs), 2):
            left, right = (all_replay_inputs[a], all_replay_inputs[b])
            common = sorted(left.keys() & right.keys())
            differing = [k for k in common if left[k] != right[k]]
            coverage.append(dict(a=a, b=b, common_case_round_slots=len(common), equal_payload_slots=len(common) - len(differing), noncomparable_slots=[dict(round=k[0], case_id=k[1], a_sha256=left[k], b_sha256=right[k]) for k in differing], a_only_slots=[list(k) for k in sorted(left.keys() - right.keys())], b_only_slots=[list(k) for k in sorted(right.keys() - left.keys())]))
        return dict(status='FAIL' if self.errors else 'INCOMPLETE' if self.incomplete or not self.reports else 'PASS', created_utc=datetime.now(timezone.utc).isoformat(), reports=self.reports, errors=self.errors, incomplete=self.incomplete, source_sha256=self.sources, replay_input_groups=dict(self.replay_groups), cross_configuration_pairs=pairs, cross_configuration_pair_coverage=coverage, limits=['Synthetic fixtures and fixed replay only; not SWE-bench or general Agent quality.', 'No TTFT or TPOT exists for these non-streaming requests; output/E2E is not decode rate.', 'Fixed replay input does not fix natural-EOS output lengths; paired E2E is not pure prefill cost.', 'Agent wall includes HTTP, tools, grader and checkpoints; tool action success is not semantic usefulness.', 'All failed requests/tasks retained; audit PASS means recorded arithmetic agrees, not 100% quality.'])


def self_test():
    response = dict(choices=[dict(message=dict(role="assistant", content="12"), finish_reason="stop")],
                    usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12))
    h = dict(attempts=1, ttft_s=None, tpot_s=None, timing_scope="non_streaming_full_response",
             request=dict(stream=False, temperature=0, reasoning_effort="none", chat_template_kwargs=dict(enable_thinking=False)),
             e2e_s=.5, protocol_ok=True, http_status=200, error=None, raw_response=canonical(response),
             response=response, message=response["choices"][0]["message"], usage=response["usage"],
             finish_reason="stop", cached_tokens=None, output_tokens_per_e2e_s=4.)
    review = Review(ROOT); review.http(h, "offline fixture")
    assert not review.errors
    detected = []
    for label in ("retry", "invented_ttft", "raw_response", "token_arithmetic", "cache_field"):
        bad = copy.deepcopy(h)
        if label == "retry": bad["attempts"] = 2
        elif label == "invented_ttft": bad["ttft_s"] = .1
        elif label == "raw_response": bad["raw_response"] = canonical(dict(choices=[], usage={}))
        elif label == "token_arithmetic": bad["output_tokens_per_e2e_s"] = 5.
        else: bad["cached_tokens"] = 10
        review = Review(ROOT); review.http(bad, "offline corruption")
        assert review.errors, label
        detected.append(label)
    assert distribution([1., 2., 3.])["p95"] == 2.9
    assert sha(canonical({"b": 2, "a": 1})) == sha(canonical({"a": 1, "b": 2}))
    return dict(status="PASS", scope="CPU synthetic Agent summary audit only; no inference",
                corruption_cases_detected=detected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    report = self_test() if args.self_test else Review(args.root).run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open_output(args.output, 'x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(json.dumps(dict(status=report['status'], reports=len(report.get('reports', [])), errors=report.get('errors', []), incomplete=report.get('incomplete', []), self_test=report.get('corruption_cases_detected'))))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get('errors') else 2 if report['status'] == 'INCOMPLETE' and (not args.allow_incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
