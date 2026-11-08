#!/usr/bin/env python3
"""Independent reconstruction of one actual failed request; no generated code execution."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ledger, checks = ({}, [])

    def read(path):
        raw = path.read_bytes()
        ledger[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return raw

    def check(name, value):
        if not value:
            raise ValueError(name)
        checks.append(name)

    def digest(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    diag = json.loads(read(path_at(ROOT, 'logs/c15-request-failure-r1.json')))
    check('diagnostic identity', diag['status'] == 'PASS' and diag['experiment'] == 'session-dspark3-agent-confirm' and (diag['session_index'] == 138) and (diag['raw_turn_index'] == 2))
    files = {}
    for name, info in diag['input_ledger'].items():
        data = read(Path(name))
        check('actual input bytes ' + name, info == ledger[name])
        if name.endswith('sessions.jsonl'):
            files['sessions'] = data
        if name.endswith('events.jsonl'):
            files['events'] = data
    row = json.loads(files['sessions'].splitlines()[diag['source_locator']['session_jsonl_line'] - 1])
    req = row['requests'][2]
    check('exact raw request', row['index'] == 138 and req['turn'] == 2 and (req['request_id'] == diag['request_id']) and (digest(req) == diag['source_locator']['request_canonical_sha256']))
    event = json.loads(files['events'].splitlines()[diag['source_locator']['model_request_event_jsonl_line'] - 1])
    check('event copy agrees', event['event'] == 'model_request' and event['session_index'] == 138 and (event['request'] == req))
    check('raw stream identity', len(req['raw_events']) == diag['source_locator']['raw_events'] == 1085 and digest(req['raw_events']) == diag['source_locator']['raw_events_canonical_sha256'])
    content, reasoning, terminals = ([], [], [])
    arguments, names, ids = (defaultdict(list), defaultdict(list), {})
    for index, raw_event in enumerate(req['raw_events']):
        if raw_event['data'] == '[DONE]':
            terminals.append(('DONE', index))
            continue
        data = json.loads(raw_event['data'])
        if data.get('usage') is not None:
            check('usage agrees', data['usage'] == diag['exact_usage'] == req['usage'])
            terminals.append(('usage', index))
        for choice in data.get('choices', []):
            delta = choice.get('delta') or {}
            content.append(delta.get('content') or '')
            reasoning.extend((delta.get(key) or '' for key in ('reasoning', 'reasoning_content')))
            if choice.get('finish_reason'):
                terminals.append((choice['finish_reason'], index))
            for tool in delta.get('tool_calls') or []:
                number, function = (tool['index'], tool.get('function') or {})
                arguments[number].append(function.get('arguments') or '')
                names[number].append(function.get('name') or '')
                if tool.get('id'):
                    ids[number] = tool['id']
    check('terminal order', terminals == [('length', 1082), ('usage', 1083), ('DONE', 1084)])
    check('exact received text', ''.join(content) == diag['exact_received_full_text'] == req['full_text'] and ''.join(reasoning) == diag['exact_received_reasoning'] == req['reasoning'])
    check('character counts', len(''.join(content)) == diag['output_character_counts']['full_text'] == 13715 and len(''.join(reasoning)) == 0)
    check('fixed output limit consumed', diag['exact_usage']['completion_tokens'] == req['payload']['max_tokens'] == diag['payload_settings']['max_tokens'] == 4096)
    check('complete transport without local truncation', req['http_status'] == 200 and req['done_received'] is True and (req['timed_out'] is False) and (req['cancelled'] is False) and (req['raw_stream_truncated'] is False) and (req['attempts'] == 1))
    for index, tool in enumerate(diag['exact_received_tool_calls']):
        text, info = (''.join(arguments[index]), diag['tool_argument_diagnostics'][index])
        check('exact tool fragments ' + str(index), text == tool['function']['arguments'] == req['tool_calls'][index]['function']['arguments'] and ''.join(names[index]) == tool['function']['name'] == 'replace_text' and (ids[index] == tool['id']))
        check('tool argument hash ' + str(index), len(text) == info['argument_chars'] and hashlib.sha256(text.encode()).hexdigest() == info['argument_sha256'])
        try:
            json.loads(text)
            error = None
        except json.JSONDecodeError as exc:
            error = dict(type=type(exc).__name__, message=exc.msg, position=exc.pos)
        check('tool JSON validity ' + str(index), (error is None) == info['json_valid'] and error == info['json_error'])
    check('failed tool IDs not executed', not set(ids.values()) & {tool['call']['id'] for tool in row['tools']} and diag['received_failed_response_tool_calls_executed'] == 0)
    check('no tool worker after failed response starts', not any((op['name'] == 'execute_tool' and op['ready_offset_s'] >= req['dispatch_offset_s'] for op in row['host_operations'])))
    check('ungraded protocol failure retained', row['terminal_reason'] == 'request_error' and row['grade'] is None and (row['task_success'] is None) and (diag['session_task_success'] is None) and (req['ok'] is False) and (req['error']['message'] == 'Tool deltas without terminal tool_calls'))
    check('root-cause uncertainty retained', any(('pre-parser' in text and 'not an independently proven' in text for text in diag['limitations'])))
    check('evidence unchanged', all((hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'] and Path(path).stat().st_size == info['bytes'] for path, info in ledger.items())))
    result = dict(status='PASS_INDEPENDENT_FAILURE_STREAM_REVIEW', utc=datetime.now(timezone.utc).isoformat(), checks=checks, evidence_ledger=ledger, reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='One actual c15 failed request and event copy; raw text/tool fragments reconstructed independently. No generated code, tool, model or full raw-tree rerun.', limitations=['Failure explanation does not repair the frozen full-run audit or certify capacity.', '4096 is total completion usage; character lengths do not establish text/tool token split.', 'Received parsed deltas show incomplete arguments at length termination, not the unavailable pre-parser model stream or proven parser root cause.', 'Initial reviewer required False for ungraded task; that failed assertion is retained, corrected to actual null. No original evidence changed.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks))))


if __name__ == '__main__':
    main()
