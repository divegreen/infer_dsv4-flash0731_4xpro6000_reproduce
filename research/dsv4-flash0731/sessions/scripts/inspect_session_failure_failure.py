#!/usr/bin/env python3
"""One-shot read-only c15 output-budget/tool-stream diagnostic.

Never executes generated text or tools. Original JSONL/event/source bytes bind to
an existing terminal analysis. Exact received text/tools and terminal SSE data
are preserved; full original stream remains in the immutable source JSONL.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='session-dspark3-agent-confirm'
PHASE='agent-open-r02-s01'
INDEX=138
TURN=2


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def need(value,message):
    if not value:raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result_root = path_at(ROOT.parent, 'scenarios/results') / NAME
    summary_path = path_at(ROOT, 'summary') / f'{NAME}-terminal-r1.json'
    summary = load_document(summary_path)
    need(summary['status'] == 'PASS', 'Terminal consistency audit required')
    ledger = {str(summary_path): dict(sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), bytes=summary_path.stat().st_size)}

    def read(relative):
        path = result_root / relative
        raw = path.read_bytes()
        info = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        matches = [v for k, v in summary['input_ledger'].items() if k.endswith('/' + NAME + '/' + relative)]
        need(len(matches) == 1 and info == matches[0], 'Source differs from terminal audited bytes: ' + relative)
        ledger[str(path)] = info
        return raw
    raw_sessions = read('sessions-' + PHASE + '/sessions.jsonl')
    selected = []
    for lineno, line in enumerate(raw_sessions.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row['index'] == INDEX:
            selected.append((lineno, line, row))
    need(len(selected) == 1, 'Failure Session must be unique')
    line_number, line, row = selected[0]
    failed = [r for r in row['requests'] if r['turn'] == TURN]
    need(len(failed) == 1, 'Failure request must be unique')
    request = failed[0]
    need(request['ok'] is False and row['terminal_reason'] == 'request_error', 'Expected recorded failure')
    events = read('sessions-' + PHASE + '/events.jsonl')
    matching = []
    for lineno, linebytes in enumerate(events.splitlines(), 1):
        event = json.loads(linebytes)
        if event.get('event') == 'model_request' and event.get('session_index') == INDEX and (event['request']['turn'] == TURN):
            need(event['request'] == request, 'Independent event copy differs from Session request')
            matching.append(lineno)
    need(len(matching) == 1, 'Exact model_request event must be unique')
    command = json.loads(read('command.json'))
    snapshot = read('source-snapshot.zip')
    source_member = 'scenarios/scripts/agent_session_client.py'
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        runtime = archive.read(source_member)
    need(hashlib.sha256(runtime).hexdigest() == command['source_sha256'][source_member], 'Executed client source mismatch')
    source_lines = runtime.decode().splitlines()
    guard_lines = [i for i, line in enumerate(source_lines, 1) if 'Tool deltas without terminal tool_calls' in line]
    need(len(guard_lines) == 1, 'Expected executed guard not unique')
    raw_events = request['raw_events']
    content = []
    reasoning = []
    tools = {}
    tool_first = None
    final_events = []
    finishes = []
    usage = None
    done = []
    for i, event in enumerate(raw_events):
        data = event['data']
        if data == '[DONE]':
            done.append(i)
            final_events.append(dict(raw_event_index=i, **event))
            continue
        obj = json.loads(data)
        if obj.get('usage') is not None:
            usage = obj['usage']
            final_events.append(dict(raw_event_index=i, **event))
        for choice in obj.get('choices', []):
            delta = choice.get('delta') or {}
            if delta.get('content') is not None:
                content.append(delta['content'])
            if delta.get('reasoning') is not None:
                reasoning.append(delta['reasoning'])
            if delta.get('reasoning_content') is not None:
                reasoning.append(delta['reasoning_content'])
            for t in delta.get('tool_calls') or []:
                if tool_first is None:
                    tool_first = dict(raw_event_index=i, **event)
                current = tools.setdefault(t['index'], dict(id='', type='function', function=dict(name='', arguments='')))
                if t.get('id'):
                    current['id'] = t['id']
                if t.get('type'):
                    current['type'] = t['type']
                function = t.get('function') or {}
                current['function']['name'] += function.get('name') or ''
                current['function']['arguments'] += function.get('arguments') or ''
            if choice.get('finish_reason') is not None:
                finishes.append(choice['finish_reason'])
                final_events.append(dict(raw_event_index=i, **event))
    reconstructed = [tools[k] for k in sorted(tools)]
    need(''.join(content) == request['full_text'] and ''.join(reasoning) == request['reasoning'], 'Raw stream text reconstruction mismatch')
    need(reconstructed == request['tool_calls'], 'Raw tool deltas reconstruction mismatch')
    need(finishes == ['length'] and usage == request['usage'] and (len(done) == 1), 'Terminal length/usage/DONE evidence mismatch')
    max_tokens = request['payload']['max_tokens']
    need(usage['completion_tokens'] == max_tokens == 4096, 'Recorded response did not exhaust the fixed output budget')
    tool_diagnostics = []
    for i, tool in enumerate(reconstructed):
        arguments = tool['function']['arguments']
        parsed = None
        error = None
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            error = dict(type=type(exc).__name__, message=exc.msg, position=exc.pos)
        tool_diagnostics.append(dict(index=i, id=tool['id'], name=tool['function']['name'], argument_chars=len(arguments), argument_sha256=hashlib.sha256(arguments.encode()).hexdigest(), json_valid=error is None, parsed_keys=sorted(parsed) if isinstance(parsed, dict) else None, json_error=error))
    need([x['json_valid'] for x in tool_diagnostics] == [True, False], 'Expected one complete and one incomplete tool argument object')
    executed = [t['call'] for t in row['tools']]
    received_ids = {tool['id'] for tool in reconstructed}
    need(not received_ids.intersection((t['id'] for t in executed)), 'A tool from failed response was executed')
    need(not any((op['name'] == 'execute_tool' and op['ready_offset_s'] >= request['dispatch_offset_s'] for op in row['host_operations'])), 'Tool worker ran during or after the failed response')
    failed_stage = next((s for s in summary['stages'] if s['phase']['id'] == PHASE))
    need(all((not s['included_as_measurement'] for s in summary['stages'])), 'Failed run promoted to formal measurements')
    output = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), experiment=NAME, phase=PHASE, session_index=INDEX, raw_turn_index=TURN, human_round_number=TURN + 1, task_id=row['task_id'], request_id=request['request_id'], source_locator=dict(session_jsonl_line=line_number, model_request_event_jsonl_line=matching[0], request_canonical_sha256=hashlib.sha256(canonical(request)).hexdigest(), raw_events=len(raw_events), raw_events_canonical_sha256=hashlib.sha256(canonical(raw_events)).hexdigest(), executed_client_sha256=hashlib.sha256(runtime).hexdigest(), client_guard_line=guard_lines[0]), request_metadata={key: request.get(key) for key in ('ok', 'http_status', 'finish_reason', 'done_received', 'error', 'timed_out', 'cancelled', 'attempts', 'raw_stream_truncated', 'first_output_s', 'ttft_tool_s', 'elapsed_until_return_s', 'payload_sha256', 'wire_body_sha256')}, payload_settings={key: request['payload'][key] for key in ('model', 'max_tokens', 'temperature', 'seed', 'stream', 'tool_choice', 'reasoning_effort', 'chat_template_kwargs')}, exact_usage=usage, first_tool_delta_event=tool_first, terminal_raw_events=final_events, exact_received_full_text=request['full_text'], exact_received_reasoning=request['reasoning'], exact_received_tool_calls=reconstructed, output_character_counts=dict(full_text=len(request['full_text']), reasoning=len(request['reasoning'])), tool_argument_diagnostics=tool_diagnostics, previous_executed_tool_names=[t['function']['name'] for t in executed], received_failed_response_tool_calls_executed=0, session_grade=row['grade'], session_task_success=row['task_success'], run_exclusion_preserved=True, frozen_classification=failed_stage['stability'], input_ledger=ledger, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), findings=['The stream ended normally at the fixed 4096 completion-token limit, with length then usage then DONE; not a timeout or local stream-byte truncation.', 'Exact tool delta reconstruction contains a complete first replace_text argument object and an incomplete second object. No tool from this failed response ran.', 'The executed client requires tool_calls finish reason before accepting tool deltas; this protocol error was preserved instead of repairing or executing partial arguments.', 'All c15 phases remain excluded from formal performance and whole-run healthy confirmation; earlier completed phases are descriptive only.'], limitations=['Provider usage gives total output tokens, not exact token count split between text and tool serialization; no split is inferred from character lengths.', 'reasoning_effort none and enable_thinking false are recorded, while visible text may contain analysis-like prose; this is not a thinking-mode benchmark.', 'The pre-parser model token stream is not recorded; this evidence identifies output-budget exhaustion and received parsed tool-stream incompleteness, not an independently proven server-parser or model root cause.', 'No retry, budget change, response repair, task execution, GPU call or original-file mutation was performed.'])
    for name, info in ledger.items():
        raw = Path(name).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == info['sha256'] and len(raw) == info['bytes'], 'Source changed during diagnostic')
    output['inputs_unchanged'] = True
    with open_output(args.output, 'x') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(dict(status='PASS', source_events=len(raw_events), output_chars=output['output_character_counts'], tool_arguments=tool_diagnostics, output=str(args.output))))


if __name__=='__main__':main()
