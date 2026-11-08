#!/usr/bin/env python3
"""CPU reproduction of installed V4 nested tool-argument schema coercion.

Execute actual selected source functions via AST; do not import vllm/torch,
contact an endpoint, initialize CUDA, change the environment or task interface.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import ast
import base64
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import regex as re
from check_cache_semantics import selected_functions

HERE=Path(__file__).resolve().parents[1]
RUN = path_at(HERE.parent, 'scenarios/results/session-tp4-agent-pilot2')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    dist = importlib.metadata.distribution('vllm')
    root = Path(dist.locate_file('vllm'))
    files = ['tool_parsers/__init__.py', 'tool_parsers/deepseekv4_engine_tool_parser.py', 'tool_parsers/utils.py', 'parser/engine/registered_adapters.py', 'parser/engine/adapters.py', 'parser/deepseek_v4.py', 'parser/engine/parser_engine.py']
    identities = {}
    checks = []

    def check(name, value):
        assert value, name
        checks.append(name)
    for rel in files:
        path = root / rel
        record = next((f for f in dist.files if str(f) == 'vllm/' + rel))
        encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
        check('wheel RECORD ' + rel, encoded == record.hash.value)
        identities[str(path)] = dict(sha256=sha(path), wheel_record_match=True)
    env = dict(json=json, math=math, re=re, contextlib=contextlib)

    class ChatTool:

        def __init__(self, d):
            self.function = SimpleNamespace(**d['function'])
    for name in ('FunctionTool', 'NamespaceTool', 'CustomTool'):
        env[name] = type(name, (), {})
    env['ChatCompletionToolsParam'] = ChatTool
    tree = ast.parse((root / 'tool_parsers/utils.py').read_text())
    alias = next((n for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', None) == '_TYPE_ALIASES'))
    env['_TYPE_ALIASES'] = ast.literal_eval(alias.value)
    extracted = {}
    extracted['utils'] = selected_functions(root / 'tool_parsers/utils.py', ['_is_json_finite', 'extract_types_from_schema', 'coerce_to_schema_type', '_is_function_tool', '_extract_tool_info', 'find_tool_properties'], env)
    extracted['engine'] = selected_functions(root / 'parser/engine/parser_engine.py', ['_coerce_value', '_coerce_dict', '_fix_arg_types', '_flush_arg_converter'], env)
    engine_type = type('ParserEngine', (), {})
    for name in ('_coerce_value', '_coerce_dict'):
        setattr(engine_type, name, staticmethod(env[name]))
    for name in ('_fix_arg_types', '_flush_arg_converter'):
        setattr(engine_type, name, env[name])
    env['ParserEngine'] = engine_type
    extracted['dsml_patterns'] = selected_functions(root / 'parser/deepseek_v4.py', ['_param_patterns'], env)
    start = '<｜DSML｜parameter'
    end = '</｜DSML｜parameter>'
    env['_PARAM_RE'], env['_PARTIAL_PARAM_RE'] = env['_param_patterns'](start, end)
    extracted['dsml_converter'] = selected_functions(root / 'parser/deepseek_v4.py', ['_dsml_arg_converter'], env)
    report = load_document(RUN / 'sessions.json')
    session = next((s for p in report['phases'] for s in p['sessions'] if s.get('task_id') == 'service_latency'))
    actual_request = session['requests'][2]
    original = next((t for t in actual_request['payload']['tools'] if t['function']['name'] == 'query_rows'))
    old_prop = original['function']['parameters']['properties']['filters']['items']['properties']['value']
    check('c02 actual advertised nested value schema is empty', old_prop == {})
    check('c02 all ten filtered attempts contain string 1024', all((isinstance(json.loads(r['tool_calls'][0]['function']['arguments'])['filters'][0]['value'], str) for r in session['requests'][2:])))
    check('unknown schema type defaults string', env['extract_types_from_schema']({}) == ['string'])
    union = dict(type=['integer', 'string', 'null', 'array'], items=dict(type=['integer', 'string', 'null']))
    task_source = HERE.parent / 'scenarios/scripts/agent_session_tasks.py'
    catalog = path_at(HERE.parent, 'scenarios/workloads/sessions/tasks/catalog.json')
    task_env = dict(MAX_SOURCE=32768, TEXT=dict(type='string', maxLength=32768))
    selected_functions(task_source, ['schema', 'tools_for'], task_env)
    actual_schemas = {}
    for task in load_document(catalog)['tasks']:
        if task['kind'] != 'data':
            continue
        task_env['context'] = lambda session, t=task: (t, None)
        for tool in task_env['tools_for'](None):
            if tool['function']['name'] not in ('query_rows', 'aggregate_table'):
                continue
            value_schema = tool['function']['parameters']['properties']['filters']['items']['properties']['value']
            actual_schemas[task['id'] + '/' + tool['function']['name']] = value_schema
            check('current tools_for explicit value union ' + task['id'] + '/' + tool['function']['name'], value_schema.get('type') == union['type'] and value_schema.get('items') == union['items'])
    variants = {'empty': {}, 'integer': {'type': 'integer'}, 'explicit_union': union, 'anyof_union': {'anyOf': [{'type': 'integer'}, {'type': 'string'}, {'type': 'null'}, {'type': 'array', 'items': {'type': ['integer', 'string', 'null']}}]}, 'current_tools_for': actual_schemas['service_latency/query_rows']}
    samples = [1024, '1024', 'cancelled', None, True, '001', 'null', [1024, 'cancelled', None], ['001', 'null']]
    rows = []

    def pipeline(schema, value, partial=False):
        tool = copy.deepcopy(original)
        tool['function']['parameters']['properties']['filters']['items']['properties']['value'] = schema
        payload = dict(table='requests', filters=[dict(column='input_tokens', op='le', value=value)], limit=50)
        dsml = ''.join((f'{start} name="{key}" string="{str(isinstance(value, str)).lower()}">' + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + end for key, value in payload.items()))
        converted = env['_dsml_arg_converter'](dsml, partial)
        engine = engine_type()
        engine._tools = [ChatTool(tool)]
        fixed = engine._fix_arg_types(converted, 'query_rows')
        return (payload, dsml, converted, fixed, engine)
    for label, schema in variants.items():
        for value in samples:
            payload, raw, converted, fixed, engine = pipeline(schema, value)
            before = json.loads(converted)['filters'][0]['value']
            after = json.loads(fixed)['filters'][0]['value']
            check(f'{label} DSML converter preserves input {value!r}', type(before) is type(value) and before == value)
            rows.append(dict(schema_name=label, schema=schema, input=value, input_python_type=type(value).__name__, after_dsml=before, after_schema_correction=after, after_python_type=type(after).__name__))

    def row(label, value):
        return next((r for r in rows if r['schema_name'] == label and type(r['input']) is type(value) and (r['input'] == value)))
    check('empty schema converts native integer into string', row('empty', 1024)['after_schema_correction'] == '1024')
    check('explicit integer preserves native numeric literal', type(row('integer', 1024)['after_schema_correction']) is int)
    check('explicit union preserves native numeric literal', type(row('explicit_union', 1024)['after_schema_correction']) is int)
    check('explicit union accepts ordinary text unchanged', row('explicit_union', 'cancelled')['after_schema_correction'] == 'cancelled')
    check('explicit union preserves null', row('explicit_union', None)['after_schema_correction'] is None)
    check('explicit union preserves mixed typed array', row('explicit_union', [1024, 'cancelled', None])['after_schema_correction'] == [1024, 'cancelled', None])
    for value in samples:
        check('actual tools_for equals tested union conversion ' + repr(value), row('current_tools_for', value)['after_schema_correction'] == row('explicit_union', value)['after_schema_correction'] and row('current_tools_for', value)['after_python_type'] == row('explicit_union', value)['after_python_type'])
    check('numeric-looking string caveat reproduced', row('explicit_union', '001')['after_schema_correction'] == 1)
    check('literal null string caveat reproduced', row('explicit_union', 'null')['after_schema_correction'] is None)
    check('array item caveat reproduced', row('explicit_union', ['001', 'null'])['after_schema_correction'] == [1, None])
    check('anyOf extracts same top-level type set', set(env['extract_types_from_schema'](variants['anyof_union'])) == set(union['type']))
    check('anyOf array-only branch items not recursively applied', row('anyof_union', ['001', 'null'])['after_schema_correction'] == ['001', 'null'])
    for label in ('empty', 'explicit_union'):
        payload, raw, converted, fixed, engine = pipeline(variants[label], 1024)
        check(label + ' partial converter on closed DSML has identical result', env['_dsml_arg_converter'](raw, True) == converted)
        engine._arg_converter = env['_dsml_arg_converter']
        engine._tool_slots = [SimpleNamespace(args=raw, name='query_rows', streamed_json='')]
        check(label + ' actual streaming flush applies identical schema correction', engine._flush_arg_converter(0) == fixed)
    typed_branches = {'oneOf': [dict(type='object', properties=dict(column=dict(const='input_tokens'), op=dict(type='string'), value=dict(type=['integer', 'null'])), required=['column', 'op', 'value']), dict(type='object', properties=dict(column=dict(const='request_id'), op=dict(type='string'), value=dict(type=['string', 'null'])), required=['column', 'op', 'value'])]}
    preserved = []
    for value in (1024, '001', 'null', None):
        item = dict(column='input_tokens' if type(value) is int else 'request_id', op='eq', value=value)
        result, changed = engine_type._coerce_value(copy.deepcopy(item), typed_branches)
        check('per-column item branch preserves raw type ' + repr(value), type(result['value']) is type(value) and result == item and (not changed))
        preserved.append(dict(input=item, output=result, changed=changed))
    check('no vllm or torch package imported', 'vllm' not in sys.modules and 'torch' not in sys.modules)
    result = dict(status='PASS_CPU_SCHEMA_COERCION_REPRODUCED_ONLINE_RAW_DSML_UNKNOWN', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, source_hashes=identities, extracted_functions=extracted, script_sha256=sha(Path(__file__)), current_interface=dict(source=str(task_source), source_sha256=sha(task_source), catalog_sha256=sha(catalog), actual_tools_for_schemas=actual_schemas), c02_evidence=dict(path=str(RUN / 'sessions.json'), sha256=sha(RUN / 'sessions.json'), session_terminal_reason=session['terminal_reason'], tool_schema=original, raw_model_dsml_recorded=False, parsed_filter_attempts=len(session['requests']) - 2), cases=rows, per_column_branch_demo=dict(schema=typed_branches, cases=preserved), minimal_numeric_demo=dict(dsml=pipeline({}, 1024)[1], before_schema_correction=pipeline({}, 1024)[2], after_empty_schema=pipeline({}, 1024)[3], after_explicit_union=pipeline(union, 1024)[3]), limitations=['AST-selected real functions with lightweight tool attribute objects; no full engine import or HTTP/GPU.', 'c02 records parsed SSE, not raw model DSML; cannot prove its original numeric/string spellings.', 'Type union fixes this primitive coercion but prefers numeric/null interpretations of ambiguous strings.', 'anyOf/oneOf type extraction is not complete JSON Schema validation or conditional branch selection.', 'Interface schema changes alter model prompts/cache identity and require a new version/experiment.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), cases=len(rows), demo=result['minimal_numeric_demo'], output=str(args.output)), ensure_ascii=False, indent=2))


if __name__=='__main__':main()
