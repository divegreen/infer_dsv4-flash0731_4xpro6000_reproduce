#!/usr/bin/env python3
"""CPU-only independent check of exact replay prefixes, salts and chat normalization.

Never opens HTTP or imports vllm/torch. Executes installed serialization classes
via AST with Pydantic BaseModel standing in for OpenAIBaseModel; no GPU KV proof.
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
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, model_serializer, model_validator
from tokenizers import Tokenizer

from check_cache_semantics import load_module, lcp, selected_functions

HERE = Path(__file__).resolve().parents[1]
SCENARIOS = HERE.parent / 'scenarios'


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runtime = SCENARIOS / 'scripts'
    source_paths = [runtime / 'agent_session_client.py', runtime / 'capacity_client.py', runtime / 'agent_benchmark.py', HERE.parent / 'scripts/benchmark_client.py']
    hashes_before = {str(p): sha_file(p) for p in source_paths}
    sys.path.insert(0, str(runtime))
    import agent_session_client as client
    old = load_module(runtime / 'agent_benchmark.py', 'stagec_legacy_agent')
    checks = []

    def check(name, value):
        assert value, name
        checks.append(name)
    phase = dict(id='cpu-prefix-review', kind='replay', arrival='closed', role='measure', sessions=6, concurrency=2, profiles=['coding', 'research', 'data'], seed=202611602, cache_scope='session')
    plan = client.validate_plan(dict(schema_version=1, experiment_id='cpu-prefix-review', phases=[phase]))
    prepared = client.prepare(plan)
    sessions = prepared['phases'][0]['sessions']
    rows = []
    for session in sessions:
        expected = client.PROFILES[session['profile']]['inputs']
        prompts = session['prompts']
        check(f'session {session['index']} exact lengths', [len(v) for v in prompts] == expected)
        check(f'session {session['index']} append-only token histories', all((a == b[:len(a)] for a, b in zip(prompts, prompts[1:]))))
        rows.append(dict(index=session['index'], profile=session['profile'], lengths=expected, adjacent_lcp=[lcp(a, b) for a, b in zip(prompts, prompts[1:])], prompt_hashes=session['prompt_hashes']))
    shared = sessions[0]['prompts'][0][:2048]
    check('all sessions have identical first 2048 token IDs', all((s['prompts'][0][:2048] == shared for s in sessions)))
    lcps = [dict(a=a['index'], b=b['index'], lcp=lcp(a['prompts'][0], b['prompts'][0])) for i, a in enumerate(sessions) for b in sessions[i + 1:]]
    check('every session pair diverges after shared preamble and before next 256 boundary', all((2048 <= r['lcp'] < 2304 for r in lcps)))
    check('all initial full prompts are distinct', len({s['prompt_hashes'][0] for s in sessions}) == 6)
    salt_modes = {}
    for scope in ('none', 'session', 'shared'):
        alt = copy.deepcopy(plan)
        alt['phases'][0]['cache_scope'] = scope
        generated = client.prepare(alt)
        check(f'{scope} cache scope does not change replay tokens', generated['phases'][0]['sessions'] == sessions)
        salts = [[client.salt_for(alt['phases'][0], i, t) for t in range(4)] for i in range(6)]
        salt_modes[scope] = salts
        check(f'{scope} salts satisfy actual ASCII validator constraints', all((0 < len(s) <= 128 and (not any((c in s for c in '@/\\\x00'))) for row in salts for s in row)))
    check('none scope creates a distinct namespace for each request', len({s for row in salt_modes['none'] for s in row}) == 24)
    check('session scope stable within turn chain and unique across sessions', all((len(set(row)) == 1 for row in salt_modes['session'])) and len({row[0] for row in salt_modes['session']}) == 6)
    check('shared scope uses one namespace for the phase', len({s for row in salt_modes['shared'] for s in row}) == 1)
    later = dict(plan['phases'][0], id='cpu-prefix-review-repeat')
    check('new phase id excludes earlier phase cache namespace', client.salt_for(later, 0, 0) != salt_modes['session'][0][0])
    dist = importlib.metadata.distribution('vllm')
    root = Path(dist.locate_file('vllm'))
    installed = ['entrypoints/generate/base/protocol.py', 'entrypoints/openai/chat_completion/protocol.py', 'renderers/online_renderer.py', 'renderers/deepseek_v4.py', 'renderers/chat_utils/chat_message.py', 'tokenizers/deepseek_v4.py', 'tokenizers/deepseek_v4_encoding.py']
    sources = {}
    for name in installed:
        path = root / name
        record = next((f for f in dist.files if str(f) == 'vllm/' + name))
        encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
        check(f'wheel RECORD matches {name}', encoded == record.hash.value)
        sources[str(path)] = dict(sha256=sha_file(path), wheel_record_match=True)
    env = dict(Any=Any, Literal=Literal, OpenAIBaseModel=BaseModel, model_serializer=model_serializer, model_validator=model_validator, copy=copy)
    extracted = {}
    for rel, cls_name in [('entrypoints/generate/base/protocol.py', 'FunctionDefinition'), ('entrypoints/openai/chat_completion/protocol.py', 'ChatCompletionToolsParam')]:
        tree = ast.parse((root / rel).read_text())
        cls = copy.deepcopy(next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)))
        module = ast.Module(body=[cls], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(root / rel), 'exec'), env)
        env[cls_name].model_rebuild(_types_namespace=env)
        extracted[cls_name] = dict(source=rel, start_line=cls.lineno, end_line=cls.end_lineno)
    encoder = load_module(root / 'tokenizers/deepseek_v4_encoding.py', 'stagec_review_encoder')
    env['encode_messages'] = encoder.encode_messages
    selected_functions(root / 'tokenizers/deepseek_v4.py', ['apply_chat_template'], env)
    token_path = Path(str(model_path() / 'tokenizer.json'))
    tok = Tokenizer.from_file(str(token_path))
    tok.no_padding()
    tok.no_truncation()
    proxy = SimpleNamespace(encode=lambda text, **kw: tok.encode(text, add_special_tokens=kw.get('add_special_tokens', False)).ids)
    histories = old.replay_histories()

    def render(messages, tools):
        return env['apply_chat_template'](proxy, messages, tools=tools, tokenize=True, enable_thinking=False, reasoning_effort='none')
    direct = [render(m, old.TOOLS) for m in histories]
    wire = json.loads(old.compact(dict(messages=histories, tools=old.TOOLS)))
    normalized = [env['ChatCompletionToolsParam'].model_validate(t).model_dump() for t in wire['tools']]
    normalized_ids = [render(m, normalized) for m in wire['messages']]
    check('normalization changes only nested schema key order, not dict equality', normalized == old.TOOLS)
    check('wire-sorted then typed-schema normalized rendering reproduces legacy lengths', [len(t) for t in normalized_ids] == [8446, 8600, 8816])
    check('normalized legacy histories preserve full preceding token prefixes', [lcp(a, b) for a, b in zip(normalized_ids, normalized_ids[1:])] == [8446, 8600])
    evidence = []
    for exp in ('scenarios-tp4-apc', 'scenarios-tp4-dspark-apc'):
        path = path_at(SCENARIOS, 'results') / exp / 'agent-replay.json'
        obj = load_document(path)
        matches = []

        def walk(x):
            if isinstance(x, dict):
                if 'http' in x and isinstance(x['http'], dict) and x['http'].get('usage'):
                    h = x['http']
                    matches.append(dict(usage=h['usage'], cached_tokens=h.get('cached_tokens'), protocol_ok=h.get('protocol_ok')))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(obj)
        check(f'{exp} old completed six successful requests retained', obj['status'] == 'COMPLETED' and len(matches) == 6 and all((m['protocol_ok'] for m in matches)))
        check(f'{exp} normalized CPU lengths match every actual usage', [m['usage']['prompt_tokens'] for m in matches] == [8446, 8600, 8816] * 2)
        evidence.append(dict(path=str(path), sha256=sha_file(path), status=obj['status'], requests=matches))
    check('source identity unchanged during CPU execution', hashes_before == {str(p): sha_file(p) for p in source_paths})
    check('no vllm/torch package imported', 'vllm' not in sys.modules and 'torch' not in sys.modules)
    result = dict(status='PASS_CPU_ONLY_ONLINE_TOKEN_IDENTITY_NOT_VERIFIED', created_utc=datetime.now(timezone.utc).isoformat(), checks=checks, script_sha256=sha_file(Path(__file__)), client_sources=hashes_before, tokenizer_sha256=sha_file(token_path), replay_sessions=rows, pairwise_lcp=lcps, salts=salt_modes, installed_sources=sources, extracted_classes=extracted, legacy_chat=dict(raw_tools_lengths=[len(t) for t in direct], wire_sorted_schema_normalized_lengths=[len(t) for t in normalized_ids], normalized_adjacent_lcp=[lcp(a, b) for a, b in zip(normalized_ids, normalized_ids[1:])], raw_parameter_key_order=list(old.TOOLS[0]['function']['parameters']), normalized_parameter_key_order=list(normalized[0]['function']['parameters']), online_evidence=evidence), limitations=['No HTTP, GPU, model weights, or KV cache execution.', 'AST Pydantic classes use BaseModel in place of OpenAIBaseModel; tests concern known declared fields.', 'Exact legacy prompt length and prefix equality corroborated, but online prompt IDs were not recorded.', 'These checks are preparatory source evidence, not proof of cache retention, DP affinity or performance.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), replay_sessions=len(rows), legacy_chat_lengths=result['legacy_chat']['wire_sorted_schema_normalized_lengths'], output=str(args.output)), indent=2))


if __name__=='__main__':main()
