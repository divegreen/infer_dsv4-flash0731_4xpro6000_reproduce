#!/usr/bin/env python3
"""CPU-only source checks: salt hash chains, V4 templates, and DP routing.

Executes selected installed functions via AST; never imports vllm/torch, contacts
a service, loads model weights, or constructs a CUDA context. These checks prove
local function semantics, not HTTP propagation, KV residency or performance.
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
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from cache_routing import session_routing_headers

HERE = Path(__file__).resolve().parents[1]
STAGE = HERE.parent / 'scenarios'
SOURCE_FILES = [
    'entrypoints/generate/base/protocol.py', 'entrypoints/generate/base/serving.py',
    'entrypoints/openai/completion/protocol.py', 'entrypoints/openai/completion/serving.py',
    'entrypoints/openai/chat_completion/protocol.py', 'entrypoints/openai/chat_completion/serving.py',
    'renderers/online_renderer.py', 'renderers/base.py', 'inputs/engine.py',
    'v1/engine/input_processor.py', 'v1/engine/output_processor.py',
    'v1/engine/core_client.py', 'v1/engine/core.py', 'v1/request.py',
    'v1/core/kv_cache_utils.py', 'v1/core/kv_cache_manager.py',
    'v1/core/kv_cache_coordinator.py', 'v1/core/single_type_kv_cache_manager.py',
    'v1/core/block_pool.py', 'v1/core/sched/scheduler.py',
    'config/cache.py', 'config/speculative.py', 'config/vllm.py',
    'v1/kv_cache_interface.py', 'v1/metrics/stats.py', 'v1/metrics/loggers.py',
    'utils/hashing.py', 'tokenizers/deepseek_v4.py',
    'tokenizers/deepseek_v4_encoding.py', 'models/deepseek_v4/attention.py',
    'models/deepseek_v4/compressor.py', 'v1/attention/backends/mla/sparse_swa.py',
    'entrypoints/serve/dev/cache/api_router.py', 'entrypoints/launchers/api_server/routers.py',
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def selected_functions(path, names, env):
    tree = ast.parse(path.read_text())
    nodes = []
    for name in names:
        hits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(hits) == 1, (path, name, len(hits))
        node = copy.deepcopy(hits[0])
        node.decorator_list = []
        nodes.append(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return {n.name: dict(start_line=n.lineno, end_line=n.end_lineno) for n in nodes}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    dist = importlib.metadata.distribution('vllm')
    root = Path(dist.locate_file('vllm'))
    result = dict(status='PASS_CPU_ONLY_GPU_NOT_VERIFIED', created_utc=datetime.now(timezone.utc).isoformat(), installed_vllm=dist.version, source_root=str(root), script_sha256=sha(Path(__file__).read_bytes()), routing_helper_sha256=sha((Path(__file__).parent / 'cache_routing.py').read_bytes()), checks=[], source_hashes={}, executed_functions={})
    for name in SOURCE_FILES:
        raw = (root / name).read_bytes()
        record = next((f for f in dist.files if str(f) == 'vllm/' + name))
        assert record.hash.value == base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        result['source_hashes'][name] = dict(sha256=sha(raw), wheel_record_match=True)

    def check(name, value):
        assert value, name
        result['checks'].append(name)

    class ValidationError(ValueError):

        def __init__(self, message, **kwargs):
            super().__init__(message)
    env = dict(hashlib=hashlib, pickle=pickle, copy=copy, sys=sys, BlockHash=lambda x: x, NONE_HASH=b'phase-c-fixed-cpu-root', VLLMValidationError=ValidationError, _MAX_CACHE_SALT_LENGTH=128, _CACHE_SALT_FORBIDDEN_CHARS=frozenset('@/\\\x00'))
    targets = {'entrypoints/generate/base/protocol.py': ['validate_cache_salt'], 'utils/hashing.py': ['sha256'], 'v1/core/kv_cache_utils.py': ['_gen_mm_extra_hash_keys', '_gen_lora_extra_hash_keys', '_gen_prompt_embeds_extra_hash_keys', 'generate_block_hash_extra_keys', 'hash_block_tokens', 'get_request_block_hasher'], 'tokenizers/deepseek_v4.py': ['apply_chat_template'], 'v1/engine/core_client.py': ['get_core_engine_for_request']}
    for rel, names in targets.items():
        if rel.endswith('core_client.py'):
            continue
        result['executed_functions'][rel] = selected_functions(root / rel, names, env)
    validate = env['validate_cache_salt']
    for value in [None, 'c01-s0001', 'a' * 128]:
        validate(value)
    for value in ['', 1, True, 'a' * 129, 'x/y', 'x@y', 'x\\y', 'x\x00y']:
        try:
            validate(value)
        except ValidationError:
            pass
        else:
            raise AssertionError(f'Invalid salt accepted: {value!r}')
    check('actual salt validator accepts None/1..128 and rejects forbidden values', True)

    def request(tokens, salt, session='s1'):
        return SimpleNamespace(all_token_ids=list(tokens), num_tokens=len(tokens), cache_salt=salt, session_id=session, block_hashes=[], mm_features=[], lora_request=None, prompt_embeds=None)
    hasher = env['get_request_block_hasher'](4, env['sha256'])
    a = request(range(512), 'session-a')
    ha = hasher(a)
    check('same token IDs and salt reproduce all full-block hashes', ha == hasher(request(range(512), 'session-a')))
    check('different salt separates every subsequent chained hash', all((x != y for x, y in zip(ha, hasher(request(range(512), 'session-b'))))))
    check('session_id alone is not in default prefix hash', ha == hasher(request(range(512), 'session-a', 'other-session')))
    check('None salt shares deterministic namespace', hasher(request(range(512), None)) == hasher(request(range(512), None, 'other')))
    ext = request(range(768), 'session-a')
    check('same salt append-only extension preserves old hash chain', hasher(ext)[:len(ha)] == ha)
    ext.block_hashes = ha[:]
    check('incremental full-block hash generation equals fresh chain', ha + hasher(ext) == hasher(request(range(768), 'session-a')))
    edited = list(range(512))
    edited[257] += 1000
    he = hasher(request(edited, 'session-a'))
    check('changed middle token preserves only earlier blocks', he[:64] == ha[:64] and all((x != y for x, y in zip(he[64:], ha[64:]))))
    check('partial last hash block is omitted', len(hasher(request(range(515), 'session-a'))) == 128)
    k0, _ = env['generate_block_hash_extra_keys'](a, 0, 4, 0)
    k1, _ = env['generate_block_hash_extra_keys'](a, 4, 8, 0)
    check('salt extra key present only in first hash block', k0 == (('cache_salt', 'session-a'),) and k1 is None)
    result['hash_checks'] = dict(hash_unit=4, fixed_test_root=True, caution='Artificial fixed NONE_HASH and text-only request stand-ins; no KV store or GPU engine instantiated.')
    for i in range(12):
        check(f'routing session {i}: default empty and sticky4 rank stable', session_routing_headers('none', i) == {} and session_routing_headers('sticky4', i) == {'X-data-parallel-rank': str(i % 4)})
    for mode, index in [('bad', 0), ('sticky4', -1), ('sticky4', True), ('sticky4', 1.0), ('none', '0')]:
        try:
            session_routing_headers(mode, index)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid routing input accepted')
    check('routing helper rejects unknown mode and invalid session index', True)
    tree = ast.parse((root / 'v1/engine/core_client.py').read_text())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DPLBAsyncMPClient'))
    fn = copy.deepcopy(next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_core_engine_for_request')))
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    env['get_late_interaction_engine_index'] = lambda pooling_params, num_engines: None
    exec(compile(ast.fix_missing_locations(code), str(root / 'v1/engine/core_client.py'), 'exec'), env)
    result['executed_functions']['v1/engine/core_client.py'] = {'DPLBAsyncMPClient.get_core_engine_for_request': dict(start_line=fn.lineno, end_line=fn.end_lineno)}
    router = SimpleNamespace(streaming_reqs=set(), reqs_in_flight={}, core_engines=[0, 1, 2, 3], lb_engines=[[0, 0, 0.0] for _ in range(4)], engine_inflight=defaultdict(int), client_count=1, eng_start_index=0)
    route = env['get_core_engine_for_request']
    selected = []
    for i in range(8):
        req = SimpleNamespace(resumable=False, request_id=f'req{i}', data_parallel_rank=None, pooling_params=None, cache_salt='same-session', session_id='same-session')
        chosen = route(router, req)
        selected.append(chosen)
        router.reqs_in_flight.pop(req.request_id)
        router.engine_inflight[chosen] -= 1
        router.lb_engines = [[0, 0, 0.0] for _ in range(4)]
    check('actual DP selector does not keep equal-load completed turns on same salt/session', selected == [0, 1, 2, 3, 0, 1, 2, 3])
    req = SimpleNamespace(resumable=False, request_id='sticky', data_parallel_rank=2, pooling_params=None)
    check('actual DP selector honors valid explicit rank', route(router, req) == 2)
    result['routing_checks'] = dict(equal_load_simulated_sequence=selected, caution='CPU routing simulation with refreshed zero load; not an actual service routing trace or cache-hit guarantee.')
    encoding = load_module(root / 'tokenizers/deepseek_v4_encoding.py', 'stagec_v4_encoding')
    agent = load_module(STAGE / 'scripts/agent_benchmark.py', 'stagec_old_agent')
    from tokenizers import Tokenizer
    tokenizer_path = Path(str(model_path() / 'tokenizer.json'))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    env['encode_messages'] = encoding.encode_messages
    proxy = SimpleNamespace(encode=lambda text, **kw: tokenizer.encode(text, add_special_tokens=kw.get('add_special_tokens', False)).ids)

    def render(messages, tools=agent.TOOLS, **kwargs):
        return env['apply_chat_template'](proxy, messages, tools=tools, tokenize=True, enable_thinking=False, reasoning_effort='none', **kwargs)
    histories = agent.replay_histories()
    ids = [render(m) for m in histories]
    result['old_chat_replay'] = dict(token_lengths=[len(t) for t in ids], adjacent_lcp=[lcp(a, b) for a, b in zip(ids, ids[1:])], token_sha256=[sha(json.dumps(t, separators=(',', ':')).encode()) for t in ids], tokenizer_sha256=sha(tokenizer_path.read_bytes()), source_sha256=sha((STAGE / 'scripts/agent_benchmark.py').read_bytes()), scope='Actual installed Python wrapper function + encoding + tokenizer; online Rust renderer equivalence not checked.')
    check('chat rendering deterministic for frozen history/tools/thinking', all((render(m) == t for m, t in zip(histories, ids))))
    changed = copy.deepcopy(histories[0])
    changed[0]['content'] = 'Changed instructions.\n' + changed[0]['content']
    system_lcp = lcp(ids[0], render(changed))
    changed_tools = copy.deepcopy(agent.TOOLS)
    changed_tools[0]['function']['description'] += ' Additional detail.'
    tools_lcp = lcp(ids[0], render(histories[0], tools=changed_tools))
    check('changing leading system text invalidates long token prefix', system_lcp < len(ids[0]))
    check('changing tool schema can invalidate long token prefix', tools_lcp < len(ids[0]))
    result['old_chat_replay'].update(changed_system_lcp=system_lcp, changed_tools_lcp=tools_lcp)
    check('no vllm or torch package imported', 'vllm' not in sys.modules and 'torch' not in sys.modules)
    result['limitations'] = ['No GPU, service, model weights or HTTP tests', 'Salt chain equality does not guarantee cached KV retention, common hybrid boundary or same DP engine', 'Python template test is not proof of online renderer byte identity', 'Same-salt cross-session sharing is intentional in a shared-namespace experiment only']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), source_files=len(result['source_hashes']), old_chat_replay=result['old_chat_replay'], output=str(args.output)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
