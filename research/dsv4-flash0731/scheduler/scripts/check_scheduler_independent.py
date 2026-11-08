#!/usr/bin/env python3
"""Independent focused controls over the explicitly fake parent fixture.

No vLLM/GPU import: combines source review and narrow object-lifetime assertions.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode=True
PHASE=Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-log', type=Path, action='append', default=[],
                        help='Optional existing author-test log to bind to this review; never required for the independent CPU controls.')
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    spec = importlib.util.spec_from_file_location('D_scheduler_independent_fixture', PHASE / 'scripts/test_research_scheduler.py')
    t = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = t
    spec.loader.exec_module(t)
    module = t.load_module()

    def check(name, value):
        assert value, name
        checks.append(name)
    with patch.dict(os.environ, {'RESEARCH_D_TRACE_DIR': '', 'RESEARCH_RUN_DIR': ''}), contextlib.redirect_stdout(io.StringIO()):
        s = module.FairShareResearchScheduler(vllm_config=t.config())
        longs = [t.request('long-a'), t.request('long-b')]
        short = t.request('short', 512)
        blocked = t.request('blocked', status='WAITING_FOR_REMOTE_KVS')
        kv = t.request('kv')
        s.running[:] = longs
        s.waiting.extend([blocked, short])
        s.kv_holding_waiting.append(kv)
        s.requests = {r.request_id: r for r in [*longs, short, blocked, kv]}
        qid = id(s.waiting)
        rid = id(s.running)
        kid = id(s.kv_holding_waiting)
        states = {id(r): dict(r.__dict__) for r in s.requests.values()}
        s._prepare_policy()
        check('running/waiting/KV queue objects retained', id(s.running) == rid and id(s.waiting) == qid and (id(s.kv_holding_waiting) == kid))
        check('blocked waiting position and KV queue unchanged', s.waiting[0] is blocked and list(s.kv_holding_waiting) == [kv])
        check('policy retains every request state before parent allocation', all((r.__dict__ == states[id(r)] for r in s.requests.values())))
        check('two long prefills reserve 512 via cap1792', s.scheduler_config.long_prefill_token_threshold == 1792)
        check('cadence counters untouched by policy', s.local_prefill_interval == 8 and s.last_prefill_step == 0 and (not s.prefill_capacity_bound))
        s.scheduler_config.long_prefill_token_threshold = 2048
        with patch.object(t.FakeParent, 'schedule', side_effect=RuntimeError('independent controlled exception')):
            try:
                s.schedule()
            except RuntimeError:
                pass
            else:
                raise AssertionError('exception lost')
        check('temporary cap restored on parent exception', s.scheduler_config.long_prefill_token_threshold == 2048)
        observed = module.ObserveResearchScheduler(vllm_config=t.config(fair=False))
        sentinel = object()
        with patch.object(t.FakeParent, 'schedule', return_value=sentinel) as call:
            actual = observed.schedule(throttle_prefills=True)
        check('Observe returns exact parent output object', actual is sentinel)
        check('Observe forwards cadence throttle unchanged', call.call_args.kwargs == {'throttle_prefills': True})
        s2 = module.FairShareResearchScheduler(vllm_config=t.config())
        request = t.request('decode', 512, computed=526, prefill=False, placeholders=3, output=16)
        check('real5-token async pending demand', s2._pending(request) == 5)
        request.next_decode_eligible_step = 2
        check('next-step decode eligibility respected', s2._pending(request) == 0)
        request.next_decode_eligible_step = 0
        request.num_computed_tokens = 642
        check('async final placeholders do not overschedule', s2._pending(request) == 0)
    installed = Path(importlib.metadata.distribution('vllm').locate_file('vllm/v1/core/sched'))
    source = (PHASE / 'scripts/research_scheduler.py').read_text()
    parent = (installed / 'scheduler.py').read_text()
    check('actual FCFS parent chooses last running victim', 'preempted_req = self.running.pop()' in parent)
    check('candidate discloses changed FCFS victim', 'last-entry preemption victim' in source)
    check('soft reserve/no admission guarantee disclosed', 'reservation_is_soft=True' in source and 'do not guarantee admission' in source)
    files = [Path(__file__), PHASE / 'scripts/research_scheduler.py', PHASE / 'scripts/test_research_scheduler.py', *args.reference_log, installed / 'scheduler.py', installed / 'async_scheduler.py', installed / 'request_queue.py']
    ledger = {str(f.resolve()): dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size) for f in files}
    result = dict(status='PASS_INDEPENDENT_CPU_SCHEDULER_SCOPE_REVIEW', utc=datetime.now(timezone.utc).isoformat(), checks=checks, evidence_ledger=ledger, reference_logs=[str(p) for p in args.reference_log], author_tests_reused='Only explicitly supplied reference log bytes are inventoried; author tests are not rerun or presumed to have passed.', limits=['Dynamic controls still use explicitly fake parent allocator; no GPU/KV/cadence engine execution proven.', 'Actual runtime initialization and output need GPU evidence and source-bound self-report.', 'Floor/many running prefills/full slots or KV constraints can exhaust soft reservation; no starvation guarantee.', 'Normal timing excludes synchronous diagnostic trace.'])
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks))))


if __name__=='__main__':main()
