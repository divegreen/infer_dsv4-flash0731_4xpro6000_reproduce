#!/usr/bin/env python3
"""CPU policy checks, optionally with installed Request/FCFSQueue/config types.

The fake parent is an explicit token-budget/admission fixture, not a vLLM KV
allocator or engine. It does not prove GPU/HTTP/quality/performance correctness.
--real-types imports vLLM CPU data types with CUDA_VISIBLE_DEVICES empty.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

from collections import deque
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REAL_TYPES = '--real-types' in sys.argv
if REAL_TYPES:
    sys.argv.remove('--real-types')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Hide all GPUs for CPU-only real-type checks')
    from vllm.config.scheduler import SchedulerConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
else:
    FCFSRequestQueue = deque


class Obj:
    def __init__(self, **values):
        self.__dict__.update(values)


class StubRequest:
    def __init__(self, request_id, prompt):
        self.request_id, self.num_prompt_tokens = request_id, prompt
        self.num_tokens = self.num_tokens_with_spec = prompt
        self.max_tokens, self.spec_token_ids = 128, []
        self.resumable, self.streaming_queue = False, None
        self.mm_features, self.has_encoder_inputs = [], False
        self.lora_request = self.pooling_params = None
        self.structured_output_request = self.prompt_embeds = None
        self.kv_hints = self.kv_transfer_params = self.ec_transfer_params = None
        self.priority = 0


def request(request_id, prompt=65536, *, computed=0, prefill=True,
            placeholders=0, next_step=0, arrival=0., status='WAITING', output=0):
    r = (Request(request_id, [1] * prompt, SamplingParams(max_tokens=128), None)
         if REAL_TYPES else StubRequest(request_id, prompt))
    if output:
        if REAL_TYPES:
            r.append_output_token_ids([2] * output)
        else:
            r.num_tokens += output
            r.num_tokens_with_spec += output
    r.num_computed_tokens, r.is_prefill_chunk = computed, prefill
    r.num_output_placeholders, r.num_in_flight_tokens = placeholders, 0
    r.next_decode_eligible_step, r.num_preemptions = next_step, 0
    r.arrival_time = arrival
    r.status = RequestStatus[status] if REAL_TYPES else status
    return r


def config(*, fair=True):
    fields = dict(policy='fcfs', async_scheduling=True, enable_chunked_prefill=True,
                  max_num_batched_tokens=4096, max_num_scheduled_tokens=None,
                  max_num_seqs=64, max_num_active_seqs=64,
                  long_prefill_token_threshold=2048 if fair else 0,
                  long_prefill_token_threshold_adaptive=False,
                  prefill_schedule_interval=8)
    schedule = (SchedulerConfig(max_model_len=131072, is_encoder_decoder=False, **fields)
                if REAL_TYPES else Obj(**fields))
    return Obj(parallel_config=Obj(tensor_parallel_size=4, pipeline_parallel_size=1,
                   data_parallel_size=1, decode_context_parallel_size=1,
                   prefill_context_parallel_size=1), scheduler_config=schedule,
               model_config=Obj(runner_type='generate', is_encoder_decoder=False,
                                is_diffusion=False, max_model_len=131072),
               cache_config=Obj(enable_prefix_caching=False), speculative_config=None,
               lora_config=None, kv_transfer_config=None, ec_transfer_config=None,
               aux_output_config=Obj(enabled=False))


class FakeParent:
    """Simplified no-KV, no-cadence token budget fixture; asserts delegation."""
    def __init__(self, *args, **kwargs):
        self.vllm_config = kwargs.get('vllm_config', args[0] if args else None)
        self.scheduler_config = self.vllm_config.scheduler_config
        self.parallel_config = self.vllm_config.parallel_config
        self.max_num_running_reqs = self.max_num_active_reqs = 64
        self.max_num_scheduled_tokens, self.max_model_len = 4096, 131072
        self.adaptive_long_prefill_threshold = False
        self.local_prefill_interval = self.scheduler_config.prefill_schedule_interval
        self.current_step = self.num_executed_steps = self.last_prefill_step = 0
        self.prefill_capacity_bound = False
        self.requests, self.running = {}, []
        self.waiting, self.kv_holding_waiting = FCFSRequestQueue(), FCFSRequestQueue()
        self.deferred_waiting = set()
        self.num_sampled_tokens_per_step = 1
        self.parent_calls = []

    def add_request(self, r):
        self.requests[r.request_id] = r
        self.waiting.append(r)

    def schedule(self, throttle_prefills=False):
        self.parent_calls.append(dict(cap=self.scheduler_config.long_prefill_token_threshold,
            running=list(self.running), waiting=list(self.waiting), throttle=throttle_prefills))
        self.current_step += 1
        count = len(self.running) + len(self.waiting) + len(self.kv_holding_waiting)
        cap = self.scheduler_config.long_prefill_token_threshold if count > 1 else 4096
        budget, scheduled, admitted = 4096, {}, []
        for r in list(self.running):
            if (r.num_output_placeholders > 0 and r.num_computed_tokens + 2
                    - r.num_output_placeholders >= r.num_prompt_tokens + r.max_tokens):
                continue
            if self.current_step < r.next_decode_eligible_step:
                continue
            pending = r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens
            n = max(0, min(pending, cap or 4096, budget))
            if n:
                scheduled[r.request_id], budget = n, budget - n
        while self.waiting and budget > 0 and len(self.running) < self.max_num_active_reqs:
            r = self.waiting[0]
            if r in self.deferred_waiting:
                break
            n = max(0, min(r.num_tokens-r.num_computed_tokens, cap or 4096, budget))
            if not n:
                break
            self.waiting.popleft()
            self.running.append(r)
            r.status = RequestStatus.RUNNING if REAL_TYPES else 'RUNNING'
            admitted.append(Obj(req_id=r.request_id))
            scheduled[r.request_id], budget = n, budget - n
        for rid, n in scheduled.items():
            r = self.requests[rid]
            r.num_computed_tokens += n
            r.num_in_flight_tokens += n
            r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens+r.num_output_placeholders
            if not r.is_prefill_chunk:
                r.num_output_placeholders += 1
        if scheduled:
            self.num_executed_steps += 1
        return Obj(num_scheduled_tokens=scheduled, total_num_scheduled_tokens=sum(scheduled.values()),
                   scheduled_new_reqs=admitted, preempted_req_ids=set(), finished_req_ids=set())


def load_module():
    names = ['vllm', 'vllm.v1', 'vllm.v1.core', 'vllm.v1.core.sched',
             'vllm.v1.core.sched.async_scheduler']
    modules = {name: types.ModuleType(name) for name in names}
    modules[names[-1]].AsyncScheduler = FakeParent
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('research_scheduler_tested',
                   Path(__file__).with_name('research_scheduler.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def setUp(self):
        self.env = patch.dict(os.environ, {'RESEARCH_D_TRACE_DIR': '', 'RESEARCH_RUN_DIR': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def scheduler(self, *, observe=False):
        cls = self.mod.ObserveResearchScheduler if observe else self.mod.FairShareResearchScheduler
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(vllm_config=config(fair=not observe))

    def seed(self, s, *, running=(), waiting=()):
        s.running[:] = running
        s.waiting.extend(waiting)
        s.requests.update((r.request_id, r) for r in [*running, *waiting])

    def test_observe_exact_parent_output_and_state(self):
        s = self.scheduler(observe=True)
        self.seed(s, running=[request('long')], waiting=[request('short', 512)])
        reference = copy.deepcopy(s)
        expected = FakeParent.schedule(reference, throttle_prefills=True)
        result = s.schedule(throttle_prefills=True)
        self.assertEqual(result.num_scheduled_tokens, expected.num_scheduled_tokens)
        self.assertEqual([r.request_id for r in s.running], [r.request_id for r in reference.running])
        self.assertEqual(s._snapshot(), reference._snapshot())
        self.assertEqual(s.parent_calls[-1]['cap'], 0)

    def test_two_long_six_short_budget_and_identity(self):
        s = self.scheduler()
        longs = [request('long-a'), request('long-b')]
        shorts = [request('short-'+str(i), 512) for i in range(6)]
        self.seed(s, running=longs, waiting=shorts)
        initial = {r.request_id:r for r in [*longs, *shorts]}
        output = s.schedule()
        self.assertEqual(s.parent_calls[-1]['cap'], 1792)
        self.assertEqual(output.num_scheduled_tokens, {'long-a':1792, 'long-b':1792, 'short-0':512})
        self.assertEqual({r.request_id:r for r in [*s.running, *s.waiting]}, initial)
        self.assertEqual(s.scheduler_config.long_prefill_token_threshold, 2048)

    def test_decode_deficit_and_pending_async_stop(self):
        s = self.scheduler()
        r = request('decode', 512, computed=526, prefill=False, placeholders=3, output=16)
        self.assertEqual(s._pending(r), 5)
        r.next_decode_eligible_step = 100
        self.assertEqual(s._pending(r), 0)
        r.next_decode_eligible_step = 0
        r.num_computed_tokens = 642
        self.assertEqual(s._pending(r), 0)
        r.num_computed_tokens, r.num_output_placeholders = 528, 0
        self.assertEqual(s._pending(r), 0)

    def test_decode_scheduled_before_long_and_uses_real_deficit(self):
        s = self.scheduler()
        decode = request('decode', 512, computed=526, prefill=False, placeholders=3, output=16)
        self.seed(s, running=[request('a'), request('b'), decode], waiting=[request('short', 512)])
        out = s.schedule()
        self.assertIs(s.parent_calls[-1]['running'][0], decode)
        self.assertEqual(s.parent_calls[-1]['cap'], (4096-5-512)//2)
        self.assertEqual(out.num_scheduled_tokens['decode'], 5)

    def test_extreme_concurrency_floor_rotates_without_dropping(self):
        s = self.scheduler()
        initial = [request(f'long-{i:02}') for i in range(64)]
        self.seed(s, running=initial)
        seen = set()
        for _ in range(4):
            out = s.schedule()
            seen.update(out.num_scheduled_tokens)
            self.assertEqual(s.parent_calls[-1]['cap'], 256)
            self.assertEqual(out.total_num_scheduled_tokens, 4096)
            self.assertEqual({id(r) for r in s.running}, {id(r) for r in initial})
        self.assertEqual(seen, {r.request_id for r in initial})

    def test_age_promotion_waiting_and_running(self):
        s = self.scheduler()
        old = request('aged')
        short = request('fresh-short',512)
        decode = request('fresh-decode',512,computed=512,prefill=False,output=1)
        self.seed(s, running=[decode,old], waiting=[short])
        s.current_step = 64
        s._research_first_seen = {old:0, short:64, decode:64}
        s._prepare_policy()
        self.assertIs(s.running[0],old)
        s.running.remove(old)
        s.waiting.append(old)
        s._prepare_policy()
        self.assertIs(s.waiting[0],old)

    def test_aged_long_waiting_soft_reservation(self):
        s = self.scheduler()
        old = request('old')
        self.seed(s,running=[request('active-a'),request('active-b')],waiting=[old])
        s.current_step=64
        s._research_first_seen[old]=0
        s._prepare_policy()
        self.assertEqual(s.scheduler_config.long_prefill_token_threshold,1920)

    def test_deferred_and_kv_holding_queues_not_reordered(self):
        s=self.scheduler()
        blocked=request('blocked',status='WAITING')
        long=request('long')
        short=request('short',512)
        kv=request('kv-holder')
        self.seed(s,waiting=[long,blocked,short])
        s.deferred_waiting.add(blocked)
        s.kv_holding_waiting.append(kv)
        s.requests[kv.request_id]=kv
        s._prepare_policy()
        self.assertIs(s.waiting[1],blocked)
        self.assertEqual(s.deferred_waiting,{blocked})
        self.assertEqual(list(s.kv_holding_waiting),[kv])

    def test_parent_preempt_resume_preserves_objects_and_counters(self):
        s=self.scheduler()
        victim=request('victim')
        self.seed(s,running=[request('decoder',512,computed=512,prefill=False,output=1),victim])
        def preempt(parent,**kwargs):
            parent.current_step+=1
            r=parent.running.pop()
            r.num_computed_tokens=0
            r.num_preemptions+=1
            r.status=RequestStatus.PREEMPTED if REAL_TYPES else 'PREEMPTED'
            parent.waiting.appendleft(r)
            return Obj(num_scheduled_tokens={},total_num_scheduled_tokens=0,
                       scheduled_new_reqs=[],preempted_req_ids={r.request_id},finished_req_ids=set())
        with patch.object(FakeParent,'schedule',preempt):
            s.schedule()
        self.assertIs(s.waiting[0],victim)
        self.assertEqual(victim.num_preemptions,1)
        s.schedule()
        self.assertIs(s.requests['victim'],victim)
        self.assertIn(victim,s.running)
        self.assertEqual(victim.num_preemptions,1)

    def test_exception_restores_cap_and_records_error(self):
        with tempfile.TemporaryDirectory(prefix='d-scheduler-error-') as d:
            os.environ['RESEARCH_D_TRACE_DIR']=d
            s=self.scheduler()
            self.addCleanup(s._research_trace.close)
            self.seed(s,running=[request('a'),request('b')],waiting=[request('short',512)])
            with patch.object(FakeParent,'schedule',side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                    s.schedule()
            self.assertEqual(s.scheduler_config.long_prefill_token_threshold,2048)
            rows=[json.loads(x) for x in next(Path(d).glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual(rows[-1]['type'],'schedule_error')
            self.assertEqual(rows[-1]['decision']['dynamic_cap'],1792)

    def test_trace_schema_snapshot_identity_and_optional_sets(self):
        with tempfile.TemporaryDirectory(prefix='d-scheduler-trace-') as d:
            os.environ['RESEARCH_D_TRACE_DIR']=os.environ['RESEARCH_RUN_DIR']=d
            s=self.scheduler()
            self.addCleanup(s._research_trace.close)
            r=request('trace-id',512)
            self.seed(s,waiting=[r])
            s.schedule()
            rows=[json.loads(x) for x in next(Path(d).glob('*.jsonl')).read_text().splitlines()]
            row=rows[-1]
            self.assertEqual(row['type'],'step')
            self.assertEqual(row['before']['requests']['trace-id']['num_computed_tokens'],0)
            self.assertEqual(row['after']['requests']['trace-id']['num_computed_tokens'],512)
            self.assertEqual(row['after']['requests']['trace-id']['num_output_placeholders'],1)
            self.assertEqual(row['output']['admitted_request_ids'],['trace-id'])
            self.assertNotIn('prompt_token_ids',json.dumps(rows))
            self.assertEqual(len(list(Path(d).glob('scheduler-identity-*.json'))),1)
            empty=Obj(num_scheduled_tokens={},total_num_scheduled_tokens=0,
                      scheduled_new_reqs=[],preempted_req_ids=None,finished_req_ids=None)
            with patch.object(FakeParent,'schedule',return_value=empty):
                s.schedule()
            with self.assertRaises(FileExistsError):
                self.scheduler()

    def test_trace_off_does_not_open_files_per_step(self):
        s=self.scheduler()
        self.seed(s,running=[request('long')])
        with (patch('builtins.open',side_effect=AssertionError('unexpected per-step file IO')),
              patch.object(Path,'open',side_effect=AssertionError('unexpected Path IO'))):
            s.schedule()

    def test_reject_config_and_request_before_delegation(self):
        for location,key,value in [('parallel_config','pipeline_parallel_size',2),
                                  ('cache_config','enable_prefix_caching',True),
                                  ('scheduler_config','policy','priority'),
                                  ('scheduler_config','long_prefill_token_threshold_adaptive',True)]:
            c=config()
            setattr(getattr(c,location),key,value)
            with self.assertRaises(ValueError):
                self.mod.FairShareResearchScheduler(vllm_config=c)
        s=self.scheduler()
        for key,value in [('resumable',True),('priority',1),('lora_request',object()),
                          ('mm_features',[object()]),('structured_output_request',object()),
                          ('prompt_embeds',object()),('kv_hints',object())]:
            r=request('unsupported',512)
            setattr(r,key,value)
            with self.assertRaises(ValueError):
                s.add_request(r)
        self.assertEqual(s.requests,{})
        good=request('good',512)
        s.add_request(good)
        self.assertIs(s.requests['good'],good)

    def test_lifetime_prunes_finished_object_but_reused_id_is_new(self):
        s=self.scheduler()
        old=request('reuse')
        self.seed(s,running=[old])
        s.schedule()
        s.running.clear()
        s.requests.clear()
        new=request('reuse')
        self.seed(s,waiting=[new])
        s._prepare_policy()
        self.assertNotIn(old,s._research_service)
        self.assertIn(new,s._research_first_seen)
        self.assertNotIn(new,s._research_service)


if __name__=='__main__':
    print('CPU_FIXTURE_SCOPE: '+('installed Request/FCFSQueue/SchedulerConfig; ' if REAL_TYPES else 'stub data types; ')
          +'fake parent token allocator, no vLLM engine/GPU/HTTP/performance validation',flush=True)
    unittest.main(verbosity=2)
