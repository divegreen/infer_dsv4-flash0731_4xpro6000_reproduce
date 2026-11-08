"""Narrow Stage D research policies on the frozen vLLM AsyncScheduler.

No model/KV/output state is emulated here: the installed parent remains the
only allocator and scheduler-output producer. Normal performance disables step
tracing. This private API extension is not a general production scheduler.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


CAP = 2048
MIN_CAP = 256
SHORT_PROMPT = 1024
WAITING_RESERVE = 512
AGE_STEPS = 64
TRACE_ENV = "RESEARCH_D_TRACE_DIR"


def _status(request):
    value = request.status
    return getattr(value, "name", str(value))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_config(config, *, fair):
    parallel, schedule = config.parallel_config, config.scheduler_config
    model, cache = config.model_config, config.cache_config
    failures = []
    for key, expected in (("tensor_parallel_size", 4), ("pipeline_parallel_size", 1),
                          ("data_parallel_size", 1), ("decode_context_parallel_size", 1),
                          ("prefill_context_parallel_size", 1)):
        if getattr(parallel, key) != expected:
            failures.append(f"{key} must equal {expected}")
    if (model.runner_type != "generate" or model.is_encoder_decoder
            or model.is_diffusion):
        failures.append("only causal text generation is supported")
    for key in ("speculative_config", "lora_config", "kv_transfer_config", "ec_transfer_config"):
        if getattr(config, key) is not None:
            failures.append(f"{key} is unsupported")
    if config.aux_output_config.enabled or cache.enable_prefix_caching:
        failures.append("auxiliary output and prefix caching must be disabled")
    if (schedule.policy != "fcfs" or not schedule.async_scheduling
            or not schedule.enable_chunked_prefill):
        failures.append("requires FCFS, async scheduling and chunked prefill")
    if (schedule.max_num_batched_tokens != 4096
            or schedule.max_num_scheduled_tokens not in (None, 4096)
            or schedule.max_num_seqs != 64
            or schedule.max_num_active_seqs not in (None, 64)):
        failures.append("requires 4096 token budget and 64 running/active slots")
    if fair and (schedule.long_prefill_token_threshold != CAP
                 or schedule.long_prefill_token_threshold_adaptive):
        failures.append("FairShare requires original cap2048 and adaptive=False")
    if failures:
        raise ValueError("Stage D scheduler: " + "; ".join(failures))


class ObserveResearchScheduler(AsyncScheduler):
    """Exact parent schedule delegation plus optional request-ID diagnostics."""
    POLICY = 'observe-v1'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('vllm_config', args[0] if args else None)
        _validate_config(config, fair=self.POLICY == 'fairshare-v1')
        super().__init__(*args, **kwargs)
        self._research_trace = None
        self._research_steps = 0
        self._research_service = {}
        self._research_first_seen = {}
        trace_directory = os.environ.get(TRACE_ENV)
        if trace_directory:
            self._research_trace = open_output(Path(trace_directory) / f'scheduler-steps-{os.getpid()}.jsonl', 'x', buffering=1)
        self._write_identity()

    def _write_identity(self):
        """One startup record in every custom run, not per-step perf logging."""
        module_path = Path(__file__).resolve()
        parent_path = Path(inspect.getfile(AsyncScheduler)).resolve()
        identity = dict(type='scheduler_identity', schema_version=1, policy=self.POLICY, pid=os.getpid(), unix_ns=time.time_ns(), class_name=f'{type(self).__module__}.{type(self).__name__}', module_file=str(module_path), module_sha256=_sha(module_path), parent_file=str(parent_path), parent_sha256=_sha(parent_path), async_scheduler=True, step_trace_enabled=self._research_trace is not None, config=dict(token_budget=self.max_num_scheduled_tokens, input_budget=self.scheduler_config.max_num_batched_tokens, original_cap=self.scheduler_config.long_prefill_token_threshold, adaptive=self.adaptive_long_prefill_threshold, prefill_interval=self.local_prefill_interval, max_running=self.max_num_running_reqs, max_active=self.max_num_active_reqs), constants=dict(cap=CAP, cap_floor=MIN_CAP, short_prompt=SHORT_PROMPT, waiting_reserve=WAITING_RESERVE, age_steps=AGE_STEPS))
        directory = os.environ.get('RESEARCH_RUN_DIR')
        if directory:
            with open_output(Path(directory) / f'scheduler-identity-{os.getpid()}.json', 'x') as sink:
                json.dump(identity, sink, indent=2)
                sink.write('\n')
        print('DSV4_RESEARCH_SCHEDULER ' + json.dumps(identity, sort_keys=True), flush=True)
        if self._research_trace is not None:
            self._write_trace(identity)

    def add_request(self, request):
        unsupported = request.resumable or request.streaming_queue is not None or request.mm_features or request.has_encoder_inputs or (request.lora_request is not None) or (request.pooling_params is not None) or (request.structured_output_request is not None) or (request.prompt_embeds is not None) or (request.kv_hints is not None) or request.kv_transfer_params or request.ec_transfer_params or (request.priority != 0) or request.spec_token_ids
        if unsupported:
            raise ValueError('Stage D scheduler accepts ordinary unprioritized text generation only')
        super().add_request(request)

    def _write_trace(self, row):
        self._research_trace.write(json.dumps(row, separators=(',', ':'), sort_keys=True) + '\n')

    def _snapshot(self):
        """Counts are optimistic scheduler positions, NOT GPU completion."""
        fields = ('num_prompt_tokens', 'num_tokens', 'num_tokens_with_spec', 'num_computed_tokens', 'is_prefill_chunk', 'num_output_placeholders', 'num_in_flight_tokens', 'next_decode_eligible_step', 'num_preemptions')
        return dict(current_step=self.current_step, num_executed_steps=self.num_executed_steps, running=[r.request_id for r in self.running], waiting=[r.request_id for r in self.waiting], kv_holding_waiting=[r.request_id for r in self.kv_holding_waiting], deferred_waiting=sorted((r.request_id for r in self.deferred_waiting)), prefill_capacity_bound=self.prefill_capacity_bound, last_prefill_step=self.last_prefill_step, local_prefill_interval=self.local_prefill_interval, requests={key: dict({field: getattr(r, field) for field in fields}, status=_status(r)) for key, r in self.requests.items()})

    def _prepare_policy(self):
        return None

    def _after_policy(self, output):
        pass

    def schedule(self, throttle_prefills=False):
        self._research_steps += 1
        trace = self._research_trace is not None
        begin_mono = time.monotonic_ns() if trace else None
        begin_unix = time.time_ns() if trace else None
        before = self._snapshot() if trace else None
        original_cap = self.scheduler_config.long_prefill_token_threshold
        decision = None
        try:
            decision = self._prepare_policy()
            output = super().schedule(throttle_prefills=throttle_prefills)
            self._after_policy(output)
        except BaseException as exc:
            if trace:
                self._write_trace(dict(type='schedule_error', schema_version=1, policy=self.POLICY, scheduler_step=self._research_steps, begin_monotonic_ns=begin_mono, begin_unix_ns=begin_unix, end_monotonic_ns=time.monotonic_ns(), end_unix_ns=time.time_ns(), before=before, decision=decision, error=dict(type=type(exc).__name__, message=str(exc))))
            raise
        finally:
            self.scheduler_config.long_prefill_token_threshold = original_cap
        if trace:
            self._write_trace(dict(type='step', schema_version=1, policy=self.POLICY, scheduler_step=self._research_steps, begin_monotonic_ns=begin_mono, begin_unix_ns=begin_unix, end_monotonic_ns=time.monotonic_ns(), end_unix_ns=time.time_ns(), throttle_prefills=throttle_prefills, before=before, after=self._snapshot(), decision=decision, output=dict(num_scheduled_tokens=dict(output.num_scheduled_tokens), total_num_scheduled_tokens=output.total_num_scheduled_tokens, admitted_request_ids=[r.req_id for r in output.scheduled_new_reqs], preempted_request_ids=sorted(output.preempted_req_ids or ()), finished_request_ids=sorted(output.finished_req_ids or ()))))
        return output


class FairShareResearchScheduler(ObserveResearchScheduler):
    """Rotate prefills and leave bounded budget for waiting short requests.

    Parent cadence, admission, KV allocation, preemption, async placeholders and
    output production remain unchanged. Reordering RUNNING does change the FCFS
    last-entry preemption victim: that is an explicit property of this policy.
    Age promotion and cap_floor avoid an unbounded policy preference under
    finite load; they do not guarantee admission with full slots/unavailable KV.
    """

    POLICY = "fairshare-v1"

    def _pending(self, request):
        # Match the frozen parent's async eligibility and real token deficit.
        if (request.num_output_placeholders > 0 and request.num_computed_tokens + 2
                - request.num_output_placeholders >= request.num_prompt_tokens + request.max_tokens):
            return 0
        if self.current_step + 1 < request.next_decode_eligible_step:
            return 0
        return max(0, min(request.num_tokens_with_spec + request.num_output_placeholders
                          - request.num_computed_tokens,
                          self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step))

    @staticmethod
    def _prefill(request):
        return request.is_prefill_chunk or request.num_computed_tokens < request.num_prompt_tokens

    def _prepare_policy(self):
        step = self.current_step + 1
        # Key by Request object, not reusable external ID; parent owns lifetime.
        alive = set(self.requests.values())
        self._research_first_seen = {r:s for r,s in self._research_first_seen.items() if r in alive}
        self._research_service = {r:s for r,s in self._research_service.items() if r in alive}
        for request in alive:
            self._research_first_seen.setdefault(request, self.current_step)

        def last(request):
            return self._research_service.get(request,
                self._research_first_seen.get(request, self.current_step))

        def aged(request):
            return step - last(request) >= AGE_STEPS

        def order(request, *, waiting=False):
            kind = (1 if request.num_prompt_tokens <= SHORT_PROMPT else 2)
            if not waiting and not self._prefill(request):
                kind = 0
            return (0 if aged(request) else 1, kind if not aged(request) else 0,
                    last(request), request.arrival_time, request.request_id)

        # Do not replace or hide any request, and never reset counters.
        self.running.sort(key=order)
        old_waiting = list(self.waiting)
        eligible = [r for r in old_waiting if r not in self.deferred_waiting
                    and _status(r) in ("WAITING", "PREEMPTED")]
        ordered_waiting = iter(sorted(eligible, key=lambda r: order(r, waiting=True)))
        eligible_set = set(eligible)
        self.waiting.clear()
        self.waiting.extend(next(ordered_waiting) if r in eligible_set else r for r in old_waiting)
        # Leave kv_holding_waiting and deferred_waiting entirely to the parent.
        prefills = [r for r in self.running if self._prefill(r) and self._pending(r) > 0]
        decode = sum(self._pending(r) for r in self.running if not self._prefill(r))
        short_waiting = [r for r in eligible if r.num_prompt_tokens <= SHORT_PROMPT]
        reserve = min(WAITING_RESERVE, sum(max(0, r.num_tokens-r.num_computed_tokens)
                                          for r in short_waiting))
        if not short_waiting and any(aged(r) for r in eligible):
            reserve = MIN_CAP
        budget = min(self.max_num_scheduled_tokens, self.scheduler_config.max_num_batched_tokens)
        cap = min(CAP, max(MIN_CAP, (budget - decode - reserve) // max(1, len(prefills))))
        self.scheduler_config.long_prefill_token_threshold = cap
        if self._research_trace is None:
            return None
        return dict(dynamic_cap=cap, decode_demand=decode, waiting_reserve=reserve,
                    running_prefills=len(prefills), running_order=[r.request_id for r in self.running],
                    waiting_order=[r.request_id for r in self.waiting],
                    aged_promotions=sorted(r.request_id for r in alive if aged(r)),
                    reservation_is_soft=True)

    def _after_policy(self, output):
        for request_id, tokens in output.num_scheduled_tokens.items():
            if tokens > 0 and request_id in self.requests:
                self._research_service[self.requests[request_id]] = self.current_step
