#!/usr/bin/env python3
"""Finite, local-only streaming completion benchmark; no model/GPU imports.

TTFT is time to first nonempty text, not first SSE event. TPOT is the
request-level estimate (E2E - TTFT) / (completion_tokens - 1), not token ITL.
Use --prepare-only to validate exact CPU-tokenized workloads without HTTP.
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
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from urllib.parse import urlsplit

import aiohttp
from tokenizers import Tokenizer

TIMEOUT_SECONDS = 300
MAX_CONCURRENCY = 32
PARAGRAPHS = (
    "A community library keeps books about science, history, and art. Volunteers sort returned books, help readers find reliable sources, and record which shelves need attention. A useful plan separates urgent repairs from optional improvements.",
    "An orchard team records rainfall, soil moisture, and harvest dates. They compare observations from several weeks before deciding when to water. A single unusually hot day is evidence about that day, not proof of a permanent trend.",
    "A school project compares walking, cycling, and taking the bus. Students consider travel time, cost, safety, and weather. They write down assumptions so another group can reproduce the comparison and explain disagreements.",
    "A small workshop receives twelve orders on Monday and eight on Tuesday. Some items need painting while others need assembly. The coordinator tracks unfinished work and checks quality before marking an order complete.",
    "A museum guide explains that maps simplify reality. A city map emphasizes streets, while a climate map emphasizes temperature and rainfall. The best map depends on the question the visitor wants to answer.",
    "A research group measures water temperature with two instruments. They record units, calibration dates, and uncertainty. When results disagree, they check the measurement procedure instead of immediately choosing the preferred number.",
)
SUFFIX = (
    "\n\nTasks: summarize the main ideas in three short points; explain one useful comparison; "
    "calculate the total of twelve and eight; and propose a practical next step. "
    "Write clear, connected English.\nAnswer:"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def endpoints(value: str) -> list[str]:
    result = []
    for raw in value.split(","):
        parsed = urlsplit(raw.strip())
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") not in ("", "/v1")):
            raise argparse.ArgumentTypeError(
                "Use comma-separated http://127.0.0.1:PORT[/v1] URLs only."
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        if port is None:
            raise argparse.ArgumentTypeError("Each endpoint needs an explicit port.")
        url = f"http://127.0.0.1:{port}/v1/completions"
        if url in result:
            raise argparse.ArgumentTypeError("Duplicate endpoint.")
        result.append(url)
    if not 1 <= len(result) <= 2:
        raise argparse.ArgumentTypeError("Supply one endpoint or two replica endpoints.")
    return result


def cases_for(suite: str) -> list[dict]:
    if suite == "long":
        return [
            dict(name="p16384_o128_c1", input_tokens=16384, concurrency=1, requests=2,
                 output_tokens=128, ignore_eos=True, arrival_pattern="closed_loop"),
            dict(name="p16384_o128_c4", input_tokens=16384, concurrency=4, requests=8,
                 output_tokens=128, ignore_eos=True, arrival_pattern="closed_loop_multibatch"),
            dict(name="p32768_o128_c1", input_tokens=32768, concurrency=1, requests=2,
                 output_tokens=128, ignore_eos=True, arrival_pattern="closed_loop"),
        ]
    if suite == "smoke":
        return [
            dict(name="smoke_math", prompt_text="Question: What is 7 + 5? Reply with only the number.\nAnswer:",
                 expected_hint="12", concurrency=1, requests=1, output_tokens=16, ignore_eos=False),
            dict(name="smoke_fact", prompt_text="Question: What is the capital of France? Reply with only the city name.\nAnswer:",
                 expected_hint="Paris", concurrency=1, requests=1, output_tokens=16, ignore_eos=False),
            dict(name="smoke_fixed_512_16", input_tokens=512, concurrency=1,
                 requests=2, output_tokens=16, ignore_eos=True),
        ]
    if suite == "full":
        return [
            dict(name="p512_o256_c1_n4", input_tokens=512, concurrency=1, requests=4, output_tokens=256, ignore_eos=True),
            dict(name="p512_o128_c8_n32", input_tokens=512, concurrency=8, requests=32, output_tokens=128, ignore_eos=True),
            dict(name="p512_o128_c32_n128", input_tokens=512, concurrency=32, requests=128, output_tokens=128, ignore_eos=True),
            dict(name="p4096_o128_c8_n32", input_tokens=4096, concurrency=8, requests=32, output_tokens=128, ignore_eos=True),
        ]
    return [
        dict(name="p512_o64_c1", input_tokens=512, concurrency=1, requests=2, output_tokens=64, ignore_eos=True),
        dict(name="p4096_o128_c1", input_tokens=4096, concurrency=1, requests=2, output_tokens=128, ignore_eos=True),
        dict(name="p512_o64_c8", input_tokens=512, concurrency=8, requests=8, output_tokens=64, ignore_eos=True),
        dict(name="p512_o64_c32", input_tokens=512, concurrency=32, requests=32, output_tokens=64, ignore_eos=True),
    ]


def make_prompt(tokenizer: Tokenizer, case: dict, seed: int, index: int, phase: str) -> list[int]:
    if "prompt_text" in case:
        return tokenizer.encode(case["prompt_text"], add_special_tokens=False).ids
    variant = hashlib.sha256(f"{seed}:{case['name']}:{phase}:{index}".encode()).hexdigest()
    rng = random.Random(int(variant, 16))
    paragraphs = list(PARAGRAPHS)
    rng.shuffle(paragraphs)
    prefix = tokenizer.encode(
        f"Read the following notes carefully. Exercise reference {variant[:12]}.\n\n",
        add_special_tokens=False,
    ).ids
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False).ids
    body = tokenizer.encode("\n\n".join(paragraphs) + "\n\n", add_special_tokens=False).ids
    need = case["input_tokens"] - len(prefix) - len(suffix)
    if need <= 0 or not body:
        raise ValueError("Prompt framing exceeds requested input length.")
    ids = prefix + (body * math.ceil(need / len(body)))[:need] + suffix
    if len(ids) != case["input_tokens"]:
        raise AssertionError("Incorrect exact token length.")
    return ids


def prepare(args: argparse.Namespace) -> tuple[dict, dict]:
    tokenizer_file = args.tokenizer_path.expanduser().resolve() / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    cases = cases_for(args.suite)
    prompts: dict[str, dict] = {}
    banks = {}
    for case in cases:
        case["arrival_pattern"] = "single_burst" if case["requests"] == case["concurrency"] else "closed_loop"
        if args.suite == "full":
            case["arrival_pattern"] = "closed_loop_multibatch"
        case["warmup_requests"] = max(case["concurrency"], len(args.base_url))
        for phase, count in (("warmup", case["warmup_requests"]), ("measure", case["requests"])):
            bank = []
            for index in range(count):
                ids = make_prompt(tokenizer, case, args.seed, index, phase)
                prompt_hash = digest_ids(ids)
                prompts[prompt_hash] = dict(
                    token_ids=ids, input_tokens=len(ids),
                    decoded_excerpt=tokenizer.decode(ids, skip_special_tokens=False)[:400],
                    sha256=prompt_hash,
                )
                bank.append(prompt_hash)
            banks[(case["name"], phase)] = bank
    report = dict(
        schema_version=1, status="PREPARED", created_at=utc_now(),
        validation_scope="Transport, actual token usage and nonempty text only; semantic quality requires human review.",
        suite=args.suite, rounds=args.rounds, model=args.model,
        endpoints=args.base_url, seed=args.seed,
        timeout_seconds=TIMEOUT_SECONDS,
        client_transport=dict(connection_policy="fresh_per_request" if args.fresh_connections else "pooled_default",
                              automatic_retries=False,
                              description="Fresh local TCP connection per request; connection setup remains inside E2E/TTFT." if args.fresh_connections else "aiohttp default keepalive pool reused across cases and rounds."),
        tokenizer=dict(path=str(tokenizer_file), sha256=hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()),
        client=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=sys.version, aiohttp=importlib.metadata.version("aiohttp"),
                    tokenizers=importlib.metadata.version("tokenizers")),
        metric_definitions=dict(
            ttft_s="Request start to first SSE choice with nonempty text; includes client/network and server time.",
            e2e_s="Request start to the [DONE] SSE event, including final usage delivery.",
            tpot_s="(e2e_s - ttft_s) / (actual completion_tokens - 1), for completion_tokens > 1; not token ITL.",
            throughput="Successful measured completion tokens / complete measured batch wall seconds; failures reported separately.",
            request_start="After concurrency semaphore acquisition, immediately before HTTP POST; client queue delay is separate.",
            percentiles="Linear interpolation between sorted successful request measurements; p95 is descriptive at small N.",
        ),
        server_requirements=["Disable prefix caching for all compared servers.",
                             "Keep model, precision, sampling and server settings comparable; this client does not alter them."],
        caveats=["Smoke expected_hint is for human review, not an automatic semantic-quality assertion.",
                 "No chunk interval is reported as token ITL; token emission boundaries are not available.",
                 "This is a finite concurrency test, not an open-loop arrival-rate benchmark.",
                 "C8/N8 and C32/N32 are single bursts, not steady-state throughput or p95 estimates with large sample counts.",
                 "Warmups precede each case at the same concurrency; first-run compilation may need additional server preparation.",
                 "Identical prompt token IDs are reused across full-suite rounds; prefix caching must stay disabled."],
        case_specs=cases, prompts=prompts, runs=[],
    )
    if args.suite == "full":
        report["suite_purpose"] = "Candidate sustained-load and longer-output validation; finite multibatch runs do not establish a steady-state SLA."
        report["metric_definitions"]["client_queue_s"] = (
            "Client task wait for a concurrency semaphore slot; excluded from TTFT/E2E. "
            "This is not the server's request queue time."
        )
        report["metric_definitions"]["e2e_s"] = (
            "Immediately before entering session.post to the [DONE] SSE event; includes "
            "HTTP/client/network overhead, server queueing and generation, and final usage delivery. "
            "Excludes client_queue_s; not a server-only latency measurement."
        )
        report["warmup_policy"] = (
            "Every round independently warms each case immediately before measuring it, "
            "at that case's concurrency and exact output_tokens; warmup results are stored "
            "separately and excluded from measured wall time and throughput."
        )
        report["caveats"].append(
            "Requests beyond concurrency wait in the client and are replenished as slots "
            "free up; measured batch wall time includes this client scheduling."
        )
    return report, banks


async def sse_events(content):
    """Assemble SSE events across arbitrary transport chunks; support CRLF/comments."""
    data_lines = []
    while True:
        raw = await content.readline()
        if not raw:
            if data_lines:
                yield "\n".join(data_lines)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)


async def request_one(session, endpoint: str, model: str, case: dict,
                      prompt: dict, seed: int, request_id: str) -> dict:
    payload = dict(model=model, prompt=prompt["token_ids"], max_tokens=case["output_tokens"],
                   temperature=0, seed=seed, n=1, stream=True, echo=False,
                   stream_options={"include_usage": True},
                   ignore_eos=case["ignore_eos"], add_special_tokens=False)
    result = dict(request_id=request_id, endpoint=endpoint, prompt_sha256=prompt["sha256"],
                  expected_prompt_tokens=prompt["input_tokens"], requested_completion_tokens=case["output_tokens"],
                  ignore_eos=case["ignore_eos"], seed=seed, started_at=utc_now(),
                  ok=False, usage=None, ttft_s=None, e2e_s=None, tpot_s=None,
                  finish_reason=None, text_excerpt="", text_characters=0, sse_text_events=0,
                  done_received=False, http_status=None, error=None)
    start = time.perf_counter()
    text_parts = []
    try:
        async with session.post(endpoint, json=payload, allow_redirects=False) as response:
            result["http_status"] = response.status
            if response.status != 200:
                details = (await response.content.read(8192)).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status}: {details}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("Response is not text/event-stream.")
            async for event in sse_events(response.content):
                now = time.perf_counter()
                if event.strip() == "[DONE]":
                    result["done_received"] = True
                    result["e2e_s"] = now - start
                    break
                chunk = json.loads(event)
                if not isinstance(chunk, dict):
                    raise ValueError("Non-object SSE payload.")
                if chunk.get("error"):
                    raise RuntimeError(f"Streaming error: {chunk['error']}")
                if chunk.get("usage") is not None:
                    result["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("Unexpected extra completion choice.")
                    text = choice.get("text") or ""
                    if not isinstance(text, str):
                        raise ValueError("Completion text is not a string.")
                    if text:
                        if result["ttft_s"] is None:
                            result["ttft_s"] = now - start
                        text_parts.append(text)
                        result["sse_text_events"] += 1
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
            if not result["done_received"]:
                raise ValueError("Stream ended without [DONE].")
            usage = result["usage"]
            if not isinstance(usage, dict):
                raise ValueError("Missing actual usage; token counts will not be inferred from chunks.")
            prompt_count, output_count = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if type(prompt_count) is not int or type(output_count) is not int or output_count < 1:
                raise ValueError("Invalid actual prompt/completion token usage.")
            if prompt_count != prompt["input_tokens"]:
                raise ValueError(f"Prompt usage mismatch: expected {prompt['input_tokens']}, got {prompt_count}.")
            if case["ignore_eos"] and output_count != case["output_tokens"]:
                raise ValueError(f"Output usage mismatch: expected {case['output_tokens']}, got {output_count}.")
            if result["ttft_s"] is None:
                raise ValueError("No nonempty completion text received.")
            if output_count > 1:
                result["tpot_s"] = (result["e2e_s"] - result["ttft_s"]) / (output_count - 1)
            result["ok"] = True
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc) or type(exc).__name__)
    finally:
        result["elapsed_until_return_s"] = time.perf_counter() - start
        text = "".join(text_parts)
        result["text_excerpt"] = text[:600]
        result["text_characters"] = len(text)
        result["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(results: list[dict], wall_s: float) -> dict:
    successful = [r for r in results if r["ok"]]
    output_tokens = sum(r["usage"]["completion_tokens"] for r in successful)
    input_tokens = sum(r["usage"]["prompt_tokens"] for r in successful)
    summary = dict(requests=len(results), successful=len(successful), failed=len(results) - len(successful),
                   all_successful=len(successful) == len(results), wall_s=wall_s,
                   successful_input_tokens=input_tokens, successful_output_tokens=output_tokens,
                   successful_requests_per_s=len(successful) / wall_s if wall_s else None,
                   output_tokens_per_s=output_tokens / wall_s if wall_s else None,
                   input_tokens_per_s=input_tokens / wall_s if wall_s else None)
    for metric in ("ttft_s", "e2e_s", "tpot_s"):
        values = [r[metric] for r in successful if r[metric] is not None]
        summary[metric] = dict(n=len(values), p50=percentile(values, 0.5), p95=percentile(values, 0.95))
    return summary


async def run_batch(session, args, report, banks, case, phase: str, round_index: int) -> dict:
    semaphore = asyncio.Semaphore(case["concurrency"])
    batch_start = time.perf_counter()
    async def run(index: int, prompt_hash: str):
        queued = time.perf_counter()
        async with semaphore:
            queue_s = time.perf_counter() - queued
            result = await request_one(
                session, args.base_url[index % len(args.base_url)], args.model, case,
                report["prompts"][prompt_hash], (args.seed + index) % (2 ** 31),
                f"r{round_index}:{case['name']}:{phase}:{index}",
            )
            result["client_queue_s"] = queue_s
            return result
    results = await asyncio.gather(*(run(i, h) for i, h in enumerate(banks[(case["name"], phase)])))
    wall_s = time.perf_counter() - batch_start
    return dict(phase=phase, summary=summarize(results, wall_s), requests=results)


def save_report(path: Path, report: dict) -> None:
    """Atomically update only the output this process previously reserved."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def benchmark(args, report, banks) -> bool:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY, limit_per_host=MAX_CONCURRENCY,
                                     force_close=args.fresh_connections)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        for round_index in range(1, args.rounds + 1):
            for case in report["case_specs"]:
                run = dict(round=round_index, case=case["name"], started_at=utc_now())
                report["runs"].append(run)
                print(f"round={round_index} case={case['name']} phase=warmup", flush=True)
                run["warmup"] = await run_batch(session, args, report, banks, case, "warmup", round_index)
                save_report(args.output, report)
                if not run["warmup"]["summary"]["all_successful"]:
                    run["status"] = "FAILED_WARMUP"
                    return False
                print(f"round={round_index} case={case['name']} phase=measure", flush=True)
                run["measure"] = await run_batch(session, args, report, banks, case, "measure", round_index)
                run["status"] = "PASSED" if run["measure"]["summary"]["all_successful"] else "FAILED"
                run["finished_at"] = utc_now()
                save_report(args.output, report)
                print(json.dumps(dict(case=case["name"], round=round_index,
                                      summary=run["measure"]["summary"])), flush=True)
                if run["status"] != "PASSED":
                    return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', type=endpoints, default='http://127.0.0.1:8000/v1', help='One URL or two comma-separated local replica URLs; deterministic round-robin routing.')
    parser.add_argument('--output', type=Path, required=True, help='New JSON result file; existing files are never overwritten.')
    parser.add_argument('--model', default='dsv4')
    parser.add_argument('--suite', choices=('smoke', 'screen', 'full', 'long'), default='screen')
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--rounds', type=int, default=None, help='Default: 2 for full, 1 otherwise; range 1–10.')
    parser.add_argument('--tokenizer-path', type=Path, default=Path(str(model_path())))
    parser.add_argument('--prepare-only', action='store_true', help='Create CPU-tokenized workload manifest without HTTP requests.')
    parser.add_argument('--fresh-connections', action='store_true', help='Use a new TCP connection for every request; no automatic retries. Record as a separate transport protocol.')
    args = parser.parse_args()
    args.rounds = args.rounds if args.rounds is not None else 2 if args.suite == 'full' else 1
    if not 1 <= args.rounds <= 10:
        parser.error('--rounds must be in 1–10')
    if not 0 <= args.seed < 2 ** 31:
        parser.error('--seed must be in 0–2147483647')
    args.output = args.output.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_output(args.output, 'x', encoding='utf-8') as f:
            json.dump(dict(status='INITIALIZING', created_at=utc_now()), f)
            f.write('\n')
    except FileExistsError:
        print(f'Refusing to overwrite existing result: {args.output}', file=sys.stderr)
        return 2
    report = dict(status='FAILED', created_at=utc_now())
    try:
        report, banks = prepare(args)
        save_report(args.output, report)
        if args.prepare_only:
            print(f'Prepared {len(report['prompts'])} unique exact-token prompts: {args.output}')
            return 0
        report['status'] = 'RUNNING'
        save_report(args.output, report)
        passed = asyncio.run(benchmark(args, report, banks))
        report['status'] = 'PASSED' if passed else 'FAILED'
        report['finished_at'] = utc_now()
        save_report(args.output, report)
        return 0 if passed else 1
    except KeyboardInterrupt:
        report['status'] = 'INTERRUPTED'
        report['finished_at'] = utc_now()
        save_report(args.output, report)
        return 130
    except Exception as exc:
        report['status'] = 'FAILED'
        report['fatal_error'] = dict(type=type(exc).__name__, message=str(exc))
        report['finished_at'] = utc_now()
        save_report(args.output, report)
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
