"""CPU-only summaries of saved scenario schema v2; never contacts a model server.

Only PASSED suites/runs and validated formal stages enter comparison tables.
Warmup/profile/failed/incomplete records remain visible in the audit ledger.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import ast
import bisect
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CACHE_PHASES = {"cold", "exact_replay", "fork_first", "warm_serial", "warm_parallel"}
CACHE_ORDER = ["cold", "exact_replay", "fork_first", "warm_serial", "warm_parallel"]
LATENCIES = ("ttft_s", "tpot_s", "e2e_s")
GAUGES = {"vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"}
COUNTERS = {"vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
            "vllm:num_preemptions_total", "vllm:request_success_total",
            "vllm:prompt_tokens_total", "vllm:generation_tokens_total"}
HISTOGRAMS = {"vllm:request_num_preemptions", "vllm:request_prefill_kv_computed_tokens",
              "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds",
              "vllm:request_decode_time_seconds", "vllm:request_inference_time_seconds",
              "vllm:e2e_request_latency_seconds", "vllm:time_to_first_token_seconds",
              "vllm:inter_token_latency_seconds", "vllm:request_prompt_tokens",
              "vllm:request_generation_tokens"}
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+.*)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"(?:,|$)')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stable_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * q
    low, high = math.floor(x), math.ceil(x)
    return a[low] + (a[high] - a[low]) * (x - low)


def distribution(values):
    return {"n": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95),
            "max": max(values) if values else None}


def divide(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def read_json(path, ledger=None):
    raw = path.read_bytes()
    if ledger is not None:
        ledger[str(path.resolve())] = {"sha256": digest(raw), "bytes": len(raw)}
    return json.loads(raw)


def metric_wanted(name):
    return (name in COUNTERS or name in GAUGES
            or (name.startswith("vllm:spec_decode_") and name.endswith("_total"))
            or any(name == h + tail for h in HISTOGRAMS for tail in ("_sum", "_count")))


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.fullmatch(line)
        if not m or not metric_wanted(m[1]):
            continue
        labels = []
        if m[2]:
            label_text = m[2][1:-1]
            position = 0
            for lm in LABEL.finditer(label_text):
                require(lm.start() == position, "Malformed Prometheus labels")
                labels.append((lm[1], json.loads('"' + lm[2] + '"')))
                position = lm.end()
            require(position == len(label_text), "Incomplete Prometheus labels")
        key = (m[1], tuple(sorted(labels)))
        value = float(m[3])
        require(math.isfinite(value), f"Nonfinite metric {key}")
        require(key not in result, f"Duplicate metric {key}")
        result[key] = value
    return result


def metric_derivations(totals):
    def val(name):
        return totals.get("vllm:" + name)
    return {
        "prefix_hit_fraction": divide(val("prefix_cache_hits_total"), val("prefix_cache_queries_total")),
        "spec_accepted_fraction": divide(val("spec_decode_num_accepted_tokens_total"), val("spec_decode_num_draft_tokens_total")),
        "spec_mean_accepted_plus_one": (1 + val("spec_decode_num_accepted_tokens_total") / val("spec_decode_num_drafts_total")
            if val("spec_decode_num_accepted_tokens_total") is not None and (val("spec_decode_num_drafts_total") or 0) > 0 else None),
        "histogram_means": {h: divide(totals.get(h + "_sum"), totals.get(h + "_count")) for h in sorted(HISTOGRAMS)},
    }


def metrics_delta(before, after):
    series, gauges, warnings = [], [], []
    values = defaultdict(list)
    for key in sorted(before.keys() | after.keys()):
        name, labels = key
        a, b = before.get(key), after.get(key)
        item = {"name": name, "labels": dict(labels), "before": a, "after": b}
        if name in GAUGES:
            gauges.append(item)
            continue
        delta = b - a if a is not None and b is not None else None
        if delta is not None and delta < -1e-8:
            warnings.append(f"Counter decreased: {name} {dict(labels)}")
            delta = None
        elif delta is not None:
            delta = max(0., delta)
        if a is None or b is None:
            warnings.append(f"Unmatched metric series: {name} {dict(labels)}")
        item["delta"] = delta
        series.append(item)
        values[name].append(delta)
    totals = {name: sum(v) if all(x is not None for x in v) else None for name, v in values.items()}
    return {"status": "PASS" if not warnings else "INCOMPLETE", "warnings": warnings,
            "series": series, "gauges": gauges, "totals": totals, "derived": metric_derivations(totals)}


def combine_metrics(items):
    names = set().union(*(x["totals"] for x in items))
    totals = {n: sum(x["totals"][n] for x in items)
              if all(x["totals"].get(n) is not None for x in items) else None for n in sorted(names)}
    return {"status": "PASS" if all(x["status"] == "PASS" for x in items) else "INCOMPLETE",
            "totals": totals, "derived": metric_derivations(totals),
            "aggregation": "Sum matched per-stage counter deltas; histogram means=sum(delta_sum)/sum(delta_count); no histogram p95 estimated."}


def load_gpu(path, tz, ledger):
    if not path.exists():
        return {}, ["gpu.csv missing"]
    raw = path.read_bytes()
    ledger[str(path.resolve())] = {"sha256": digest(raw), "bytes": len(raw), "snapshot_may_be_appending": True}
    devices, warnings = defaultdict(list), []
    for number, row in enumerate(csv.reader(io.StringIO(raw.decode())), 1):
        if not row or row[0].startswith("#"):
            continue
        try:
            require(len(row) == 7, "Expected seven GPU CSV columns")
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
            gpu = int(row[1])
            require(gpu in range(4), "GPU outside authorized 0..3")
            values = []
            for x in row[3:]:
                try:
                    v = float(x)
                    values.append(v if math.isfinite(v) and v >= 0 else None)
                except ValueError:
                    values.append(None)
            devices[gpu].append({"t": stamp, "uuid": row[2].strip(),
                                 **dict(zip(("memory_mib", "utilization_pct", "power_w", "temperature_c"), values))})
        except (ValueError, TypeError) as exc:
            warnings.append(f"GPU row {number}: {exc}")
    for gpu, samples in devices.items():
        samples.sort(key=lambda x: x["t"])
        require(len({x["uuid"] for x in samples}) == 1, f"GPU {gpu} UUID changed")
        require(all(a["t"] < b["t"] for a, b in zip(samples, samples[1:])), f"GPU {gpu} duplicate timestamps")
    return dict(devices), warnings


def integrate_gpu(devices, start, end, expected, max_gap):
    duration, per_gpu = end - start, []
    require(duration > 0, "Invalid GPU time window")
    for gpu in expected:
        samples = devices.get(gpu, [])
        times = [x["t"] for x in samples]
        lo = max(0, bisect.bisect_right(times, start) - 1)
        hi = min(len(samples), bisect.bisect_left(times, end) + 1)
        local = samples[lo:hi]
        inside = [x for x in local if start <= x["t"] <= end]
        measures = {}
        max_observed_gap = 0.
        for field in ("memory_mib", "utilization_pct", "power_w", "temperature_c"):
            integral = coverage = 0.
            for a, b in zip(local, local[1:]):
                left, right = max(start, a["t"]), min(end, b["t"])
                if right <= left:
                    continue
                gap = b["t"] - a["t"]
                max_observed_gap = max(max_observed_gap, gap)
                if gap > max_gap or a[field] is None or b[field] is None:
                    continue
                v0 = a[field] + (b[field] - a[field]) * (left - a["t"]) / gap
                v1 = a[field] + (b[field] - a[field]) * (right - a["t"]) / gap
                integral += (v0 + v1) * .5 * (right - left)
                coverage += right - left
            observed = [x[field] for x in inside if x[field] is not None]
            measures[field] = {"integral": integral, "coverage_s": coverage,
                "coverage_fraction": coverage / duration, "time_weighted_mean": divide(integral, coverage),
                "observed_min": min(observed) if observed else None, "observed_max": max(observed) if observed else None}
        per_gpu.append({"gpu": gpu, "uuid": samples[0]["uuid"] if samples else None,
                        "samples_inside": len(inside), "max_overlapping_sample_gap_s": max_observed_gap,
                        "measurements": measures})
    complete = bool(per_gpu) and all(x["measurements"]["power_w"]["coverage_fraction"] >= 1 - 1e-7 for x in per_gpu)
    covered_energy = sum(x["measurements"]["power_w"]["integral"] for x in per_gpu)
    return {"status": "COMPLETE_SAMPLED_WINDOW" if complete else "PARTIAL_OR_MISSING",
            "window_s": duration, "energy_j": covered_energy if complete else None,
            "covered_energy_j": covered_energy, "devices": per_gpu}


def combine_gpu(items):
    duration = sum(x["window_s"] for x in items)
    energy = sum(x["energy_j"] for x in items) if all(x["energy_j"] is not None for x in items) else None
    grouped = defaultdict(list)
    for item in items:
        for device in item["devices"]:
            grouped[device["gpu"]].append(device)
    devices = []
    for gpu, group in sorted(grouped.items()):
        measures = {}
        for field in group[0]["measurements"]:
            rows = [x["measurements"][field] for x in group]
            coverage, integral = sum(x["coverage_s"] for x in rows), sum(x["integral"] for x in rows)
            lows = [x["observed_min"] for x in rows if x["observed_min"] is not None]
            highs = [x["observed_max"] for x in rows if x["observed_max"] is not None]
            measures[field] = {"integral": integral, "coverage_s": coverage, "coverage_fraction": coverage / duration,
                "time_weighted_mean": divide(integral, coverage), "observed_min": min(lows) if lows else None,
                "observed_max": max(highs) if highs else None}
        devices.append({"gpu": gpu, "samples_inside": sum(x["samples_inside"] for x in group),
                        "max_overlapping_sample_gap_s": max(x["max_overlapping_sample_gap_s"] for x in group), "measurements": measures})
    return {"status": "COMPLETE_SAMPLED_WINDOW" if energy is not None else "PARTIAL_OR_MISSING",
            "window_s": duration, "energy_j": energy, "covered_energy_j": sum(x["covered_energy_j"] for x in items),
            "devices": devices}


def request_summary(rows, wall, poisson):
    success = [r for r in rows if r["ok"]]
    output = sum(r["usage"]["completion_tokens"] for r in success)
    result = {"requests": len(rows), "successful": len(success), "failed": len(rows) - len(success),
        "wall_s": wall, "input_tokens": sum(r["usage"]["prompt_tokens"] for r in success),
        "output_tokens": output, "output_tokens_per_s": divide(output, wall),
        "requests_per_s": divide(len(success), wall),
        "latencies": {k: distribution([r[k] for r in success]) for k in LATENCIES},
        "dispatch_lag_s": distribution([r["dispatch_lag_s"] for r in success]),
        "dispatch_lag_interpretation": "external scheduled-arrival dispatch delay" if poisson else "wait from common batch origin, including semaphore queue; not external arrival latency",
        "scheduled_latencies": None, "exploratory_slo": None}
    if poisson:
        result["scheduled_latencies"] = {k: distribution([r[k] for r in success]) for k in ("scheduled_ttft_s", "scheduled_e2e_s")}
        passed = sum(r["scheduled_ttft_s"] <= 1. and r["tpot_s"] <= .05 and r["scheduled_e2e_s"] <= 10. for r in success)
        result["exploratory_slo"] = {"scheduled_ttft_limit_s": 1., "tpot_limit_s": .05, "scheduled_e2e_limit_s": 10.,
            "passed": passed, "fraction_all_requests": divide(passed, len(rows)), "goodput_requests_per_s": divide(passed, wall),
            "quality_gate": "transport/token-count only; not semantic quality or customer SLA"}
    return result


def validate_stage(report, run, stage, checked_prompts):
    case = next(c for c in report["cases"] if c["name"] == run["case"])
    phase = stage["phase"]
    bank = report["banks"][f'{run["round"]}/{run["case"]}/{phase}']
    rows = stage["requests"]
    require(len(rows) == len(bank) > 0, "Stage count does not match bank")
    expected_n = 1 if phase in ("cold", "exact_replay", "fork_first") else case["n"]
    require(len(rows) == expected_n, "Formal request count does not match case")
    wall = stage["summary"]["wall_s"]
    require(finite(wall) and wall > 0, "Invalid batch wall time")
    require(abs((stage["end_unix"] - stage["start_unix"]) - wall) < .1, "UTC and monotonic durations disagree")
    expected_concurrency = 8 if phase == "warm_parallel" else 1 if phase in ("cold", "exact_replay", "fork_first") else case["c"]
    require(stage["concurrency_limit"] == expected_concurrency, "Concurrency mismatch")
    require(stage.get("offered_rate") == case.get("rate"), "Offered rate mismatch")
    offsets = [r["planned_offset_s"] for r in rows]
    require(all(finite(x) and x >= 0 for x in offsets) and offsets == sorted(offsets), "Invalid arrival offsets")
    require(same(stage["planned_arrival_span_s"], offsets[-1]), "Arrival span mismatch")
    seen = set()
    for i, (row, item) in enumerate(zip(rows, bank)):
        require(row["request_id"] == f'{run["round"]}/{run["case"]}/{phase}/{i}', "Request ID mismatch")
        require(row["request_id"] not in seen, "Duplicate request")
        seen.add(row["request_id"])
        key = item["prompt_sha256"]
        prompt = report["prompts"][key]
        if key not in checked_prompts:
            require(digest(json.dumps(prompt["token_ids"], separators=(",", ":")).encode()) == key, "Prompt hash mismatch")
            require(prompt["sha256"] == key and prompt["input_tokens"] == len(prompt["token_ids"]), "Prompt length mismatch")
            checked_prompts.add(key)
        require(row["prompt_sha256"] == key and row["input_tokens"] == prompt["input_tokens"], "Input identity mismatch")
        require(row["requested_output_tokens"] == item["output_tokens"] and row["seed"] == item["seed"], "Output budget/seed mismatch")
        if not row["ok"]:
            continue
        require(row.get("error") is None and row["http_status"] == 200 and row["done_received"], "False successful protocol status")
        require(bool(row["text"]) and digest(row["text"].encode()) == row["text_sha256"], "Text/hash mismatch")
        usage = row["usage"]
        require(all(type(usage[k]) is int and usage[k] > 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")), "Invalid usage")
        require(usage["prompt_tokens"] == prompt["input_tokens"] and usage["completion_tokens"] == item["output_tokens"], "Actual usage mismatch")
        require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Total usage mismatch")
        require(row["finish_reason"] == "length", "Unexpected controlled-output finish reason")
        for k in (*LATENCIES, "dispatch_lag_s", "scheduled_ttft_s", "scheduled_e2e_s", "elapsed_s", "planned_offset_s"):
            require(finite(row[k]) and row[k] >= 0, f"Invalid latency {k}")
        require(row["ttft_s"] <= row["e2e_s"] <= row["elapsed_s"] + 1e-8, "Invalid request timing order")
        require(same(row["tpot_s"], (row["e2e_s"] - row["ttft_s"]) / (usage["completion_tokens"] - 1)), "TPOT mismatch")
        require(same(row["scheduled_ttft_s"], row["ttft_s"] + row["dispatch_lag_s"]), "Scheduled TTFT mismatch")
        require(same(row["scheduled_e2e_s"], row["e2e_s"] + row["dispatch_lag_s"]), "Scheduled E2E mismatch")
        if not stage.get("offered_rate"):
            require(row["planned_offset_s"] == 0, "Non-Poisson schedule is not zero-offset")
        require(stage["start_unix"] - .01 <= row["start_unix"] <= stage["end_unix"] + .01, "Request outside stage")
        require(row["start_unix"] + row["elapsed_s"] <= stage["end_unix"] + .1, "Request finishes after stage boundary")
        expected_send = stage["start_unix"] + row["planned_offset_s"] + row["dispatch_lag_s"]
        require(abs(row["start_unix"] - expected_send) < .1, "Actual dispatch disagrees with planned arrival and lag")
    raw = stage["summary"]
    computed = request_summary(rows, wall, bool(stage.get("offered_rate")))
    for key in ("requests", "successful", "failed", "output_tokens"):
        require(raw[key] == computed[key], f"Recorded summary {key} mismatch")
    require(same(raw["output_tokens_per_s"], computed["output_tokens_per_s"]), "Recorded throughput mismatch")
    for key in LATENCIES:
        for quantile in ("p50", "p95", "max"):
            require(raw[key][quantile] == computed["latencies"][key][quantile] or same(raw[key][quantile], computed["latencies"][key][quantile]), f"Recorded {key} {quantile} mismatch")
    fingerprint = stable_hash({"case": case, "phase": phase, "round": run["round"], "bank": bank,
                               "concurrency": stage["concurrency_limit"], "planned_offsets": [r["planned_offset_s"] for r in rows],
                               "protocol": report["protocol"], "tokenizer_sha256": report["tokenizer_sha256"]})
    return computed, fingerprint


def non_profile_ast_hash(source):
    """Compare executable AST excluding only workloads(suite=='profile') branch.

    This permits the documented profile-only case addition without ignoring
    request, metric, timing, sampling or non-profile workload code changes.
    """
    tree = ast.parse(source)
    removed = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "workloads":
            kept = []
            for statement in node.body:
                test = statement.test if isinstance(statement, ast.If) else None
                profile_test = (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "suite" and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "profile" and not statement.orelse)
                if profile_test:
                    removed += 1
                else:
                    kept.append(statement)
            node.body = kept
    require(removed <= 1, "Unexpected multiple profile workload branches")
    return digest(ast.dump(tree, include_attributes=False).encode()), removed


def source_audit(directory, command, report):
    key = "scenarios/scripts/scenario_client.py"
    expected = command["source_sha256"][key]
    require(report["client_sha256"] == expected, "Report client SHA differs from command record")
    with zipfile.ZipFile(directory / "source-snapshot.zip") as archive:
        source = archive.read(key)
        require(digest(source) == expected, "Client source snapshot mismatch")
    ast_hash, removed = non_profile_ast_hash(source)
    return expected, ast_hash, removed


def collect(results, selected, gpu_timezone, max_gap):
    ledger, rounds, skipped, errors, experiments, warnings = {}, [], [], [], [], []
    accumulation = defaultdict(list)
    matches = defaultdict(dict)
    found_directories = {p.name for p in results.iterdir() if p.is_dir()}
    for name in sorted(selected - found_directories):
        errors.append({"experiment_id": name, "error": "Requested experiment directory missing"})
    for directory in sorted(p for p in results.iterdir() if p.is_dir() and (not selected or p.name in selected)):
        exp_record = {"experiment_id": directory.name, "directory": str(directory.resolve())}
        experiments.append(exp_record)
        try:
            command = read_json(directory / "command.json", ledger)
            archive_path = directory / "source-snapshot.zip"
            archive_bytes = archive_path.read_bytes()
            ledger[str(archive_path.resolve())] = {"sha256": digest(archive_bytes), "bytes": len(archive_bytes)}
            config = read_json(directory / "config.json", ledger)
            experiment = read_json(directory / "experiment.json", ledger)
            status = read_json(directory / "status.json", ledger) if (directory / "status.json").exists() else None
            final = bool(status and status.get("exit_code") == 0 and not status.get("timed_out") and status.get("supervisor_error") is None and experiment.get("status") == "PASS")
            exp_record.update(config=config, git_commit=command.get("git_commit"), launcher_status=experiment.get("status"),
                              supervisor_status=status, final_execution_verified=final)
            gpus = [int(x) for x in command["environment"]["CUDA_VISIBLE_DEVICES"].split(",")]
            require(set(gpus) <= set(range(4)), "GPU mask outside 0..3")
            devices, gpu_warnings = load_gpu(directory / "gpu.csv", ZoneInfo(gpu_timezone), ledger)
            exp_record["gpu_warnings"] = gpu_warnings
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            errors.append({"experiment": directory.name, "error": repr(exc)})
            continue
        for report_path in sorted(directory.glob("scenario-*.json")):
            report_error_start = len(errors)
            try:
                report = read_json(report_path, ledger)
                suite = report["suite"]
                require(report["schema_version"] == 2, "Unsupported report schema")
                require(suite in ("decode", "context", "arrival", "cache", "profile"), "Unknown scenario suite")
                client_sha, client_ast_sha, profile_branches_removed = source_audit(directory, command, report)
                checked_prompts, seen, prior_prompt_cases = set(), set(), defaultdict(set)
                if report["status"] == "PASSED":
                    wanted = [(r, c["name"]) for r in range(1, report["rounds"] + 1) for c in report["cases"]]
                    require([(r["round"], r["case"]) for r in report["runs"]] == wanted, "PASSED report has incomplete or reordered rounds/cases")
                for run in report["runs"]:
                    run_key = (run["round"], run["case"])
                    require(run_key not in seen, "Duplicate round/case")
                    seen.add(run_key)
                    phase_names = [x["phase"] for x in run["stages"]]
                    require(len(phase_names) == len(set(phase_names)), "Duplicate stage phase")
                    if run["status"] == "PASSED":
                        require(phase_names == (CACHE_ORDER if suite == "cache" else ["warmup", "measure"]), "PASSED run missing or reordered phases")
                        require(all(a["end_unix"] <= b["start_unix"] + .01 for a, b in zip(run["stages"], run["stages"][1:])), "Stage time windows overlap or are reordered")
                    if suite == "cache":
                        prior_prompt_cases.clear()  # The client resets once per cache case/round.
                    for stage in run["stages"]:
                        phase = stage["phase"]
                        identity = {"experiment_id": directory.name, "config": config["name"], "suite": suite,
                                    "case": run["case"], "round": run["round"], "phase": phase}
                        earlier_cases = set()
                        shared_count = 0
                        for request in stage.get("requests", []):
                            previous = prior_prompt_cases[request["prompt_sha256"]] - {run["case"]}
                            if previous:
                                shared_count += 1
                                earlier_cases.update(previous)
                            prior_prompt_cases[request["prompt_sha256"]].add(run["case"])
                        reason = ("profile excluded" if suite == "profile" or experiment.get("profile") else
                                  "warmup excluded" if phase == "warmup" else
                                  "suite/run not PASSED" if report["status"] != "PASSED" or run["status"] != "PASSED" else None)
                        if reason:
                            skipped.append({**identity, "reason": reason, "recorded_summary": stage.get("summary"),
                                "failures": [{"request_id": r.get("request_id"), "error": r.get("error"), "http_status": r.get("http_status")} for r in stage.get("requests", []) if not r.get("ok")]})
                            continue
                        try:
                            require(phase in (CACHE_PHASES if suite == "cache" else {"measure"}), "Unknown formal phase")
                            summary, fingerprint = validate_stage(report, run, stage, checked_prompts)
                            require(summary["failed"] == 0, "PASSED formal stage contains failed requests")
                            if suite == "cache":
                                require(run["reset"]["status"] == 200 and json.loads(run["reset"]["body"])["success"] is True, "Cache reset not verified")
                            before_path, after_path = directory / stage["metrics_before"], directory / stage["metrics_after"]
                            for path in (before_path, after_path):
                                require(path.resolve().parent == directory.resolve(), "Metric path outside run directory")
                            metric_inputs = []
                            for path in (before_path, after_path):
                                raw = path.read_bytes(); ledger[str(path.resolve())] = {"sha256": digest(raw), "bytes": len(raw)}
                                metric_inputs.append(parse_metrics(raw.decode()))
                            metrics = metrics_delta(*metric_inputs)
                            metric_checks = {}
                            for name, raw_count in (("vllm:prompt_tokens_total", summary["input_tokens"]),
                                                    ("vllm:generation_tokens_total", summary["output_tokens"]),
                                                    ("vllm:request_success_total", summary["successful"])):
                                measured = metrics["totals"].get(name)
                                matched = measured is not None and same(measured, raw_count)
                                metric_checks[name] = {"delta": measured, "raw_requests_total": raw_count, "matched": matched}
                                if not matched:
                                    metrics["warnings"].append(f"Snapshot delta differs from raw requests: {name}; asynchronous metric boundaries may differ")
                                    metrics["status"] = "INCOMPLETE"
                            metrics["raw_request_crosscheck"] = metric_checks
                            warnings.extend({**identity, "warning": value} for value in metrics["warnings"])
                            gpu = integrate_gpu(devices, stage["start_unix"], stage["end_unix"], gpus, max_gap)
                            gpu["output_tokens_per_joule"] = divide(summary["output_tokens"], gpu["energy_j"])
                            gpu["joules_per_output_token"] = divide(gpu["energy_j"], summary["output_tokens"])
                            cache_enabled = False if "--no-enable-prefix-caching" in config["args"] else True if "--enable-prefix-caching" in config["args"] else None
                            history = {"same_prompt_requests_seen_in_earlier_cases": shared_count, "earlier_cases": sorted(earlier_cases),
                                       "apc_explicitly_enabled": cache_enabled,
                                       "caveat": "No reset between arrival cases; later offered-rate case may inherit exact-prompt cache when APC is enabled." if shared_count else None}
                            if shared_count:
                                warnings.append({**identity, "warning": history["caveat"], "history": history})
                            row = {**identity, "final_execution_verified": final, "client_sha256": client_sha,
                                "non_profile_client_ast_sha256": client_ast_sha, "profile_workload_branches_excluded_from_ast_hash": profile_branches_removed,
                                "comparison_cohort": stable_hash({"client_ast_sha256": client_ast_sha, "tokenizer_sha256": report["tokenizer_sha256"], "protocol": report["protocol"]})[:16],
                                "tokenizer_sha256": report["tokenizer_sha256"], "input_fingerprint": fingerprint,
                                "declared_rounds": report["rounds"], "selected_gpus": gpus,
                                "source_report": str(report_path.resolve()), "start_unix": stage["start_unix"], "end_unix": stage["end_unix"],
                                "offered_rate": stage.get("offered_rate"), "concurrency_limit": stage["concurrency_limit"],
                                "planned_arrival_span_s": stage["planned_arrival_span_s"], "summary": summary,
                                "planned_sample_arrival_rate": divide(len(stage["requests"]) - 1, stage["planned_arrival_span_s"]) if stage.get("offered_rate") else None,
                                "actual_input_output_shape_counts": dict(Counter(f'{r["usage"]["prompt_tokens"]}/{r["usage"]["completion_tokens"]}' for r in stage["requests"] if r["ok"])),
                                "cache_history": history, "metrics": metrics, "gpu": gpu}
                            rounds.append(row)
                            accumulation[(directory.name, suite, run["case"], phase)].append((row, stage["requests"]))
                        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                            errors.append({**identity, "error": repr(exc)})
            except (OSError, ValueError, KeyError, TypeError, SyntaxError, zipfile.BadZipFile) as exc:
                errors.append({"experiment_id": directory.name, "report": str(report_path), "error": repr(exc)})
            if len(errors) > report_error_start:
                # Prevent a good surviving round of an invalid suite from
                # appearing as a complete/final multi-round comparison.
                rejected = [r for r in rounds if r["source_report"] == str(report_path.resolve())]
                rounds[:] = [r for r in rounds if r["source_report"] != str(report_path.resolve())]
                for r in rejected:
                    skipped.append({k: r[k] for k in ("experiment_id", "config", "suite", "case", "round", "phase")}
                                   | {"reason": "whole suite quarantined after validation error", "recorded_summary": r["summary"], "failures": []})
                for key in list(accumulation):
                    accumulation[key] = [x for x in accumulation[key] if x[0]["source_report"] != str(report_path.resolve())]
                    if not accumulation[key]:
                        del accumulation[key]
    for r in rounds:
        matches[(r["suite"], r["case"], r["phase"], r["round"])][r["experiment_id"]] = {
            "input_fingerprint": r["input_fingerprint"], "client_sha256": r["client_sha256"],
            "tokenizer_sha256": r["tokenizer_sha256"], "non_profile_client_ast_sha256": r["non_profile_client_ast_sha256"],
            "comparison_cohort": r["comparison_cohort"]}
    fairness = []
    for key, values in sorted(matches.items()):
        cohort_match = len({v["comparison_cohort"] for v in values.values()}) == 1
        input_match = len({v["input_fingerprint"] for v in values.values()}) == 1
        sha_match = len({v["client_sha256"] for v in values.values()}) == 1
        state = ("SINGLE_CONFIGURATION_NOT_CROSS_VERIFIED" if len(values) < 2 else
                 "UNCOMPARABLE_COHORTS" if not cohort_match else
                 "INPUT_MISMATCH" if not input_match else
                 "MATCHED_INPUTS_AND_SOURCE" if sha_match else "MATCHED_INPUTS_AND_NON_PROFILE_AST")
        item = {"suite": key[0], "case": key[1], "phase": key[2], "round": key[3],
                "experiments": values, "status": state, "matched": None if len(values) < 2 else cohort_match and input_match}
        fairness.append(item)
        if state == "INPUT_MISMATCH":
            errors.append({"error": "Cross-configuration workload mismatch", "comparison": item})
        elif state in ("UNCOMPARABLE_COHORTS", "MATCHED_INPUTS_AND_NON_PROFILE_AST"):
            warnings.append({"warning": state, "comparison": item})
        for r in rounds:
            if (r["suite"], r["case"], r["phase"], r["round"]) == key:
                r["comparison_status"] = state
    combined = []
    for _, items in sorted(accumulation.items()):
        rr = [x[0] for x in items]
        rows = [request for _, requests in items for request in requests]
        common = {k: rr[0][k] for k in ("experiment_id", "config", "suite", "case", "phase", "client_sha256", "non_profile_client_ast_sha256", "tokenizer_sha256", "comparison_cohort", "offered_rate", "concurrency_limit", "selected_gpus")}
        summary = request_summary(rows, sum(x["summary"]["wall_s"] for x in rr), bool(rr[0]["offered_rate"]))
        gpu = combine_gpu([x["gpu"] for x in rr]); gpu["output_tokens_per_joule"] = divide(summary["output_tokens"], gpu["energy_j"])
        gpu["joules_per_output_token"] = divide(gpu["energy_j"], summary["output_tokens"])
        shapes = Counter()
        for item in rr:
            shapes.update(item["actual_input_output_shape_counts"])
        combined.append({**common, "rounds": [x["round"] for x in rr], "final_execution_verified": all(x["final_execution_verified"] for x in rr),
            "coverage": {"expected_rounds": rr[0]["declared_rounds"], "included_rounds": len(rr), "complete": len(rr) == rr[0]["declared_rounds"]},
            "comparison_status": ",".join(sorted({x["comparison_status"] for x in rr})),
            "round_output_tokens_per_s": [x["summary"]["output_tokens_per_s"] for x in rr],
            "input_fingerprints": [x["input_fingerprint"] for x in rr], "summary": summary,
            "actual_input_output_shape_counts": dict(shapes), "cache_history_by_round": [x["cache_history"] for x in rr],
            "planned_sample_arrival_rate": divide(sum(x["summary"]["requests"] - 1 for x in rr), sum(x["planned_arrival_span_s"] for x in rr)) if rr[0]["offered_rate"] else None,
            "metrics": combine_metrics([x["metrics"] for x in rr]), "gpu": gpu})
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": digest(Path(__file__).read_bytes()), "results_directory": str(results.resolve()),
        "definitions": {"selection": "PASSED report + PASSED run + validated formal phase; exclude warmup/profile. Unfinished overall experiments remain explicitly provisional.",
            "aggregation": "Successful actual completion tokens / sum(batch wall_s); latency percentiles recomputed from pooled successful raw requests, never mean of percentiles.",
            "poisson": "Only offered-rate measure stages use scheduled latencies and exploratory goodput; closed-loop dispatch lag starts at common batch origin and is not exogenous arrival delay.",
            "metrics": "Exact matched-label before/after counter deltas; missing/decreasing series are null. Histogram means use sum/count; no bucket-derived exact p95.",
            "gpu": f"Per-GPU timestamps in {gpu_timezone}; boundary-clipped linear interpolation and trapezoidal integration; gaps > {max_gap}s skipped, no extrapolation. Energy is sampled whole-board power incl. idle/queue; excludes CPU/system. Tokens/J only for full window coverage on all selected GPUs.",
            "gpu_uncertainty": "Requested 1-second sampling is not exact cadence. True between-sample peaks and short-window energy are unresolved; boundary interpolation, timestamp alignment, and workload transitions introduce unquantified error. Observed memory/utilization maxima are sampled, not hardware peaks.",
            "semantic_quality": "Controlled ignore_eos performance workload; no semantic accuracy or production SLA inferred."},
        "experiments": experiments, "rounds": rounds, "combined": combined,
        "audit": {"status": "PASS" if not errors else "FAILED", "errors": errors, "warnings": warnings, "excluded_stages": skipped,
                  "cross_configuration_inputs": fairness, "source_files": ledger}}


def flatten(row):
    s = row["summary"]
    output = {k: row[k] for k in ("experiment_id", "config", "suite", "case", "phase", "final_execution_verified", "comparison_cohort", "comparison_status", "client_sha256", "non_profile_client_ast_sha256", "tokenizer_sha256", "offered_rate", "concurrency_limit")}
    output["selected_gpus"] = ",".join(map(str, row["selected_gpus"]))
    output["metrics_status"] = row["metrics"]["status"]
    output["round"] = row.get("round", ",".join(map(str, row.get("rounds", []))))
    output.update({k: s[k] for k in ("requests", "successful", "failed", "wall_s", "input_tokens", "output_tokens", "output_tokens_per_s", "requests_per_s")})
    for key, stats in s["latencies"].items():
        output.update({key + "_" + q: stats[q] for q in ("p50", "p95", "max")})
    output.update({"dispatch_lag_s_" + q: s["dispatch_lag_s"][q] for q in ("p50", "p95", "max")})
    for k in ("scheduled_ttft_s", "scheduled_e2e_s"):
        output.update({k + "_" + q: s["scheduled_latencies"][k][q] if s["scheduled_latencies"] else None for q in ("p50", "p95", "max")})
    output["example_slo_pass"] = s["exploratory_slo"]["passed"] if s["exploratory_slo"] else None
    output["example_goodput_rps"] = s["exploratory_slo"]["goodput_requests_per_s"] if s["exploratory_slo"] else None
    output["planned_sample_arrival_rate"] = row.get("planned_sample_arrival_rate")
    output["actual_input_output_shape_counts"] = json.dumps(row["actual_input_output_shape_counts"], sort_keys=True)
    output.update({"gpu_energy_j": row["gpu"]["energy_j"], "output_tokens_per_joule": row["gpu"]["output_tokens_per_joule"],
                   "joules_per_output_token": row["gpu"]["joules_per_output_token"], "gpu_coverage_status": row["gpu"]["status"]})
    for gpu in row["gpu"]["devices"]:
        prefix = f'gpu{gpu["gpu"]}_'
        output[prefix + "samples_inside"] = gpu["samples_inside"]
        output[prefix + "max_sample_gap_s"] = gpu["max_overlapping_sample_gap_s"]
        for field, value in gpu["measurements"].items():
            output[prefix + field + "_mean"] = value["time_weighted_mean"]
            output[prefix + field + "_sampled_max"] = value["observed_max"]
            output[prefix + field + "_coverage_fraction"] = value["coverage_fraction"]
    for name in sorted(COUNTERS | {"vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"}):
        output[name + "_delta"] = row["metrics"]["totals"].get(name)
    for name in sorted(HISTOGRAMS):
        output[name + "_sum_delta"] = row["metrics"]["totals"].get(name + "_sum")
        output[name + "_count_delta"] = row["metrics"]["totals"].get(name + "_count")
        output[name + "_mean"] = row["metrics"]["derived"]["histogram_means"][name]
    output["prefix_hit_fraction"] = row["metrics"]["derived"]["prefix_hit_fraction"]
    return output


def markdown(report):
    lines = ["# 场景化实验 CPU 汇总", "", f'生成时间：{report["created_utc"]}。审计：**{report["audit"]["status"]}**。', "",
             "仅下表已通过的正式 stage 进入比较；warmup、profile、失败/未完成 suite 在 JSON audit 中保留。整体实验尚未退出成功的已完成 suite 标记为 provisional。", "",
             "吞吐=各轮实际成功输出 token 总和 / 各轮 wall 总和；p50/p95 合并逐请求计算。表中延迟单位秒。缓存各 phase 独立，不跨 phase 合并。comparison_cohort见CSV/JSON；不同cohort不可直接混排行。client原始hash差异始终保留；仅profile workload分支变化可由其余源码AST一致单独验证，不静默忽略。", "",
             "| 实验 | suite / case / phase | 轮 | N / 成功 / 失败 | tok/s | TTFT p50 / p95 | TPOT p50 / p95 | E2E p50 / p95 | 状态 / 比较证据 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    def f(x):
        return "—" if x is None else f"{x:.4f}"
    for r in report["combined"]:
        s, lat = r["summary"], r["summary"]["latencies"]
        pairs = [f'{f(lat[k]["p50"])} / {f(lat[k]["p95"])}' for k in LATENCIES]
        lines.append(f'| {r["experiment_id"]} | {r["suite"]} / {r["case"]} / {r["phase"]} | {r["rounds"]} | {s["requests"]}/{s["successful"]}/{s["failed"]} | {f(s["output_tokens_per_s"])} | ' + " | ".join(pairs) + f' | {"final" if r["final_execution_verified"] else "provisional"} / {r["comparison_status"]} |')
    lines += ["", "## Poisson 计划到达指标", "", "下表仅对外生 Poisson 到达有意义；固定并发的batch起点排队不作SLO。阈值为scheduled TTFT≤1s、TPOT≤50ms、scheduled E2E≤10s，仅探索性，不含语义质量门槛。", "",
        "| 实验 / case | rate | scheduled TTFT p50 / p95 | scheduled E2E p50 / p95 | dispatch p95 | SLO通过 / N | goodput req/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["combined"]:
        s = r["summary"]
        if s["scheduled_latencies"]:
            a, b = s["scheduled_latencies"]["scheduled_ttft_s"], s["scheduled_latencies"]["scheduled_e2e_s"]
            lines.append(f'| {r["experiment_id"]} / {r["case"]} | {r["offered_rate"]} | {f(a["p50"])} / {f(a["p95"])} | {f(b["p50"])} / {f(b["p95"])} | {f(s["dispatch_lag_s"]["p95"])} | {s["exploratory_slo"]["passed"]}/{s["requests"]} | {f(s["exploratory_slo"]["goodput_requests_per_s"])} |')
    lines += ["", "## 缓存、服务阶段与 GPU 采样", "", "各轮的完整label counter差分、histogram sum/count、均值及每卡memory/util/power见JSON/CSV。这里不从histogram bucket生成精确p95。", "",
        "| 实验 / case / phase | prefix hits / queries | preemptions | queue / prefill / decode均值(s) | 所选GPU采样J | output tok/J | metrics状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in report["combined"]:
        m, means = r["metrics"]["totals"], r["metrics"]["derived"]["histogram_means"]
        timing = " / ".join(f(means["vllm:request_" + x + "_time_seconds"]) for x in ("queue", "prefill", "decode"))
        lines.append(f'| {r["experiment_id"]} / {r["case"]} / {r["phase"]} | {f(m.get("vllm:prefix_cache_hits_total"))} / {f(m.get("vllm:prefix_cache_queries_total"))} | {f(m.get("vllm:num_preemptions_total"))} | {timing} | {f(r["gpu"]["energy_j"])} | {f(r["gpu"]["output_tokens_per_joule"])} | {r["metrics"]["status"]} |')
    lines += ["", "GPU能量按每卡实际时间戳、阶段边界插值及梯形积分；超过gap阈值不插值、不外推。所有所选GPU完整覆盖才给tok/J，具体mask见CSV/JSON。整卡功率包含空闲/排队，不含CPU或整机；约1秒采样无法捕捉真实峰值，短阶段和边界误差未量化。", "",
              "注意：arrival 的rate2/rate6在同轮使用相同输入且中间不reset；APC开启时rate6可能继承rate2完整prompt缓存。各stage实际复用情况见cache_history与prefix counter差分，不把差异纯归因到达率。", "",
              f'排除stage数：{len(report["audit"]["excluded_stages"])}；审计错误数：{len(report["audit"]["errors"])}。每轮独立行见 rounds.csv；合并行见 combined.csv；来源哈希、provisional状态及全部排除原因见 summary.json。', ""]
    return "\n".join(lines)


def write_outputs(output, report):
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in (('summary.json', json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'), ('summary.md', markdown(report))):
        tmp = output / (filename + '.tmp')
        write_text(tmp, content)
        tmp.replace(output / filename)
    for name in ('rounds', 'combined'):
        rows = [flatten(x) for x in report[name]]
        fields = list(dict.fromkeys((k for row in rows for k in row))) if rows else ['experiment_id', 'suite', 'case', 'phase']
        tmp = output / (name + '.csv.tmp')
        with open_output(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(output / (name + '.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=path_at(ROOT, 'results'))
    parser.add_argument('--output', type=Path, default=path_at(ROOT, 'summary'))
    parser.add_argument('--experiment', action='append', default=[])
    parser.add_argument('--gpu-timezone', default='Etc/UTC')
    parser.add_argument('--max-gpu-gap-seconds', type=float, default=3.0)
    args = parser.parse_args()
    require(args.max_gpu_gap_seconds > 0, 'GPU gap threshold must be positive')
    require(args.output.resolve() != args.results.resolve() and args.results.resolve() not in args.output.resolve().parents, 'Summary output must be outside raw results')
    report = collect(args.results, set(args.experiment), args.gpu_timezone, args.max_gpu_gap_seconds)
    write_outputs(args.output, report)
    print(json.dumps({'status': report['audit']['status'], 'formal_stages': len(report['rounds']), 'combined_rows': len(report['combined']), 'excluded_stages': len(report['audit']['excluded_stages']), 'errors': report['audit']['errors'], 'output': str(args.output)}, ensure_ascii=False))
    return 0 if report['audit']['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
