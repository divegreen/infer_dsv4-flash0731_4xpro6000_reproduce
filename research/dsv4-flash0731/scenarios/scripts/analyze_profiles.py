"""Read bounded Kineto traces; validate the expected s06 manifest.

Kernel durations can overlap, so their sum is never reported as wall time.
Name-based categories are heuristic; raw names and per-rank results remain.
The default manifest is the three-case, four-rank TP4 profile used by s06.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_CASES = (
    "profile_prefill_16k",
    "profile_decode",
    "profile_context_mix",
)
DEFAULT_RANKS = (0, 1, 2, 3)
TRACE_ID = re.compile(
    r"^(?P<case>.+?)_dp(?P<dp>\d+)_pp(?P<pp>\d+)_tp(?P<tp>\d+)"
    r"_dcp(?P<dcp>\d+)_ep(?P<ep>\d+)_rank(?P<rank>\d+)(?:\.[0-9]+)?\.pt\.trace\.json(?:\.gz)?$"
)


def category(name):
    """Assign a descriptive, name-based category.

    GEMV is checked before GEMM, and GEMM before sparse attention, so names
    such as SparseGemm are not classified as attention merely because they
    contain the word sparse.
    """
    lower = name.lower()
    if re.search(r"nccl|all_?reduce|all_?gather|reduce_?scatter", lower):
        return "communication"
    if re.search(r"marlin|moe|expert", lower):
        return "moe_explicit_or_marlin"
    if re.search(r"gemv|gemvx", lower):
        return "gemv_unspecified"
    if re.search(r"gemm|matmul|cutlass", lower):
        return "gemm_unspecified"
    if re.search(r"sparse|attention|mla|flash_fwd", lower):
        return "attention_explicit"
    return "other"


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def union_duration(intervals):
    total = 0.0
    right = None
    for left, end in sorted(intervals):
        if right is None or left >= right:
            total += end - left
        else:
            total += max(0.0, end - right)
        right = end if right is None else max(right, end)
    return total


def trace_identity(path):
    match = TRACE_ID.match(path.name)
    if match is None:
        return None
    values = match.groupdict()
    return {key: int(value) if key != "case" else value for key, value in values.items()}


def cpu_iterations(events):
    """Return de-duplicated host execute_context scopes.

    CPU user annotations carry an External id in these traces. If a trace has
    no id, the exact (name, ts, dur) scope is the conservative fallback. GPU
    annotations are intentionally not used: they are repeated across streams.
    """
    result = []
    seen = set()
    for event in events:
        if (not isinstance(event, dict) or event.get("cat") != "user_annotation"
                or event.get("ph") != "X"):
            continue
        name = str(event.get("name", ""))
        if not name.startswith("execute_context"):
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        external_id = args.get("External id")
        key = ("external", external_id) if external_id is not None else (
            "scope", name, event.get("ts"), event.get("dur"))
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "name": name,
            "ts": event.get("ts"),
            "duration_ms": event.get("dur", 0.0) / 1000.0
            if finite_number(event.get("dur")) else None,
            "external_id": external_id,
        })
    result.sort(key=lambda item: (
        item["ts"] if finite_number(item["ts"]) else float("inf"), item["name"]))
    return result


def inspect(path, expected_identity=None):
    data = path.read_bytes()
    trace = json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)
    if not isinstance(trace, dict):
        raise ValueError("trace root must be an object")
    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")

    kernels = []
    invalid_kernel_events = 0
    non_x_kernel_events = 0
    nonfinite_kernel_events = 0
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "kernel":
            continue
        if event.get("ph") != "X":
            non_x_kernel_events += 1
            invalid_kernel_events += 1
            continue
        ts, duration = event.get("ts"), event.get("dur")
        if (not finite_number(ts) or not finite_number(duration) or duration <= 0):
            nonfinite_kernel_events += 1
            invalid_kernel_events += 1
            continue
        kernels.append(event)

    validation_errors = []
    identity = expected_identity or trace_identity(path)
    distributed = trace.get("distributedInfo")
    if not isinstance(distributed, dict):
        validation_errors.append("missing distributedInfo")
        distributed = {}
    rank = distributed.get("rank")
    world_size = distributed.get("world_size")
    if identity is not None and rank != identity["rank"]:
        validation_errors.append(
            f"distributed rank {rank!r} != filename rank {identity['rank']}")

    names = defaultdict(lambda: dict(calls=0, duration_us=0.0))
    categories = defaultdict(float)
    devices = defaultdict(list)
    for event in kernels:
        name = str(event.get("name", "<unnamed>"))
        names[name]["calls"] += 1
        names[name]["duration_us"] += event["dur"]
        categories[category(name)] += event["dur"]
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        device = args.get("device", event.get("pid"))
        devices[str(device)].append((event["ts"], event["ts"] + event["dur"]))
    total = sum(item["duration_us"] for item in names.values())
    cpu = cpu_iterations(events)
    return dict(
        path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        events=len(events), identity=identity, distributed_info={
            "rank": rank, "world_size": world_size, "backend": distributed.get("backend")},
        invalid_kernel_events=invalid_kernel_events,
        non_x_kernel_events=non_x_kernel_events,
        nonfinite_kernel_events=nonfinite_kernel_events,
        cuda_kernel_events=len(kernels),
        cuda_verified=bool(kernels) and not invalid_kernel_events,
        kernel_duration_sum_ms=total / 1000,
        categories=[dict(category=cat, summed_ms=value / 1000,
                         fraction_of_kernel_sum=value / total if total else None)
                    for cat, value in sorted(categories.items(), key=lambda kv: -kv[1])],
        devices=[dict(
            device=device,
            first_to_last_kernel_ms=(max(end for _, end in intervals)
                                     - min(start for start, _ in intervals)) / 1000,
            kernel_interval_union_ms=union_duration(intervals) / 1000)
                 for device, intervals in devices.items()],
        cpu_iterations=cpu,
        cpu_iteration_count=len(cpu),
        cpu_iteration_counts={
            name: sum(item["name"] == name for item in cpu)
            for name in sorted({item["name"] for item in cpu})
        },
        top_kernels=[dict(
            name=name,
            category=category(name),
            calls=value["calls"],
            summed_ms=value["duration_us"] / 1000,
            mean_us=value["duration_us"] / value["calls"],
            fraction_of_kernel_sum=value["duration_us"] / total)
                     for name, value in sorted(
                         names.items(), key=lambda kv: -kv[1]["duration_us"])[:50]],
        validation_errors=validation_errors,
    )


def validate_manifest(paths, expected_cases=DEFAULT_CASES, expected_ranks=DEFAULT_RANKS):
    expected_cases = tuple(expected_cases)
    expected_ranks = tuple(expected_ranks)
    errors = []
    identities = []
    seen = set()
    for path in paths:
        identity = trace_identity(path)
        if identity is None:
            errors.append(f"filename does not match TP4 trace schema: {path.name}")
            continue
        key = (identity["case"], identity["rank"])
        if key in seen:
            errors.append(f"duplicate case/rank trace: {key}")
        seen.add(key)
        identities.append(identity)
    expected = {(case, rank) for case in expected_cases for rank in expected_ranks}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing:
        errors.append(f"missing case/rank traces: {missing}")
    if unexpected:
        errors.append(f"unexpected case/rank traces: {unexpected}")
    if len(paths) != len(expected):
        errors.append(f"expected {len(expected)} traces, found {len(paths)}")
    return dict(
        expected_cases=list(expected_cases), expected_ranks=list(expected_ranks),
        expected_trace_count=len(expected), found_trace_count=len(paths),
        identities=sorted(identities, key=lambda item: (item["case"], item["rank"])),
        complete=not errors, errors=errors,
    )


def analyze(paths, expected_cases=DEFAULT_CASES, expected_ranks=DEFAULT_RANKS):
    paths = sorted(set(paths))
    manifest = validate_manifest(paths, expected_cases, expected_ranks)
    traces = []
    errors = list(manifest["errors"])
    for path in paths:
        try:
            item = inspect(path, trace_identity(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        traces.append(item)
        errors.extend(f"{path.name}: {error}" for error in item["validation_errors"])
        if item["invalid_kernel_events"]:
            errors.append(f"{path.name}: {item['invalid_kernel_events']} invalid kernel events")
        if not item["cuda_verified"]:
            errors.append(f"{path.name}: no fully valid CUDA kernel set")
    status = "PASS" if paths and manifest["complete"] and not errors else "NOT VERIFIED"
    return dict(
        status=status,
        scope="Independent profiled requests only; never included in normal performance ranking",
        manifest=manifest,
        errors=errors,
        limitations=[
            "Kernel names give heuristic categories, not exact model-layer attribution.",
            "Durations summed across streams/ranks can overlap and are not E2E time.",
            "First-to-last kernel interval excludes leading/trailing capture overhead.",
            "CPU execute_context scopes are de-duplicated by External id; GPU annotations are not used because they repeat across streams.",
            "Limited iterations are not a steady-state utilization or full-request profile.",
        ],
        traces=traces,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-cases', default=','.join(DEFAULT_CASES), help='Comma-separated case names in the required manifest.')
    parser.add_argument('--expected-ranks', default=','.join((str(x) for x in DEFAULT_RANKS)), help='Comma-separated integer ranks in the required manifest.')
    args = parser.parse_args()
    try:
        expected_cases = tuple((x for x in args.expected_cases.split(',') if x))
        expected_ranks = tuple((int(x) for x in args.expected_ranks.split(',') if x))
    except ValueError as exc:
        parser.error(f'invalid expected manifest: {exc}')
    paths = sorted(set(args.directory.rglob('*.pt.trace.json')) | set(args.directory.rglob('*.pt.trace.json.gz')))
    result = analyze(paths, expected_cases, expected_ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.output, json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], traces=len(result['traces']), cuda_events=sum((t['cuda_kernel_events'] for t in result['traces'])), errors=len(result['errors']))))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == "__main__":
    raise SystemExit(main())
