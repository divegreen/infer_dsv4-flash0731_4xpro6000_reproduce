#!/usr/bin/env python3
"""Rebuild measured tables from immutable per-experiment evidence, using stdlib.

No model, torch, GPU, network or subprocess calls. Only summary.csv, summary.json
and comparison.md are replaced. --self-test exercises synthetic CPU fixtures.
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
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_LINE = re.compile(r"\bERROR\b|(?:\w*Error|\w*Exception):|Traceback \(most recent call last\)")
OOM = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError:|CUDA error: out of memory", re.I)
METRICS = ("ttft_s", "tpot_s", "e2e_s")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def unique(items):
    return list(dict.fromkeys(items))


class Evidence:
    """Read each source once; retain its hash and any read/parse uncertainty."""

    def __init__(self, root: Path):
        self.root = root
        native_root = Path(__file__).resolve().parents[1]
        self.key_root = workspace_path() if root.resolve() == native_root else root
        self.sources = {}
        self.warnings = []

    def key(self, path: Path) -> str:
        """Use one relocatable key for native centralized and legacy fixture evidence."""
        return os.path.relpath(path, self.key_root)

    def text(self, path: Path) -> str | None:
        if not path.is_file():
            return None
        relative = self.key(path)
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
        except OSError as error:
            self.warnings.append(f"{relative}: read failed: {error}")
            return None
        changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        self.sources[relative] = dict(sha256=sha256(raw), bytes_read=len(raw),
                                      changed_during_read=changed)
        if changed:
            self.warnings.append(f"{relative}: changed during read; regenerate after experiment finishes")
        return raw.decode("utf-8", errors="replace")

    def json(self, path: Path, default=None):
        text = self.text(path)
        if text is None:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as error:
            self.warnings.append(f"{self.key(path)}: invalid/incomplete JSON: {error}")
            return default


def parse_gpu(text: str | None) -> dict:
    records = {}
    malformed = 0
    for values in csv.reader(io.StringIO(text or "")):
        if not values or values[0].startswith("#"):
            continue
        if len(values) != 7:
            malformed += 1
            continue
        timestamp, index, uuid, memory, utilization, power, temperature = [v.strip() for v in values]
        try:
            index = int(index)
            memory = float(memory)
            if not math.isfinite(memory):
                raise ValueError("nonfinite memory")
        except ValueError:
            malformed += 1
            continue
        row = records.setdefault(str(index), dict(uuid=uuid, samples=0,
                                 memory_sampled_max_mib=0, first_timestamp=timestamp))
        row["samples"] += 1
        row["last_timestamp"] = timestamp
        row["memory_sampled_max_mib"] = max(row["memory_sampled_max_mib"], memory)
        for label, raw in (("utilization_sampled_max_pct", utilization),
                           ("power_sampled_max_w", power), ("temperature_sampled_max_c", temperature)):
            try:
                number = finite(float(raw))
            except ValueError:
                number = None
            if number is not None:
                row[label] = max(row.get(label, number), number)
    maxima = [row["memory_sampled_max_mib"] for row in records.values()]
    return dict(per_gpu=records, memory_sampled_max_mib=max(maxima) if maxima else None,
                malformed_rows=malformed, sampling_interval_note="Nominal 3 seconds plus nvidia-smi overhead",
                scope="Whole experiment, including startup/warmup; sampled maximum is NOT an instantaneous peak")


def log_actor(line: str) -> str:
    match = re.search(r"\(([^()]*?)\s+pid=(\d+)\)", line)
    return f"{match[1]} pid={match[2]}" if match else "unlabelled"


def parse_logs(server: str | None, process: str | None) -> dict:
    models, kv, available, graphs, errors = [], [], [], [], []
    oom = False
    transport = []
    for source, text in (("server.log", server), ("process.log", process)):
        for lineno, raw in enumerate((text or "").splitlines(), 1):
            line = ANSI.sub("", raw)
            actor = log_actor(line)
            context = dict(source=source, line=lineno, actor=actor)
            match = re.search(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", line)
            if match:
                models.append(dict(context, model_gib=float(match[1]), loading_seconds=float(match[2])))
            match = re.search(r"GPU KV cache size:\s*([\d,]+) tokens", line)
            if match:
                kv.append(dict(context, tokens=int(match[1].replace(",", ""))))
            match = re.search(r"Available KV cache memory:\s*([\d.]+) GiB", line)
            if match:
                available.append(dict(context, gib=float(match[1])))
            match = re.search(r"CUDA graph pool memory:\s*([\d.]+) GiB \(actual\)", line)
            if match:
                graphs.append(dict(context, gib=float(match[1])))
            if ERROR_LINE.search(line):
                errors.append(dict(context, text=line[:1500]))
            if OOM.search(line):
                oom = True
            if "NCCL" in line:
                transport.extend(re.findall(r"(?:via|using)\s+((?:SHM|P2P|NET)/[^\s,]+)", line))
    distinct_errors = {}
    for entry in errors:
        distinct_errors[(entry["source"], entry["text"])] = entry
    return dict(model_loading=models, kv_cache_tokens=kv, available_kv_memory=available,
                cuda_graph_actual_memory=graphs, oom_detected=oom,
                error_line_count=len(errors), errors=list(distinct_errors.values())[-16:],
                nccl_transports=unique(transport),
                model_memory_note="vLLM per-worker loading log; not total process memory",
                kv_note="Logged cache capacity per engine/line; never blindly summed across TP workers")


def lifecycle(status: dict, experiment: dict) -> str:
    if status.get("timed_out"):
        return "TIMED OUT"
    if status:
        if status.get("exit_code") == 0:
            return "COMPLETED" if experiment.get("status") != "FAILED" else "INCONSISTENT STATUS"
        return "FAILED"
    if experiment.get("status") == "FAILED":
        return "FAILED (supervisor status unavailable)"
    return "INCOMPLETE OR RUNNING (no final supervisor status)"


def request_errors(phase) -> list:
    return [dict(request_id=row.get("request_id"), error=row.get("error"),
                 http_status=row.get("http_status"))
            for row in (phase or {}).get("requests", []) if not row.get("ok")]


def benchmark_rows(evidence: Evidence, folder: Path, common: dict) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("benchmark*.json")):
        report = evidence.json(path, {})
        if not isinstance(report, dict):
            evidence.warnings.append(f"{evidence.key(path)}: expected benchmark object")
            continue
        specifications = {c["name"]: c for c in report.get("case_specs", []) if "name" in c}
        semantic = report.get("semantic_validation", {})
        semantic_status = semantic.get("status", "NOT VERIFIED") if isinstance(semantic, dict) else "NOT VERIFIED"
        for run in report.get("runs", []):
            case = specifications.get(run.get("case"), {})
            measure = run.get("measure") or {}
            # Only an explicitly measured phase is allowed to contribute metrics.
            if measure.get("phase", "measure") != "measure":
                evidence.warnings.append(f"{evidence.key(path)}: ignored non-measure phase in {run.get('case')}")
                measure = {}
            summary = measure.get("summary") or {}
            count = summary.get("requests")
            successful = summary.get("successful")
            failed = summary.get("failed")
            transport = "NOT MEASURED"
            if summary:
                transport = "PASS" if count and count == successful and failed == 0 else "FAILED OR PARTIAL"
            elif run.get("status") == "FAILED_WARMUP":
                transport = "WARMUP FAILED; NOT MEASURED"
            reqs = measure.get("requests", [])
            prompt_hashes = [r.get("prompt_sha256") for r in reqs]
            output_hashes = [r.get("text_sha256") for r in reqs]
            row = dict(common, row_type="measurement", benchmark_file=evidence.key(path),
                       suite=report.get("suite"), benchmark_status=report.get("status"),
                       case=run.get("case"), round=run.get("round"), phase="measure",
                       case_status=run.get("status"), transport_status=transport,
                       semantic_status=semantic_status, semantic_validation=semantic,
                       input_tokens=case.get("input_tokens"), output_tokens_requested=case.get("output_tokens"),
                       concurrency=case.get("concurrency"), arrival_pattern=case.get("arrival_pattern", "not recorded"),
                       ignore_eos=case.get("ignore_eos"), requests=count, successful=successful, failed=failed,
                       measured_request_errors=request_errors(measure), warmup_errors=request_errors(run.get("warmup")),
                       benchmark_fatal_error=report.get("fatal_error"),
                       measured_prompt_hashes=prompt_hashes, measured_output_hashes=output_hashes,
                       tokenizer_sha256=report.get("tokenizer", {}).get("sha256"),
                       benchmark_client_sha256=report.get("client", {}).get("sha256"),
                       benchmark_source_sha256=evidence.sources.get(evidence.key(path), {}).get("sha256"),
                       source_metric_definitions=report.get("metric_definitions", {}),
                       source_validation_scope=report.get("validation_scope"))
            for key in ("wall_s", "successful_input_tokens", "successful_output_tokens",
                        "successful_requests_per_s", "output_tokens_per_s", "input_tokens_per_s"):
                row[key] = finite(summary.get(key))
            for metric in METRICS:
                values = summary.get(metric) or {}
                for stat in ("n", "p50", "p95"):
                    row[f"{metric}_{stat}"] = finite(values.get(stat))
            rows.append(row)
        if not report.get("runs") and report:
            rows.append(dict(common, row_type="measurement", benchmark_file=evidence.key(path),
                             suite=report.get("suite"), benchmark_status=report.get("status"),
                             phase="measure", transport_status="NOT MEASURED", semantic_status="NOT VERIFIED",
                             benchmark_fatal_error=report.get("fatal_error")))
    return rows


def collectives(evidence: Evidence, folder: Path, common: dict, process: str | None) -> list[dict]:
    metrics, source = [], None
    for name in ("collectives.json", "collectives-recovered.json"):
        data = evidence.json(folder / name)
        if data is not None:
            metrics = data.get("metrics", []) if isinstance(data, dict) else data
            source = evidence.key(folder / name)
            if isinstance(metrics, list) and metrics:
                break
            metrics = []
    if not metrics:
        source = evidence.key(folder / "process.log")
        for line in (process or "").splitlines():
            # Rank-0 records are complete single-line JSON; never parse a shared total file.
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict) and "operation" in item and "median_us" in item:
                metrics.append(item)
    result = []
    for metric in metrics:
        if not isinstance(metric, dict) or "operation" not in metric:
            continue
        row = dict(common, row_type="collective", collective_source=source, **metric)
        row["collective_source_sha256"] = evidence.sources.get(source, {}).get("sha256")
        trials = [finite(v) for v in row.get("trials_seconds_per_call", [])]
        if trials and all(v is not None for v in trials):
            row["recomputed_median_us"] = statistics.median(trials) * 1e6
            row["median_consistent"] = math.isclose(row["recomputed_median_us"], row.get("median_us", -1), rel_tol=1e-8)
            if not row["median_consistent"]:
                evidence.warnings.append(f"{source}: median inconsistent with trials for {row['operation']}")
        result.append(row)
    return result


def build_summary(root: Path) -> dict:
    evidence = Evidence(root)
    experiments, measured, nccl = ([], [], [])
    results = path_at(root, 'results')
    native_root = Path(__file__).resolve().parents[1]
    allowed = {e['id'] for e in load_document(native_root / 'experiments/catalog.json')['experiments'] if e['topic'] == 'architecture'} if root.resolve() == native_root else None
    for folder in sorted(results.iterdir()) if results.is_dir() else []:
        if allowed is not None and folder.name not in allowed:
            continue
        if not folder.is_dir():
            continue
        if not any(((folder / name).is_file() for name in ('command.json', 'status.json', 'experiment.json'))):
            continue
        command = evidence.json(folder / 'command.json', {}) or {}
        status = evidence.json(folder / 'status.json', {}) or {}
        experiment = evidence.json(folder / 'experiment.json', {}) or {}
        config = evidence.json(folder / 'config.json', {}) or experiment.get('configuration', {})
        server_command = evidence.json(folder / 'server-command.json', {}) or {}
        server = evidence.text(folder / 'server.log') or ''
        for replica_log in sorted(folder.glob('replica-*/server.log')):
            server += '\n' + evidence.text(replica_log)
        process = evidence.text(folder / 'process.log')
        logs = parse_logs(server, process)
        gpu = parse_gpu(evidence.text(folder / 'gpu.csv'))
        provenance = dict(command, **status)
        environment = dict(provenance.get('environment', {}), **server_command.get('extra_environment', {}))
        common = dict(experiment_id=folder.name, experiment_state=lifecycle(status, experiment), configuration_name=config.get('name'), start_utc=provenance.get('start_utc'), end_utc=provenance.get('end_utc'), exit_code=status.get('exit_code'), timed_out=status.get('timed_out'), startup_seconds=finite(experiment.get('startup_seconds')), elapsed_seconds=finite(status.get('elapsed_seconds', experiment.get('elapsed_seconds'))), git_commit=provenance.get('git_commit'), git_branch=provenance.get('git_branch'), git_dirty=bool(provenance['git_status']) if 'git_status' in provenance else None, reference_id=provenance.get('reference_id'), gpu_memory_sampled_max_mib=gpu['memory_sampled_max_mib'], gpu_memory_is_peak=False, model_gib_per_rank={r['actor']: r['model_gib'] for r in logs['model_loading']}, kv_cache_tokens_by_engine={r['actor']: r['tokens'] for r in logs['kv_cache_tokens']}, oom_detected=logs['oom_detected'], nccl_transports=logs['nccl_transports'], nccl_p2p_level=environment.get('NCCL_P2P_LEVEL', 'unset/default'), visible_devices=environment.get('CUDA_VISIBLE_DEVICES'), config_snapshot_sha256=evidence.sources.get(evidence.key(folder / 'config.json'), {}).get('sha256'))
        error_summary = unique([str(v) for v in (experiment.get('error'), status.get('supervisor_error')) if v])
        error_summary.extend((r['text'] for r in logs['errors'][-5:]))
        common['error_summary'] = error_summary
        record = dict(common, row_type='experiment', config=config, environment=environment, server_command=server_command.get('command'), command=provenance.get('command'), hypothesis=provenance.get('objective_hypothesis'), recorded_source_sha256=provenance.get('source_sha256', {}), git_status=provenance.get('git_status'), gpu_samples=gpu, log_evidence=logs, experiment_recorded_status=experiment.get('status'))
        experiments.append(record)
        measured.extend(benchmark_rows(evidence, folder, common))
        nccl.extend(collectives(evidence, folder, common, process))
    return dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(), generator_sha256=sha256(Path(__file__).read_bytes()), root=str(root), method='Read-only evidence snapshot; no reruns, no warmup performance, no inferred semantic accuracy', caveats=['No accepted project baseline is inferred from reference_id.', 'GPU maxima are nominal 3-second samples over the entire run, not instantaneous peaks.', 'TTFT uses first nonempty text; TPOT is request-level estimate, not individual-token ITL.', 'Partial measured cases are preserved even if another case or experiment failed.', 'Missing supervisor status means incomplete OR running; live process state was not inspected.', 'Collective input_payload_GBps is algorithm input payload rate, not PCIe physical bandwidth.', 'Public results/collectives-4.json is deliberately ignored.', 'Source files may change during ongoing experiments; rerun after completion.'], experiments=experiments, measurements=measured, collectives=nccl, sources=evidence.sources, warnings=evidence.warnings)


def cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])


def markdown(summary: dict) -> str:
    lines = ["# 自动生成的实测对比表", "", f"生成时间：{summary['generated_utc']}。由 `scripts/summarize_results.py` 重建；原始结果不修改。",
             "", "此文件仅汇总实测，不作最终方案推荐。transport PASS 表示 HTTP/流式/实际 token 数检查通过；语义准确率不由此推断。",
             "缺失字段显示 —，失败不填成 0；暖机指标不进入性能表。运行中结果是快照，需结束后重新生成。", "", "## 实验状态与容量", ""]
    lines.append(table(["实验 ID", "状态", "启动 s", "模型 GiB/worker", "KV tokens/engine", "GPU 采样最大 MiB", "Git", "OOM"],
                       [(r["experiment_id"], r["experiment_state"], r["startup_seconds"],
                         r["model_gib_per_rank"], r["kv_cache_tokens_by_engine"], r["gpu_memory_sampled_max_mib"],
                         (r["git_commit"] or "未记录")[:12] + (" dirty" if r["git_dirty"] else ""), r["oom_detected"])
                        for r in summary["experiments"]]))
    lines += ["", "GPU 数值来自全程约 3 秒间隔采样，包含启动与暖机，**不是瞬时峰值**；worker 显存日志和 KV 容量保留原 actor，不跨 TP rank 相加。", "", "## 实际计时：仅 measure", ""]
    lines.append(table(["实验 / suite", "case / round", "输入/输出上限", "C / 请求数", "arrival", "成功/失败", "输出 tok/s", "TTFT p50/p95 ms", "TPOT p50/p95 ms", "E2E p50/p95 s", "transport", "语义"],
                       [(f"{r['experiment_id']} / {r.get('suite')}", f"{r.get('case')} / {r.get('round')}",
                         f"{cell(r.get('input_tokens'))}/{cell(r.get('output_tokens_requested'))}",
                         f"{cell(r.get('concurrency'))}/{cell(r.get('requests'))}", r.get("arrival_pattern"),
                         f"{cell(r.get('successful'))}/{cell(r.get('failed'))}", r.get("output_tokens_per_s"),
                         "/".join(cell(r.get(f"ttft_s_{s}") * 1000 if r.get(f"ttft_s_{s}") is not None else None) for s in ("p50", "p95")),
                         "/".join(cell(r.get(f"tpot_s_{s}") * 1000 if r.get(f"tpot_s_{s}") is not None else None) for s in ("p50", "p95")),
                         "/".join(cell(r.get(f"e2e_s_{s}")) for s in ("p50", "p95")),
                         r["transport_status"], r["semantic_status"]) for r in summary["measurements"]]))
    lines += ["", "C 为客户端并发上限。single_burst 不是稳态吞吐；小样本 p95 仅作描述。TTFT 是首个非空文本，TPOT 是请求级估计，不是 token ITL。完整请求错误、输入/输出哈希与来源哈希见 `summary.json`。", "", "## NCCL：逐实验的实测记录", ""]
    lines.append(table(["实验 ID", "NCCL 路径", "world", "操作", "每 rank 输入 bytes", "中位 µs", "payload GB/s", "各 trial µs"],
                       [(r["experiment_id"], ",".join(r["nccl_transports"]) or "未记录", r.get("world_size"),
                         r.get("operation"), r.get("input_bytes_per_rank"), r.get("median_us"), r.get("input_payload_GBps"),
                         ", ".join(f"{t * 1e6:.3f}" for t in r.get("trials_seconds_per_call", [])))
                        for r in summary["collectives"]]))
    lines += ["", "每个 trial 包含若干次调用，原始次数保留于 JSON/CSV。payload GB/s 不是物理 PCIe 总线带宽；不从集合通信微基准直接推断模型 EP 性能。忽略曾被覆盖的公共 `collectives-4.json`，只使用各实验目录的 JSON 或原始 process.log。", "", "## 失败与未完成记录", ""]
    failures = [(r["experiment_id"], r["experiment_state"], "；".join(r["error_summary"]) or "没有终态或错误摘要")
                for r in summary["experiments"] if r["experiment_state"] != "COMPLETED" or r["error_summary"]]
    lines.append(table(["实验 ID", "状态", "原始错误摘要"], failures) if failures else "当前快照没有记录到失败。")
    request_failures = [(r["experiment_id"], r.get("case"), r.get("measured_request_errors"), r.get("warmup_errors"), r.get("benchmark_fatal_error"))
                        for r in summary["measurements"] if r.get("measured_request_errors") or r.get("warmup_errors") or r.get("benchmark_fatal_error")]
    if request_failures:
        lines += ["", table(["实验 ID", "case", "measure 错误", "warmup 错误（不计性能）", "client fatal"], request_failures)]
    if summary["warnings"]:
        lines += ["", "## 读取警告", ""] + [f"- {message}" for message in summary["warnings"]]
    lines += ["", "复现汇总：`python3 research/dsv4-flash0731/scripts/summarize_results.py`。这只读取已有结果，不运行 GPU。", ""]
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def output_paths(root: Path) -> tuple[Path, Path, Path]:
    native = root.resolve() == Path(__file__).resolve().parents[1]
    folder = workspace_path('results/analysis/architecture/doc') if native else path_at(root, 'results')
    report = folder / 'comparison.md' if native else path_at(root, 'comparison.md')
    return folder / 'summary.csv', folder / 'summary.json', report


def write_outputs(root: Path, summary: dict) -> None:
    rows = summary['experiments'] + summary['measurements'] + summary['collectives']
    preferred = ['row_type', 'experiment_id', 'experiment_state', 'suite', 'case', 'round', 'phase', 'transport_status', 'semantic_status', 'input_tokens', 'output_tokens_requested', 'concurrency', 'requests', 'successful', 'failed', 'output_tokens_per_s', 'input_tokens_per_s', 'successful_requests_per_s']
    omitted = {'config', 'environment', 'server_command', 'command', 'gpu_samples', 'log_evidence', 'git_status', 'source_metric_definitions'}
    fields = preferred + sorted(set().union(*(r.keys() for r in rows)) - set(preferred) - omitted) if rows else preferred
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value, (dict, list)) else value for key, value in row.items() if key not in omitted})
    csv_path, json_path, report_path = output_paths(root)
    atomic_write(csv_path, stream.getvalue())
    atomic_write(json_path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_write(report_path, markdown(summary))


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix='dsv4-summary-test-') as directory:
        root = Path(directory)

        def put(name, value):
            path = path_at(root, 'results') / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_text(path, value if isinstance(value, str) else json.dumps(value))
        put('old/command.json', {'environment': {'CUDA_VISIBLE_DEVICES': '0,1,2,3'}})
        put('old/status.json', {'exit_code': 0})
        put('old/experiment.json', {'status': 'PASS', 'startup_seconds': 12.5})
        base = {'suite': 'smoke', 'status': 'PASSED', 'case_specs': [{'name': 'case1', 'concurrency': 1}], 'runs': [{'case': 'case1', 'round': 1, 'status': 'PASSED', 'warmup': {'summary': {'output_tokens_per_s': 9999}}, 'measure': {'phase': 'measure', 'summary': {'requests': 1, 'successful': 1, 'failed': 0, 'output_tokens_per_s': 5.5, 'ttft_s': {'p50': 0.1, 'p95': 0.2}}, 'requests': []}}]}
        put('old/benchmark.json', base)
        put('old/server.log', '(Worker_TP0 pid=4) Model loading took 37.8 GiB memory and 2.5 seconds\n(EngineCore pid=2) GPU KV cache size: 120,140 tokens\n')
        put('old/gpu.csv', '2026/11/05,0,GPU-a,100,50,40,30\n2026/11/05,0,GPU-a,200,70,60,35\n# timeout\npartial')
        put('new/command.json', {'git_commit': 'abc', 'git_status': ' M file'})
        put('new/benchmark-screen.json', base)
        put('new/benchmark-smoke.json', {'suite': 'smoke', 'status': 'FAILED', 'runs': [{'case': 'failed', 'round': 1, 'status': 'FAILED_WARMUP', 'warmup': {'summary': {'output_tokens_per_s': 8888}, 'requests': [{'ok': False, 'error': {'message': 'timeout'}}]}}]})
        put('failed/status.json', {'exit_code': 1})
        put('failed/server.log', 'torch.OutOfMemoryError: CUDA out of memory\n')
        put('broken/command.json', '{incomplete')
        metric = dict(operation='all_reduce', median_us=2.0, input_bytes_per_rank=8, trials_seconds_per_call=[1e-06, 2e-06, 3e-06], input_payload_GBps=0.004)
        put('comm/status.json', {'exit_code': 0})
        put('comm/collectives-recovered.json', [metric])
        put('comm/process.log', 'NCCL INFO Channel via SHM/direct/direct\n')
        put('comm-log/status.json', {'exit_code': 0})
        put('comm-log/process.log', json.dumps(metric) + '\n')
        put('collectives-4.json', [{'operation': 'WRONG_SHARED_OVERWRITE', 'median_us': 999999}])
        originals = {p: p.read_bytes() for p in path_at(root, 'results').rglob('*') if p.is_file()}
        summary = build_summary(root)
        assert len(summary['measurements']) == 3
        assert len(summary['collectives']) == 2
        assert all((row['operation'] == 'all_reduce' for row in summary['collectives']))
        assert all((row['median_consistent'] for row in summary['collectives']))
        old = next((r for r in summary['measurements'] if r['experiment_id'] == 'old'))
        assert old['output_tokens_per_s'] == 5.5 and old['transport_status'] == 'PASS'
        assert old['semantic_status'] == 'NOT VERIFIED' and old['ttft_s_p50'] == 0.1
        assert old['gpu_memory_sampled_max_mib'] == 200 and old['gpu_memory_is_peak'] is False
        assert old['model_gib_per_rank'] == {'Worker_TP0 pid=4': 37.8}
        assert old['kv_cache_tokens_by_engine'] == {'EngineCore pid=2': 120140}
        failed_warmup = next((r for r in summary['measurements'] if r.get('case') == 'failed'))
        assert failed_warmup['output_tokens_per_s'] is None and failed_warmup['warmup_errors']
        assert 'INCOMPLETE OR RUNNING' in failed_warmup['experiment_state']
        assert next((r for r in summary['experiments'] if r['experiment_id'] == 'failed'))['oom_detected']
        assert summary['warnings']
        write_outputs(root, summary)
        assert '9999' not in path_at(root, 'comparison.md').read_text()
        assert 'warmup 错误' in path_at(root, 'comparison.md').read_text()
        assert len(list(csv.DictReader((path_at(root, 'results') / 'summary.csv').open()))) == len(summary['experiments']) + 5
        assert all((path.read_bytes() == raw for path, raw in originals.items()))
    print('PASS: old/new benchmark schemas, warmup exclusion, partial failures, incomplete JSON, source immutability, GPU/log extraction, per-ID NCCL, CSV/JSON/Markdown')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.root.resolve()
    summary = build_summary(root)
    write_outputs(root, summary)
    print(json.dumps(dict(experiments=len(summary['experiments']), measured_rows=len(summary['measurements']), collective_rows=len(summary['collectives']), warnings=summary['warnings'], outputs=[str(path) for path in output_paths(root)]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
