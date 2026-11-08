"""Build compact research tables from immutable measured runs, excluding warmup."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return load_document(path)


def counters(path):
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        match = re.match(r'(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total)(\{[^}]*\})? ([\d.eE+-]+)$', line)
        if match:
            values[(match[1], match[2])] = float(match[3])
    return values


def main():
    data = {'screen': [], 'full': [], 'long': [], 'speculative': [], 'chat': []}
    for folder in sorted(path_at(ROOT, 'results').iterdir()):
        if not folder.is_dir() or folder.name not in {e['id'] for e in load_document(ROOT / 'experiments/catalog.json')['experiments'] if e['topic'] == 'architecture'}:
            continue
        for suite in ('screen', 'full', 'long'):
            path = folder / f'benchmark-{suite}.json'
            if not path.is_file():
                continue
            report = load(path)
            cases = {}
            for run in report.get('runs', []):
                measurement = run.get('measure', {})
                summary = measurement.get('summary', {})
                if not summary.get('all_successful'):
                    continue
                cases.setdefault(run['case'], []).append(summary)
            for case, rows in cases.items():
                wall = sum((x['wall_s'] for x in rows))
                data[suite].append({'experiment': folder.name, 'case': case, 'rounds': len(rows), 'report_status': report.get('status'), 'connection_policy': report.get('client_transport', {}).get('connection_policy', 'pooled_default'), 'requests': sum((x['successful'] for x in rows)), 'output_tokens_per_s': sum((x['successful_output_tokens'] for x in rows)) / wall, 'throughput_min': min((x['output_tokens_per_s'] for x in rows)), 'throughput_max': max((x['output_tokens_per_s'] for x in rows)), 'ttft_p50_ms_by_round': [x['ttft_s']['p50'] * 1000 for x in rows], 'tpot_p50_ms_by_round': [x['tpot_s']['p50'] * 1000 for x in rows], 'e2e_p95_s_by_round': [x['e2e_s']['p95'] for x in rows], 'source': str(path.relative_to(ROOT))})
            before, after = (counters(folder / f'metrics-before-{suite}.txt'), counters(folder / f'metrics-after-{suite}.txt'))
            if before and after:
                delta = {}
                for key in after:
                    value = after[key] - before.get(key, 0)
                    if value < 0:
                        raise ValueError(f'Counter reset: {folder.name}/{suite}/{key}')
                    delta[key[0]] = delta.get(key[0], 0) + value
                drafts = delta.get('vllm:spec_decode_num_drafts_total', 0)
                proposed = delta.get('vllm:spec_decode_num_draft_tokens_total', 0)
                accepted = delta.get('vllm:spec_decode_num_accepted_tokens_total', 0)
                if drafts and proposed:
                    data['speculative'].append({'experiment': folder.name, 'suite': suite, 'scope': 'suite warmup and measured requests together; not measure-only', 'drafts': drafts, 'draft_tokens': proposed, 'accepted_tokens': accepted, 'acceptance_fraction': accepted / proposed, 'accepted_length_including_bonus': 1 + accepted / drafts})
        if (folder / 'chat-probe.json').is_file():
            chat = load(folder / 'chat-probe.json')
            data['chat'].append({'experiment': folder.name, 'status': chat['status'], 'summary': chat.get('summary')})
    write_text(path_at(ROOT, 'results/analysis.json'), json.dumps(data, indent=2) + '\n')
    lines = ['# 可重建的精简结果表', '', '源：各实验独立 JSON。只汇总成功的正式 measure，暖机性能不进入表；不把性能等同模型准确率。FAILED套件的成功部分仅用于保留证据，不能当作完整通过的比较对象；失败请求详见comparison.md。', '']
    for suite in ('screen', 'full', 'long'):
        lines += [f'## {suite}', '', '| 实验 | 套件状态 / 连接协议 | 负载 | 成功轮数 / 请求数 | 合计输出 tok/s | 各轮吞吐范围 | TTFT p50 ms（各轮） | TPOT p50 ms（各轮） |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in data[suite]:
            ttft = '/'.join((f'{x:.1f}' for x in row['ttft_p50_ms_by_round']))
            tpot = '/'.join((f'{x:.2f}' for x in row['tpot_p50_ms_by_round']))
            lines.append(f'| {row['experiment']} | {row['report_status']} / {row['connection_policy']} | {row['case']} | {row['rounds']} / {row['requests']} | {row['output_tokens_per_s']:.1f} | {row['throughput_min']:.1f}–{row['throughput_max']:.1f} | {ttft} | {tpot} |')
        lines += ['']
    lines += ['## DSpark 接受统计', '', '以下 counter 差分覆盖整套负载的暖机与正式请求，不能当成某个 case 的正式测量接受率。', '', '| 实验 / suite | 草稿 token 数 | 接受数 | 接受率 | 含 bonus 平均接受长度 |', '| --- | --- | --- | --- | --- |']
    for row in data['speculative']:
        lines.append(f'| {row['experiment']} / {row['suite']} | {row['draft_tokens']:.0f} | {row['accepted_tokens']:.0f} | {row['acceptance_fraction']:.2%} | {row['accepted_length_including_bonus']:.3f} |')
    lines += ['', '## 聊天小样本验证', '']
    for row in data['chat']:
        lines.append(f'- {row['experiment']}: {row['status']}；{json.dumps(row['summary'], ensure_ascii=False)}')
    lines += ['', '合计吞吐=各轮输出token总数/各轮墙钟时间总和；范围仅描述重复波动，不是统计置信区间。screen为有限突发，full为有限闭环多波，long为特定长度验证，均不是长期SLA。', '']
    write_text(workspace_path('results/analysis/architecture/doc/analysis-tables.md'), '\n'.join(lines))
    print(json.dumps({key: len(value) for key, value in data.items()}))


if __name__ == "__main__":
    main()
