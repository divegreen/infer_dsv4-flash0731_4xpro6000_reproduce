"""Regenerate a descriptive capacity experiment registry from actual files."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return load_document(path) if path.is_file() else {}


def main():
    lines = ['# 阶段 A 实验登记', '', '根据实际plan、command/status、experiment和capacity checkpoint生成。计划不代表执行，客户端PASSED仍需supervisor终态和独立审计。旧s01–s10见原experiments.md。', '', '| ID | 状态 | 已完成 / 计划phase | commit | config | 来源 |', '| --- | --- | ---: | --- | --- | --- |']
    for path in sorted(path_at(ROOT, 'workloads/capacity').glob('a*.json')):
        plan = read(path)
        name = plan['experiment_id']
        folder = path_at(ROOT, 'results') / name
        command, status = (read(folder / 'command.json'), read(folder / 'status.json'))
        client, experiment = (read(folder / 'capacity.json'), read(folder / 'experiment.json'))
        config = read(folder / 'config.json')
        if status:
            state = 'COMPLETED' if status.get('exit_code') == 0 and (not status.get('timed_out')) and (client.get('status') == 'PASSED') and (experiment.get('status') == 'PASS') else 'FAILED / inspect evidence'
        elif command:
            state = 'IN PROGRESS / verify live process'
        else:
            state = 'NOT STARTED'
        commit = command.get('git_commit', '')[:8] or '—'
        dirty = ' + recorded dirty files' if command.get('git_status') else ''
        sources = f'[plan](workloads/capacity/{path.name})'
        if command:
            sources += f' / [command](results/{name}/command.json)'
        if status:
            sources += f' / [status](results/{name}/status.json)'
        if client:
            sources += f' / [requests](results/{name}/capacity.json)'
        lines.append(f'| {name} | {state} | {len(client.get('runs', []))} / {len(plan['phases'])} | {commit}{dirty} | {config.get('name', '见队列计划')} | {sources} |')
    lines += ['', '容量判定以独立capacity-summary和审计为准；COMPLETED仅表示完整协议执行成功，不表示SLO或稳定性通过。runtime、计划及实际输入hash可从source-snapshot和客户端记录独立重算。', '']
    write_text(path_at(ROOT, 'capacity-experiments.md'), '\n'.join(lines))
    print('Wrote capacity-experiments.md from actual execution evidence')


if __name__ == '__main__':
    main()
