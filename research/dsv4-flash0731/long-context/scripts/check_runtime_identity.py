#!/usr/bin/env python3
"""Read-only CPU verification of Stage B terminal runtime and input identities.

Never contacts GPUs, HTTP services or model code. Requires an existing PASS
integrity audit for each run and refuses to overwrite either output file.
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
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / 'scenarios'
GROUPS = {
    'screen': ['context-tp4-screen', 'context-tp2pp2-screen', 'context-dspark3-screen',
               'context-dp4ep-screen', 'context-tp4-th2048-screen', 'context-tp4-th2048-cad8-screen'],
    'confirmation': ['context-tp2pp2-confirm', 'context-tp4-th2048-cad8-confirm'],
    'profile': ['context-tp4-profile', 'context-tp4-th2048-profile', 'context-tp4-th2048-cad8-profile'],
}
COMMON_SOURCES = ('scenarios/scripts/capacity_client.py',
                  'scripts/benchmark_client.py',
                  'scenarios/scripts/run_logged.py',
                  'metadata/environment.json', 'metadata/model-inventory.json',
                  'project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py')
ORDINARY_ENTRY = 'scenarios/scripts/serve_capacity.py'
PROFILE_ENTRY = 'scenarios/scripts/serve_long_context_profile.py'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path, ledger):
    path = path.resolve()
    raw = path.read_bytes()
    info = dict(sha256=sha(raw), bytes=len(raw))
    require(str(path) not in ledger or ledger[str(path)] == info, 'Evidence changed: '+str(path))
    ledger[str(path)] = info
    return raw


def read_json(path, ledger):
    return json.loads(read(path, ledger))


def inspect(name, group, ledger):
    run = path_at(STAGE, 'results') / name
    data = {key: read_json(run / key, ledger) for key in ('command.json', 'status.json', 'experiment.json', 'capacity.json', 'config.json', 'server-command.json')}
    command, status, experiment, report = (data[key] for key in ('command.json', 'status.json', 'experiment.json', 'capacity.json'))
    require(status.get('exit_code') == 0 and status.get('end_utc') and (not status.get('timed_out')) and (not status.get('supervisor_error')), name + ': unsuccessful/nonterminal supervisor')
    require(experiment.get('status') == 'PASS' and report.get('status') == 'PASSED', name + ': nonterminal or failed client')
    audit = read_json(path_at(ROOT, 'logs') / f'audit-{name}-r1.json', ledger)
    require(audit.get('status') == 'PASS' and (not audit.get('errors')) and (not audit.get('incomplete')), name + ': missing complete PASS audit')
    require(status.get('command') == command.get('command'), name + ': supervisor command mismatch')
    entry = PROFILE_ENTRY if group == 'profile' else ORDINARY_ENTRY
    require(any((str(arg).endswith(entry) for arg in command['command'])), name + ': wrong executed entrypoint')
    snapshot = run / 'source-snapshot.zip'
    read(snapshot, ledger)
    with zipfile.ZipFile(snapshot) as archive:
        source = {key: sha(archive.read(key)) for key in (*COMMON_SOURCES, entry)}
    for key, value in source.items():
        require(value == command['source_sha256'].get(key) == status['source_sha256'].get(key), name + ': snapshot hash mismatch ' + key)
    require(report['client_sha256'] == source[COMMON_SOURCES[0]] and report['base_helper_sha256'] == source[COMMON_SOURCES[1]], name + ': actual client/helper SHA mismatch')
    if group == 'profile':
        require(report['profile_harness_sha256'] == source[entry], name + ': profile harness SHA mismatch')
        require(experiment.get('excluded_from_performance') is True, name + ': profile not excluded')
        require([phase['phase']['role'] for phase in report['runs']] == ['warmup'] * 6 + ['profile'], name + ': wrong profile roles')
        require(sum((len(p['requests']) for p in report['runs'])) == 26 and len(report['runs'][-1]['requests']) == 7, name + ': wrong profile request count')
    config, server = (data['config.json'], data['server-command.json'])
    require(server['extra_environment'] == config['env'], name + ': launch env mismatch')
    expected_env = {'NCCL_DEBUG': 'WARN'}
    if name == 'context-dspark3-screen':
        expected_env['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    if group == 'profile':
        expected_env['VLLM_SERVER_DEV_MODE'] = '1'
    require(config['env'] == expected_env, name + ': undeclared extra environment')
    model_path = server['command'][server['command'].index('serve') + 1]
    workload = run / Path(report['workload_file']).name
    require(sha(read(workload, ledger)) == report['workload_sha256'], name + ': workload hash mismatch')
    identities = []
    for phase in report['runs']:
        rows = []
        for row in sorted(phase['requests'], key=lambda item: item['index']):
            identity = {key: row[key] for key in ('index', 'shape_name', 'input_tokens', 'requested_output_tokens', 'seed', 'ignore_eos', 'prompt_sha256')}
            if phase['phase']['mode'] != 'closed-loop':
                identity['planned_offset_s'] = row['planned_offset_s']
            rows.append(identity)
        identities.append(dict(id=phase['phase']['id'], mode=phase['phase']['mode'], seed=phase['phase']['seed'], requests=len(rows), sha256=sha(canonical(rows))))
    return dict(experiment=name, group=group, git_commit=command['git_commit'], snapshot_members=len(command['source_sha256']), source_sha256=source, tokenizer_sha256=report['tokenizer_sha256'], workload_sha256=report['workload_sha256'], plan_identity_sha256=sha(canonical({key: value for key, value in report['plan'].items() if key != 'experiment_id'})), phase_identities=identities, supervisor_environment=command['environment'], extra_environment=config['env'], model_path=model_path, server_executable=server['command'][0], client_python=report['python'], client_aiohttp=report['aiohttp'], transport=report['transport'], audit_totals=audit['totals'], audit_warnings=audit['warnings'])


def verify(ledger):
    records = [inspect(name, group, ledger) for group, names in GROUPS.items() for name in names]
    source_checks = {}
    for key in COMMON_SOURCES:
        values = {record['source_sha256'][key] for record in records}
        require(len(values) == 1, 'Common source/metadata drift: '+key)
        source_checks[key] = dict(identical=True, sha256=values.pop())
    for key, groups in ((ORDINARY_ENTRY, ('screen', 'confirmation')), (PROFILE_ENTRY, ('profile',))):
        selected = [r for r in records if r['group'] in groups]
        values = {r['source_sha256'][key] for r in selected}
        require(len(values) == 1, 'Entrypoint source drift: '+key)
        source_checks[key] = dict(identical=True, sha256=values.pop(), experiments=len(selected))
    common = ('tokenizer_sha256', 'supervisor_environment', 'model_path', 'server_executable',
              'client_python', 'client_aiohttp', 'transport')
    common_checks = {key: len({canonical(record[key]) for record in records}) == 1 for key in common}
    require(all(common_checks.values()), 'Common runtime metadata drift')
    workload_checks = {}
    for group in GROUPS:
        selected = [record for record in records if record['group'] == group]
        for key in ('workload_sha256', 'phase_identities', 'plan_identity_sha256'):
            require(len({canonical(record[key]) for record in selected}) == 1, group+': input identity mismatch '+key)
        workload_checks[group] = dict(status='PASS', experiments=len(selected),
                                     requests_per_experiment=sum(p['requests'] for p in selected[0]['phase_identities']),
                                     workload_sha256=selected[0]['workload_sha256'],
                                     phase_identities=selected[0]['phase_identities'])
    totals = Counter()
    for record in records:
        totals.update({key: record['audit_totals'][key] for key in ('phases', 'requests', 'prompts', 'metrics_samples')})
    for path, info in ledger.items():
        require(sha(Path(path).read_bytes()) == info['sha256'], 'Evidence changed during review: '+path)
    return dict(status='PASS', reviewed_utc=datetime.now(timezone.utc).isoformat(),
                scope='CPU-only terminal evidence, runtime and matched-workload identity; no performance qualification',
                totals=dict(totals), common_source_comparisons=source_checks,
                common_runtime_comparisons=common_checks, workload_group_comparisons=workload_checks,
                records=records, evidence_ledger=ledger,
                limitations=['Profile harness is compared only among three Profile runs; ordinary launcher only among eight performance runs.',
                             'DSpark3 explicitly adds VLLM_USE_V2_MODEL_RUNNER=1; this is a declared architecture treatment.',
                             'All three Profile runs explicitly add VLLM_SERVER_DEV_MODE=1 for local profiling controls; ordinary performance runs do not.',
                             'Snapshot membership and Git commits may differ while executed components match.',
                             'Captured metadata equality does not freshly hash all model weights or exclude shared-host noise.',
                             'Confirmation tests only .5 req/s; even finite stability confirmation cannot establish a maximum capacity boundary.',
                             '>16K has no defined joint TTFT/E2E Research SLO; eligible-subset success is not whole-mix SLO capacity.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [Path(str(args.output_prefix) + suffix) for suffix in ('.json', '.md')]
    require(not any((path.exists() for path in outputs)), 'Output exists; use a new review ID')
    ledger = {}
    read(Path(__file__), ledger)
    report = verify(ledger)
    lines = ['# 阶段 B 全部运行身份核对', '', '状态：PASS。仅CPU读取终态证据，未访问GPU或模型服务。', '', f'11项合计{report['totals']['phases']}阶段、{report['totals']['requests']}请求、{report['totals']['metrics_samples']}份指标样本。', '', 'client、基础prompt helper、supervisor、environment/model-inventory记录在全部11项中分别一致。普通8项的serve_capacity一致，3项Profile的独立harness一致；没有要求两种不同职责的launcher相同。', '', '三组各自的完整prepared workload、归一化计划与逐请求输入身份均匹配：', '', '| 组 | 实验数 | 每项请求 | workload SHA256 |', '| --- | ---: | ---: | --- |']
    for group, check in report['workload_group_comparisons'].items():
        lines.append(f'| {group} | {check['experiments']} | {check['requests_per_experiment']} | {check['workload_sha256']} |')
    warnings = [(record['experiment'], warning) for record in report['records'] for warning in record['audit_warnings']]
    lines += ['', '全部tokenizer、supervisor环境、模型路径、解释器、aiohttp与transport一致。DSpark3声明的V2 runner环境差异、三个Profile共同的VLLM_SERVER_DEV_MODE=1诊断开关均单独保留；完整snapshot成员数或Git提交不同不等于执行代码改变。', '', f'保留{len(warnings)}条独立审计warning，完整内容见JSON逐项记录。Profile不得进入正常性能排名；确认只测0.5 req/s，不称最大稳定极限；>16K没有联合Research SLO。', '', '记录相同不证明性能改善、长期SLA或排除共享主机噪声。完整SHA、输入身份及证据ledger见同名JSON。']
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with open_output(outputs[0], 'x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    with open_output(outputs[1], 'x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], totals=report['totals'], output=str(outputs[0])), indent=2))


if __name__ == '__main__':
    main()
