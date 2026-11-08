#!/usr/bin/env python3
"""CPU-only Research SLO reanalysis of immutable historical arrival evidence."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ['scenarios-tp4-nocache', 'scenarios-tp4-dspark3-retry2', 'scenarios-tp2pp2', 'scenarios-dp4ep']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*p
    a, b = math.floor(x), math.ceil(x)
    return values[a] + (values[b]-values[a])*(x-a)


def stats(values):
    return dict(n=len(values), p50=percentile(values,.5), p95=percentile(values,.95),
                p99=percentile(values,.99), max=max(values) if values else None)


def tier(n):
    for boundary, label, limit in [(1024,'le1k',1.),(4096,'le4k',3.),(16384,'le16k',8.)]:
        if n <= boundary:
            return label, limit
    return 'gt16k', None


def enrich(row, poisson):
    result = dict(row)
    name, limit = tier(row['input_tokens'])
    result.update(input_tier=name, ttft_limit_s=limit)
    usage = row.get('usage') or {}
    o = usage.get('completion_tokens')
    good = row['ok']
    if good:
        assert type(o) is int and o > 0
        assert usage['prompt_tokens'] == row['input_tokens']
        assert usage.get('total_tokens',row['input_tokens']+o) == row['input_tokens']+o
        assert o == row['requested_output_tokens']
        assert row['http_status'] == 200 and row['done_received'] and row['error'] is None
        assert digest(row['text'].encode()) == row['text_sha256']
        assert row['e2e_s'] >= row['ttft_s'] >= 0
        assert math.isclose(row['tpot_s'],(row['e2e_s']-row['ttft_s'])/(o-1),abs_tol=1e-9)
        for key in ('ttft_s','e2e_s'):
            assert math.isclose(row['scheduled_'+key],row[key]+row['dispatch_lag_s'],abs_tol=1e-9)
    prefix='scheduled_' if poisson else ''
    ttft,e2e=row.get(prefix+'ttft_s'),row.get(prefix+'e2e_s')
    result['evaluation_ttft_s'],result['evaluation_e2e_s']=ttft,e2e
    e2e_limit=limit+.05*(o-1) if limit is not None and type(o) is int and o>0 else None
    result.update(e2e_limit_s=e2e_limit, ttft_ratio=ttft/limit if ttft is not None and limit else None,
                  e2e_ratio=e2e/e2e_limit if e2e is not None and e2e_limit else None)
    result['research_slo_pass']=bool(good and limit and ttft<=limit and e2e<=e2e_limit
                                       and (o==1 or row['tpot_s']<=.05)) if limit else None
    result['legacy_slo_pass']=bool(good and ttft<=1 and e2e<=10 and (o==1 or row['tpot_s']<=.05))
    # Old client's legacy fields included semaphore batch delay even for closed-loop.
    result['original_client_legacy_slo_pass']=bool(good and row['scheduled_ttft_s']<=1
        and row['scheduled_e2e_s']<=10 and row['tpot_s']<=.05)
    return result


def aggregate(rows, wall):
    good=[r for r in rows if r['ok']]
    eligible=[r for r in rows if r['ttft_limit_s'] is not None]
    applicable=[r for r in good if r['ttft_limit_s'] is not None]
    metrics={k:stats([r[k] for r in good if r.get(k) is not None]) for k in
             ('evaluation_ttft_s','evaluation_e2e_s','tpot_s','dispatch_lag_s','ttft_ratio','e2e_ratio')}
    p95=None
    if applicable:
        p95=all(metrics[k]['p95'] is not None and metrics[k]['p95']<=bound for k,bound in
                [('ttft_ratio',1),('e2e_ratio',1),('tpot_s',.05)])
    passes=sum(r['research_slo_pass'] is True for r in rows)
    legacy=sum(r['legacy_slo_pass'] for r in rows)
    return dict(requests=len(rows),successful=len(good),failed=len(rows)-len(good),
        input_output_shapes=dict(Counter(f'{r["input_tokens"]}/{r["requested_output_tokens"]}' for r in rows)),
        wall_s=wall, output_tokens=sum(r['usage']['completion_tokens'] for r in good),
        output_tokens_per_s=sum(r['usage']['completion_tokens'] for r in good)/wall,
        requests_per_s=len(good)/wall, metrics=metrics, p95_slo_pass=p95,
        research_slo_eligible=len(eligible),research_slo_passes=passes,
        research_slo_joint_pass_rate=passes/len(eligible) if eligible else None,
        research_slo_goodput_requests_per_s=passes/wall,
        legacy_slo_passes=legacy,legacy_slo_joint_pass_rate=legacy/len(rows) if rows else None,
        legacy_slo_goodput_requests_per_s=legacy/wall,
        original_client_legacy_slo_passes=sum(r['original_client_legacy_slo_pass'] for r in rows))


def build(root, names):
    sources, groups, rounds, prompt_sets = ({}, defaultdict(list), [], {})
    for name in names:
        directory = path_at(root, 'results') / name
        original = {}
        for file in ['scenario-arrival.json', 'status.json', 'experiment.json']:
            p = directory / file
            raw = p.read_bytes()
            sources[str(p.relative_to(root))] = digest(raw)
            original[file] = json.loads(raw)
        assert original['status.json']['exit_code'] == 0 and (not original['status.json']['timed_out'])
        assert original['experiment.json']['status'] == 'PASS'
        report = original['scenario-arrival.json']
        assert report['status'] == 'PASSED'
        for h, p in report['prompts'].items():
            assert h == digest(json.dumps(p['token_ids'], separators=(',', ':')).encode()) == p['sha256']
            assert len(p['token_ids']) == p['input_tokens']
        for run in report['runs']:
            assert run['status'] == 'PASSED'
            for stage in run['stages']:
                if stage['phase'] != 'measure':
                    continue
                poisson = stage.get('offered_rate') is not None
                bank = report['banks'][f'{run['round']}/{run['case']}/measure']
                assert len(bank) == len(stage['requests'])
                for r, b in zip(stage['requests'], bank):
                    assert all((r[k] == b[k] for k in ('prompt_sha256', 'seed')))
                    assert r['requested_output_tokens'] == b['output_tokens']
                    assert r['input_tokens'] == report['prompts'][r['prompt_sha256']]['input_tokens']
                signature = digest(json.dumps(bank, sort_keys=True, separators=(',', ':')).encode())
                paired = (run['round'], run['case'])
                assert paired not in prompt_sets or prompt_sets[paired] == signature
                prompt_sets[paired] = signature
                rows = [enrich(r, poisson) for r in stage['requests']]
                wall = stage['summary']['wall_s']
                summary = aggregate(rows, wall)
                assert summary['original_client_legacy_slo_passes'] == stage['summary']['example_slo_pass']
                summary['by_input_tier'] = {t: aggregate([r for r in rows if r['input_tier'] == t], wall) for t in sorted({r['input_tier'] for r in rows})}
                summary['overall_ratio_p95_slo_pass'] = summary['p95_slo_pass']
                eligible = [s for s in summary['by_input_tier'].values() if s['research_slo_eligible']]
                summary['p95_slo_pass'] = all((x['p95_slo_pass'] is True for x in eligible)) if eligible else None
                entry = dict(experiment=name, case=run['case'], round=run['round'], offered_rate=stage.get('offered_rate'), evaluation_basis='scheduled' if poisson else 'HTTP request', planned_arrival_span_s=stage.get('planned_arrival_span_s'), summary=summary)
                rounds.append(entry)
                groups[name, run['case']].append((entry, rows))
    combined = []
    for (name, case), parts in groups.items():
        rows = [r for _, rr in parts for r in rr]
        wall = sum((x['summary']['wall_s'] for x, _ in parts))
        summary = aggregate(rows, wall)
        summary['by_input_tier'] = {t: aggregate([r for r in rows if r['input_tier'] == t], wall) for t in sorted({r['input_tier'] for r in rows})}
        summary['overall_ratio_p95_slo_pass'] = summary['p95_slo_pass']
        eligible = [s for s in summary['by_input_tier'].values() if s['research_slo_eligible']]
        summary['p95_slo_pass'] = all((x['p95_slo_pass'] is True for x in eligible)) if eligible else None
        combined.append(dict(experiment=name, case=case, rounds=[p['round'] for p, _ in parts], offered_rate=parts[0][0]['offered_rate'], evaluation_basis=parts[0][0]['evaluation_basis'], summary=summary, per_round_p95_pass=[p['summary']['p95_slo_pass'] for p, _ in parts], stability='NOT ASSESSABLE: old finite 48-request Poisson, no fixed long horizon or queue time series'))
    for relative, h in sources.items():
        assert digest((root / relative).read_bytes()) == h
    return dict(status='PASS', scope='Independent CPU recomputation and source integrity, not capacity or quality certification', created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(Path(__file__).read_bytes()), definitions=dict(slo='input<=1024/4096/16384 TTFT<=1/3/8s; TPOT<=.05; E2E<=TTFT_limit+.05*(actual output-1)', primary='P95 normalized TTFT/E2E and TPOT independently per eligible input tier', percentiles='Linear interpolation over pooled successful raw requests; no mean of percentiles; no P99 gate', goodput='Successful requests jointly meeting per-request Research SLO / sum(original full cohort wall)', legacy='Retained 1s TTFT/.05 TPOT/10s E2E; also preserve exact old client legacy count separately', closed_loop='Uses HTTP time only; original-client legacy field retains its historical semaphore-delay inclusion'), limitations=['Only 48 requests per Poisson round: 30/10/8 per shape; tails descriptive, no CI.', 'At most two rates were tested; no maximum stable rate or long-window stability certification.', 'All four old configurations are APC off, fixed-output synthetic prompts, no new quality inference.', 'Old finite-request horizon differs from phase-A fixed-duration plan; rates guide screening only.'], source_sha256=sources, rounds=rounds, combined=combined, rate_guidance='Start matched screen at 0.5/1/2/4 req/s. All old 2 req/s mixed cases fail at least one tier P95; include sub-2 rates rather than claiming capacity >=2. Higher rates still inform no-SLO throughput; do not certify stability from old drain-dominated finite tests.')


def markdown(report):
    lines=['# 历史 arrival 在阶段 A Research SLO 下的独立复算','',
      '只读 s01/s03r2/s04/s08 原始结果；不覆盖旧评价。PASS表示复算/源完整性通过，不表示SLO或稳定容量通过。',
      '','Poisson用计划到达起算；C64闭环用HTTP发起起算。Goodput分母为完整旧stage墙钟，包含尾部完成时间。P95逐输入档位判定；P99仅描述。',
      '', '| 实验 | case | N | Research满足率 | Research Goodput req/s | legacy Goodput req/s | 逐档P95 | 输出tok/s |',
      '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in report['combined']:
        s=row['summary']
        lines.append(f'| {row["experiment"]} | {row["case"]} | {s["requests"]} | {100*s["research_slo_joint_pass_rate"]:.2f}% | {s["research_slo_goodput_requests_per_s"]:.3f} | {s["legacy_slo_goodput_requests_per_s"]:.3f} | {s["p95_slo_pass"]} | {s["output_tokens_per_s"]:.1f} |')
    lines+=['','| 实验 / case | 档位N | TTFT P50/P95/P99 (s) | TPOT P95 (ms) | E2E ratio P95 | P95通过 |',
            '| --- | --- | --- | ---: | ---: | --- |']
    for row in report['combined']:
        for name,s in row['summary']['by_input_tier'].items():
            m=s['metrics'];t=m['evaluation_ttft_s']
            lines.append(f'| {row["experiment"]} / {row["case"]} | {name} / {s["requests"]} | {t["p50"]:.3f}/{t["p95"]:.3f}/{t["p99"]:.3f} | {1000*m["tpot_s"]["p95"]:.2f} | {m["e2e_ratio"]["p95"]:.3f} | {s["p95_slo_pass"]} |')
    lines+=['','建议从0.5/1/2/4 req/s匹配粗扫起步：旧2 req/s各架构均至少一个档位P95失败，故需要低于2的点。6 req/s可说明过载/无SLO吞吐变化，但旧数据无固定长窗口、无连续waiting队列证据，不能推出稳定性或精确容量。',
       '','各Poisson轮只有48请求（30/10/8三个shape），两轮合并的尾分位数仍仅作描述，无置信保证。新旧负载形状概率虽相近，有限样本mix和窗口不同，不直接混排名。',
       '','完整分轮数值、源文件SHA256、原client legacy原值、验证定义见同名JSON。复现：',
       '', '```bash', '.venv-vllm/bin/python research/dsv4-flash0731/scenarios/scripts/recompute_capacity_old.py --output-prefix /tmp/capacity-old-new-id', '```','']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output-prefix', type=Path, required=True)
    a = p.parse_args()
    outj = a.output_prefix.with_suffix('.json')
    outm = a.output_prefix.with_suffix('.md')
    if outj.exists() or outm.exists():
        raise FileExistsError('Use a unique output prefix')
    report = build(a.root, DEFAULT_IDS)
    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open_output(outj, 'x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    with open_output(outm, 'x') as f:
        f.write(markdown(report))
    print(json.dumps(dict(status=report['status'], combined=len(report['combined']), rounds=len(report['rounds']), requests=sum((r['summary']['requests'] for r in report['combined'])), json=str(outj), markdown=str(outm))))


if __name__=='__main__':main()
