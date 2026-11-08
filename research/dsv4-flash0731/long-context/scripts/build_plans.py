"""Create immutable Stage B screening plans; refuses existing outputs."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
STAGE = HERE.parent / 'scenarios'
MIX = [dict(name=f'p{p}_o128', input_tokens=p, output_tokens=128, weight=w)
       for p, w in ((512, 60), (16384, 15), (32768, 15), (65536, 10))]


def new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(path, 'x') as out:
        json.dump(obj, out, indent=2)
        out.write('\n')


def phases():
    result = [dict(id=f'warm-p{s["input_tokens"]}', mode='closed-loop', role='warmup',
                   requests=1, concurrency=1, seed=202611610+i, shape_mix=[s])
              for i, s in enumerate(MIX)]
    result.append(dict(id='warm-c8', mode='closed-loop', role='warmup', requests=8,
                       concurrency=8, seed=202611620, shape_mix=MIX))
    for rate in (.5, 1.):
        result.append(dict(id=f'context_mix-r{str(rate).replace(".", "p")}-screen',
                           mode='poisson', role='screen', rate=rate, duration_s=90,
                           seed=202611621, shape_mix=MIX))
    for trial in (1, 2):
        arrivals = [dict(offset_s=t, shape_name='p512_o128') for t in range(96)]
        arrivals += [dict(offset_s=t, shape_name='p65536_o128') for t in range(8, 81, 12)]
        arrivals.sort(key=lambda a: (a['offset_s'], a['shape_name'] != 'p65536_o128'))
        result.append(dict(id=f'interference64-trace-{trial}', mode='trace', role='interference',
                           duration_s=96, seed=202611630+trial, arrivals=arrivals, shape_mix=MIX))
    arrivals = [dict(offset_s=0., shape_name='p65536_o128') for _ in range(8)]
    arrivals += [dict(offset_s=.5+.5*i, shape_name='p512_o128') for i in range(16)]
    result.append(dict(id='kv_residency-c8long', mode='trace', role='stress', duration_s=12,
                       seed=202611640, arrivals=arrivals, shape_mix=MIX))
    return result


def main():
    jobs = []
    for name, base, threshold in [('context-tp4-screen', 'tp4-marlin-nocache-128k', 0), ('context-tp2pp2-screen', 'tp2-pp2-marlin-nocache-128k', 0), ('context-dspark3-screen', 'tp4-marlin-dspark3-nocache-128k', 0), ('context-dp4ep-screen', 'dp4-ep-marlin-nocache-128k', 0), ('context-tp4-th2048-screen', 'tp4-marlin-nocache-128k', 2048)]:
        config = load_document(path_at(STAGE, 'configs') / f'{base}.json')
        config['name'] = name
        config['args'] += ['--enable-chunked-prefill', '--long-prefill-token-threshold', str(threshold)]
        new(path_at(STAGE, 'configs') / f'{name}.json', config)
        new(path_at(STAGE, 'workloads/long-context') / f'{name}.json', dict(schema_version=1, experiment_id=name, model='dsv4', base_url='http://127.0.0.1:8260', tokenizer_path=str(model_path()), seed=20261106, timeout_s=300, max_inflight=128, metrics_interval_s=1, phases=phases()))
        jobs.append(dict(id=name, config=f'{name}.json', capacity_plan=f'phase-b/{name}.json', timeout=2400, reference='phase-a-final-7c09c61-and-b01-matched-workload', hypothesis='Measure long-short interference and KV residency; matched lengths/arrivals; threshold=' + str(threshold)))
    new(path_at(STAGE, 'workloads/long-context/series-screen.json'), dict(jobs=jobs))
    print(json.dumps(dict(status='CREATED', jobs=[j['id'] for j in jobs]), indent=2))


if __name__ == '__main__':
    main()
