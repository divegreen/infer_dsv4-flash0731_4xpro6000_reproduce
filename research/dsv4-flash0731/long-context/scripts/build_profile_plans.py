"""Write the bounded, paired diagnostic profile configs/plans; never run them."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent/'scenarios'
sys.path.insert(0, str(STAGE/'scripts'))
import serve_long_context_profile as harness


def write(path, obj):
    with open_output(path, 'x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def main():
    source = load_document(path_at(STAGE, 'workloads/long-context/context-tp4-screen.json'))
    shape = [s for s in source['phases'][5]['shape_mix'] if s['input_tokens'] in (512, 65536)]
    profile = dict(id='profile_interference64', mode='trace', role='profile', duration_s=4, seed=202611650, shape_mix=shape, arrivals=[dict(offset_s=t, shape_name='p65536_o128' if i == 0 else 'p512_o128') for i, t in enumerate(harness.OFFSETS)])
    warm = dict(deepcopy(profile), id='warm_interference64', role='warmup', seed=202611651)
    for name, config_name in [('context-tp4-profile', 'context-tp4-screen'), ('context-tp4-th2048-profile', 'context-tp4-th2048-screen')]:
        plan = deepcopy(source)
        plan.update(experiment_id=name, timeout_s=180, max_inflight=16, phases=deepcopy(source['phases'][:5]) + [deepcopy(warm), deepcopy(profile)])
        config = load_document(path_at(STAGE, 'configs') / f'{config_name}.json')
        config['name'] = name
        config['env']['VLLM_SERVER_DEV_MODE'] = '1'
        config['args'] += ['--enable-logging-iteration-details', '--profiler-config', json.dumps(harness.profiler_options(path_at(STAGE, 'results') / name))]
        normalized = harness.validate_profile_plan(plan)
        command = harness.server_command(config, normalized, path_at(STAGE, 'results') / name)
        write(path_at(STAGE, 'configs') / f'{name}.json', config)
        write(path_at(STAGE, 'workloads/long-context') / f'{name}.json', plan)
        print(json.dumps(dict(id=name, requests=26, command=command)))


if __name__ == '__main__':
    main()
