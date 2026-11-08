"""Foreground launcher for a verified local configuration; no benchmark side effects."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import json
import os
from pathlib import Path
import shlex


def main():
    root = Path(__file__).resolve().parents[1]
    project = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Name in configs/ (without .json), or an explicit configuration path')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--port', type=int, default=8260)
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    devices = args.gpus.split(',')
    if len(devices) != len(set(devices)) or not set(devices) <= {'0', '1', '2', '3'}:
        parser.error('Use unique physical GPU IDs from 0,1,2,3.')
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be in 1024..65535.')
    path = Path(args.config)
    if not path.is_file():
        path = path_at(root, 'configs') / (args.config.removesuffix('.json') + '.json')
    config = load_document(path)
    options = config['args']
    if not isinstance(options, list) or not all((isinstance(item, str) for item in options)):
        parser.error('Configuration args must be a list of strings.')
    values = {}
    for index, item in enumerate(options):
        if item.startswith('-'):
            if not item.startswith('--'):
                parser.error('Configuration options must use long names, such as --tensor-parallel-size.')
            name, equal, value = item.partition('=')
            if name in values:
                parser.error(f'Repeated configuration option: {name}')
            values[name] = value if equal else options[index + 1] if index + 1 < len(options) else None

    def size(flag):
        try:
            value = int(values.get(flag, '1'))
        except (TypeError, ValueError):
            parser.error(f'{flag} requires a positive integer.')
        if value < 1:
            parser.error(f'{flag} requires a positive integer.')
        return value
    if size('--tensor-parallel-size') * size('--pipeline-parallel-size') * size('--data-parallel-size') != len(devices):
        parser.error('Selected GPU count must equal TP × PP × DP of this local configuration.')
    if any((flag in values for flag in ('--host', '--port', '--model', '--served-model-name'))):
        parser.error('Address and model identity are owned by this launcher.')
    binary = project / '.venv-vllm/bin/vllm'
    model = Path(str(model_path()))
    if not args.print_command and (not binary.is_file() or not (model / 'config.json').is_file()):
        parser.error('Expected local vLLM environment or model is missing.')
    command = [str(binary), 'serve', str(model), '--host', '127.0.0.1', '--port', str(args.port), '--served-model-name', 'dsv4', *options]
    if '--reasoning-parser' not in values:
        command += ['--reasoning-parser', 'deepseek_v4']
    environment = os.environ.copy()
    environment.update(config.get('env', {}))
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus, VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    print(f'Configuration: {path.resolve()}', flush=True)
    print(f'CUDA_VISIBLE_DEVICES={args.gpus} CUDA_DEVICE_ORDER=PCI_BUS_ID', flush=True)
    print(shlex.join(command), flush=True)
    if not args.print_command:
        os.execve(binary, command, environment)


if __name__ == "__main__":
    main()
