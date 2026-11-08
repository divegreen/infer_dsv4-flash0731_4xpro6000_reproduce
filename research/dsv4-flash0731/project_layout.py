"""Explicit path resolution for native inputs and disposable research outputs.

No cwd changes or pathlib monkey patches. Historical source directories describe
which tools are reused; this module owns where their data live.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
PROJECT = RESEARCH.parents[1]
EXPERIMENTS = RESEARCH / 'experiments'
RESULTS = PROJECT / 'results'
TOPICS = ('scenarios', 'long-context', 'sessions', 'scheduler')


def workspace_path(relative=''):
    return PROJECT / relative


def model_path():
    path = Path(os.environ.get('DSV4_MODEL_PATH', '~/models/dsv4-flash0731')).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def path_at(base, relative):
    """Route a logical study path; arbitrary test/temp roots keep normal semantics."""
    path = Path(base) / relative
    try:
        parts = path.relative_to(RESEARCH).parts
    except ValueError:
        return path
    topic = parts[0] if parts and parts[0] in TOPICS else 'architecture'
    rest = parts[1:] if topic != 'architecture' else parts
    if not rest:
        return path
    kind, *tail = rest
    if kind in ('configs', 'workloads'):
        return EXPERIMENTS / topic / kind / Path(*tail)
    if kind == 'results':
        # Historical architecture analysis.json is derived, not a raw run.
        if tail and tail[0] == 'analysis.json':
            return RESULTS / 'analysis/architecture' / Path(*tail)
        return RESULTS / 'runs' / Path(*tail)
    if kind in ('summary', 'logs', 'figures', 'evidence'):
        return RESULTS / 'analysis' / topic / kind / Path(*tail)
    if len(rest) == 1 and kind.endswith('.md'):
        return (RESEARCH if topic == 'architecture' else RESEARCH / topic) / 'doc' / kind
    return path


def expand_document(value, run_id=None):
    if isinstance(value, dict):
        run_id = value.get('experiment_id', value.get('name', run_id))
        return {k: expand_document(v, run_id) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_document(v, run_id) for v in value]
    if isinstance(value, str):
        if value.startswith('~/models/dsv4-flash0731'):
            return str(model_path()) + value[len('~/models/dsv4-flash0731'):]
        value = value.replace('${PROJECT_ROOT}', str(PROJECT))
        value = value.replace('${MODEL_PATH}', str(model_path()))
        if '${RUN_DIR}' in value:
            target = os.environ.get('RESEARCH_RUN_DIR')
            if target is None and run_id:
                target = str(RESULTS / 'runs' / run_id)
            if target is None:
                raise ValueError('Profiler input requires an experiment identity or RESEARCH_RUN_DIR')
            value = value.replace('${RUN_DIR}', target)
        return value
    return value


def load_document(path):
    path = Path(path)
    value = json.loads(path.read_text())
    # Results and historical evidence are never rewritten while reading.
    if path.resolve().is_relative_to(EXPERIMENTS):
        value = expand_document(value)
    return value


def write_text(path, text, *args, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.write_text(text, *args, **kwargs)


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.write_bytes(data)


def open_output(path, mode='w', *args, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode, *args, **kwargs)


def ensure_layout():
    for name in ('runs', 'analysis', 'preserved', 'setup', 'validation', 'stage-state'):
        (RESULTS / name).mkdir(parents=True, exist_ok=True)


def snapshot_inputs():
    """Shared runtime dependencies that every supervisor must bind to its evidence."""
    return [Path(__file__), EXPERIMENTS / 'catalog.json', PROJECT / 'tools/reproduce.py']
