"""Explicit, reproducible DP routing headers for the four-engine experiment.

This does not select GPUs, launch a server, or prove cache residency. The caller
must use a DP=4 deployment for sticky4 and preserve the same session index on
every turn. Salt and routing remain separate experimental variables.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output


def session_routing_headers(dp_routing: str, session_index: int) -> dict[str, str]:
    if dp_routing not in ('none', 'sticky4'):
        raise ValueError('dp_routing must be none or sticky4')
    if type(session_index) is not int or session_index < 0:
        raise ValueError('session_index must be a non-negative integer')
    if dp_routing == 'none':
        return {}
    return {'X-data-parallel-rank': str(session_index % 4)}
