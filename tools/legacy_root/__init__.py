"""Legacy paper-era flat modules (relocated here 2026-11-14).

Self-bootstrapping: importing ``tools.legacy_root`` puts this directory on
``sys.path`` so the intra-package flat imports (``from blocks import ...``,
``from config_schema import ...``) keep working exactly as they did when these
files lived at the repository root.
"""
import sys as _sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
