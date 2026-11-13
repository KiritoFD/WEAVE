"""Compatibility shim: the paper-era flat modules moved to tools/legacy_root/ (2026-11-14).

`python -m weave` and `import weave` keep working by re-exporting from the
legacy package. Current research code lives under src/ and exp/.
"""
from tools.legacy_root.model import WEAVE, build_model_from_config

__all__ = [
    "WEAVE",
    "build_model_from_config",
]
