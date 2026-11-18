"""Compatibility shim: the paper-era flat modules moved to tools/legacy_root/ (2026-11-14).

`python -m weave` and `import weave` keep working via PEP 562 lazy re-export from
the legacy package (avoiding eager CUDA/PyTorch DLL allocation during lightweight CLI/test runs).
Current research code lives under src/ and exp/.
"""
from __future__ import annotations

__all__ = [
    "WEAVE",
    "build_model_from_config",
]


def __getattr__(name: str):
    if name in ("WEAVE", "build_model_from_config"):
        from tools.legacy_root import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
