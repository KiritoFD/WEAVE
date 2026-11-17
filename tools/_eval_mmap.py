"""Wrapper: run tools/eval_images_manifest.py with torch.load mmap=True.

Workaround for intermittent access-violation crashes in torch 2.13
serialization.load_tensor on this machine (commit-charge pressure).
mmap=True keeps storages file-backed instead of committed RAM.
 Falls back to a normal load if the file is not in zip format.

Usage: python tools/_eval_mmap.py <same args as eval_images_manifest.py>
"""
import runpy
import sys
from pathlib import Path

import torch

_orig_load = torch.load


def _patched_load(f, *args, **kwargs):
    use_mmap = isinstance(f, (str, Path)) and kwargs.get("mmap") is None
    if use_mmap:
        kwargs["mmap"] = True
    try:
        return _orig_load(f, *args, **kwargs)
    except RuntimeError as exc:
        if use_mmap and "mmap" in str(exc).lower():
            kwargs.pop("mmap", None)
            return _orig_load(f, *args, **kwargs)
        raise


torch.load = _patched_load

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "eval_images_manifest.py"
    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")
