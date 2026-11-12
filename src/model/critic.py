"""Style critics and probes used as inference-time guidance.

LatentStyleNet is the R1 classifier whose target-margin gradient drives the
Oracle and R1 edit paths. It is frozen everywhere; nothing here is trained.
"""
from __future__ import annotations

from tools.v6_next.probe_style import LatentStyleNet  # noqa: F401

__all__ = ["LatentStyleNet"]
