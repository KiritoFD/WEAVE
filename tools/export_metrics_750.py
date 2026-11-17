#!/usr/bin/env python3
"""Compatibility entry point delegating to docs/performance/build_metrics.py.

All 750-task metrics (Cross600, Same150, ArtFID/KID, latency, Pareto flags) are
consolidated in docs/performance/metrics.{csv,json}.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "docs/performance/build_metrics.py"
    runpy.run_path(str(target), run_name="__main__")
