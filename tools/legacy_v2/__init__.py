"""WEAVE-v2: distribution-level style supervision + multi-scale single-step generator.

本包默认完全惰性：只有显式启用 ``v2.enabled`` 才会构建 v2 组件。
旧 ``flow.py`` / ``model.py`` 的生产默认路径不做任何改动。
"""
from __future__ import annotations

from .config import V2Config

__all__ = ["V2Config"]
