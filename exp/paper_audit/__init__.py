"""CLEAVE Paper Final Audit Package (F0-F8, Zero Training / Zero New Learnable Parameters)."""
from __future__ import annotations

from pathlib import Path

from src.utils.runtime import REPO

AUDIT_OUT = REPO / "runs/paper_audit"
AUDIT_DOC_DIR = REPO / "docs/system/paper-audit"
