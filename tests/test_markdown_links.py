"""Guard: active Markdown documentation (root, docs/, exp/, src/, scripts/) must contain 0 broken relative links."""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_PREFIXES = (
    "docs/archive/",
    "archives/",
    "aaai2027_v4/",
    "deliver/",
    "tools/",
    "datasets/",
    "runs/",
    "weights_local/",
    ".git/",
    ".zcode/",
)

LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def test_active_markdown_relative_links_exist() -> None:
    broken: list[str] = []
    for path in sorted(PROJECT_ROOT.rglob("*.md")):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith(SKIP_PREFIXES):
            continue
        text = FENCE_RE.sub("", path.read_text(encoding="utf-8", errors="ignore"))
        for match in LINK_RE.finditer(text):
            label, raw = match.group(1), match.group(2).strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1].strip()
            if " " in raw:
                raw = raw.split()[0]
            if not raw or raw.startswith(("http://", "https://", "mailto:", "#", "file:")):
                continue
            target_rel = raw.split("#")[0]
            if not target_rel:
                continue
            target = (path.parent / target_rel).resolve()
            if not target.exists():
                broken.append(f"{rel}: [{label}]({raw})")
    assert not broken, "Broken relative Markdown links in active docs:\n  " + "\n  ".join(broken)
