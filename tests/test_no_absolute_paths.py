"""防回归护栏：活跃代码（src/、exp/、tests/、根目录配置及新脚本）不允许出现硬编码绝对路径。

历史归档与冻结证据目录（archives/、aaai2027_v4/、docs/、deliver/、runs/、tools/）
及机器环境备忘（AGENTS.md）按 2026-11-14 重组约定整体豁免；历史遗留的 scripts/
待改造文件登记在 ``KNOWN_DEBT`` 中，改完后必须同步删除对应条目。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WHITELIST_PREFIXES = (
    "archives/",
    "aaai2027_v4/",
    "docs/",
    "deliver/",
    "runs/",
    "tools/",
)

MACHINE_LOCAL_FILES = {
    "AGENTS.md",
}

KNOWN_DEBT = {
    "scripts/_audit_stylealigned_cross.py",
    "scripts/_print_cross_table.py",
    "scripts/_scan_ablation_sources.py",
    "scripts/build_submission_pack.py",
    "scripts/dev/BASELINE_PROGRESS.md",
    "scripts/dev/pack_code_zip.py",
    "scripts/dev/pack_subpack.py",
    "scripts/dev/prepare_submission_v6.py",
    "scripts/dev/register_new_style.py",
    "scripts/pull_submission_pack_remote.bat",
    "scripts/register_other5.py",
    "scripts/scripts/_audit_stylealigned_cross.py",
    "scripts/scripts/_print_cross_table.py",
    "scripts/scripts/_repro_dino_eval.py",
    "scripts/scripts/_scan_ablation_sources.py",
    "scripts/scripts/build_submission_pack.py",
    "scripts/scripts/pull_submission_pack_remote.bat",
    "scripts/scripts/register_other5.py",
}

_MNT_PREFIX = "/" + "mnt/[a-z]/"
_HOME_PREFIX = "/" + "home/"
_ROOT_PREFIX = "/" + "root/"
ABSOLUTE_PATH_RE = re.compile(
    "(?<![A-Za-z0-9_])[A-Za-z]:[\\\\/]"
    + "|" + _MNT_PREFIX
    + "|" + _HOME_PREFIX
    + "|" + _ROOT_PREFIX
)


def _repo_files() -> list[Path]:
    """git 跟踪的文件 + exp/ 下活跃 Python/JSON/Shell 源码。"""
    files: set[str] = set()
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        files.update(n for n in out.stdout.splitlines() if n.strip())
    except (OSError, subprocess.CalledProcessError):
        files.update(
            p.relative_to(PROJECT_ROOT).as_posix()
            for p in PROJECT_ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        )
    for sub in ("src", "exp", "tests"):
        for p in (PROJECT_ROOT / sub).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                files.add(p.relative_to(PROJECT_ROOT).as_posix())
    return [PROJECT_ROOT / name for name in sorted(files)]


def find_violations(files: Iterable[Path]) -> dict[str, list[int]]:
    """返回 {相对路径: 命中行号列表}。"""
    violations: dict[str, list[int]] = {}
    for path in files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ABSOLUTE_PATH_RE.search(line):
                try:
                    rel = path.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    rel = path.as_posix()
                violations.setdefault(rel, []).append(lineno)
    return violations


def _is_allowed(rel: str) -> bool:
    return (
        rel.startswith(WHITELIST_PREFIXES)
        or rel in MACHINE_LOCAL_FILES
        or rel in KNOWN_DEBT
    )


def test_no_absolute_paths_outside_known_debt() -> None:
    violations = find_violations(_repo_files())
    unexpected = {rel: lines for rel, lines in violations.items() if not _is_allowed(rel)}
    assert not unexpected, (
        "发现未登记的绝对路径（新增代码请改用 paths.py / 相对路径）：\n"
        + "\n".join(f"  {rel}: 行 {lines}" for rel, lines in sorted(unexpected.items()))
    )


def test_known_debt_entries_are_current() -> None:
    """改完的文件必须从 KNOWN_DEBT 移除，避免清单腐化。"""
    violations = find_violations(_repo_files())
    stale = sorted(name for name in KNOWN_DEBT if name not in violations)
    assert not stale, "以下条目已无绝对路径，请从 KNOWN_DEBT 删除：\n" + "\n".join(
        f"  {name}" for name in stale
    )


def test_guard_catches_injected_drive_path(tmp_path: Path) -> None:
    """护栏自检：注入一条盘符路径必须被检测到。"""
    injected = tmp_path / "injected.py"
    injected.write_text('PATH = r"' + "G" + ":" + "\\\\" + 'data\\\\images"\n', encoding="utf-8")
    found = find_violations([injected])
    assert injected.as_posix() in found, f"注入的盘符路径未被检测到: {found}"
