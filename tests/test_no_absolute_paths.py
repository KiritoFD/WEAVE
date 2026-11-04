"""防回归护栏：仓库内不允许出现硬编码绝对路径。

历史记录/论文产物/文档目录整体放行；B3-B6 尚未改造的文件登记在
``KNOWN_DEBT`` 中，改完后必须同步删除对应条目（条目过期同样判失败）。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 历史资料 / 论文产物目录整体放行
WHITELIST_PREFIXES = (
    "EXPERIMENT_ARCHAEOLOGY/",
    "archives/",
    "aaai2027_v4/",
    "docs/",
)

# 本机 MCP 工具配置，不属于 WEAVE 运行路径
MACHINE_LOCAL_FILES = {
    "config/mcporter.json",
    "config/config/mcporter.json",
}

# B3-B6 待改造清单（每批改完后删除对应条目）
KNOWN_DEBT = {
    "BASELINE_PROGRESS.md",
    "Plan.md",
    "REPOSITORY_INVENTORY.md",
    "configs/exp_rebuttal_hh_head_p256.json",
    "configs/exp_rebuttal_w03_p256.json",
    "configs/exp_rebuttal_w10_p256.json",
    "configs/configs/exp_rebuttal_hh_head_p256.json",
    "configs/configs/exp_rebuttal_w03_p256.json",
    "configs/configs/exp_rebuttal_w10_p256.json",
    "pack_code_zip.py",
    "pack_subpack.py",
    "prepare_submission_v6.py",
    "probes/probe_velocity_adain.py",
    "register_new_style.py",
    "scripts/_audit_stylealigned_cross.py",
    "scripts/_print_cross_table.py",
    "scripts/_repro_dino_eval.py",
    "scripts/_scan_ablation_sources.py",
    "scripts/build_submission_pack.py",
    "scripts/dev/BASELINE_PROGRESS.md",
    "scripts/dev/pack_code_zip.py",
    "scripts/dev/pack_subpack.py",
    "scripts/dev/prepare_submission_v6.py",
    "scripts/dev/register_new_style.py",
    "scripts/eval_other5.py",
    "scripts/pull_submission_pack_remote.bat",
    "scripts/register_other5.py",
    "scripts/run_bench_inference_latency.bat",
    "scripts/scripts/_audit_stylealigned_cross.py",
    "scripts/scripts/_print_cross_table.py",
    "scripts/scripts/_repro_dino_eval.py",
    "scripts/scripts/_scan_ablation_sources.py",
    "scripts/scripts/build_submission_pack.py",
    "scripts/scripts/eval_other5.py",
    "scripts/scripts/pull_submission_pack_remote.bat",
    "scripts/scripts/register_other5.py",
    "scripts/scripts/run_bench_inference_latency.bat",
    "tools/12h/build256.sh",
    "tools/12h/eval_path.sh",
    "tools/12h/eval_path_attr.sh",
    "tools/12h/eval_path_trt.sh",
    "tools/12h/eval_v4_trt.sh",
    "tools/12h/run_adain_attrib.sh",
    "tools/12h/run_after_a4.sh",
    "tools/12h/run_curves.sh",
    "tools/12h/run_v3s.sh",
    "tools/12h/trt_parity.py",
    "tools/12h/v4_arms_ab.sh",
    "tools/12h/v4_eval_ckpts.sh",
    "tools/12h/v4_full_eval.sh",
    "tools/12h/v4_full_eval_all.sh",
    "tools/12h/v4_full_lowstrength.sh",
    "tools/12h/v4_refpool_eval.py",
    "tools/12h/v4_refpool_gen.sh",
    "tools/12h/v4_refpool_rest.sh",
    "tools/12h/v4_seed_check.sh",
    "tools/12h/v4_sweep.sh",
    "tools/12h/v4_train.sh",
    "tools/12h/v4_train_logged.sh",
    "tools/audit/audit_core.py",
    "tools/audit/probe_env.py",
    "tools/batch_eval_wfi.py",
    "tools/build_other5_dataset.py",
    "tools/build_ref_style_tokens.py",
    "tools/check_dinov2_attn.py",
    "tools/check_output_sup_grad.py",
    "tools/diag_three_way.py",
    "tools/encode_wikiart27_sd15.py",
    "tools/extract_dino_cache.py",
    "tools/fast_vae_encode_d5.py",
    "tools/gen_pseudo_targets.py",
    "tools/infer_adain.py",
    "tools/infer_latent_wct.py",
    "tools/infer_sd_variants.py",
    "tools/infer_wct.py",
    "tools/preencode_d5_packed.py",
    "tools/prepare_d5_512_clean.py",
    "tools/prepare_samam_eval.py",
    "tools/taesd_encode_d5.py",
    "tools/trt/bench_vae_decoder.py",
    "tools/trt/build_trt_engines.py",
    "tools/trt/convert_onnx_fp16.py",
    "tools/trt/export_vae_decoder_onnx.py",
    "tools/trt/make_real_latents.py",
    "tools/trt/offline_check_onnx.py",
    "tools/trt/run_window.sh",
    "tools/vae_encode_p2a256.py",
    "tools/vae_encode_r5.py",
    "tools/vae_recon_probe.py",
    "v4/audit_data.py",
    "v4/evaluate.py",
    "weave_eval_backfill.sh",
    "weave_finish_trt.sh",
    "weave_launch.sh",
    "weave_run_v2_v3.sh",
}

# 盘符路径 / WSL 挂载 / 家目录；负向后行断言避免误伤 http:// 等 URL。
# 拆成片段拼接，避免本文件源码被自身规则命中。
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
    """git 跟踪 + 未忽略的新文件；git 不可用时退回 rglob。"""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        names: Iterable[str] = out.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        names = (
            p.relative_to(PROJECT_ROOT).as_posix()
            for p in PROJECT_ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        )
    return [PROJECT_ROOT / name for name in names if name.strip()]


def find_violations(files: Iterable[Path]) -> dict[str, list[int]]:
    """返回 {相对路径: 命中行号列表}。"""
    violations: dict[str, list[int]] = {}
    for path in files:
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
    injected.write_text('PATH = r"' + "G" + ':' + "\\\\" + 'data\\\\images"\n', encoding="utf-8")
    found = find_violations([injected])
    assert injected.as_posix() in found, f"注入的盘符路径未被检测到: {found}"
