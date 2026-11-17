"""WEAVE V19 / PRISM v2 Rapid-Iteration Master Sequencer (4090).

Usage:
  python -m exp.v19.run_v19 --canary --pool ../pseudo_gt_24h/orig_mean_fp16
  python -m exp.v19.run_v19 --all --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from . import OUT, ROOT, STYLES, read_json, run_measure_v19, save_json, utc
from src.utils.raw_io import export_raw_pngs
from src.utils.runtime import sanitize_cuda_env

DEFAULT_POOL = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")

TEACHER_ARMS = [
    "V19_T1_aniso_r1",
    "V19_T2_s10_vgg_nosid",
    "V19_T3_hybrid_pro_nosid",
    "V19_T3_hybrid_wrong",
    "V19_T4_sweet_vgg_nosid",
    "V19_T5_sweet_hybrid_nosid",
    "V19_T5_sweet_wrong",
]

STUDENT_ARMS = [
    "V19_S1_global_nosid",
    "V19_S2_prism1_l075",
    "V19_S2_prism1_v2",
    "V19_S2_prism1_l130",
    "V19_S2b_prism1_blind_same",
    "V19_S2_prism1_wrong",
    "V19_S3_oneshot_exemplar",
]

ALL_V19_ARMS = TEACHER_ARMS + STUDENT_ARMS


def run_canary(device: str = "cuda"):
    """15-second end-to-end verification of S10-BF16 autograd, Luma-Chroma orthogonality, VGG critic & Student."""
    from src.utils import load_s10_decoder, decode_s10
    from .fiber_duality import (
        BalancedVGGCriticHead,
        StyleFiberEncoder,
        UnnormalizedVGGFeatureExtractor,
        apply_latent_chroma_shift,
        compute_latent_luma_chroma_priors,
        extract_color_fiber_27d,
    )
    from .aniso_teacher import project_anisotropic_trust
    from .routed_student import SpatiallyRoutedChordNet

    print("[V19 Canary] 1/3 Checking Luma-Chroma orthogonality & Anisotropic Trust Projection...", flush=True)
    z = torch.randn(5, 4, 64, 64, device=device) * 0.6
    priors = compute_latent_luma_chroma_priors(z.cpu(), torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0.5, 0.4, -0.3, 0.2]))
    P_Y = priors["P_Y"].to(device)
    P_C = priors["P_C"].to(device)
    zy = torch.einsum("ij,bjhw->bihw", P_Y, z)
    zc = torch.einsum("ij,bjhw->bihw", P_C, z)
    ortho_err = float((zy * zc).sum(dim=1).abs().max().item())
    assert ortho_err < 1e-5, f"Luma-Chroma not orthogonal: {ortho_err}"
    z_shift = apply_latent_chroma_shift(z, torch.tensor([4, 3, 1, 0, 2], device=device), priors, alpha=0.5)
    luma_drift = float((torch.einsum("ij,bjhw->bihw", P_Y, z_shift - z)).abs().max().item())
    assert luma_drift < 1e-5, f"Chroma shift leaked into luminance: {luma_drift}"
    z_proj = project_anisotropic_trust(z + 0.8, z, P_Y, P_C)
    assert torch.isfinite(z_proj).all()

    print("[V19 Canary] 2/3 Checking S10-BF16 autograd -> VGG-16 + Oklab critic...", flush=True)
    dec_fp16 = load_s10_decoder(device)
    dec_bf16 = copy.deepcopy(dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)
    vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval()
    zr = z[:2].detach().clone().requires_grad_(True)
    rgb = decode_s10(dec_bf16, zr, batch=2, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
    feat = vgg_ext(rgb, multi_scale=True)
    head = BalancedVGGCriticHead(in_dim=feat.shape[1], num_styles=5).to(device)
    logits = head(feat)
    logits[:, 4].sum().backward()
    assert zr.grad is not None and torch.isfinite(zr.grad).all() and float(zr.grad.abs().max()) > 0.0
    del dec_fp16, dec_bf16, vgg_ext, head
    torch.cuda.empty_cache()

    print("[V19 Canary] 3/3 Checking SpatiallyRoutedChordNet zero-chord exact identity...", flush=True)
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device)
    # Perturb out_conv so zero-chord check specifically tests the continuous chord-norm gate!
    with torch.no_grad():
        stu.out_conv.weight.normal_(0, 0.02)
    c64 = torch.randn(2, 4, 64, 64, device=device)
    d16 = torch.randn(2, 384, 16, 16, device=device)
    chord = torch.zeros(2, 128, device=device)
    chord[0] = F.normalize(torch.randn(128, device=device), dim=-1)
    delta = stu(z[:2], zy[:2], c64, d16, chord)
    assert float(delta[0].abs().max()) > 0.0 and float(delta[1].abs().max()) == 0.0, "Zero-chord identity failed!"
    delta.square().mean().backward()
    print(f"V19_CANARY_ALL_OK (Student params={stu.n_params()/1e6:.2f}M)", flush=True)


def pick_best_teacher_mode() -> str:
    mode_map = {
        "V19_T4_sweet_vgg_nosid": "s10_vgg",
        "V19_T5_sweet_hybrid_nosid": "hybrid",
    }
    best_arm, best_score = "V19_T5_sweet_hybrid_nosid", -1.0
    for arm, m in mode_map.items():
        sp = OUT / "measured" / arm / "summary_v19.json"
        if sp.exists():
            c = read_json(sp)["cross600"]
            pen = 0.0 if c["dino_c"] >= 0.695 else -0.25
            sc = float(c["csd_top1"]) + pen
            if sc > best_score:
                best_score, best_arm = sc, arm
    chosen = mode_map[best_arm]
    print(f"[V19 Select] Best Sweet-Spot Teacher = {best_arm} (mode={chosen}, score={best_score:.4f})", flush=True)
    return chosen


def export_selected_pngs(arm_names: list[str]):
    """Export 750 RGB PNGs from raw/*.pt via 8-thread lossless C-zlib writer (~1.5s/arm)."""
    out_img_root = OUT / "png_750"
    for arm in arm_names:
        raw_dir = OUT / "raw" / arm
        if not (raw_dir / "meta.json").exists():
            continue
        dst = out_img_root / arm
        cnt = export_raw_pngs(raw_dir, dst, max_workers=8)
        print(f"[V19 PNG] Exported {cnt} PNGs for {arm} -> {dst}", flush=True)


def build_v19_report():
    summaries = {}
    for arm in ALL_V19_ARMS:
        sp = OUT / "measured" / arm / "summary_v19.json"
        if sp.exists():
            summaries[arm] = read_json(sp)

    gate1 = read_json(OUT / "gate1_critic_and_fiber_audit.json") if (OUT / "gate1_critic_and_fiber_audit.json").exists() else None
    master = {"utc": utc(), "gate1": gate1, "summaries": summaries}
    save_json(OUT / "v19_master_summary.json", master)

    lines = [
        "# PRISM v2 / V19 实验实测结果汇总（免源标签 + 各向异性 S10-VGG 教师 + 语义路由单步学生）",
        "",
        "| 实验臂 | Cross600 Top-1 ↑ | Margin ↑ | DINO-S ↑ | CLIP-S ↑ | Affinity ↑ | DINO-C ↑ | LPIPS ↓ | SrcLock ↓ | Same150 ↑ | ER / Imp / Min / Roc / Uki (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, s in summaries.items():
        c = s["cross600"]
        sm = s["same150"]
        pt = s["per_target_cross120"]
        p_str = " / ".join(f"{pt[st]['csd_top1']*100:.1f}" for st in STYLES)
        lines.append(
            f"| {arm} | **{c['csd_top1']*100:.2f}%** | {c['csd_margin']:+.4f} | "
            f"{c['dino_s']:.4f} | {c['clip_s']:.4f} | {c['style_affinity']:.4f} | "
            f"{c['dino_c']:.4f} | {c['lpips']:.4f} | {c['pred_eq_src']*100:.1f}% | "
            f"{sm['csd_top1']*100:.1f}% | {p_str} |"
        )
    (OUT / "V19_RESULTS_AUTO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Export PNGs for sweet-spot teacher, flagship 1-step student, and wrong-ID control
    export_arms = ["V19_T5_sweet_hybrid_nosid", "V19_S2_prism1_v2", "V19_S2_prism1_l130", "V19_S2_prism1_wrong"]
    export_selected_pngs([a for a in export_arms if a in summaries])
    print("V19_REPORT_DONE", flush=True)


def run_stage(stage: str, pool_dir: str, stu_steps: int = 3600, device: str = "cuda"):
    from .fiber_duality import build_or_load_v19_foundation
    from .aniso_teacher import run_sprint1_teachers
    from .routed_student import train_and_eval_v19_students

    if stage == "sprint0_foundation":
        build_or_load_v19_foundation(pool_dir, device=device)
    elif stage == "sprint1_teachers":
        bundle = build_or_load_v19_foundation(pool_dir, device=device)
        run_sprint1_teachers(bundle, device=device)
    elif stage == "sprint2_students":
        bundle = build_or_load_v19_foundation(pool_dir, device=device)
        best_tmode = pick_best_teacher_mode()
        train_and_eval_v19_students(bundle, teacher_mode=best_tmode, steps=stu_steps, device=device)
    elif stage.startswith("measure:"):
        arm_name = stage.split(":", 1)[1]
        run_measure_v19(arm_name, device=device)
    elif stage == "report":
        build_v19_report()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_all_pipeline(pool_dir: str, stu_steps: int = 3600):
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "pipeline_status.json"
    status = read_json(status_path) if status_path.exists() else {}

    stages = (
        ["sprint0_foundation", "sprint1_teachers"]
        + [f"measure:{a}" for a in TEACHER_ARMS]
        + ["sprint2_students"]
        + [f"measure:{a}" for a in STUDENT_ARMS]
        + ["report"]
    )

    t_pipe = time.time()
    for stg in stages:
        if status.get(stg, {}).get("state") == "done":
            print(f"[V19 Pipe] SKIP {stg} (already done)", flush=True)
            continue
        print(f"\n[V19 Pipe] === START {stg} ({utc()}) ===", flush=True)
        t0 = time.time()
        status[stg] = {"state": "running", "utc_start": utc()}
        save_json(status_path, status)

        cmd = [
            sys.executable, "-u", "-m", "exp.v19.run_v19",
            "--stage", stg,
            "--pool", pool_dir,
            "--stu-steps", str(stu_steps),
        ]
        r = subprocess.run(cmd, cwd=ROOT, env=sanitize_cuda_env())
        elapsed = round((time.time() - t0) / 60.0, 2)
        if r.returncode != 0:
            status[stg] = {"state": f"failed:{r.returncode}", "minutes": elapsed}
            save_json(status_path, status)
            raise RuntimeError(f"Stage {stg} failed with rc={r.returncode}")

        status[stg] = {"state": "done", "minutes": elapsed, "utc_end": utc()}
        save_json(status_path, status)
        print(f"[V19 Pipe] === DONE {stg} ({elapsed:.2f} min, total {(time.time()-t_pipe)/60:.1f} min) ===", flush=True)

    print("V19_PIPELINE_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--stu-steps", type=int, default=3600)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.canary:
        run_canary(device=a.device)
    elif a.stage:
        run_stage(a.stage, a.pool, stu_steps=a.stu_steps, device=a.device)
    elif a.all:
        run_all_pipeline(a.pool, stu_steps=a.stu_steps)
    else:
        ap.error("Specify --canary, --stage <name>, or --all")


if __name__ == "__main__":
    main()
