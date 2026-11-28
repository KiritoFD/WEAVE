"""V36 CLEAVE-One Diagnostic Suite (D0-D4):
  - D0: Checkpoint comparison (P1 30k / P1 42k / P2 50k) on test750.
  - D1: Diagnostic Oracle Draft Check (closed-set V35 E-Core drafts -> frozen V36 45M P-Core).
        NOTE: E-Core has seen Core-5; strictly diagnostic upper-bound check, NEVER official One-shot score.
  - D2: Style MLP Attribution (fixed Reference-A3 I2 startpoint, swap/zero ONLY StyleMLPEncoder).
  - D3: Direct Reference-A3 (0-NFE) baseline vs full 6-step V36 P-Core output.
  - D4: Seen-Dev (131 seen movements) vs Artist-OOD vs Movement-OOD (Core-5) paired transfer comparison.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from exp.v18 import tasks_and_sources
from exp.v29_biglittle import RawWriterV29, compute_dino_top5_mean
from exp.v36_art_foundation import CACHE_DIR, FIG_DIR, MEAS_DIR, ROOT
from exp.v36_art_foundation.eval_v36 import (
    _compute_v36_csls_k10,
    load_v36_models,
    reconstruct_reference_a3,
    sample_v36_ode,
)
from src.eval.runner import run_measure_batch_subprocess
from src.utils import FeatureStack, auto_referee_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, utc


@torch.no_grad()
def _sample_v36_style_dropped(
    pcore,
    z_start: torch.Tensor,
    y16: torch.Tensor,
    q32: torch.Tensor,
    eps: torch.Tensor,
    *,
    tau0: float = 0.50,
    steps: int = 6,
    wc: float = 1.6,
    gamma_val: float = 0.85,
    device: str = "cuda",
) -> torch.Tensor:
    """Integrate V36 ODE with StyleMLPEncoder completely dropped (only content CFG wc active)."""
    B = z_start.shape[0]
    z = (1.0 - tau0) * eps + tau0 * z_start if tau0 > 0 else eps.clone()
    dt = (1.0 - tau0) / float(steps)
    f0 = torch.zeros(B, dtype=torch.bool, device=device)
    f1 = torch.ones(B, dtype=torch.bool, device=device)
    g = torch.full((B,), gamma_val, device=device)
    l1 = torch.ones(B, device=device)
    l0 = torch.zeros(B, device=device)
    zeros_s = torch.zeros(B, 256, device=device)

    prev_v = None
    for i in range(steps):
        t_scalar = tau0 + i * dt
        z2 = z.repeat(2, 1, 1, 1)
        y2 = y16.repeat(2, 1, 1, 1)
        q2 = q32.repeat(2, 1, 1, 1)
        s2 = torch.cat([zeros_s, zeros_s], dim=0)
        t2 = torch.full((B * 2,), t_scalar, device=device)
        ds2 = torch.ones(B * 2, dtype=torch.bool, device=device)
        dc2 = torch.cat([f0, f1], dim=0)
        ls2 = torch.cat([l1, l0], dim=0)
        g2 = g.repeat(2)
        with bf16_autocast(device):
            v_0c, v_00 = pcore(
                z2, y2, q2, s2, t2, drop_style=ds2, drop_content=dc2, layout_scale=ls2, gamma=g2,
            ).float().chunk(2, dim=0)
        v_cur = v_00 + wc * (v_0c - v_00)
        if i == 0 or prev_v is None:
            z = z + dt * v_cur
        else:
            z = z + dt * (1.5 * v_cur - 0.5 * prev_v)
        prev_v = v_cur
    return z


def _diag_raw_dir(tag: str) -> Path:
    return ROOT / "runs/v36_art_foundation/raw_diag" / tag


def _diag_meas_dir(tag: str) -> Path:
    return MEAS_DIR / "diag" / tag


@torch.no_grad()
def _generate_pending_arm(
    arm_tag: str,
    *,
    z_start_750: torch.Tensor,
    e_style_750: torch.Tensor | None,
    q32_750: torch.Tensor,
    y16_750: torch.Tensor,
    tasks: list[dict],
    dec,
    pcore=None,
    tau0: float = 0.50,
    wj: float = 3.8,
    wc: float = 1.6,
    steps: int = 6,
    drop_style_zero: bool = False,
    wrong: bool = False,
    device: str = "cuda",
) -> tuple[Path, Path]:
    raw_dir = _diag_raw_dir(arm_tag)
    m_dir = _diag_meas_dir(arm_tag)
    if (m_dir / "summary_v36.json").exists() or (m_dir / "eval.json").exists():
        return raw_dir, m_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    writer = RawWriterV29(arm_tag, dec=dec, device=device)
    writer.path = raw_dir
    eps_750 = torch.randn(z_start_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    bs = 24
    for st in range(0, len(tasks), bs):
        ed = min(st + bs, len(tasks))
        sl = slice(st, ed)
        z0 = z_start_750[sl].to(device).float()
        if steps <= 0 or pcore is None:
            zo = z0
        elif drop_style_zero:
            zo = _sample_v36_style_dropped(
                pcore, z0, y16_750[sl].to(device).float(), q32_750[sl].to(device).float(),
                eps_750[sl].to(device).float(), tau0=tau0, steps=steps, wc=wc, device=device,
            )
        else:
            assert e_style_750 is not None
            zo = sample_v36_ode(
                pcore, z0, y16_750[sl].to(device).float(), q32_750[sl].to(device).float(),
                e_style_750[sl].to(device).float(), eps_750[sl].to(device).float(),
                tau0=tau0, steps=steps, wj=wj, wc=wc, device=device,
            )
        writer.write_batch(zo, tasks[sl])
    writer.finalize(arm=arm_tag, lambda_val=tau0, wrong=wrong)
    return raw_dir, m_dir


def _finalize_measured_arm(arm_tag: str, meta: dict[str, Any]) -> dict[str, Any]:
    m_dir = _diag_meas_dir(arm_tag)
    sum_p = m_dir / "summary_v36.json"
    if sum_p.exists():
        return read_json(sum_p)
    sa = read_json(m_dir / "summary_arm.json")
    ev = read_json(m_dir / "eval.json")
    s_c, s_s = sa["cross600"], sa["same150"]
    out = {
        **meta,
        "arm": arm_tag,
        "cross600": {
            "csd_top1": float(s_c["csd_top1"]),
            "csd_margin": float(s_c["csd_margin"]),
            "dino_c": float(s_c["dino_c"]),
            "dino_s": float(s_c["dino_s"]),
            "dino_s_top5": compute_dino_top5_mean(m_dir),
            "csls_k10": _compute_v36_csls_k10(ev["rows"]),
            "clip_s": float(s_c["clip_s"]),
            "style_affinity": float(s_c["style_affinity"]),
            "lpips": float(s_c["lpips"]),
        },
        "same150": {"csd_top1": float(s_s["csd_top1"]), "dino_c": float(s_s["dino_c"])},
    }
    save_json(sum_p, out)
    raw_dir = _diag_raw_dir(arm_tag)
    for bf in raw_dir.glob("batch_*.pt"):
        bf.unlink()
    return out


def _flush_measure(pending: list[tuple[Path, Path, str]], metas: dict[str, dict[str, Any]], device: str) -> list[dict[str, Any]]:
    need = [(r, m, n) for r, m, n in pending if not (m / "summary_v36.json").exists()]
    if need:
        run_measure_batch_subprocess(need, device=device, write_png=False)
    return [_finalize_measured_arm(n, metas[n]) for _, _, n in pending]


def _load_diag_assets():
    tasks, z_src_750, _ = tasks_and_sources()
    ea_assets = torch.load(CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    q32_750 = ea_assets["q32_750"]
    y16_750 = ea_assets["y16_750"]
    refs_d = ea_assets["refs150"]
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = pick_wrong_cross_tid(sid_750, tid_750)
    idx_ok = torch.tensor([int(tid_750[i].item()) * 30 + (i % 30) for i in range(len(tasks))], dtype=torch.long)
    idx_wr = torch.tensor([int(tid_wrong[i].item()) * 30 + (i % 30) for i in range(len(tasks))], dtype=torch.long)
    return tasks, z_src_750, q32_750, y16_750, refs_d, idx_ok, idx_wr


def run_diag_d0(device: str = "cuda") -> dict[str, Any]:
    """D0: Compare P1 30k vs P1 42k vs existing P2 50k across tau0."""
    tasks, z_src_750, q32_750, y16_750, refs_d, idx_ok, idx_wr = _load_diag_assets()
    dec = load_s10_decoder(device)

    specs = [
        ("p1_30k", ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p1_step030000.pt", (0.0, 0.35, 0.50)),
        ("p1_42k", ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p1_final.pt", (0.0, 0.35, 0.50, 0.65)),
    ]
    pending: list[tuple[Path, Path, str]] = []
    metas: dict[str, dict[str, Any]] = {}
    pairs_causal: list[tuple[str, str]] = []

    for ck_tag, ck_path, taus in specs:
        pcore, style_enc, priors = load_v36_models(ck_path, device)
        e_refs = style_enc(refs_d["stat_1561"].to(device).float()).cpu()
        z_refs = refs_d["z_ref"]
        z_start_ok = reconstruct_reference_a3(z_src_750.float(), z_refs[idx_ok].float(), priors, mode="I2").cpu()
        z_start_wr = reconstruct_reference_a3(z_src_750.float(), z_refs[idx_wr].float(), priors, mode="I2").cpu()
        for t0 in taus:
            ok_tag = f"D0_{ck_tag}_I2_t{int(round(t0 * 100)):02d}"
            wr_tag = f"{ok_tag}_wrong"
            r_ok, m_ok = _generate_pending_arm(
                ok_tag, z_start_750=z_start_ok, e_style_750=e_refs[idx_ok], q32_750=q32_750, y16_750=y16_750,
                tasks=tasks, dec=dec, pcore=pcore, tau0=t0, wj=3.8, wc=1.6, steps=6, wrong=False, device=device,
            )
            r_wr, m_wr = _generate_pending_arm(
                wr_tag, z_start_750=z_start_wr, e_style_750=e_refs[idx_wr], q32_750=q32_750, y16_750=y16_750,
                tasks=tasks, dec=dec, pcore=pcore, tau0=t0, wj=3.8, wc=1.6, steps=6, wrong=True, device=device,
            )
            pending.extend([(r_ok, m_ok, ok_tag), (r_wr, m_wr, wr_tag)])
            metas[ok_tag] = {"diag": "D0", "ckpt_tag": ck_tag, "tau0": t0, "wrong": False}
            metas[wr_tag] = {"diag": "D0", "ckpt_tag": ck_tag, "tau0": t0, "wrong": True}
            pairs_causal.append((ok_tag, wr_tag))
        del pcore, style_enc
        gc.collect()
        torch.cuda.empty_cache()

    del dec
    gc.collect()
    torch.cuda.empty_cache()

    rows = {r["arm"]: r for r in _flush_measure(pending, metas, device)}
    summary = []
    for ok_tag, wr_tag in pairs_causal:
        ro, rw = rows[ok_tag], rows[wr_tag]
        ro["wrong_ctrl_top1"] = rw["cross600"]["csd_top1"]
        ro["causal_delta_pp"] = round((ro["cross600"]["csd_top1"] - rw["cross600"]["csd_top1"]) * 100.0, 2)
        summary.append(ro)
        print(f"[D0 {ok_tag}] Top1={ro['cross600']['csd_top1']:.4f} D-C={ro['cross600']['dino_c']:.4f} D-S={ro['cross600']['dino_s']:.4f} Causal={ro['causal_delta_pp']:+.2f}pp", flush=True)

    p2_p = MEAS_DIR / "v36_main_p2_I2_tau_sweep.json"
    p2_rows = read_json(p2_p) if p2_p.exists() else []
    rep = {"utc": utc(), "d0_new": summary, "p2_50k_ref": p2_rows}
    save_json(MEAS_DIR / "v36_diag_d0_checkpoints.json", rep)
    return rep


@torch.no_grad()
def _save_d1_oracle_grid(pcore, style_enc, priors, z_e_750: torch.Tensor, device: str = "cuda") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dec = load_s10_decoder(device)
    tasks, z_src_750, _ = tasks_and_sources()
    ea = torch.load(CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    q32_750, y16_750, refs_d = ea["q32_750"], ea["y16_750"], ea["refs150"]
    e_refs = style_enc(refs_d["stat_1561"].to(device).float())
    z_refs = refs_d["z_ref"].to(device).float()

    # Fixed 5 cross-style tasks (1 per source style -> rotated Core-5 targets)
    picks = [1, 182, 363, 544, 600]  # (0->1, 1->2, 2->3, 3->4, 4->0)
    cell, pad = 160, 8
    cols = 4
    canvas = Image.new("RGB", (cols * (cell + pad) + pad, len(picks) * (cell + pad) + 32), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "V36 D1 Oracle Grid (tau0=0.65, wj=3.8): Src | E-Core100+A3 Draft | V36(Draft) | V36(A3 Src)", fill=(235, 235, 240))
    eps_b = torch.randn(len(picks), 4, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(EVAL_SEED))

    idx_t = torch.tensor([int(tasks[i]["target_style_id"]) * 30 + (i % 30) for i in picks], dtype=torch.long)
    zc = z_src_750[picks].to(device).float()
    ze = z_e_750[picks].to(device).float()
    y16 = y16_750[picks].to(device).float()
    q32 = q32_750[picks].to(device).float()
    zr = z_refs[idx_t]
    es = e_refs[idx_t]
    ze_a3 = reconstruct_reference_a3(ze, zr, priors, mode="I2")
    zs_a3 = reconstruct_reference_a3(zc, zr, priors, mode="I2")

    zo_draft = sample_v36_ode(pcore, ze_a3, y16, q32, es, eps_b, tau0=0.65, steps=6, wj=3.8, wc=1.6, device=device)
    zo_a3src = sample_v36_ode(pcore, zs_a3, y16, q32, es, eps_b, tau0=0.65, steps=6, wj=3.8, wc=1.6, device=device)

    col_imgs = [
        decode_s10(dec, zc, batch=len(picks)),
        decode_s10(dec, ze_a3, batch=len(picks)),
        decode_s10(dec, zo_draft, batch=len(picks)),
        decode_s10(dec, zo_a3src, batch=len(picks)),
    ]
    for ri in range(len(picks)):
        y0 = 32 + ri * (cell + pad)
        for ci, batch_rgb in enumerate(col_imgs):
            im = Image.fromarray((batch_rgb[ri].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((cell, cell), Image.Resampling.BILINEAR)
            canvas.paste(im, (pad + ci * (cell + pad), y0))
    out_png = FIG_DIR / "v36_diag_d1_oracle_grid.png"
    canvas.save(out_png)
    del dec
    return out_png


def run_diag_d1(device: str = "cuda") -> dict[str, Any]:
    """D1 (Diagnostic Oracle): Feed closed-set V35 E-Core s=1.00 drafts into frozen V36 P-Core."""
    tasks, z_src_750, q32_750, y16_750, refs_d, idx_ok, idx_wr = _load_diag_assets()
    ec_pack = torch.load(ROOT / "runs/v35_final/p2/cached_latents/ecore_s100.pt", map_location="cpu", weights_only=False)
    id_to_idx = {tid: i for i, tid in enumerate(ec_pack["ids"])}
    z_e_750 = torch.stack([ec_pack["z"][id_to_idx[t["task_id"]]] for t in tasks], dim=0).float()

    p2_ckpt = ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
    p1_ckpt = ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p1_final.pt"

    pcore_p2, style_p2, priors = load_v36_models(p2_ckpt, device)
    dec = load_s10_decoder(device)
    e_refs_p2 = style_p2(refs_d["stat_1561"].to(device).float()).cpu()
    z_refs = refs_d["z_ref"]
    ze_a3_ok = reconstruct_reference_a3(z_e_750, z_refs[idx_ok].float(), priors, mode="I2").cpu()

    _save_d1_oracle_grid(pcore_p2, style_p2, priors, z_e_750, device=device)

    pending: list[tuple[Path, Path, str]] = []
    metas: dict[str, dict[str, Any]] = {}

    def _add(tag: str, z_st: torch.Tensor, es, pcore_m, *, tau0: float, steps: int, note: str, wrong: bool = False):
        r_d, m_d = _generate_pending_arm(
            tag, z_start_750=z_st, e_style_750=es, q32_750=q32_750, y16_750=y16_750,
            tasks=tasks, dec=dec, pcore=pcore_m, tau0=tau0, wj=3.8, wc=1.6, steps=steps, wrong=wrong, device=device,
        )
        pending.append((r_d, m_d, tag))
        metas[tag] = {"diag": "D1_ORACLE_CLOSED_SET_DRAFT", "tau0": tau0, "steps": steps, "note": note, "wrong": wrong}

    _add("D1_raw_ecore100_0NFE", z_e_750, None, None, tau0=1.0, steps=0, note="Raw E-Core s=1.00 draft (0-NFE)")
    _add("D1_raw_ecore100_A3_0NFE", ze_a3_ok, None, None, tau0=1.0, steps=0, note="E-Core s=1.00 + A3 I2 draft (0-NFE)")

    for t0 in (0.50, 0.65, 0.75):
        ti = int(round(t0 * 100))
        _add(f"D1_p2_ecore100_t{ti:02d}", z_e_750, e_refs_p2[idx_ok], pcore_p2, tau0=t0, steps=6, note="V36 P2 on raw E-Core100")
        _add(f"D1_p2_ecore100A3_t{ti:02d}", ze_a3_ok, e_refs_p2[idx_ok], pcore_p2, tau0=t0, steps=6, note="V36 P2 on E-Core100+A3")
    _add("D1_p2_ecore100_t65_wrongStyle", z_e_750, e_refs_p2[idx_wr], pcore_p2, tau0=0.65, steps=6, note="Fixed E-Core100, wrong e_style", wrong=True)
    _add("D1_p2_ecore100A3_t65_wrongStyle", ze_a3_ok, e_refs_p2[idx_wr], pcore_p2, tau0=0.65, steps=6, note="Fixed E-Core100+A3, wrong e_style", wrong=True)

    del pcore_p2, style_p2
    gc.collect()
    torch.cuda.empty_cache()

    pcore_p1, style_p1, _ = load_v36_models(p1_ckpt, device)
    e_refs_p1 = style_p1(refs_d["stat_1561"].to(device).float()).cpu()
    _add("D1_p1_ecore100_t65", z_e_750, e_refs_p1[idx_ok], pcore_p1, tau0=0.65, steps=6, note="V36 P1(42k) on raw E-Core100")
    _add("D1_p1_ecore100A3_t65", ze_a3_ok, e_refs_p1[idx_ok], pcore_p1, tau0=0.65, steps=6, note="V36 P1(42k) on E-Core100+A3")

    del pcore_p1, style_p1, dec
    gc.collect()
    torch.cuda.empty_cache()

    rows = _flush_measure(pending, metas, device)
    for ro in rows:
        print(f"[D1 {ro['arm']}] Top1={ro['cross600']['csd_top1']:.4f} D-C={ro['cross600']['dino_c']:.4f} D-S={ro['cross600']['dino_s']:.4f} Margin={ro['cross600']['csd_margin']:+.4f}", flush=True)
    rep = {"utc": utc(), "diagnostic_warning": "Closed-set E-Core saw Core-5; Oracle draft upper bound only.", "rows": rows}
    save_json(MEAS_DIR / "v36_diag_d1_oracle.json", rep)
    return rep


def run_diag_d2_d3(device: str = "cuda") -> dict[str, Any]:
    """D2: Fix A3 I2 startpoint and swap/zero ONLY StyleMLPEncoder. D3: Direct A3 (0-NFE) decode baseline."""
    tasks, z_src_750, q32_750, y16_750, refs_d, idx_ok, idx_wr = _load_diag_assets()
    p2_ckpt = ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
    p1_ckpt = ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p1_final.pt"

    pcore_p2, style_p2, priors = load_v36_models(p2_ckpt, device)
    dec = load_s10_decoder(device)
    e_refs_p2 = style_p2(refs_d["stat_1561"].to(device).float()).cpu()
    z_a3_ok = reconstruct_reference_a3(z_src_750.float(), refs_d["z_ref"][idx_ok].float(), priors, mode="I2").cpu()

    pending: list[tuple[Path, Path, str]] = []
    metas: dict[str, dict[str, Any]] = {}

    def _add(tag: str, z_st: torch.Tensor, es, pcore_m, *, tau0: float, steps: int, drop_zero: bool = False, note: str = "", wrong: bool = False):
        r_d, m_d = _generate_pending_arm(
            tag, z_start_750=z_st, e_style_750=es, q32_750=q32_750, y16_750=y16_750,
            tasks=tasks, dec=dec, pcore=pcore_m, tau0=tau0, wj=3.8, wc=1.6, steps=steps,
            drop_style_zero=drop_zero, wrong=wrong, device=device,
        )
        pending.append((r_d, m_d, tag))
        metas[tag] = {"diag": "D3" if tag.startswith("D3") else "D2", "tau0": tau0, "steps": steps, "note": note, "wrong": wrong}

    # D3: 0-NFE Reference-A3 I2 direct & Raw source
    _add("D3_refA3_I2_direct_0NFE", z_a3_ok, None, None, tau0=1.0, steps=0, note="0-NFE Reference-A3 I2 direct decode")
    _add("D3_raw_src_0NFE", z_src_750.float(), None, None, tau0=1.0, steps=0, note="0-NFE raw source latent decode")

    # D2: Fixed A3 startpoint, swap/zero StyleMLP only
    for t0 in (0.35, 0.50):
        ti = int(round(t0 * 100))
        _add(f"D2_p2_A3ok_styleWrong_t{ti:02d}", z_a3_ok, e_refs_p2[idx_wr], pcore_p2, tau0=t0, steps=6, note="Fixed A3(correct), wrong e_style", wrong=True)
        _add(f"D2_p2_A3ok_styleZero_t{ti:02d}", z_a3_ok, None, pcore_p2, tau0=t0, steps=6, drop_zero=True, note="Fixed A3(correct), drop_style=True")

    del pcore_p2, style_p2
    gc.collect()
    torch.cuda.empty_cache()

    pcore_p1, style_p1, _ = load_v36_models(p1_ckpt, device)
    e_refs_p1 = style_p1(refs_d["stat_1561"].to(device).float()).cpu()
    _add("D2_p1_A3ok_styleWrong_t50", z_a3_ok, e_refs_p1[idx_wr], pcore_p1, tau0=0.50, steps=6, note="P1(42k) fixed A3(correct), wrong e_style", wrong=True)

    del pcore_p1, style_p1, dec
    gc.collect()
    torch.cuda.empty_cache()

    rows = _flush_measure(pending, metas, device)
    for ro in rows:
        print(f"[{ro['arm']}] Top1={ro['cross600']['csd_top1']:.4f} D-C={ro['cross600']['dino_c']:.4f} D-S={ro['cross600']['dino_s']:.4f} LPIPS={ro['cross600']['lpips']:.4f}", flush=True)
    rep = {"utc": utc(), "rows": rows}
    save_json(MEAS_DIR / "v36_diag_d2_d3.json", rep)
    return rep


@torch.no_grad()
def run_diag_d4(device: str = "cuda", *, n_pairs: int = 300) -> dict[str, Any]:
    """D4: Seen-Dev (131 seen styles) vs Artist-OOD (unseen artists) vs Movement-OOD (Core-5 unseen styles)."""
    ready = read_json(CACHE_DIR / "v36_caches_ready.json")
    manifest = read_json(CACHE_DIR / "curation_manifest.json")
    p2_ckpt = ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
    p1_ckpt = ROOT / "runs/v36_art_foundation/checkpoints/v36_main_p1_final.pt"

    pcore_p2, style_p2, priors = load_v36_models(p2_ckpt, device)
    pcore_p1, style_p1, _ = load_v36_models(p1_ckpt, device)
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    mb = auto_referee_batch(device)

    def _np(x) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()

    def _extract_rgb_feats(z_lat: torch.Tensor) -> dict[str, np.ndarray]:
        d_l, c_l, cl_l = [], [], []
        for s in range(0, len(z_lat), 24):
            rgb = decode_s10(dec, z_lat[s:s + 24].to(device).float(), batch=min(24, len(z_lat) - s), output_cpu=True)
            imgs = [Image.fromarray((rgb[i].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)) for i in range(rgb.shape[0])]
            fo = stack.extract(imgs, batch=mb)
            d_l.append(_np(fo["dino"]))
            c_l.append(_np(fo["csd"]))
            cl_l.append(_np(fo["clip"]))
        return {"dino": np.concatenate(d_l, 0), "csd": np.concatenate(c_l, 0), "clip": np.concatenate(cl_l, 0)}

    split_specs = [
        ("seen_dev", ready["dev_info"], manifest["seen_dev"]),
        ("artist_ood", ready["art_info"], manifest["artist_ood"]),
        ("movement_ood", ready["mv_info"], manifest["movement_ood"]),
    ]

    rng = np.random.default_rng(EVAL_SEED)
    results: dict[str, Any] = {}

    for sp_name, sp_info, sp_recs in split_specs:
        N = int(sp_info["n"])
        mm_z = np.lib.format.open_memmap(ROOT / sp_info["paths"]["z_gt"], mode="r", dtype=np.float16, shape=(N, 4, 64, 64))
        mm_q = np.lib.format.open_memmap(ROOT / sp_info["paths"]["q32"], mode="r", dtype=np.float16, shape=(N, 32, 16, 16))
        mm_y = np.lib.format.open_memmap(ROOT / sp_info["paths"]["y16"], mode="r", dtype=np.float16, shape=(N, 1, 16, 16))
        mm_st = np.lib.format.open_memmap(ROOT / sp_info["paths"]["stat"], mode="r", dtype=np.float16, shape=(N, 1561))

        styles = [r["style"] for r in sp_recs]
        c_idx = rng.permutation(N)[: min(n_pairs, N)]
        r_ok_idx = np.empty_like(c_idx)
        r_wr_idx = np.empty_like(c_idx)
        for k, ci in enumerate(c_idx):
            cs = styles[ci]
            cand_ok = [j for j in range(N) if styles[j] != cs]
            rk = cand_ok[int(rng.integers(0, len(cand_ok)))]
            r_ok_idx[k] = rk
            ts = styles[rk]
            cand_wr = [j for j in range(N) if styles[j] != cs and styles[j] != ts]
            r_wr_idx[k] = cand_wr[int(rng.integers(0, len(cand_wr)))]

        zc = torch.from_numpy(np.array(mm_z[c_idx])).float()
        y16 = torch.from_numpy(np.array(mm_y[c_idx])).float()
        q32 = torch.from_numpy(np.array(mm_q[c_idx])).float()
        zr_ok = torch.from_numpy(np.array(mm_z[r_ok_idx])).float()
        zr_wr = torch.from_numpy(np.array(mm_z[r_wr_idx])).float()
        st_ok = torch.from_numpy(np.array(mm_st[r_ok_idx])).float()

        fc = _extract_rgb_feats(zc)
        fr_ok = _extract_rgb_feats(zr_ok)
        fr_wr = _extract_rgb_feats(zr_wr)
        z_a3 = reconstruct_reference_a3(zc.to(device), zr_ok.to(device), priors, mode="I2").cpu()
        eps = torch.randn(zc.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

        def _gen_ode(pcore_m, style_m, tau0: float) -> torch.Tensor:
            outs = []
            es_all = style_m(st_ok.to(device)).cpu()
            for s in range(0, len(zc), 24):
                sl = slice(s, min(s + 24, len(zc)))
                zo = sample_v36_ode(
                    pcore_m, z_a3[sl].to(device), y16[sl].to(device), q32[sl].to(device),
                    es_all[sl].to(device), eps[sl].to(device), tau0=tau0, steps=6, wj=3.8, wc=1.6, device=device,
                )
                outs.append(zo.cpu())
            return torch.cat(outs, 0)

        arms_latents = [
            ("refA3_direct_0NFE", z_a3),
            ("p1_42k_I2_t35", _gen_ode(pcore_p1, style_p1, 0.35)),
            ("p1_42k_I2_t50", _gen_ode(pcore_p1, style_p1, 0.50)),
            ("p2_50k_I2_t35", _gen_ode(pcore_p2, style_p2, 0.35)),
            ("p2_50k_I2_t50", _gen_ode(pcore_p2, style_p2, 0.50)),
        ]
        sp_rows = []
        for arm_lbl, z_out in arms_latents:
            fo = _extract_rgb_feats(z_out)
            dino_c = float((fo["dino"] * fc["dino"]).sum(-1).mean())
            dino_s = float((fo["dino"] * fr_ok["dino"]).sum(-1).mean())
            clip_s = float((fo["clip"] * fr_ok["clip"]).sum(-1).mean())
            csd_tgt = (fo["csd"] * fr_ok["csd"]).sum(-1)
            csd_wr = (fo["csd"] * fr_wr["csd"]).sum(-1)
            csd_src = (fo["csd"] * fc["csd"]).sum(-1)
            row = {
                "split": sp_name,
                "arm": arm_lbl,
                "n": int(len(zc)),
                "dino_c": round(dino_c, 4),
                "dino_s": round(dino_s, 4),
                "clip_s": round(clip_s, 4),
                "csd_tgt_cos": round(float(csd_tgt.mean()), 4),
                "csd_2way_follow": round(float((csd_tgt > csd_wr).mean()), 4),
                "csd_margin_vs_wrong": round(float((csd_tgt - csd_wr).mean()), 4),
                "csd_gain_vs_src": round(float((csd_tgt > csd_src).mean()), 4),
            }
            sp_rows.append(row)
            print(f"[D4 {sp_name:12s} | {arm_lbl:18s}] 2Way={row['csd_2way_follow']*100:.2f}% Margin={row['csd_margin_vs_wrong']:+.4f} D-C={row['dino_c']:.4f} D-S={row['dino_s']:.4f}", flush=True)
        results[sp_name] = sp_rows

    del pcore_p2, style_p2, pcore_p1, style_p1, dec, stack
    gc.collect()
    torch.cuda.empty_cache()

    rep = {"utc": utc(), "n_pairs": n_pairs, "splits": results}
    save_json(MEAS_DIR / "v36_diag_d4_seen_vs_ood.json", rep)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "d0", "d1", "d23", "d4"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sanitize_cuda_env()
    MEAS_DIR.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {"utc": utc()}
    if a.which in ("all", "d1"):
        out["d1"] = run_diag_d1(a.device)
    if a.which in ("all", "d23"):
        out["d23"] = run_diag_d2_d3(a.device)
    if a.which in ("all", "d4"):
        out["d4"] = run_diag_d4(a.device)
    if a.which in ("all", "d0"):
        out["d0"] = run_diag_d0(a.device)
    save_json(MEAS_DIR / "v36_diagnostics_summary.json", out)
    print("[V36 Diagnostics] ALL REQUESTED DIAGNOSTICS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
