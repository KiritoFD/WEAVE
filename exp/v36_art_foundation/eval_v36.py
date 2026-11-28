"""V36 Evaluation & Inference Suite:
- Reference-A3 (zero E-Core, zero class ID): extracts low-freq luminance (u_y) and chrominance (P_C)
  moments directly from the single reference painting latent z_ref, clamps scale ratios in [0.5, 2.0],
  and constructs I0 / I1 / I2 analytic starting points cleanly.
- Stage P3: Seen-Dev calibration of start mode (I0 vs I1 vs I2 Reference-A3) & CFG -> frozen config.
- Stage P4: Full official test750 (Core-5 strictly unseen movements in 1-shot reference mode + wrong ctrl),
  plus Artist-OOD, Movement-OOD, Instance-Swap, Content-Swap, SG-25, E2E throughput, and visual grids.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v29_biglittle import RawWriterV29, compute_dino_top5_mean, csd_prototypes
from exp.v34_frontier.stage_a_startpoint import decompose_py_luma
from exp.v36_art_foundation import CACHE_DIR, FIG_DIR, MEAS_DIR, ROOT, SEED_V36
from exp.v36_art_foundation.models import FoundationPCoreNet, StyleMLPEncoder
from src.eval.runner import run_measure_subprocess
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, utc


@torch.no_grad()
def reconstruct_reference_a3(
    z_src: torch.Tensor,
    z_ref: torch.Tensor,
    priors: dict[str, torch.Tensor],
    *,
    mode: str = "I2",
    alpha_y: float = 1.0,
    alpha_c: float = 1.0,
    ratio_min: float = 0.5,
    ratio_max: float = 2.0,
    kernel_size: int = 19,
    sigma: float = 4.5,
) -> torch.Tensor:
    """Single-reference Reference-A3 analytic starting point (zero E-Core, zero class ID).

    Modes:
      - I0: raw source latent z_src
      - I1: chrominance P_C moment matching only (P_Y untouched)
      - I2: joint L_low (along u_y) + P_C redistribution (L_high untouched), std ratios clamped [0.5, 2.0]
    """
    if mode == "I0":
        return z_src
    dev = z_src.device
    zf = z_src.float()
    rf = z_ref.to(dev).float()
    p_y = priors["P_Y"].to(dev).float()
    p_c = priors["P_C"].to(dev).float()
    u_y = priors["u_y"].to(dev).float()

    l_low_s, l_high_s, zy_s = decompose_py_luma(zf, p_y, kernel_size=kernel_size, sigma=sigma)
    l_low_r, _, _ = decompose_py_luma(rf, p_y, kernel_size=kernel_size, sigma=sigma)
    zc_s = torch.einsum("ij,bjhw->bihw", p_c, zf)
    zc_r = torch.einsum("ij,bjhw->bihw", p_c, rf)

    if mode == "I2" and alpha_y > 0.0:
        sl_s = torch.einsum("c,bchw->bhw", u_y, l_low_s).unsqueeze(1)
        sl_r = torch.einsum("c,bchw->bhw", u_y, l_low_r).unsqueeze(1)
        mu_s, sd_s = sl_s.mean(dim=(2, 3), keepdim=True), sl_s.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        mu_r, sd_r = sl_r.mean(dim=(2, 3), keepdim=True), sl_r.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        scale_y = (sd_r / sd_s).clamp(ratio_min, ratio_max)
        sl_target = (sl_s - mu_s) * scale_y + mu_r
        l_low_target = sl_target * u_y.view(1, 4, 1, 1)
        zy_out = zy_s + alpha_y * (l_low_target - l_low_s)
    else:
        zy_out = zy_s

    if mode in ("I1", "I2") and alpha_c > 0.0:
        mu_cs, sd_cs = zc_s.mean(dim=(2, 3), keepdim=True), zc_s.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        mu_cr, sd_cr = zc_r.mean(dim=(2, 3), keepdim=True), zc_r.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        scale_c = (sd_cr / sd_cs).clamp(ratio_min, ratio_max)
        zc_target = torch.einsum("ij,bjhw->bihw", p_c, (zc_s - mu_cs) * scale_c + mu_cr)
        zc_out = zc_s + alpha_c * (zc_target - zc_s)
    else:
        zc_out = zc_s

    return zy_out + zc_out


def load_v36_models(ckpt_path: Path, device: str = "cuda"):
    priors = torch.load(CACHE_DIR / "v36_seen_train_priors.pt", map_location="cpu", weights_only=False)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pcore = FoundationPCoreNet(
        base_dim=208, style_dim=256, cond_dim=384,
        use_gamma=True, normalize_style=True, per_layer_adagn=True, use_checkpoint=False,
    ).to(device).eval()
    pcore.load_state_dict(ck["pcore_ema"], strict=True)
    style_enc = StyleMLPEncoder(1561, 512, 256, priors["stat_mean"], priors["stat_std"]).to(device).eval()
    style_enc.load_state_dict(ck["style_enc_ema"], strict=True)
    return pcore, style_enc, priors


@torch.no_grad()
def sample_v36_ode(
    pcore: FoundationPCoreNet,
    z_start: torch.Tensor,
    y16: torch.Tensor,
    q32: torch.Tensor,
    e_s: torch.Tensor,
    eps: torch.Tensor,
    *,
    tau0: float = 0.50,
    steps: int = 6,
    wj: float = 3.2,
    wc: float = 1.6,
    gamma_val: float = 0.85,
    device: str = "cuda",
) -> torch.Tensor:
    """Fast 1-NFE/step DPM-Solver-2M ODE integrator with two-axis CFG (ws + wc)."""
    B = z_start.shape[0]
    z = (1.0 - tau0) * eps + tau0 * z_start if tau0 > 0 else eps.clone()
    dt = (1.0 - tau0) / float(steps)
    f0 = torch.zeros(B, dtype=torch.bool, device=device)
    f1 = torch.ones(B, dtype=torch.bool, device=device)
    g = torch.full((B,), gamma_val, device=device)
    l1 = torch.ones(B, device=device)
    l0 = torch.zeros(B, device=device)
    zeros_s = torch.zeros_like(e_s)

    def _vel(zz: torch.Tensor, t_scalar: float) -> torch.Tensor:
        if abs(wc - 1.0) > 1e-3:
            z3 = zz.repeat(3, 1, 1, 1)
            y3 = y16.repeat(3, 1, 1, 1)
            q3 = q32.repeat(3, 1, 1, 1)
            s3 = torch.cat([e_s, zeros_s, zeros_s], dim=0)
            t3 = torch.full((B * 3,), t_scalar, device=device)
            ds3 = torch.cat([f0, f1, f1], dim=0)
            dc3 = torch.cat([f0, f0, f1], dim=0)
            ls3 = torch.cat([l1, l1, l0], dim=0)
            g3 = g.repeat(3)
            with bf16_autocast(device):
                v_sc, v_0c, v_00 = pcore(z3, y3, q3, s3, t3, drop_style=ds3, drop_content=dc3, layout_scale=ls3, gamma=g3).float().chunk(3, dim=0)
            vr = v_00 + wc * (v_0c - v_00) + wj * (v_sc - v_0c)
        else:
            z2 = zz.repeat(2, 1, 1, 1)
            y2 = y16.repeat(2, 1, 1, 1)
            q2 = q32.repeat(2, 1, 1, 1)
            s2 = torch.cat([e_s, zeros_s], dim=0)
            t2 = torch.full((B * 2,), t_scalar, device=device)
            ds2 = torch.cat([f0, f1], dim=0)
            dc2 = torch.cat([f0, f0], dim=0)
            ls2 = l1.repeat(2)
            g2 = g.repeat(2)
            with bf16_autocast(device):
                v_sc, v_0c = pcore(z2, y2, q2, s2, t2, drop_style=ds2, drop_content=dc2, layout_scale=ls2, gamma=g2).float().chunk(2, dim=0)
            vr = v_0c + wj * (v_sc - v_0c)
        s_ref = v_sc.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
        s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
        return 0.55 * vr * (s_ref / s_raw) + 0.45 * vr

    prev_v = None
    for i in range(steps):
        v_cur = _vel(z, tau0 + i * dt)
        if i == 0 or prev_v is None:
            z = z + dt * v_cur
        else:
            z = z + dt * (1.5 * v_cur - 0.5 * prev_v)
        prev_v = v_cur
    return z


@torch.no_grad()
def calibrate_stage_p3(ckpt_path: Path, *, device: str = "cuda") -> dict:
    """Stage P3: Calibrate I0 / I1 / I2 (Reference-A3) strictly on Seen-Dev using frozen S10 + VGG style/content proxy."""
    out_p = MEAS_DIR / "p3_seen_dev_calibration.json"
    if out_p.exists():
        return read_json(out_p)
    MEAS_DIR.mkdir(parents=True, exist_ok=True)

    pcore, style_enc, priors = load_v36_models(ckpt_path, device)
    dec = load_s10_decoder(device)
    from exp.v35_cleave_one.losses import FrozenVGGStyleCritic
    vgg = FrozenVGGStyleCritic(device=device)

    ready = read_json(CACHE_DIR / "v36_caches_ready.json")
    dev_info = ready["dev_info"]
    N_d = int(dev_info["n"])
    mm_z = np.lib.format.open_memmap(ROOT / dev_info["paths"]["z_gt"], mode="r", dtype=np.float16, shape=(N_d, 4, 64, 64))
    mm_y = np.lib.format.open_memmap(ROOT / dev_info["paths"]["y16"], mode="r", dtype=np.float16, shape=(N_d, 1, 16, 16))
    mm_q = np.lib.format.open_memmap(ROOT / dev_info["paths"]["q32"], mode="r", dtype=np.float16, shape=(N_d, 32, 16, 16))
    mm_st = np.lib.format.open_memmap(ROOT / dev_info["paths"]["stat"], mode="r", dtype=np.float16, shape=(N_d, 1561))

    idx_c = np.arange(0, min(32, N_d))
    idx_ra = (idx_c + 123) % N_d
    idx_rb = (idx_c + 317) % N_d
    zc = torch.from_numpy(np.array(mm_z[idx_c])).to(device).float()
    y16 = torch.from_numpy(np.array(mm_y[idx_c])).to(device).float()
    q32 = torch.from_numpy(np.array(mm_q[idx_c])).to(device).float()
    zra = torch.from_numpy(np.array(mm_z[idx_ra])).to(device).float()
    zrb = torch.from_numpy(np.array(mm_z[idx_rb])).to(device).float()
    ea = style_enc(torch.from_numpy(np.array(mm_st[idx_ra])).to(device).float())
    eb = style_enc(torch.from_numpy(np.array(mm_st[idx_rb])).to(device).float())

    def _vgg_chunks(z_in: torch.Tensor, fn) -> torch.Tensor:
        outs = []
        for s in range(0, z_in.shape[0], 8):
            xb = decode_s10(dec, z_in[s:s + 8], batch=8, output_cpu=False)
            outs.append(fn(xb))
        return torch.cat(outs, dim=0)

    vc_gt = _vgg_chunks(zc, vgg.gray_content_vec)
    vra = _vgg_chunks(zra, vgg.style_vec)
    vrb = _vgg_chunks(zrb, vgg.style_vec)

    rng_e = torch.Generator(device=device).manual_seed(EVAL_SEED)
    eps = torch.randn(zc.shape, device=device, generator=rng_e)

    rows = []
    for mode in ("I0", "I1", "I2"):
        for wj in (2.8, 3.2, 3.8):
            za_s = reconstruct_reference_a3(zc, zra, priors, mode=mode)
            zb_s = reconstruct_reference_a3(zc, zrb, priors, mode=mode)
            zo_a = sample_v36_ode(pcore, za_s, y16, q32, ea, eps, tau0=0.50, steps=6, wj=wj, wc=1.6, device=device)
            zo_b = sample_v36_ode(pcore, zb_s, y16, q32, eb, eps, tau0=0.50, steps=6, wj=wj, wc=1.6, device=device)
            vxa = _vgg_chunks(zo_a, vgg.style_vec)
            vxb = _vgg_chunks(zo_b, vgg.style_vec)
            ca = _vgg_chunks(zo_a, vgg.gray_content_vec)
            cb = _vgg_chunks(zo_b, vgg.gray_content_vec)
            daa = (vxa * vra).sum(-1)
            dab = (vxa * vrb).sum(-1)
            dbb = (vxb * vrb).sum(-1)
            dba = (vxb * vra).sum(-1)
            follow = 0.5 * ((daa > dab).float().mean().item() + (dbb > dba).float().mean().item())
            margin = 0.5 * ((daa - dab).mean().item() + (dbb - dba).mean().item())
            c_err = 0.5 * (F.l1_loss(ca, vc_gt).item() + F.l1_loss(cb, vc_gt).item())
            rows.append({
                "mode": mode, "wj": wj, "wc": 1.6,
                "vgg_follow_2way": round(follow, 4),
                "vgg_style_margin": round(margin, 5),
                "vgg_gray_content_l1": round(c_err, 5),
            })
    best = max(rows, key=lambda r: r["vgg_style_margin"] - 0.5 * r["vgg_gray_content_l1"])
    rep = {"utc": utc(), "ckpt": str(ckpt_path), "grid": rows, "best": best}
    save_json(out_p, rep)
    print(f"[V36 P3 Seen-Dev] Best start/CFG -> {best}", flush=True)
    return rep


def _compute_v36_csls_k10(rows: list[dict]) -> float:
    """Compute Cross600 CSLS@10 accuracy directly from eval.json rows."""
    cross_rows = [r for r in rows if r.get("is_cross")]
    src_feats = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)
    tasks = read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]
    stems = sorted({t["source_stem"] for t in tasks})
    proto = csd_prototypes()
    bg = np.stack([src_feats["raw_csd"][s] for s in stems])
    bg_sim = bg @ proto.T
    r_proto = np.sort(bg_sim, axis=0)[-10:].mean(0)
    x = np.array([r["csd_scores_5"] for r in cross_rows])
    y = np.array([r["tgt_style_id"] for r in cross_rows])
    pred = (2.0 * x - r_proto).argmax(axis=1)
    return round(float((pred == y).mean()), 4)


@torch.no_grad()
def _generate_test750_v36(
    ckpt_path: Path,
    *,
    tag: str,
    start_mode: str = "I2",
    tau0: float = 0.50,
    wj: float = 3.2,
    wc: float = 1.6,
    steps: int = 6,
    wrong: bool = False,
    device: str = "cuda",
) -> Path:
    m_dir = MEAS_DIR / tag
    if (m_dir / "summary_v36.json").exists():
        return m_dir

    if (m_dir / "summary_arm.json").exists():
        summary = read_json(m_dir / "summary_arm.json")
    else:
        pcore, style_enc, priors = load_v36_models(ckpt_path, device)
        dec = load_s10_decoder(device)
        tasks, z_src_750, _ = tasks_and_sources()
        ea_assets = torch.load(CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
        q32_750 = ea_assets["q32_750"]
        y16_750 = ea_assets["y16_750"]
        refs_d = ea_assets["refs150"]
        e_refs = style_enc(refs_d["stat_1561"].to(device).float()).cpu()  # (150, 256)
        z_refs = refs_d["z_ref"]  # (150, 4, 64, 64)

        sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
        tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
        eff_tid = pick_wrong_cross_tid(sid_750, tid_750) if wrong else tid_750
        eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

        raw_dir = ROOT / "runs/v36_art_foundation/raw" / tag
        raw_dir.mkdir(parents=True, exist_ok=True)
        writer = RawWriterV29(tag, dec=dec, device=device)
        writer.path = raw_dir

        bs = 24
        for st in range(0, len(tasks), bs):
            ed = min(st + bs, len(tasks))
            sl = slice(st, ed)
            r_idx_t = torch.tensor([int(eff_tid[i].item()) * 30 + (i % 30) for i in range(st, ed)], dtype=torch.long)
            zc = z_src_750[sl].to(device).float()
            y16 = y16_750[sl].to(device).float()
            q32 = q32_750[sl].to(device).float()
            zr = z_refs[r_idx_t].to(device).float()
            es = e_refs[r_idx_t].to(device)
            z_start = reconstruct_reference_a3(zc, zr, priors, mode=start_mode)
            eb = eps_750[sl].to(device).float()
            zo = sample_v36_ode(pcore, z_start, y16, q32, es, eb, tau0=tau0, steps=steps, wj=wj, wc=wc, device=device)
            writer.write_batch(zo, tasks[sl])
        writer.finalize(arm=tag, lambda_val=tau0, wrong=wrong)

        del pcore, style_enc, dec
        gc.collect()
        torch.cuda.empty_cache()

        summary = run_measure_subprocess(raw_dir, m_dir, arm_name=tag, device=device)
        for bf in raw_dir.glob("batch_*.pt"):
            bf.unlink()
    s_c = summary["cross600"]
    s_s = summary["same150"]
    ev = read_json(m_dir / "eval.json")
    rows = ev["rows"]
    out_sum = {
        "arm": tag,
        "start_mode": start_mode,
        "tau0": tau0,
        "wj": wj,
        "wc": wc,
        "steps": steps,
        "cross600": {
            "csd_top1": float(s_c["csd_top1"]),
            "csd_margin": float(s_c["csd_margin"]),
            "dino_c": float(s_c["dino_c"]),
            "dino_s": float(s_c["dino_s"]),
            "dino_s_top5": compute_dino_top5_mean(m_dir),
            "csls_k10": _compute_v36_csls_k10(rows),
            "clip_s": float(s_c["clip_s"]),
            "style_affinity": float(s_c["style_affinity"]),
            "lpips": float(s_c["lpips"]),
        },
        "same150": {"csd_top1": float(s_s["csd_top1"]), "dino_c": float(s_s["dino_c"])},
    }
    save_json(m_dir / "summary_v36.json", out_sum)
    return m_dir


@torch.no_grad()
def _save_visual_grid(
    ckpt_path: Path,
    *,
    tag: str,
    start_mode: str = "I2",
    tau0: float = 0.50,
    wj: float = 3.2,
    wc: float = 1.6,
    steps: int = 6,
    device: str = "cuda",
) -> Path:
    """Export fixed qualitative Core-5 unseen-movement + same-movement instance swap grid."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pcore, style_enc, priors = load_v36_models(ckpt_path, device)
    dec = load_s10_decoder(device)
    tasks, z_src_750, _ = tasks_and_sources()
    ea_assets = torch.load(CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    q32_750, y16_750, refs_d = ea_assets["q32_750"], ea_assets["y16_750"], ea_assets["refs150"]
    e_refs = style_enc(refs_d["stat_1561"].to(device).float())
    z_refs = refs_d["z_ref"].to(device).float()

    # Pick 5 distinct source images -> 5 Core-5 target refs (first exemplar per style)
    src_picks = [0, 150, 300, 450, 600]
    ref_picks = [0, 30, 60, 90, 120]
    cell = 160
    pad = 8
    cols = 1 + len(ref_picks)
    rows_n = len(src_picks)
    canvas = Image.new("RGB", (cols * (cell + pad) + pad, rows_n * (cell + pad) + 32), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"V36 {tag} ({start_mode}, tau0={tau0}, wj={wj}) | Core-5 Unseen Movements", fill=(235, 235, 240))

    eps_b = torch.randn(len(ref_picks), 4, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(EVAL_SEED))
    for ri, si in enumerate(src_picks):
        y0 = 32 + ri * (cell + pad)
        src_rgb = decode_s10(dec, z_src_750[si:si + 1].to(device).float(), batch=1)[0]
        src_im = Image.fromarray((src_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((cell, cell), Image.Resampling.BILINEAR)
        canvas.paste(src_im, (pad, y0))

        zc = z_src_750[si:si + 1].to(device).float().repeat(len(ref_picks), 1, 1, 1)
        y16 = y16_750[si:si + 1].to(device).float().repeat(len(ref_picks), 1, 1, 1)
        q32 = q32_750[si:si + 1].to(device).float().repeat(len(ref_picks), 1, 1, 1)
        zr = z_refs[ref_picks]
        es = e_refs[ref_picks]
        z_start = reconstruct_reference_a3(zc, zr, priors, mode=start_mode)
        zo = sample_v36_ode(pcore, z_start, y16, q32, es, eps_b, tau0=tau0, steps=steps, wj=wj, wc=wc, device=device)
        outs = decode_s10(dec, zo, batch=len(ref_picks))
        for ci in range(len(ref_picks)):
            oim = Image.fromarray((outs[ci].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((cell, cell), Image.Resampling.BILINEAR)
            canvas.paste(oim, (pad + (ci + 1) * (cell + pad), y0))

    out_png = FIG_DIR / f"{tag}_core5_grid.png"
    canvas.save(out_png)
    return out_png


def eval_v36_full(
    ckpt_path: Path,
    *,
    tag: str,
    start_mode: str = "I2",
    taus: tuple[float, ...] = (0.0, 0.35, 0.50, 0.65, 0.80),
    wj: float = 3.2,
    wc: float = 1.6,
    steps: int = 6,
    device: str = "cuda",
) -> list[dict]:
    """Run Stage P4 full test750 sweep + wrong counterfactuals + grid export."""
    sweep = []
    for t0 in taus:
        sub_tag = f"{tag}_{start_mode}_t{int(round(t0 * 100)):02d}"
        m_dir = _generate_test750_v36(ckpt_path, tag=sub_tag, start_mode=start_mode, tau0=t0, wj=wj, wc=wc, steps=steps, device=device)
        m_wrong = _generate_test750_v36(ckpt_path, tag=f"{sub_tag}_wrong", start_mode=start_mode, tau0=t0, wj=wj, wc=wc, steps=steps, wrong=True, device=device)
        s = read_json(m_dir / "summary_v36.json")
        sw = read_json(m_wrong / "summary_v36.json")
        s["wrong_ctrl_top1"] = sw["cross600"]["csd_top1"]
        s["causal_delta_pp"] = round((s["cross600"]["csd_top1"] - sw["cross600"]["csd_top1"]) * 100.0, 2)
        sweep.append(s)
        save_json(MEAS_DIR / f"{sub_tag}_report.json", s)
        print(f"[V36 P4 {sub_tag}] Top1={s['cross600']['csd_top1']:.4f} D-C={s['cross600']['dino_c']:.4f} D-S={s['cross600']['dino_s']:.4f} Causal={s['causal_delta_pp']:+.2f}pp", flush=True)

    _save_visual_grid(ckpt_path, tag=tag, start_mode=start_mode, tau0=0.50, wj=wj, wc=wc, steps=steps, device=device)
    save_json(MEAS_DIR / f"{tag}_{start_mode}_tau_sweep.json", sweep)
    return sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", default="p4", choices=["p3", "p4", "ablate_starts"])
    ap.add_argument("--tag", default="v36_main")
    ap.add_argument("--start-mode", default="I2", choices=["I0", "I1", "I2"])
    ap.add_argument("--wj", type=float, default=3.2)
    ap.add_argument("--wc", type=float, default=1.6)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sanitize_cuda_env()
    ckpt_p = Path(a.ckpt)

    if a.stage == "p3":
        calibrate_stage_p3(ckpt_p, device=a.device)
    elif a.stage == "ablate_starts":
        for mode in ("I0", "I1", "I2"):
            eval_v36_full(ckpt_p, tag=a.tag, start_mode=mode, taus=(0.50, 0.65), wj=a.wj, wc=a.wc, steps=a.steps, device=a.device)
    else:
        eval_v36_full(ckpt_p, tag=a.tag, start_mode=a.start_mode, wj=a.wj, wc=a.wc, steps=a.steps, device=a.device)


if __name__ == "__main__":
    main()
