"""F3: Complete CLEAVE-One (V36 P2 50k) Instance Control & Color-Matched Swap Audit.

Zero training / zero new learnable parameters. Evaluates cleanly under frozen FeatureStack + Haar/Sobel stroke
response + explicit Reference Layout Leakage (gray edge/structure correlation vs reference image):
  1. Reference-Swap (200 same-movement different-ref triplets from movement_ood).
  2. Color-Matched Swap (reusing V37 palette_p2_50k 64 pairs + full 128 pairs matched in low-freq Oklab palette
     but distant in HF texture, reporting Color Sim, DINO-C, 2-Way Follow, Brushstroke Haar/Sobel Delta, and Layout Leakage).
  3. A3 / Cond 2x2 Layout Leakage audit on V37_D2_A3*_P*_t50 outputs.
  4. Content-Swap (100 same-ref different-content pairs) & SG-25 (125 unseen-movement / unseen-ref pairs).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v36_art_foundation import CACHE_DIR as V36_CACHE_DIR
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from src.utils import FeatureStack, auto_referee_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc
from src.utils.structure import dwt2_haar, rgb_to_gray, rgb_to_oklab


F3_DIR = REPO / "runs/paper_audit/f3_instance_control"
V36_P2_CKPT = REPO / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
V37_PAL_PT = REPO / "runs/v37_editing/diagnostics/palette_p2_50k.pt"
V37_QUAD_JSON = REPO / "runs/v37_editing/diagnostics/core_quadrants.json"


def _sobel_mag(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gp = F.pad(gray.float(), (1, 1, 1, 1), mode="reflect")
    return (F.conv2d(gp, kx).square() + F.conv2d(gp, ky).square() + 1e-8).sqrt()


def _edge_corr(a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
    """Per-sample Sobel edge-map Pearson correlation between two [0,1] RGB batches (downgraded to 64x64)."""
    ga = F.avg_pool2d(rgb_to_gray(a01.float()), 8, 8)
    gb = F.avg_pool2d(rgb_to_gray(b01.float()), 8, 8)
    ea = _sobel_mag(ga).flatten(1)
    eb = _sobel_mag(gb).flatten(1)
    ea = ea - ea.mean(dim=1, keepdim=True)
    eb = eb - eb.mean(dim=1, keepdim=True)
    return F.cosine_similarity(ea, eb, dim=1)


def _stroke_hf_vec(rgb01: torch.Tensor) -> torch.Tensor:
    """Batch Haar high-freq magnitudes (LH, HL, HH) + Sobel RMS -> shape (B, 4)."""
    g = rgb_to_gray(rgb01.float())
    _, lh, hl, hh = dwt2_haar(g)
    sb = _sobel_mag(g)
    return torch.stack(
        [lh.abs().mean((1, 2, 3)), hl.abs().mean((1, 2, 3)), hh.abs().mean((1, 2, 3)), sb.mean((1, 2, 3))],
        dim=1,
    )


def _oklab_mean_sim(a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
    """Global Oklab mean+std similarity: negative L2 distance mapped smoothly or direct L2."""
    la = rgb_to_oklab(a01.float())
    lb = rgb_to_oklab(b01.float())
    va = torch.cat([la.mean((2, 3)), la.std((2, 3), unbiased=False)], dim=1)
    vb = torch.cat([lb.mean((2, 3)), lb.std((2, 3), unbiased=False)], dim=1)
    return (va - vb).pow(2).mean(dim=1).sqrt()


def _feat_batch(stack: FeatureStack, rgb01: torch.Tensor, mb: int) -> dict[str, torch.Tensor]:
    u8 = (rgb01.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
    out = stack.extract(imgs, batch=mb)
    return {
        "dino": out["dino"].cpu().float(),
        "csd": torch.as_tensor(out["csd"]).cpu().float(),
        "clip": out["clip"].cpu().float(),
    }


def _open_memmap_split(split_info: dict[str, Any]) -> dict[str, Any]:
    n = int(split_info["n"])
    p = split_info["paths"]
    return {
        "n": n,
        "z": np.lib.format.open_memmap(REPO / p["z_gt"], mode="r", dtype=np.float16, shape=(n, 4, 64, 64)),
        "y": np.lib.format.open_memmap(REPO / p["y16"], mode="r", dtype=np.float16, shape=(n, 1, 16, 16)),
        "q": np.lib.format.open_memmap(REPO / p["q32"], mode="r", dtype=np.float16, shape=(n, 32, 16, 16)),
        "st": np.lib.format.open_memmap(REPO / p["stat"], mode="r", dtype=np.float16, shape=(n, 1561)),
    }


@torch.no_grad()
def run_f3(device: str = "cuda") -> dict[str, Any]:
    F3_DIR.mkdir(parents=True, exist_ok=True)
    out_p = F3_DIR / "f3_summary.json"
    if out_p.exists():
        return read_json(out_p)

    ready = read_json(V36_CACHE_DIR / "v36_caches_ready.json")
    cm = read_json(V36_CACHE_DIR / "curation_manifest.json")
    mm_dev = _open_memmap_split(ready["dev_info"])
    mm_mv = _open_memmap_split(ready["mv_info"])
    tasks750, z_src_750, _ = tasks_and_sources()
    ea = torch.load(V36_CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)

    pcore, style_enc, priors = load_v36_models(V36_P2_CKPT, device)
    dec = load_s10_decoder(device)

    def _gen_ode(zc: torch.Tensor, y16: torch.Tensor, q32: torch.Tensor, zr: torch.Tensor, sr: torch.Tensor, tau0: float, seed: int) -> torch.Tensor:
        eps = torch.randn(zc.shape, generator=torch.Generator().manual_seed(seed))
        es_all = style_enc(sr.to(device).float())
        outs = []
        for st in range(0, len(zc), 16):
            sl = slice(st, min(len(zc), st + 16))
            zs = reconstruct_reference_a3(zc[sl].to(device).float(), zr[sl].to(device).float(), priors, mode="I2")
            zo = sample_v36_ode(
                pcore, zs, y16[sl].to(device).float(), q32[sl].to(device).float(),
                es_all[sl], eps[sl].to(device).float(), tau0=tau0, steps=6, wj=3.8, wc=1.6, device=device,
            )
            outs.append(decode_s10(dec, zo, batch=16, output_cpu=True).clamp(0.0, 1.0))
        return torch.cat(outs, 0)

    def _decode_z(zt: torch.Tensor) -> torch.Tensor:
        outs = []
        for st in range(0, len(zt), 16):
            outs.append(decode_s10(dec, zt[st:st + 16].to(device).float(), batch=16, output_cpu=True).clamp(0.0, 1.0))
        return torch.cat(outs, 0)

    rng = np.random.default_rng(20261128)
    gen_cache: dict[str, dict[str, Any]] = {}

    # 1. Reference-Swap (200 same-movement different-ref triplets from movement_ood, contents from seen_dev)
    mv_recs = cm["movement_ood"]
    by_mv: dict[str, list[int]] = {}
    for i, r in enumerate(mv_recs):
        by_mv.setdefault(r["style"], []).append(i)
    mvs = [k for k, v in by_mv.items() if len(v) >= 4]
    c_i, ra_i, rb_i = [], [], []
    for k in range(200):
        m = mvs[k % len(mvs)]
        pa, pb = rng.choice(by_mv[m], size=2, replace=False)
        c_i.append((k * 5) % mm_dev["n"])
        ra_i.append(int(pa))
        rb_i.append(int(pb))
    zc = torch.from_numpy(np.array(mm_dev["z"][c_i])).float()
    y16 = torch.from_numpy(np.array(mm_dev["y"][c_i])).float()
    q32 = torch.from_numpy(np.array(mm_dev["q"][c_i])).float()
    zra = torch.from_numpy(np.array(mm_mv["z"][ra_i])).float()
    zrb = torch.from_numpy(np.array(mm_mv["z"][rb_i])).float()
    sra = torch.from_numpy(np.array(mm_mv["st"][ra_i])).float()
    srb = torch.from_numpy(np.array(mm_mv["st"][rb_i])).float()
    gen_cache["ref_swap_t50"] = {
        "xc": _decode_z(zc), "xra": _decode_z(zra), "xrb": _decode_z(zrb),
        "ya": _gen_ode(zc, y16, q32, zra, sra, 0.50, EVAL_SEED + 10),
        "yb": _gen_ode(zc, y16, q32, zrb, srb, 0.50, EVAL_SEED + 10),
    }
    gen_cache["ref_swap_t35"] = {
        "xc": gen_cache["ref_swap_t50"]["xc"], "xra": gen_cache["ref_swap_t50"]["xra"], "xrb": gen_cache["ref_swap_t50"]["xrb"],
        "ya": _gen_ode(zc, y16, q32, zra, sra, 0.35, EVAL_SEED + 10),
        "yb": _gen_ode(zc, y16, q32, zrb, srb, 0.35, EVAL_SEED + 10),
    }

    # 2. Color-Matched Swap (reuse exact V37 palette_p2_50k 64 pairs + fresh 64-pair full A3+P decode at t35 & t50)
    if V37_PAL_PT.exists():
        pal_ck = torch.load(V37_PAL_PT, map_location="cpu", weights_only=False)
        pal_row = pal_ck["row"]
        p_ri = np.asarray(pal_row["ref_a"], dtype=np.int64)
        p_rj = np.asarray(pal_row["ref_b"], dtype=np.int64)
        # Reconstruct exact split_pack("seen_dev", 64) indices
        zc_p = torch.from_numpy(np.array(mm_dev["z"][:64])).float()
        y_p = torch.from_numpy(np.array(mm_dev["y"][:64])).float()
        q_p = torch.from_numpy(np.array(mm_dev["q"][:64])).float()
        zra_p = torch.from_numpy(np.array(mm_dev["z"][p_ri])).float()
        zrb_p = torch.from_numpy(np.array(mm_dev["z"][p_rj])).float()
        sra_p = torch.from_numpy(np.array(mm_dev["st"][p_ri])).float()
        srb_p = torch.from_numpy(np.array(mm_dev["st"][p_rj])).float()
        gen_cache["color_matched_v37_fixed_a3"] = {
            "xc": _decode_z(zc_p), "xra": _decode_z(zra_p), "xrb": _decode_z(zrb_p),
            "ya": _decode_z(pal_ck["outputs"][0]), "yb": _decode_z(pal_ck["outputs"][1]),
        }
        gen_cache["color_matched_full_e2e_t35"] = {
            "xc": gen_cache["color_matched_v37_fixed_a3"]["xc"],
            "xra": gen_cache["color_matched_v37_fixed_a3"]["xra"],
            "xrb": gen_cache["color_matched_v37_fixed_a3"]["xrb"],
            "ya": _gen_ode(zc_p, y_p, q_p, zra_p, sra_p, 0.35, EVAL_SEED + 20),
            "yb": _gen_ode(zc_p, y_p, q_p, zrb_p, srb_p, 0.35, EVAL_SEED + 20),
        }

    # 3. Content-Swap (100 pairs: same ref, two distinct contents c1 vs c2)
    cs_c1 = np.arange(0, 100) % mm_dev["n"]
    cs_c2 = (np.arange(0, 100) + 173) % mm_dev["n"]
    cs_r = (np.arange(0, 100) * 3) % mm_mv["n"]
    zc1 = torch.from_numpy(np.array(mm_dev["z"][cs_c1])).float()
    yc1 = torch.from_numpy(np.array(mm_dev["y"][cs_c1])).float()
    qc1 = torch.from_numpy(np.array(mm_dev["q"][cs_c1])).float()
    zc2 = torch.from_numpy(np.array(mm_dev["z"][cs_c2])).float()
    yc2 = torch.from_numpy(np.array(mm_dev["y"][cs_c2])).float()
    qc2 = torch.from_numpy(np.array(mm_dev["q"][cs_c2])).float()
    zrcs = torch.from_numpy(np.array(mm_mv["z"][cs_r])).float()
    srcs = torch.from_numpy(np.array(mm_mv["st"][cs_r])).float()
    gen_cache["content_swap_t50"] = {
        "xc1": _decode_z(zc1), "xc2": _decode_z(zc2), "xr": _decode_z(zrcs),
        "y1": _gen_ode(zc1, yc1, qc1, zrcs, srcs, 0.50, EVAL_SEED + 30),
        "y2": _gen_ode(zc2, yc2, qc2, zrcs, srcs, 0.50, EVAL_SEED + 30),
    }

    # 4. SG-25 Style-Pool Proxy (25 unseen movement_ood clusters x 5 Core-5 contents = 125 pairs)
    sg_c_idx = np.repeat(np.array([0, 150, 300, 450, 600]), 25)
    sg_r_idx = np.tile(np.linspace(0, mm_mv["n"] - 1, 25, dtype=int), 5)
    zc_sg = z_src_750[sg_c_idx].float()
    y_sg = ea["y16_750"][sg_c_idx].float()
    q_sg = ea["q32_750"][sg_c_idx].float()
    zr_sg = torch.from_numpy(np.array(mm_mv["z"][sg_r_idx])).float()
    sr_sg = torch.from_numpy(np.array(mm_mv["st"][sg_r_idx])).float()
    gen_cache["ood25_125pairs_t50"] = {
        "xc": _decode_z(zc_sg), "xr": _decode_z(zr_sg),
        "y": _gen_ode(zc_sg, y_sg, q_sg, zr_sg, sr_sg, 0.50, EVAL_SEED + 40),
    }

    # Free UNet/decoder before FeatureStack
    del pcore, style_enc, dec
    gc.collect()
    torch.cuda.empty_cache()

    stack = FeatureStack(device)
    mb = auto_referee_batch(device)
    summary: dict[str, Any] = {"utc": utc()}

    for tag in ("ref_swap_t50", "ref_swap_t35", "color_matched_v37_fixed_a3", "color_matched_full_e2e_t35"):
        if tag not in gen_cache:
            continue
        b = gen_cache[tag]
        fc = _feat_batch(stack, b["xc"], mb)
        fra = _feat_batch(stack, b["xra"], mb)
        frb = _feat_batch(stack, b["xrb"], mb)
        fya = _feat_batch(stack, b["ya"], mb)
        fyb = _feat_batch(stack, b["yb"], mb)

        csd_aa = (fya["csd"] * fra["csd"]).sum(-1)
        csd_ab = (fya["csd"] * frb["csd"]).sum(-1)
        csd_bb = (fyb["csd"] * frb["csd"]).sum(-1)
        csd_ba = (fyb["csd"] * fra["csd"]).sum(-1)
        follow = 0.5 * ((csd_aa > csd_ab).float().mean().item() + (csd_bb > csd_ba).float().mean().item())
        margin = 0.5 * ((csd_aa - csd_ab).mean().item() + (csd_bb - csd_ba).mean().item())

        dc = 0.5 * ((fya["dino"] * fc["dino"]).sum(-1).mean().item() + (fyb["dino"] * fc["dino"]).sum(-1).mean().item())
        ds = 0.5 * ((fya["dino"] * fra["dino"]).sum(-1).mean().item() + (fyb["dino"] * frb["dino"]).sum(-1).mean().item())
        ref_csd_sim = 0.5 * (csd_aa.mean().item() + csd_bb.mean().item())

        # Stroke/HF response difference & Oklab color diff & Layout Leakage
        hf_a = _stroke_hf_vec(b["ya"])
        hf_b = _stroke_hf_vec(b["yb"])
        hf_delta = float((hf_a - hf_b).abs().mean().item())
        ok_ref_diff = float(_oklab_mean_sim(b["xra"], b["xrb"]).mean().item())
        ok_out_diff = float(_oklab_mean_sim(b["ya"], b["yb"]).mean().item())
        leak_ref = 0.5 * (_edge_corr(b["ya"], b["xra"]).mean().item() + _edge_corr(b["yb"], b["xrb"]).mean().item())
        keep_src_edge = 0.5 * (_edge_corr(b["ya"], b["xc"]).mean().item() + _edge_corr(b["yb"], b["xc"]).mean().item())
        y_l2_diff = float((b["ya"] - b["yb"]).pow(2).mean((1, 2, 3)).sqrt().mean().item())

        summary[tag] = {
            "n_triplets": len(b["xc"]),
            "follow_2way": round(follow, 4),
            "csd_margin_vs_other": round(margin, 5),
            "dino_c": round(dc, 4),
            "dino_s": round(ds, 4),
            "ref_csd_sim": round(ref_csd_sim, 4),
            "out_rgb_rmse": round(y_l2_diff, 4),
            "stroke_hf_delta": round(hf_delta, 5),
            "oklab_ref_pair_dist": round(ok_ref_diff, 4),
            "oklab_out_pair_dist": round(ok_out_diff, 4),
            "src_edge_keep": round(keep_src_edge, 4),
            "ref_layout_leakage": round(leak_ref, 4),
        }

    # Content-Swap summary
    cs = gen_cache["content_swap_t50"]
    fc1 = _feat_batch(stack, cs["xc1"], mb)
    fc2 = _feat_batch(stack, cs["xc2"], mb)
    fy1 = _feat_batch(stack, cs["y1"], mb)
    fy2 = _feat_batch(stack, cs["y2"], mb)
    fr_cs = _feat_batch(stack, cs["xr"], mb)
    c11 = (fy1["dino"] * fc1["dino"]).sum(-1)
    c12 = (fy1["dino"] * fc2["dino"]).sum(-1)
    c22 = (fy2["dino"] * fc2["dino"]).sum(-1)
    c21 = (fy2["dino"] * fc1["dino"]).sum(-1)
    summary["content_swap_t50"] = {
        "n_pairs": len(cs["xc1"]),
        "content_2way_follow": round(0.5 * ((c11 > c12).float().mean().item() + (c22 > c21).float().mean().item()), 4),
        "dino_c": round(0.5 * (c11.mean().item() + c22.mean().item()), 4),
        "dino_s": round(0.5 * ((fy1["dino"] * fr_cs["dino"]).sum(-1).mean().item() + (fy2["dino"] * fr_cs["dino"]).sum(-1).mean().item()), 4),
        "ref_csd_sim": round(0.5 * ((fy1["csd"] * fr_cs["csd"]).sum(-1).mean().item() + (fy2["csd"] * fr_cs["csd"]).sum(-1).mean().item()), 4),
        "src_edge_keep": round(0.5 * (_edge_corr(cs["y1"], cs["xc1"]).mean().item() + _edge_corr(cs["y2"], cs["xc2"]).mean().item()), 4),
        "ref_layout_leakage": round(0.5 * (_edge_corr(cs["y1"], cs["xr"]).mean().item() + _edge_corr(cs["y2"], cs["xr"]).mean().item()), 4),
    }

    # OOD-25 summary
    oo = gen_cache["ood25_125pairs_t50"]
    fo_c = _feat_batch(stack, oo["xc"], mb)
    fo_r = _feat_batch(stack, oo["xr"], mb)
    fo_y = _feat_batch(stack, oo["y"], mb)
    summary["ood25_125pairs_t50"] = {
        "n_pairs": len(oo["xc"]),
        "dino_c": round(float((fo_y["dino"] * fo_c["dino"]).sum(-1).mean().item()), 4),
        "dino_s": round(float((fo_y["dino"] * fo_r["dino"]).sum(-1).mean().item()), 4),
        "clip_s": round(float((fo_y["clip"] * fo_r["clip"]).sum(-1).mean().item()), 4),
        "ref_csd_sim": round(float((fo_y["csd"] * fo_r["csd"]).sum(-1).mean().item()), 4),
        "src_edge_keep": round(float(_edge_corr(oo["y"], oo["xc"]).mean().item()), 4),
        "ref_layout_leakage": round(float(_edge_corr(oo["y"], oo["xr"]).mean().item()), 4),
    }

    # Include V37 existing A3/Cond 2x2 quadrants
    if V37_QUAD_JSON.exists():
        summary["v37_a3_cond_2x2"] = read_json(V37_QUAD_JSON)

    del stack
    gc.collect()
    torch.cuda.empty_cache()
    save_json(out_p, summary)
    print(f"[F3 DONE] saved {out_p}", flush=True)
    return summary


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_f3(a.device)


if __name__ == "__main__":
    main()
