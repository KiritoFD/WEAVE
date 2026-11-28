"""F4: Decoder & Encoder Asymmetry + Texture Retraction (Delta-z Alpha Scaling + enc20 Error Propagation).

Zero training / zero new learnable parameters. Evaluates fixed edit trajectory:
  z(alpha) = z_s + alpha * (z_edit - z_s),  alpha in {0.0, 0.25, 0.50, 0.75, 1.00, 1.25}
across:
  1. SD-VAE (D0 sd0_vae_full encoder=A0 -> decoder=SD-VAE)
  2. StyleVAE (C_edit_best paired encoder + decoder; NEVER cross-fed with A0 latents)
  3. S10B (A0 encoder -> pruned S10B decoder)
for both correct-target and wrong-target edit directions (test750), measuring:
  - Haar LL/LH/HL/HH band energy & Sobel edge change vs source
  - CSD Source Drop (target vs source movement score shift), Target Rise, Top-1, Margin, DINO-C, DINO-S, LPIPS
  - Optional enc20 (pruned A0) layer-by-layer error propagation into E-Core / P-Core / S10B.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_final import D0_FULL_VAE_DIR, S10B_CKPT, build_pcore_startpoint
from exp.v35_final.common import load_pcore_v35, load_v32_ecore_and_priors, make_v35_schedule
from src.eval.metrics import paired_cluster_ci
from src.utils import FeatureStack, auto_decode_batch, auto_infer_batch, auto_referee_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, STYLES, pick_wrong_cross_tid
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc
from src.utils.structure import dwt2_haar, rgb_to_gray


F4_DIR = REPO / "runs/paper_audit/f4_asymmetry"
P8_REPORT_PATH = REPO / "runs/v35_final/p8_final_report.json"
ALPHAS_OK = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25)
ALPHAS_WR = (0.50, 1.00)


def _find_style_vae_ckpt() -> Path | None:
    env_p = os.environ.get("WEAVE_STYLE_VAE_CKPT")
    cands = [
        Path(env_p) if env_p else None,
        REPO / "runs/vae_style_vae/C_edit_best.pt",
        REPO.parent / "Workspace/WEAVE_VAE/runs/vae_style_vae/C_edit_best.pt",
        REPO.parent / "WEAVE_VAE/runs/vae_style_vae/C_edit_best.pt",
        REPO.parent / "Workspace/WEAVE_VAE/runs/vae_style_vae/C_edit_best.pt",
    ]
    for p in cands:
        if p is not None and p.exists():
            return p
    return None


def _sobel_mag(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gp = F.pad(gray.float(), (1, 1, 1, 1), mode="reflect")
    return (F.conv2d(gp, kx).square() + F.conv2d(gp, ky).square() + 1e-8).sqrt()


@torch.no_grad()
def _build_or_load_p8_endpoints(device: str = "cuda") -> dict[str, Path]:
    cache_dir = F4_DIR / "latents_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok_p = cache_dir / "p8_s100_ok.pt"
    wr_p = cache_dir / "p8_s100_wrong.pt"
    if ok_p.exists() and wr_p.exists():
        return {"ok": ok_p, "wrong": wr_p}

    # Reuse P9 cached p8 latents if available
    p9_ok = REPO / "runs/v35_final/p9/cached_p8_latents/p8_s100.pt"
    p9_wr = REPO / "runs/v35_final/p9/cached_p8_latents/p8_s100_wrong.pt"
    if p9_ok.exists() and p9_wr.exists():
        torch.save(torch.load(p9_ok, map_location="cpu", weights_only=False), ok_p)
        torch.save(torch.load(p9_wr, map_location="cpu", weights_only=False), wr_p)
        return {"ok": ok_p, "wrong": wr_p}

    p8 = read_json(P8_REPORT_PATH)
    fcfg = p8["final_schedule_cfg"]
    ckpt_p = REPO / "runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt"
    if Path(p8["locked_ckpt"]).exists():
        ckpt_p = Path(p8["locked_ckpt"])
    tasks, z_src, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wr = pick_wrong_cross_tid(sid_750, tid_750)
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    q16_750 = assets["stiefel"]["test750"]["q32"].float()
    cg_750 = assets["stiefel"]["test750"]["cnx_gray"].float()
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()

    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=p8["locked_weight"], device=device)
    bs = auto_infer_batch(device)
    kw = make_v35_schedule(1.00, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
    lam = float(kw.pop("draft_lambda", 0.75))

    for out_p, tl in ((ok_p, tid_750), (wr_p, tid_wr)):
        chunks = []
        for st in range(0, len(tasks), bs):
            sl = slice(st, min(len(tasks), st + bs))
            ti = tl[sl].to(device)
            ze = ecore(
                z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                q16_750[sl].to(device).float(), ti,
                torch.full((len(ti),), lam, device=device), raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
            )
            ze_s = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
            u_st = match_cnx_k8_submode(cg_750[sl].to(device).float(), ti, assets["bank"])
            zo = integrate_v33_trajectory(
                pcore, ze_s, eps_750[sl].to(device).float(), y16_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                ti, u_st, steps=int(fcfg["steps"]), solver="dpm2m", **kw,
            )
            chunks.append(zo.cpu().half())
        torch.save({"ids": [t["task_id"] for t in tasks], "z": torch.cat(chunks, 0)}, out_p)

    del ecore, pcore
    gc.collect()
    torch.cuda.empty_cache()
    return {"ok": ok_p, "wrong": wr_p}


@torch.no_grad()
def _load_style_vae_paired(ckpt_path: Path, device: str):
    from diffusers import AutoencoderKL
    from tools.vae_style_vae.model import StyleVAE
    from tools.vae_suite.decoder_prune import build_decoder

    vae = AutoencoderKL.from_pretrained(str(D0_FULL_VAE_DIR), local_files_only=True).to(device)
    s10_ck = torch.load(S10B_CKPT, map_location="cpu", weights_only=False)
    dec = build_decoder(s10_ck["meta"]["widths_forward"]).to(device)
    model = StyleVAE(vae, dec).to(device)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(st["encoder"], strict=True)
    model.decoder.load_state_dict(st["decoder"], strict=True)
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def _score_rgb_batch(
    rgb01_all: torch.Tensor,
    src_u8: torch.Tensor,
    src_dino: np.ndarray,
    src_csd: np.ndarray,
    tasks: list[dict],
    tid_wrong: torch.Tensor,
    is_wrong: bool,
    stack: FeatureStack,
    ref_dino: list[torch.Tensor],
    ref_clip: torch.Tensor,
    proto_np: np.ndarray,
    mb: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(tasks)
    haar_ll, haar_lh, haar_hl, haar_hh, sob_m, sob_diff = [], [], [], [], [], []
    d_list, c_list, cl_list, lp_list = [], [], [], []

    for st in range(0, n, 32):
        ed = min(n, st + 32)
        xb = rgb01_all[st:ed].to(device).float().clamp(0.0, 1.0)
        sb_src = torch.stack([src_u8[st + i] for i in range(ed - st)], dim=0)
        src_f = sb_src.to(device).float() / 255.0

        g = rgb_to_gray(xb)
        gs = rgb_to_gray(src_f)
        ll, lh, hl, hh = dwt2_haar(g)
        sx = _sobel_mag(g)
        sx0 = _sobel_mag(gs)
        haar_ll.append(ll.abs().mean((1, 2, 3)).cpu())
        haar_lh.append(lh.abs().mean((1, 2, 3)).cpu())
        haar_hl.append(hl.abs().mean((1, 2, 3)).cpu())
        haar_hh.append(hh.abs().mean((1, 2, 3)).cpu())
        sob_m.append(sx.mean((1, 2, 3)).cpu())
        sob_diff.append((sx - sx0).abs().mean((1, 2, 3)).cpu())

        u8_b = (xb * 255.0).round().to(torch.uint8).cpu()
        imgs = [Image.fromarray(a) for a in u8_b.permute(0, 2, 3, 1).numpy()]
        fo = stack.extract(imgs, batch=mb)
        d_list.append(fo["dino"].cpu())
        c_list.append(torch.as_tensor(fo["csd"]).cpu())
        cl_list.append(fo["clip"].cpu())
        lp_list.append(stack.lpips_vs(u8_b, sb_src, batch=mb).cpu())

    df = torch.cat(d_list, 0).to(device)
    cf = torch.cat(c_list, 0).numpy()
    clf = torch.cat(cl_list, 0).to(device)
    lp = torch.cat(lp_list, 0).numpy()
    scores = torch.stack([(df @ p.T).max(1).values for p in ref_dino], 1).cpu().numpy()
    cs = cf @ proto_np.T
    clips = (clf @ ref_clip.T).cpu().numpy()
    df_np = df.cpu().numpy()

    rows = []
    tgt_rise, src_drop, wrong_follow = [], [], []
    for i, t in enumerate(tasks):
        ss = int(t["source_style_id"])
        ts = int(t["target_style_id"])
        wt = int(tid_wrong[i].item()) if is_wrong else ts
        p_cls = int(cs[i].argmax())
        src_score_src = float(src_csd[i] @ proto_np[ss])
        src_score_tgt = float(src_csd[i] @ proto_np[ts])
        out_score_src = float(cs[i, ss])
        out_score_tgt = float(cs[i, ts])
        if ss != ts:
            tgt_rise.append(out_score_tgt - src_score_tgt)
            src_drop.append(src_score_src - out_score_src)
            if is_wrong:
                wrong_follow.append(int(p_cls == wt))
        rows.append({
            "task_id": t["task_id"],
            "src_stem": t["source_stem"],
            "src_style_id": ss,
            "tgt_style_id": ts,
            "src_style": STYLES[ss],
            "tgt_style": STYLES[ts],
            "is_cross": ss != ts,
            "dino_c": float(df_np[i] @ src_dino[i]),
            "dino_s": float(scores[i, ts]),
            "style_scores_5": scores[i].tolist(),
            "clip_scores_5": clips[i].tolist(),
            "csd_scores_5": cs[i].tolist(),
            "target_top1": int(p_cls == ts),
            "target_margin": float(cs[i, ts] - np.delete(cs[i], ts).max()),
            "lpips": float(lp[i]),
            "tgt_score": out_score_tgt,
            "src_score": out_score_src,
            "wrong_tgt_score": float(cs[i, wt]) if is_wrong else 0.0,
        })

    cross = [r for r in rows if r["is_cross"]]
    same = [r for r in rows if not r["is_cross"]]
    stats = {
        "n_cross": len(cross),
        "csd_top1": round(float(np.mean([r["target_top1"] for r in cross])), 4),
        "csd_margin": round(float(np.mean([r["target_margin"] for r in cross])), 5),
        "csd_tgt_score": round(float(np.mean([r["tgt_score"] for r in cross])), 5),
        "csd_src_score": round(float(np.mean([r["src_score"] for r in cross])), 5),
        "target_rise": round(float(np.mean(tgt_rise)), 5),
        "source_drop": round(float(np.mean(src_drop)), 5),
        "dino_c": round(float(np.mean([r["dino_c"] for r in cross])), 4),
        "dino_s": round(float(np.mean([r["dino_s"] for r in cross])), 4),
        "clip_s": round(float(np.mean([r["clip_scores_5"][r["tgt_style_id"]] for r in cross])), 4),
        "lpips": round(float(np.mean([r["lpips"] for r in cross])), 4),
        "same150_top1": round(float(np.mean([r["target_top1"] for r in same])), 4),
        "haar_ll": round(float(torch.cat(haar_ll).mean().item()), 6),
        "haar_lh": round(float(torch.cat(haar_lh).mean().item()), 6),
        "haar_hl": round(float(torch.cat(haar_hl).mean().item()), 6),
        "haar_hh": round(float(torch.cat(haar_hh).mean().item()), 6),
        "sobel_mag": round(float(torch.cat(sob_m).mean().item()), 6),
        "sobel_diff_vs_src": round(float(torch.cat(sob_diff).mean().item()), 6),
    }
    if is_wrong:
        stats["wrong_tgt_follow"] = round(float(np.mean(wrong_follow)), 4)
        stats["wrong_tgt_score"] = round(float(np.mean([r["wrong_tgt_score"] for r in cross])), 5)
    return stats, rows


@torch.no_grad()
def run_f4(device: str = "cuda") -> dict[str, Any]:
    F4_DIR.mkdir(parents=True, exist_ok=True)
    out_p = F4_DIR / "f4_summary.json"
    if out_p.exists():
        return read_json(out_p)

    lat_p = _build_or_load_p8_endpoints(device=device)
    tasks, z_src, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = pick_wrong_cross_tid(sid_750, tid_750)
    z_ok = torch.load(lat_p["ok"], map_location="cpu", weights_only=False)["z"].float()
    z_wr = torch.load(lat_p["wrong"], map_location="cpu", weights_only=False)["z"].float()
    z_s = z_src.float()

    # Load test750 source PNGs & ref assets cleanly
    spec = read_json(REPO / "eval_protocol/tasks_750.json")
    src_by_stem: dict[str, torch.Tensor] = {}
    for srec in spec["sources"]:
        with Image.open(REPO / srec["source_rel_path"]) as im:
            arr = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.uint8)
        src_by_stem[srec["source_stem"]] = torch.from_numpy(arr).permute(2, 0, 1)
    src_u8 = torch.stack([src_by_stem[t["source_stem"]] for t in tasks], dim=0)

    src_feats = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)
    src_dino = np.stack([np.asarray(src_feats["raw_dino"][t["source_stem"]]) for t in tasks], axis=0)
    src_csd = np.stack([np.asarray(src_feats["raw_csd"][t["source_stem"]]) for t in tasks], axis=0)

    stack = FeatureStack(device)
    proto_np = np.asarray(stack.csd_prototypes)
    ea = torch.load(REPO / "runs/v36_art_foundation/cache/v36_eval_assets.pt", map_location="cpu", weights_only=False)
    # Build Core-5 ref_dino & ref_clip from frozen S10-decoded refs
    dec_s10 = load_s10_decoder(device)
    ref_u8_l = []
    for st in range(0, len(ea["refs150"]["z_ref"]), 24):
        rg = decode_s10(dec_s10, ea["refs150"]["z_ref"][st:st + 24].to(device).float(), batch=24, output_cpu=True).clamp(0.0, 1.0)
        ref_u8_l.append((rg * 255.0).round().to(torch.uint8))
    ref_u8 = torch.cat(ref_u8_l, 0)
    mb = auto_referee_batch(device)
    rf = stack.extract([Image.fromarray(a) for a in ref_u8.permute(0, 2, 3, 1).numpy()], batch=mb)
    ref_dino = [rf["dino"][s * 30:(s + 1) * 30].clone().to(device) for s in range(5)]
    ref_clip = torch.stack([rf["clip"][s * 30:(s + 1) * 30].mean(dim=0) for s in range(5)], dim=0)
    ref_clip = (ref_clip / ref_clip.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(device)

    # First decode S10B ok & wrong at alpha=1.0 so we have RGBs to encode into StyleVAE's paired coordinate space
    rgb_p8_ok = decode_s10(dec_s10, z_ok.to(device), batch=auto_decode_batch(device), output_cpu=True).clamp(0.0, 1.0)
    rgb_p8_wr = decode_s10(dec_s10, z_wr.to(device), batch=auto_decode_batch(device), output_cpu=True).clamp(0.0, 1.0)

    results: dict[str, Any] = {
        "utc": utc(), "decoders": {}, "protocol_revision": 2,
        "decoder_control": "s10b vs sd0_vae: identical A0 latent trajectory; decoder protocol exception",
        "stylevae_control": "paired re-encoding of S10-rendered endpoints; NOT a decoder-only causal comparison",
        "encoder_control": "latent probe plus existing full test750 enc20 acceptance results",
    }
    saved_rows: dict[tuple[str, float, str], list[dict]] = {}

    def _eval_decoder(dname: str, decode_fn, z_base: torch.Tensor, z_tgt_ok: torch.Tensor, z_tgt_wr: torch.Tensor):
        d_out: dict[str, Any] = {"ok": {}, "wrong": {}, "directional_spread": {}}
        dz_ok = z_tgt_ok - z_base
        dz_wr = z_tgt_wr - z_base
        def measure(a, direction):
            tag = f"a{int(round(a * 100)):03d}"
            arm_p = F4_DIR / "arms" / f"{dname}_{direction}_{tag}.json"
            if arm_p.exists():
                arm = read_json(arm_p)
                if arm.get("protocol_revision") == 2 and len(arm.get("rows", [])) == 750:
                    return arm["stats"], arm["rows"]
            dz = dz_ok if direction == "ok" else dz_wr
            rgb_all = decode_fn(z_base + a * dz)
            st, rows = _score_rgb_batch(
                rgb_all, src_u8, src_dino, src_csd, tasks, tid_wrong, direction == "wrong",
                stack, ref_dino, ref_clip, proto_np, mb, device,
            )
            from src.eval.metrics import enrich_rows
            enrich_rows(rows, require_full=True)
            cross_ids = [i for i, t in enumerate(tasks) if int(t["source_style_id"]) != int(t["target_style_id"])]
            sample_ids = [cross_ids[j] for j in np.linspace(0, len(cross_ids) - 1, 8, dtype=int)]
            sample_dir = F4_DIR / "samples" / f"{dname}_{direction}_{tag}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i in sample_ids:
                arr = (rgb_all[i].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(arr).save(sample_dir / f"{i:04d}.png")
            save_json(arm_p, {"protocol_revision": 2, "stats": st, "rows": rows})
            return st, rows
        for a in ALPHAS_OK:
            tag = f"a{int(round(a * 100)):03d}"
            st, rows = measure(a, "ok")
            d_out["ok"][tag] = st
            saved_rows[(dname, a, "ok")] = rows
            print(f"[F4 {dname:9s} ok {tag}] Top1={st['csd_top1']:.4f} Rise={st['target_rise']:+.4f} Drop={st['source_drop']:+.4f} HH={st['haar_hh']:.5f} C={st['dino_c']:.4f}", flush=True)
        for a in ALPHAS_WR:
            tag = f"a{int(round(a * 100)):03d}"
            st, rows = measure(a, "wrong")
            d_out["wrong"][tag] = st
            saved_rows[(dname, a, "wrong")] = rows
            ok_st = d_out["ok"][tag]
            d_out["directional_spread"][tag] = {
                "net_top1_pp": round((ok_st["csd_top1"] - st["wrong_tgt_follow"]) * 100.0, 2),
                "target_score_spread": round(ok_st["csd_tgt_score"] - st["csd_tgt_score"], 5),
                "hh_diff": round(ok_st["haar_hh"] - st["haar_hh"], 6),
            }
        results["decoders"][dname] = d_out

    # 1. S10B
    _eval_decoder(
        "s10b",
        lambda zz: decode_s10(dec_s10, zz.to(device), batch=auto_decode_batch(device), output_cpu=True).clamp(0.0, 1.0),
        z_s, z_ok, z_wr,
    )
    del dec_s10
    gc.collect()
    torch.cuda.empty_cache()

    # 2. SD-VAE (D0)
    from diffusers import AutoencoderKL
    sd_vae = AutoencoderKL.from_pretrained(str(D0_FULL_VAE_DIR), local_files_only=True).to(device).eval().requires_grad_(False)

    def _dec_sdvae(zz: torch.Tensor) -> torch.Tensor:
        outs = []
        for s in range(0, len(zz), 16):
            o = sd_vae.decode(zz[s:s + 16].to(device).float() / 0.18215).sample
            outs.append((o / 2.0 + 0.5).clamp(0.0, 1.0).cpu())
        return torch.cat(outs, 0)

    _eval_decoder("sd0_vae", _dec_sdvae, z_s, z_ok, z_wr)
    del sd_vae
    gc.collect()
    torch.cuda.empty_cache()

    # 3. StyleVAE C_edit_best (STRICTLY PAIRED Encoder + Decoder!)
    style_ckpt = _find_style_vae_ckpt()
    if style_ckpt is not None:
        sv_vae = _load_style_vae_paired(style_ckpt, device)

        def _enc_sv(rgb01: torch.Tensor) -> torch.Tensor:
            outs = []
            for s in range(0, len(rgb01), 16):
                x = rgb01[s:s + 16].to(device).float() * 2.0 - 1.0
                outs.append(sv_vae.encode(x).mean.cpu())
            return torch.cat(outs, 0)

        def _dec_sv(zz: torch.Tensor) -> torch.Tensor:
            outs = []
            for s in range(0, len(zz), 16):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = sv_vae.decode(zz[s:s + 16].to(device).float()).float()
                outs.append((o * 0.5 + 0.5).clamp(0.0, 1.0).cpu())
            return torch.cat(outs, 0)

        src_f01 = src_u8.float() / 255.0
        zc_s = _enc_sv(src_f01)
        zc_ok = _enc_sv(rgb_p8_ok)
        zc_wr = _enc_sv(rgb_p8_wr)
        _eval_decoder("style_vae_c_edit_paired", _dec_sv, zc_s, zc_ok, zc_wr)
        results["style_vae_ckpt"] = str(style_ckpt.name)
        del sv_vae
        gc.collect()
        torch.cuda.empty_cache()
    else:
        results["style_vae_note"] = "C_edit_best.pt not found in standard relative locations"

    # Paired bootstrap CI comparing S10B vs SD0_VAE & StyleVAE at alpha=0.50 and 1.00
    cis = {}
    for a in (0.50, 1.00):
        tag = f"a{int(round(a * 100)):03d}"
        if ("s10b", a, "ok") in saved_rows and ("sd0_vae", a, "ok") in saved_rows:
            cis[f"s10b_minus_sd0_{tag}"] = paired_cluster_ci(saved_rows[("s10b", a, "ok")], saved_rows[("sd0_vae", a, "ok")])
        if ("style_vae_c_edit_paired", a, "ok") in saved_rows:
            cis[f"s10b_minus_stylevae_{tag}"] = paired_cluster_ci(
                saved_rows[("s10b", a, "ok")], saved_rows[("style_vae_c_edit_paired", a, "ok")]
            )
    results["paired_ci"] = cis

    # Optional enc20 error propagation if synced into paper_audit/enc20_best.pt or runs/v35_3060_opt
    enc_ckpt = next(
        (p for p in (
            REPO / "runs/paper_audit/enc20_best.pt",
            REPO / "runs/v35_3060_opt/distillation/enc20/best.pt",
        ) if p.exists()),
        None,
    )
    if enc_ckpt is not None:
        from exp.deploy_v35.a0_prune import StudentEncoder, build_encoder
        a0_raw = load_a0_encoder(device)
        a0_raw = a0_raw.float().to(device).eval()
        def a0_f(rgb):
            return a0_raw.encode(rgb * 2 - 1).latent_dist.mean * 0.18215
        ck_e = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
        enc20 = StudentEncoder(build_encoder(ck_e["meta"]["widths"]), torch.nn.Conv2d(8, 4, 1)).to(device).eval()
        enc20.load_state_dict(ck_e["model"], strict=True)
        sub_rgb = (src_u8[::5][:50].to(device).float() / 255.0)
        z_ref_a0 = torch.cat([a0_f(sub_rgb[i:i + 10]) for i in range(0, 50, 10)], 0)
        z_ref_e20 = torch.cat([enc20(sub_rgb[i:i + 10]) for i in range(0, 50, 10)], 0)
        results["enc20_latent_error"] = {
            "n_sources": 50,
            "latent_rmse": round(float((z_ref_e20 - z_ref_a0).pow(2).mean().sqrt().item()), 5),
            "latent_rel_rmse": round(float((z_ref_e20 - z_ref_a0).pow(2).mean().sqrt().item() / z_ref_a0.std().item()), 5),
            "latent_cos": round(float(F.cosine_similarity(z_ref_e20.flatten(1), z_ref_a0.flatten(1), dim=1).mean().item()), 5),
        }

    del stack
    gc.collect()
    torch.cuda.empty_cache()
    save_json(out_p, results)
    print(f"[F4 DONE] saved {out_p}", flush=True)
    return results


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_f4(a.device)


if __name__ == "__main__":
    main()
