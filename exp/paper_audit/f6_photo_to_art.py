"""F6: Independent Photo-to-Art (P2A-256 -> 512px Unified Protocol) Benchmark.

Zero training / zero new learnable parameters. Evaluates cleanly on the 30 real photo inputs from
P2A-256 (photo_*):
  Part A (Closed-Set Core-5 P2A, 150 tasks = 30 photos x 5 Core-5 WikiArt targets):
    CLEAVE-E (lam=0.60, 0.75, 1.00), CLEAVE-P/E (s=0.60, 0.85, 1.00, 1.50), CLEAVE-One (tau0=0.35, 0.50, 0.65)
    + Identity under unified S10B + FeatureStack referee (CSD Top-1, DINO-C, DINO-S, CLIP-S, Style Affinity, LPIPS).
  Part B (Standard P2A-256 Target Transfer, 120 tasks = 30 photos x {monet, vangogh, cezanne, Hayao}):
    1-shot CLEAVE-One (tau0=0.35, 0.50) vs external baselines (Identity, AdaIN, StyleID, StyleShot, StyTr2, AesPA, Seedream)
    under identical FeatureStack referee (strictly reporting exact sample counts & missing files).
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32, match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_cleave_one.style_features import ConvNeXtExplicitAndPatchBackbone
from exp.v35_final import build_pcore_startpoint
from exp.v35_final.common import load_pcore_v35, load_v32_ecore_and_priors, make_v35_schedule
from exp.v36_art_foundation import CACHE_DIR as V36_CACHE_DIR
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from src.eval.metrics import paired_cluster_ci
from src.utils import FeatureStack, auto_decode_batch, auto_referee_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, STYLES
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc
from src.utils.structure import rgb_to_structure_64


F6_DIR = REPO / "runs/paper_audit/f6_p2a"
P2A_ASSETS = REPO / "runs/paper_audit/p2a_assets"
V36_P2_CKPT = REPO / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
P8_REPORT_PATH = REPO / "runs/v35_final/p8_final_report.json"
P2A_TARGETS = ("monet", "vangogh", "cezanne", "Hayao")
EXT_METHODS = ("identity", "adain", "styleid", "styleshot", "stytr2", "aespa", "seedream")


def _load_rgb512(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def run_f6(device: str = "cuda") -> dict[str, Any]:
    F6_DIR.mkdir(parents=True, exist_ok=True)
    out_p = F6_DIR / "f6_summary.json"
    if out_p.exists():
        return read_json(out_p)

    photo_dir = P2A_ASSETS / "sources/photo"
    photos = sorted(photo_dir.glob("*.*"))
    if not photos:
        raise FileNotFoundError(f"Missing P2A photo sources under {photo_dir}")
    n_p = len(photos)
    photo_stems = [p.stem for p in photos]
    rgb_p = torch.stack([_load_rgb512(p) for p in photos], 0)
    src_u8 = (rgb_p * 255.0).round().clamp(0, 255).to(torch.uint8)

    # Encode 30 photos cleanly
    a0 = load_a0_encoder(device).float()
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    st_ck = torch.load(REPO / "runs/v32_nodino/cache/cnx_stiefel_cache.pt", map_location="cpu", weights_only=False)
    w_stiefel = st_ck["W_stiefel_cnx"].to(device).float()
    w_luma = v18["luma_proj"]["w"].to(device).float()
    b_luma = v18["luma_proj"]["b"].to(device).float()
    cnx_ext = ConvNeXtTinyStage3Extractor(device=device).eval()

    from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
    xb = rgb_p.to(device)
    z_p = torch.cat([(a0.encode(xb[i:i + 10] * 2.0 - 1.0).latent_dist.mean * 0.18215).float().cpu() for i in range(0, n_p, 10)], 0)
    c64_p = rgb_to_structure_64(xb).cpu()
    y16_p = project_latent_y16_lcn(z_p.to(device), w_luma, b_luma).cpu()
    cg_p = cnx_ext.extract_gray_layout(xb).cpu()
    q32_p = stiefel_polar_quotient(cg_p.to(device), w_stiefel).cpu()
    del a0

    # Prepare Part B target refs (30 each from p2a_assets/refs/<tgt>)
    ref_banks_b: dict[str, list[Path]] = {t: sorted((P2A_ASSETS / "refs" / t).glob("*.*")) for t in P2A_TARGETS}
    cnx_style = ConvNeXtExplicitAndPatchBackbone(device=device)
    a0_b = load_a0_encoder(device).float()
    ref_lat_b: dict[str, torch.Tensor] = {}
    ref_stat_b: dict[str, torch.Tensor] = {}
    ref_u8_b: dict[str, torch.Tensor] = {}
    for t_name, files in ref_banks_b.items():
        if not files:
            continue
        rx = torch.stack([_load_rgb512(f) for f in files], 0).to(device)
        ref_u8_b[t_name] = (rx * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        ref_lat_b[t_name] = torch.cat([(a0_b.encode(rx[i:i + 10] * 2.0 - 1.0).latent_dist.mean * 0.18215).float().cpu() for i in range(0, len(files), 10)], 0)
        ref_stat_b[t_name] = torch.cat([cnx_style.extract_stat_1561(rx[i:i + 10]).cpu() for i in range(0, len(files), 10)], 0)
    del a0_b, cnx_style, cnx_ext
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- PART A: 30 photos x 5 Core-5 movements (150 tasks) ----------
    ea = torch.load(V36_CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    refs150 = ea["refs150"]
    tasks_a = [{"idx": k * 5 + ts, "p_idx": k, "stem": photo_stems[k], "ts": ts, "tgt": STYLES[ts]} for k in range(n_p) for ts in range(5)]
    p_idx_a = torch.tensor([t["p_idx"] for t in tasks_a], dtype=torch.long)
    tid_a = torch.tensor([t["ts"] for t in tasks_a], dtype=torch.long)
    eps_a = torch.randn((len(tasks_a), 4, 64, 64), generator=torch.Generator().manual_seed(EVAL_SEED))

    dec = load_s10_decoder(device)
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    p8 = read_json(P8_REPORT_PATH)
    fcfg = p8["final_schedule_cfg"]
    ckpt_p = Path(p8["locked_ckpt"]) if Path(p8["locked_ckpt"]).exists() else (REPO / "runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt")
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=p8["locked_weight"], device=device)
    assets = build_or_load_cnx_assets(device=device)

    part_a_imgs: dict[str, torch.Tensor] = {"Identity": src_u8[p_idx_a]}

    def _dec_u8(zo: torch.Tensor) -> torch.Tensor:
        rg = decode_s10(dec, zo.to(device), batch=auto_decode_batch(device), output_cpu=True).clamp(0.0, 1.0)
        return (rg * 255.0).round().to(torch.uint8)

    for lam in (0.60, 0.75, 1.00):
        chunks = []
        for s in range(0, len(tasks_a), 16):
            sl = slice(s, min(len(tasks_a), s + 16))
            pi = p_idx_a[sl]
            ze = ecore(z_p[pi].to(device), c64_p[pi].to(device), q32_p[pi].to(device), tid_a[sl].to(device), torch.full((len(pi),), lam, device=device), raw_stats, priors_ec, noise=eps_a[sl].to(device))
            chunks.append(ze.cpu())
        part_a_imgs[f"CLEAVE_E_l{int(round(lam * 100)):03d}"] = _dec_u8(torch.cat(chunks, 0))

    for sv in (0.60, 0.85, 1.00, 1.50):
        kw = make_v35_schedule(sv, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
        lam_d = float(kw.pop("draft_lambda", 0.75))
        chunks = []
        for s in range(0, len(tasks_a), 16):
            sl = slice(s, min(len(tasks_a), s + 16))
            pi = p_idx_a[sl]
            ti = tid_a[sl].to(device)
            ze = ecore(z_p[pi].to(device), c64_p[pi].to(device), q32_p[pi].to(device), ti, torch.full((len(pi),), lam_d, device=device), raw_stats, priors_ec, noise=eps_a[sl].to(device))
            ze_s = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
            u_st = match_cnx_k8_submode(cg_p[pi].to(device), ti, assets["bank"])
            zo = integrate_v33_trajectory(pcore, ze_s, eps_a[sl].to(device), y16_p[pi].to(device), q32_p[pi].to(device), ti, u_st, steps=int(fcfg["steps"]), solver="dpm2m", **kw)
            chunks.append(zo.cpu())
        part_a_imgs[f"CLEAVE_PE_s{int(round(sv * 100)):03d}"] = _dec_u8(torch.cat(chunks, 0))

    del ecore, pcore
    gc.collect()
    torch.cuda.empty_cache()

    pcore_v36, style_v36, priors_v36 = load_v36_models(V36_P2_CKPT, device)
    e_core5 = style_v36(refs150["stat_1561"].to(device).float()).cpu()
    z_core5 = refs150["z_ref"]
    for t0 in (0.35, 0.50, 0.65):
        chunks = []
        for s in range(0, len(tasks_a), 16):
            sl = slice(s, min(len(tasks_a), s + 16))
            pi = p_idx_a[sl]
            ri = torch.tensor([int(tid_a[i].item()) * 30 + (p_idx_a[i].item() % 30) for i in range(sl.start, sl.stop)], dtype=torch.long)
            zs = reconstruct_reference_a3(z_p[pi].to(device), z_core5[ri].to(device).float(), priors_v36, mode="I2")
            zo = sample_v36_ode(pcore_v36, zs, y16_p[pi].to(device), q32_p[pi].to(device), e_core5[ri].to(device), eps_a[sl].to(device), tau0=t0, steps=6, wj=3.8, wc=1.6, device=device)
            chunks.append(zo.cpu())
        part_a_imgs[f"CLEAVE_One_t{int(round(t0 * 100)):02d}"] = _dec_u8(torch.cat(chunks, 0))

    # ---------- PART B: 30 photos x 4 P2A-256 styles (monet, vangogh, cezanne, Hayao) ----------
    tasks_b = []
    for k in range(n_p):
        for t_name in P2A_TARGETS:
            if t_name in ref_lat_b:
                r_idx = k % len(ref_lat_b[t_name])
                tasks_b.append({"p_idx": k, "stem": photo_stems[k], "tgt": t_name, "r_idx": r_idx})
    p_idx_b = torch.tensor([t["p_idx"] for t in tasks_b], dtype=torch.long)
    eps_b = torch.randn((len(tasks_b), 4, 64, 64), generator=torch.Generator().manual_seed(EVAL_SEED + 6))

    part_b_imgs: dict[str, list[torch.Tensor | None]] = {}
    for t0 in (0.35, 0.50):
        out_u8 = []
        for s in range(0, len(tasks_b), 16):
            sl = slice(s, min(len(tasks_b), s + 16))
            grp = tasks_b[sl]
            pi = p_idx_b[sl]
            zr = torch.stack([ref_lat_b[g["tgt"]][g["r_idx"]] for g in grp], 0).to(device).float()
            sr = torch.stack([ref_stat_b[g["tgt"]][g["r_idx"]] for g in grp], 0).to(device).float()
            zs = reconstruct_reference_a3(z_p[pi].to(device), zr, priors_v36, mode="I2")
            zo = sample_v36_ode(pcore_v36, zs, y16_p[pi].to(device), q32_p[pi].to(device), style_v36(sr), eps_b[sl].to(device), tau0=t0, steps=6, wj=3.8, wc=1.6, device=device)
            out_u8.append(_dec_u8(zo))
        cat_u8 = torch.cat(out_u8, 0)
        part_b_imgs[f"CLEAVE_One_t{int(round(t0 * 100)):02d}"] = [cat_u8[i] for i in range(len(tasks_b))]

    # Save sample grids / preview PNGs for visual inspection & F2
    sample_dir = F6_DIR / "sample_pngs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for k in range(min(6, n_p)):
        Image.fromarray(src_u8[k].permute(1, 2, 0).numpy()).save(sample_dir / f"src_{photo_stems[k]}.png")
    for arm_lbl in ("CLEAVE_E_l075", "CLEAVE_PE_s085", "CLEAVE_PE_s100", "CLEAVE_One_t50"):
        arr = part_a_imgs[arm_lbl]
        for k in range(min(6, n_p)):
            ti = k * 5 + (k % 5)
            Image.fromarray(arr[ti].permute(1, 2, 0).numpy()).save(sample_dir / f"{arm_lbl}_{photo_stems[k]}_to_{STYLES[k % 5]}.png")

    # Core-5 reference u8 for FeatureStack
    ref_core5_u8 = torch.cat([_dec_u8(z_core5[s:s + 24]) for s in range(0, len(z_core5), 24)], 0)

    del pcore_v36, style_v36, dec
    gc.collect()
    torch.cuda.empty_cache()

    # Load P2A-256 external outputs
    for meth in EXT_METHODS:
        m_dir = P2A_ASSETS / "baselines" / meth
        if not m_dir.exists():
            continue
        lst: list[torch.Tensor | None] = []
        for g in tasks_b:
            # P2A filename convention: photo_<suffix>__to__<tgt>.jpg
            raw_stem = g["stem"]
            short = raw_stem[6:] if raw_stem.startswith("photo_") else raw_stem
            fn = m_dir / f"photo__{short}__to__{g['tgt']}.jpg"
            if not fn.exists():
                fn = m_dir / f"{raw_stem}_to_{g['tgt']}.jpg"
            if fn.exists():
                with Image.open(fn) as im:
                    ar = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.uint8)
                lst.append(torch.from_numpy(ar).permute(2, 0, 1))
            else:
                lst.append(None)
        part_b_imgs[meth] = lst

    # Persist actual outputs for LPIPS-only repair; repair never reruns generation.
    cache = F6_DIR / "lpips_cache"
    cache.mkdir(exist_ok=True)
    torch.save({"src_u8": src_u8, "source_ids": photo_stems}, cache / "sources.pt")
    for name, images in part_a_imgs.items():
        torch.save({"rgb_u8": images, "tasks": [
            {"task_id": f"F6_A_{t['idx']:04d}", "source_pos": t["p_idx"],
             "source_id": t["stem"], "target": t["tgt"]} for t in tasks_a],
            "expected_tasks": len(tasks_a)}, cache / f"part_a_{name}.pt")
    for name, images in part_b_imgs.items():
        valid = [i for i, im in enumerate(images) if im is not None]
        if valid:
            torch.save({"rgb_u8": torch.stack([images[i] for i in valid]), "tasks": [
                {"task_id": f"F6_B_{i:04d}", "source_pos": tasks_b[i]["p_idx"],
                 "source_id": tasks_b[i]["stem"], "target": tasks_b[i]["tgt"]} for i in valid],
                "expected_tasks": len(tasks_b)}, cache / f"part_b_{name}.pt")

    # Evaluate Part A & Part B under frozen FeatureStack
    stack = FeatureStack(device)
    mb = auto_referee_batch(device)
    proto_np = np.asarray(stack.csd_prototypes)

    def _ex(u8_t: torch.Tensor) -> dict[str, np.ndarray]:
        fo = stack.extract([Image.fromarray(a) for a in u8_t.permute(0, 2, 3, 1).numpy()], batch=mb)
        return {"dino": fo["dino"].cpu().numpy(), "csd": np.asarray(fo["csd"]), "clip": fo["clip"].cpu().numpy()}

    f_src = _ex(src_u8)
    f_ref_core5 = _ex(ref_core5_u8)
    rd_c5 = [torch.from_numpy(f_ref_core5["dino"][s * 30:(s + 1) * 30]).to(device) for s in range(5)]
    rc_c5 = torch.stack([torch.from_numpy(f_ref_core5["clip"][s * 30:(s + 1) * 30]).mean(0) for s in range(5)], 0)
    rc_c5 = (rc_c5 / rc_c5.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(device)

    part_a_res: dict[str, Any] = {}
    for arm_name, u8_all in part_a_imgs.items():
        fo = _ex(u8_all)
        df = torch.from_numpy(fo["dino"]).to(device)
        clf = torch.from_numpy(fo["clip"]).to(device)
        ds_m = torch.stack([(df @ p.T).max(1).values for p in rd_c5], 1).cpu().numpy()
        cs_m = fo["csd"] @ proto_np.T
        cl_m = (clf @ rc_c5.T).cpu().numpy()
        # FeatureStack normalizes generated uint8 only; references must be RGB01.
        lp = stack.lpips_vs(u8_all, src_u8[p_idx_a].float().div(255.0), batch=mb).cpu().numpy()

        top1 = np.array([int(cs_m[i].argmax() == tasks_a[i]["ts"]) for i in range(len(tasks_a))])
        dc = np.array([float(fo["dino"][i] @ f_src["dino"][tasks_a[i]["p_idx"]]) for i in range(len(tasks_a))])
        ds = np.array([float(ds_m[i, tasks_a[i]["ts"]]) for i in range(len(tasks_a))])
        cl = np.array([float(cl_m[i, tasks_a[i]["ts"]]) for i in range(len(tasks_a))])
        per_t = {}
        for sid, st in enumerate(STYLES):
            mask = np.array([t["ts"] == sid for t in tasks_a])
            per_t[st] = {
                "csd_top1": round(float(top1[mask].mean()), 4),
                "dino_c": round(float(dc[mask].mean()), 4),
                "dino_s": round(float(ds[mask].mean()), 4),
                "clip_s": round(float(cl[mask].mean()), 4),
                "lpips": round(float(lp[mask].mean()), 4),
            }
        part_a_res[arm_name] = {
            "n_tasks": len(tasks_a),
            "csd_top1": round(float(top1.mean()), 4),
            "dino_c": round(float(dc.mean()), 4),
            "dino_s": round(float(ds.mean()), 4),
            "clip_s": round(float(cl.mean()), 4),
            "style_affinity": round(0.5 * (float(ds.mean()) + float(cl.mean())), 4),
            "lpips": round(float(lp.mean()), 4),
            "per_target_30": per_t,
        }

    # Part B scoring
    f_ref_b = {t: _ex(u8) for t, u8 in ref_u8_b.items()}
    part_b_res: dict[str, Any] = {}
    for meth, img_list in part_b_imgs.items():
        valid_i = [i for i, im in enumerate(img_list) if im is not None]
        if not valid_i:
            continue
        u8_v = torch.stack([img_list[i] for i in valid_i], 0)
        src_v = torch.stack([src_u8[tasks_b[i]["p_idx"]] for i in valid_i], 0)
        fo = _ex(u8_v)
        lp = stack.lpips_vs(u8_v, src_v.float().div(255.0), batch=mb).cpu().numpy()

        dc_l, ds_set_l, ds_1shot_l, csd_sim_l, cl_l = [], [], [], [], []
        for j, idx_t in enumerate(valid_i):
            g = tasks_b[idx_t]
            rb = f_ref_b[g["tgt"]]
            dc_l.append(float(fo["dino"][j] @ f_src["dino"][g["p_idx"]]))
            ds_all = rb["dino"] @ fo["dino"][j]
            ds_set_l.append(float(ds_all.max()))
            ds_1shot_l.append(float(ds_all[g["r_idx"]]))
            csd_sim_l.append(float((rb["csd"] @ fo["csd"][j]).mean()))
            cl_mean = rb["clip"].mean(0)
            cl_mean = cl_mean / max(1e-8, float(np.linalg.norm(cl_mean)))
            cl_l.append(float(fo["clip"][j] @ cl_mean))

        ds_mean = float(np.mean(ds_set_l))
        cl_mean_val = float(np.mean(cl_l))
        part_b_res[meth] = {
            "n_valid": len(valid_i),
            "n_expected": len(tasks_b),
            "dino_c": round(float(np.mean(dc_l)), 4),
            "dino_s_set_max": round(ds_mean, 4),
            "dino_s_1shot": round(float(np.mean(ds_1shot_l)), 4),
            "csd_ref_sim": round(float(np.mean(csd_sim_l)), 4),
            "clip_s": round(cl_mean_val, 4),
            "style_affinity": round(0.5 * (ds_mean + cl_mean_val), 4),
            "lpips": round(float(lp.mean()), 4),
        }

    del stack
    gc.collect()
    torch.cuda.empty_cache()

    rep = {"utc": utc(), "n_photos": n_p, "part_a_core5_p2a": part_a_res, "part_b_p2a256_targets": part_b_res}
    save_json(out_p, rep)
    print(f"[F6 DONE] saved {out_p}", flush=True)
    return rep


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_f6(a.device)


if __name__ == "__main__":
    main()
