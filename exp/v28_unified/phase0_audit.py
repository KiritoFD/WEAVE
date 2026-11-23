"""Phase 0: 100% Train RGB Mapping Verification, Enc->S10 Ceiling (200 ER + 200 Uki), and ema_s4000 Loss Audit."""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

from exp.v18 import STYLES
from exp.v27_single.unified.model import UnifiedEditor
from exp.v27_single.unified.teacher_gate import task_plan
from exp.v28_unified import (
    GATE_OLD_OUT,
    OUT,
    ROOT,
    SD0_VAE_DIR,
    SEED_PLAN,
    UNIFIED_OLD_OUT,
    assert_vram_budget,
)
from exp.v28_unified.data_map import load_rgb_512_tensor, resolve_train_rgb_paths
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc
from src.utils.structure import dwt2_haar, extract_color_fiber_27d, local_contrast_norm, rgb_to_gray


def _haar_hf_lf(rgb01: torch.Tensor) -> torch.Tensor:
    y = rgb_to_gray(rgb01.float())
    ll, lh, hl, hh = dwt2_haar(y)
    hf = (lh.square() + hl.square() + hh.square()).mean(dim=(1, 2, 3))
    lf = (ll - ll.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(1, 2, 3)).clamp_min(1e-6)
    return torch.log((hf + 1e-5) / lf)


def _to_pils(rgb01: torch.Tensor) -> list[Image.Image]:
    u8 = (rgb01.detach().cpu().float().clamp(0.0, 1.0) * 255.0).round().byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in u8]


@torch.inference_mode()
def audit_enc_s10_ceiling(train: dict, paths_4400: list, n_per_style: int = 200, device: str = "cuda") -> dict:
    """Evaluate 200 ER + 200 Uki real paintings through Enc -> S10 roundtrip."""
    seed_all(SEED_PLAN + 10)
    gen = torch.Generator().manual_seed(SEED_PLAN + 10)
    picks = []
    for sid in (0, 4):
        ids = (train["sid"] == sid).nonzero().flatten()
        perm = ids[torch.randperm(len(ids), generator=gen)[:n_per_style]]
        picks.append(perm)
    sel = torch.cat(picks)
    sel_paths = [paths_4400[i] for i in sel.tolist()]
    sel_sids = train["sid"][sel].numpy()

    vae = AutoencoderKL.from_pretrained(str(SD0_VAE_DIR), local_files_only=True).to(device).eval()
    sf = float(vae.config.scaling_factor)
    dec = load_s10_decoder(device)

    rgb_orig_list, rgb_rt_list, z_enc_list, z_stored_list = [], [], [], []
    for sl_idx in sel.split(16):
        batch_paths = [paths_4400[i] for i in sl_idx.tolist()]
        rgb = load_rgb_512_tensor(batch_paths).to(device)
        z_enc = vae.encode(rgb * 2.0 - 1.0).latent_dist.mean * sf
        rgb_rt = decode_s10(dec, z_enc, batch=8, output_cpu=True).clamp(0.0, 1.0)
        rgb_orig_list.append(rgb.cpu())
        rgb_rt_list.append(rgb_rt)
        z_enc_list.append(z_enc.cpu())
        z_stored_list.append(train["z"][sl_idx].float())

    del vae, dec
    gc.collect()
    torch.cuda.empty_cache()

    rgb_orig = torch.cat(rgb_orig_list, dim=0)
    rgb_rt = torch.cat(rgb_rt_list, dim=0)
    z_enc_all = torch.cat(z_enc_list, dim=0)
    z_stored_all = torch.cat(z_stored_list, dim=0)

    stack = FeatureStack(device)
    proto = stack.csd_prototypes
    pils_orig = _to_pils(rgb_orig)
    pils_rt = _to_pils(rgb_rt)
    f_orig = stack.extract(pils_orig, batch=16)
    f_rt = stack.extract(pils_rt, batch=16)
    u8_orig = (rgb_orig * 255.0).round().byte()
    u8_rt = (rgb_rt * 255.0).round().byte()
    lpips_vals = stack.lpips_vs(u8_rt, u8_orig, batch=16).cpu().numpy()

    # Ref-bank DINO-S using existing reference_bank
    from src.utils import reference_bank
    rbank = reference_bank(device)
    rd = rbank["dino"]
    dino_s_orig = torch.stack([(f_orig["dino"].to(device) @ p.T).max(1).values for p in rd], dim=1).cpu().numpy()
    dino_s_rt = torch.stack([(f_rt["dino"].to(device) @ p.T).max(1).values for p in rd], dim=1).cpu().numpy()
    dino_c_rt = F.cosine_similarity(f_rt["dino"].float(), f_orig["dino"].float()).cpu().numpy()

    csd_pred_orig = (f_orig["csd"] @ proto.T).argmax(axis=1)
    csd_pred_rt = (f_rt["csd"] @ proto.T).argmax(axis=1)
    haar_orig = _haar_hf_lf(rgb_orig).numpy()
    haar_rt = _haar_hf_lf(rgb_rt).numpy()
    ok_orig = extract_color_fiber_27d(rgb_orig).numpy()
    ok_rt = extract_color_fiber_27d(rgb_rt).numpy()

    del stack
    gc.collect()
    torch.cuda.empty_cache()

    out = {
        "n_per_style": n_per_style,
        "latent_enc_vs_stored_mae": round(float((z_enc_all - z_stored_all).abs().mean()), 6),
        "styles": {},
    }
    for sid in (0, 4):
        m = sel_sids == sid
        name = STYLES[sid]
        out["styles"][name] = {
            "n": int(m.sum()),
            "orig_csd_top1": round(float((csd_pred_orig[m] == sid).mean()), 4),
            "s10_roundtrip_csd_top1": round(float((csd_pred_rt[m] == sid).mean()), 4),
            "top1_retention_delta_pp": round(float(((csd_pred_rt[m] == sid).mean() - (csd_pred_orig[m] == sid).mean()) * 100.0), 2),
            "orig_dino_s": round(float(dino_s_orig[m, sid].mean()), 4),
            "s10_roundtrip_dino_s": round(float(dino_s_rt[m, sid].mean()), 4),
            "dino_s_delta": round(float(dino_s_rt[m, sid].mean() - dino_s_orig[m, sid].mean()), 4),
            "s10_roundtrip_dino_c": round(float(dino_c_rt[m].mean()), 4),
            "s10_roundtrip_lpips": round(float(lpips_vals[m].mean()), 4),
            "haar_hf_lf_delta": round(float((haar_rt[m] - haar_orig[m]).mean()), 4),
            "oklab_27d_mae": round(float(np.abs(ok_rt[m] - ok_orig[m]).mean()), 4),
        }
    return out


@torch.inference_mode()
def audit_student_transfer_loss(train: dict, paths_4400: list, device: str = "cuda") -> dict:
    """Decompose ema_s4000 transfer loss vs teacher hiC/hiS on the exact 150 probe tasks."""
    pilot_p = GATE_OLD_OUT / "pilot_endpoints.pt"
    ckpt_p = UNIFIED_OLD_OUT / "ema_s4000.pt"
    assert pilot_p.exists(), f"Missing {pilot_p}"
    assert ckpt_p.exists(), f"Missing {ckpt_p}"

    pilot = torch.load(pilot_p, map_location="cpu", weights_only=False)
    src_idx = pilot["src"].long()
    tid = pilot["tid"].long()
    z_src = train["z"][src_idx].float()
    c64_src = train["c_64"][src_idx].float()
    z_hic = pilot["hiC"].float()
    z_his = pilot["hiS"].float()

    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    q32_src = c21["train"]["q32"][src_idx].float()
    saved = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model = UnifiedEditor(saved["raw_stats"].shape[-1]).to(device).eval()
    model.load_state_dict(saved["ema"])
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}

    dec = load_s10_decoder(device)
    # Decode source S10, original RGB, teacher hiC/hiS, and student at 4 lambdas
    rgb_src_orig = torch.cat([load_rgb_512_tensor([paths_4400[i] for i in sl.tolist()]) for sl in src_idx.split(25)], 0)
    rgb_src_s10 = decode_s10(dec, z_src.to(device), batch=16, output_cpu=True).clamp(0.0, 1.0)
    rgb_hic = decode_s10(dec, z_hic.to(device), batch=16, output_cpu=True).clamp(0.0, 1.0)
    rgb_his = decode_s10(dec, z_his.to(device), batch=16, output_cpu=True).clamp(0.0, 1.0)

    stu_latents = {}
    stu_rgbs = {}
    for lam in (0.45, 0.70, 0.85, 1.00):
        zs_out = []
        for st in range(0, len(src_idx), 16):
            sl = slice(st, min(st + 16, len(src_idx)))
            z_out = model(
                z_src[sl].to(device),
                c64_src[sl].to(device),
                q32_src[sl].to(device),
                tid[sl].to(device),
                torch.full((len(src_idx[sl]),), lam, device=device, dtype=torch.float32),
                raw_stats,
                priors,
            )
            zs_out.append(z_out.cpu())
        z_all = torch.cat(zs_out, 0)
        stu_latents[lam] = z_all
        stu_rgbs[lam] = decode_s10(dec, z_all.to(device), batch=16, output_cpu=True).clamp(0.0, 1.0)

    del dec, model
    gc.collect()
    torch.cuda.empty_cache()

    stack = FeatureStack(device)
    proto = stack.csd_prototypes
    from src.utils import reference_bank
    rbank = reference_bank(device)
    rd = rbank["dino"]
    rc = rbank["clip"]

    f_orig = stack.extract(_to_pils(rgb_src_orig), batch=16)
    f_s10_src = stack.extract(_to_pils(rgb_src_s10), batch=16)
    u8_s10_src = (rgb_src_s10 * 255.0).round().byte()

    def _eval_rgb(name: str, rgb: torch.Tensor, z_lat: torch.Tensor, ref_lat: torch.Tensor | None = None) -> dict:
        feats = stack.extract(_to_pils(rgb), batch=16)
        csd_scores = feats["csd"] @ proto.T
        pred = csd_scores.argmax(axis=1)
        tid_np = tid.numpy()
        hit = pred == tid_np
        dino_s_mat = torch.stack([(feats["dino"].to(device) @ p.T).max(1).values for p in rd], dim=1).cpu().numpy()
        dino_s = float(dino_s_mat[np.arange(len(tid_np)), tid_np].mean())
        clip_mat = (feats["clip"].to(device) @ rc.T).cpu().numpy()
        clip_s = float(clip_mat[np.arange(len(tid_np)), tid_np].mean())
        dino_c_orig = float(F.cosine_similarity(feats["dino"].float(), f_orig["dino"].float()).mean())
        dino_c_s10 = float(F.cosine_similarity(feats["dino"].float(), f_s10_src["dino"].float()).mean())
        u8 = (rgb * 255.0).round().byte()
        lpips_s10 = float(stack.lpips_vs(u8, u8_s10_src, batch=16).mean())
        haar_val = float(_haar_hf_lf(rgb).mean())

        res = {
            "arm": name,
            "csd_top1": round(float(hit.mean()), 4),
            "er_hits_30": int(hit[tid_np == 0].sum()),
            "uki_hits_30": int(hit[tid_np == 4].sum()),
            "per_target": {STYLES[s]: round(float(hit[tid_np == s].mean()), 4) for s in range(5)},
            "dino_s": round(dino_s, 4),
            "clip_s": round(clip_s, 4),
            "dino_c_vs_orig": round(dino_c_orig, 4),
            "dino_c_vs_s10_src": round(dino_c_s10, 4),
            "lpips_vs_s10_src": round(lpips_s10, 4),
            "haar_hf_lf": round(haar_val, 4),
        }
        if ref_lat is not None:
            P_Y = priors["P_Y"].cpu()
            P_C = priors["P_C"].cpu()
            dz = z_lat - ref_lat
            dy = torch.einsum("ij,bjhw->bihw", P_Y, dz)
            dc = torch.einsum("ij,bjhw->bihw", P_C, dz)
            res["vs_teacher_latent"] = {
                "l1": round(float(dz.abs().mean()), 4),
                "luma_l1": round(float(dy.abs().mean()), 4),
                "chroma_l1": round(float(dc.abs().mean()), 4),
                "high_freq_energy_ratio": round(float(
                    sum(b.square().mean() for b in dwt2_haar(z_lat)[1:])
                    / sum(b.square().mean() for b in dwt2_haar(ref_lat)[1:]).clamp_min(1e-8)
                ), 4),
            }
        return res

    results = {
        "teacher_hiC": _eval_rgb("teacher_hiC", rgb_hic, z_hic),
        "teacher_hiS": _eval_rgb("teacher_hiS", rgb_his, z_his),
    }
    for lam in (0.45, 0.70, 0.85, 1.00):
        ref_t = z_hic if lam <= 0.45 else z_his
        row = _eval_rgb(f"ema_s4000_l{int(round(lam*100)):03d}", stu_rgbs[lam], stu_latents[lam], ref_lat=ref_t)
        teacher_ref = results["teacher_hiC"] if lam <= 0.45 else results["teacher_hiS"]
        row["dino_s_drop_vs_teacher"] = round(row["dino_s"] - teacher_ref["dino_s"], 4)
        row["dino_s_retention_ratio"] = round(row["dino_s"] / max(teacher_ref["dino_s"], 1e-6), 4)
        row["haar_retention_delta"] = round(row["haar_hf_lf"] - teacher_ref["haar_hf_lf"], 4)
        results[f"stu_l{int(round(lam*100)):03d}"] = row

    del stack
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_phase0(device: str = "cuda") -> dict:
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ccf = torch.load(ROOT / "runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)
    train = ccf["train"]

    # 1. 100% RGB mapping audit (4,400 train + 1,500 planned sources)
    paths_4400 = resolve_train_rgb_paths(train["ids"], train["sid"], require_full=True)
    selected, _, _ = task_plan(train)
    paths_1500 = [paths_4400[i] for i in selected.tolist()]
    assert len(paths_1500) == 1500 and all(p is not None and p.exists() for p in paths_1500)
    mapping_summary = {
        "train_4400_mapped": len(paths_4400),
        "planned_1500_mapped": len(paths_1500),
        "d5_count": sum(1 for s in train["ids"] if str(s).startswith("d5__")),
        "wikiart_count": sum(1 for s in train["ids"] if str(s).startswith("wikiart__")),
        "all_files_exist": True,
    }
    print(f"[Phase 0] RGB mapping verified: {mapping_summary}", flush=True)

    # 2. 200 ER + 200 Uki Enc->S10 roundtrip ceiling
    ceiling = audit_enc_s10_ceiling(train, paths_4400, n_per_style=200, device=device)
    print(f"[Phase 0] Enc->S10 ceiling: {ceiling}", flush=True)

    # 3. 150-task exact decomposition of ema_s4000 vs teacher
    transfer = audit_student_transfer_loss(train, paths_4400, device=device)
    print(f"[Phase 0] Student transfer decomposition done in {time.monotonic()-t0:.1f}s", flush=True)

    vram_gb = assert_vram_budget(device=device, tag="phase0")
    report = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "peak_vram_gb": round(vram_gb, 3),
        "mapping": mapping_summary,
        "enc_s10_ceiling_200x2": ceiling,
        "student_transfer_loss_150": transfer,
    }
    save_json(OUT / "phase0_audit.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_phase0(device=args.device)


if __name__ == "__main__":
    main()
