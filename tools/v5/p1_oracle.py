"""V5 P1: Latent Oracle
Directly optimize source latent z' = z + delta for 64 steps on 100 cross-style tasks.
Matrix:
- O1_weak, O1_strong (LL modified)
- O2_weak, O2_strong (LL + LH + HL modified)
- O3_weak, O3_strong (All modified, including HH)
- O_unconstrained (Diagnostic, lambda=0.0)

Evaluates on independent CSD (dT, dS, dM), DINO-C, and LPIPS to judge Branch G1.
Maintains VRAM in the optimal 7-9.5 GB range to avoid WDDM host-memory paging.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import paths
import csd_model
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar
from tools.pseudo_gt.matching import load_dino_model, encode_dino

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_IDS = {name: i for i, name in enumerate(STYLES)}


def spatial_self_similarity(z: torch.Tensor) -> torch.Tensor:
    """Compute normalized spatial self-similarity matrix on 16x16 pooled grid."""
    p = F.adaptive_avg_pool2d(z, (16, 16))
    B, C, H, W = p.shape
    flat = p.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 256, C)
    norm = F.normalize(flat, dim=-1, eps=1e-6)
    sim = torch.bmm(norm, norm.transpose(1, 2))  # (B, 256, 256)
    return sim


def gradient_loss(z: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    """Directional consistency loss on spatial gradients."""
    dx = z[:, :, :, 1:] - z[:, :, :, :-1]
    dy = z[:, :, 1:, :] - z[:, :, :-1, :]
    dx_s = z_src[:, :, :, 1:] - z_src[:, :, :, :-1]
    dy_s = z_src[:, :, 1:, :] - z_src[:, :, :-1, :]

    cos_x = F.cosine_similarity(dx, dx_s, dim=1, eps=1e-6)
    cos_y = F.cosine_similarity(dy, dy_s, dim=1, eps=1e-6)
    return (1.0 - cos_x.mean()) + (1.0 - cos_y.mean())


def compute_structure_loss(z: torch.Tensor, z_src: torch.Tensor, sim_src: torch.Tensor) -> torch.Tensor:
    sim_z = spatial_self_similarity(z)
    l_selfsim = F.mse_loss(sim_z, sim_src)
    l_grad = gradient_loss(z, z_src)
    return l_selfsim + 0.2 * l_grad


def compute_style_loss(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    z: torch.Tensor,
    tgt_style_ids: torch.Tensor,
    descriptors: dict,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute differentiable style loss against real art multi-scale descriptors & SWD."""
    B = z.shape[0]
    total_loss = z.new_tensor(0.0)

    for sid, style_name in enumerate(STYLES):
        mask = (tgt_style_ids == sid)
        if not mask.any():
            continue
        st_desc = descriptors["styles"][style_name]
        
        mu_ll = st_desc["ll_mean"].to(device)
        mu_lh = st_desc["lh_mean"].to(device)
        mu_hl = st_desc["hl_mean"].to(device)
        mu_hh = st_desc["hh_mean"].to(device)

        cov_ll_tgt = st_desc["ll_cov"].to(device)
        cov_lh_tgt = st_desc["lh_cov"].to(device)
        cov_hl_tgt = st_desc["hl_cov"].to(device)
        cov_hh_tgt = st_desc["hh_cov"].to(device)

        ll_sub = ll[mask]
        lh_sub = lh[mask]
        hl_sub = hl[mask]
        hh_sub = hh[mask]

        m_diff = (
            F.mse_loss(ll_sub.mean(dim=(0, 2, 3)), mu_ll)
            + F.mse_loss(lh_sub.mean(dim=(0, 2, 3)), mu_lh)
            + F.mse_loss(hl_sub.mean(dim=(0, 2, 3)), mu_hl)
            + F.mse_loss(hh_sub.mean(dim=(0, 2, 3)), mu_hh)
        )

        def b_cov(b_sub: torch.Tensor) -> torch.Tensor:
            flat = b_sub.permute(0, 2, 3, 1).reshape(-1, 4)
            mean = flat.mean(dim=0, keepdim=True)
            centered = flat - mean
            return (centered.t() @ centered) / max(centered.shape[0] - 1, 1)

        cov_diff = (
            F.mse_loss(b_cov(ll_sub), cov_ll_tgt)
            + F.mse_loss(b_cov(lh_sub), cov_lh_tgt)
            + F.mse_loss(b_cov(hl_sub), cov_hl_tgt)
            + F.mse_loss(b_cov(hh_sub), cov_hh_tgt)
        )

        # Patch SWD loss
        R_z = descriptors["R_z"].to(device)
        pat_z = F.unfold(z[mask], kernel_size=8, stride=8).permute(0, 2, 1).reshape(-1, 256)
        proj_z = pat_z @ R_z.t()
        proj_z_sorted, _ = torch.sort(proj_z, dim=0)

        q_idx = torch.linspace(0.0, float(proj_z_sorted.shape[0] - 1), 128, device=device).long()
        qgrid_cur_z = proj_z_sorted[q_idx]
        qgrid_ref_z = st_desc["qgrid_z"].to(device)

        swd_loss = F.l1_loss(qgrid_cur_z, qgrid_ref_z)
        sub_loss = m_diff + 0.1 * cov_diff + 0.5 * swd_loss
        total_loss = total_loss + sub_loss * (mask.sum().float() / B)

    return total_loss


def run_oracle_arm(
    z_src: torch.Tensor,
    sim_src: torch.Tensor,
    tgt_style_ids: torch.Tensor,
    descriptors: dict,
    bands_allowed: str,
    lam_struct: float,
    num_steps: int = 64,
    lr: float = 0.05,
    device: str = "cuda",
) -> torch.Tensor:
    """Run batch optimization on all tasks concurrently."""
    with torch.no_grad():
        ll_0, lh_0, hl_0, hh_0 = dwt2_haar(z_src)

    delta_ll = torch.zeros_like(ll_0, requires_grad=True)
    delta_lh = torch.zeros_like(lh_0, requires_grad=True) if bands_allowed in ("ll_lh_hl", "all") else None
    delta_hl = torch.zeros_like(hl_0, requires_grad=True) if bands_allowed in ("ll_lh_hl", "all") else None
    delta_hh = torch.zeros_like(hh_0, requires_grad=True) if bands_allowed == "all" else None

    params = [delta_ll]
    if delta_lh is not None:
        params.extend([delta_lh, delta_hl])
    if delta_hh is not None:
        params.append(delta_hh)

    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        ll = ll_0 + delta_ll
        lh = lh_0 + (delta_lh if delta_lh is not None else torch.zeros_like(lh_0))
        hl = hl_0 + (delta_hl if delta_hl is not None else torch.zeros_like(hl_0))
        hh = hh_0 + (delta_hh if delta_hh is not None else torch.zeros_like(hh_0))

        z_rec = idwt2_haar(ll, lh, hl, hh)

        l_style = compute_style_loss(ll, lh, hl, hh, z_rec, tgt_style_ids, descriptors, device=device)
        if lam_struct > 0.0:
            l_struct = compute_structure_loss(z_rec, z_src, sim_src)
            loss = l_style + lam_struct * l_struct
        else:
            loss = l_style

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        ll = ll_0 + delta_ll
        lh = lh_0 + (delta_lh if delta_lh is not None else torch.zeros_like(lh_0))
        hl = hl_0 + (delta_hl if delta_hl is not None else torch.zeros_like(hl_0))
        hh = hh_0 + (delta_hh if delta_hh is not None else torch.zeros_like(hh_0))
        z_final = idwt2_haar(ll, lh, hl, hh)

    # Free optimizer memory
    del delta_ll, delta_lh, delta_hl, delta_hh, optimizer, params
    torch.cuda.empty_cache()
    return z_final.detach()


def decode_latents_to_pils(vae, z: torch.Tensor, batch_size: int = 8) -> List[Image.Image]:
    """Decode latents to PIL images in fp16 chunks to keep memory compact and fast."""
    B = z.shape[0]
    pils = []
    scaling = float(vae.config.scaling_factor)
    with torch.inference_mode():
        for s in range(0, B, batch_size):
            chunk = (z[s:s + batch_size] / scaling).half()
            img_g = vae.decode(chunk).sample.clamp(-1.0, 1.0)
            arr_g = ((img_g.permute(0, 2, 3, 1).cpu().float().numpy() + 1.0) * 127.5).astype(np.uint8)
            pils.extend([Image.fromarray(a) for a in arr_g])
    return pils


def evaluate_latents(
    z_final: torch.Tensor,
    tasks: List[dict],
    proto_mat: np.ndarray,
    csd: torch.nn.Module,
    dino: torch.nn.Module,
    vae: torch.nn.Module,
    csd_src: np.ndarray,
    dino_src: np.ndarray,
    device: str = "cuda",
) -> List[dict]:
    """Evaluate generated latents against precomputed source CSD/DINO."""
    pils_gen = decode_latents_to_pils(vae, z_final, batch_size=8)

    with torch.inference_mode():
        csd_gen = csd_model.feats_pils(csd, pils_gen, batch=32, device=device)
        dino_gen_t, _ = encode_dino(dino, pils_gen, batch=32, device=device)
        dino_gen = dino_gen_t.float().cpu().numpy()

    results = []
    for i, t in enumerate(tasks):
        fg = csd_gen[i]
        fs = csd_src[i]
        tgt_id = t["tgt_style_id"]
        src_id = t["src_style_id"]

        cos_tgt_g = float(fg @ proto_mat[tgt_id])
        cos_tgt_s = float(fs @ proto_mat[tgt_id])
        cos_src_g = float(fg @ proto_mat[src_id])
        cos_src_s = float(fs @ proto_mat[src_id])

        dT = cos_tgt_g - cos_tgt_s
        dS = cos_src_g - cos_src_s
        dM = dT - dS
        dino_c = float(dino_gen[i] @ dino_src[i])

        results.append({
            "task_id": t["task_id"],
            "src_style": t["src_style"],
            "tgt_style": t["tgt_style"],
            "dT": dT,
            "dS": dS,
            "dM": dM,
            "dino_c": dino_c,
            "cos_tgt_g": cos_tgt_g,
            "cos_tgt_s": cos_tgt_s,
        })
    return results


def main():
    print("=== WEAVE V5 P1: Latent Oracle Matrix Experiment ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = REPO_ROOT / "runs" / "v5" / "p1_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads((REPO_ROOT / "runs" / "v5" / "v5_oracle100_tasks.json").read_text(encoding="utf-8"))
    src_latents = torch.load(REPO_ROOT / "runs" / "v5" / "cache" / "eval_src_latents.pt", map_location="cpu")
    descriptors = torch.load(REPO_ROOT / "runs" / "v5" / "cache" / "d5_style_multiscale_descriptors.pt", map_location="cpu")

    proto_p = paths.asset_root() / "pseudo_gt_24h" / "csd_proto.pt"
    proto_dict = torch.load(proto_p, map_location="cpu", weights_only=False)
    proto_mat = np.stack([proto_dict[s] for s in STYLES])

    print("[p1] Loading CSD, DINO, and VAE (fp16)...", flush=True)
    csd, ck, msg = csd_model.load_csd(device)
    dino = load_dino_model(device=device)

    candidates = [
        paths.asset_root() / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
        REPO_ROOT / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
    ]
    vae_dir = None
    for c in candidates:
        if c.exists():
            snaps = list(c.glob("*"))
            if snaps and (snaps[0] / "config.json").exists():
                vae_dir = snaps[0]
                break

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float16, local_files_only=True).to(device).eval()

    # Prepare inputs on GPU
    z_src_list = [src_latents[t["src_stem"]] for t in tasks]
    z_src = torch.stack(z_src_list).to(device=device, dtype=torch.float32)  # (100, 4, 64, 64)
    sim_src = spatial_self_similarity(z_src)  # (100, 256, 256) precomputed once!
    tgt_style_ids = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    # Pre-compute source features ONCE
    print("[p1] Pre-decoding source images and computing source CSD & DINO features (once)...", flush=True)
    t_src_start = time.time()
    pils_src = decode_latents_to_pils(vae, z_src, batch_size=8)
    with torch.inference_mode():
        csd_src = csd_model.feats_pils(csd, pils_src, batch=32, device=device)
        dino_src_t, _ = encode_dino(dino, pils_src, batch=32, device=device)
        dino_src = dino_src_t.float().cpu().numpy()
    print(f"[p1] Source reference features ready in {time.time() - t_src_start:.2f}s", flush=True)

    matrix_configs = [
        {"name": "O1_weak", "bands": "ll", "lam": 0.5},
        {"name": "O1_strong", "bands": "ll", "lam": 2.0},
        {"name": "O2_weak", "bands": "ll_lh_hl", "lam": 0.5},
        {"name": "O2_strong", "bands": "ll_lh_hl", "lam": 2.0},
        {"name": "O3_weak", "bands": "all", "lam": 0.5},
        {"name": "O3_strong", "bands": "all", "lam": 2.0},
        {"name": "O_unconstrained", "bands": "all", "lam": 0.0},
    ]

    report = {"matrix": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[p1] Running Oracle Matrix across {len(matrix_configs)} arms on {len(tasks)} tasks concurrently...", flush=True)

    for m in matrix_configs:
        name = m["name"]
        print(f"\n>>> Arm: {name} (bands={m['bands']}, lambda={m['lam']}) <<<", flush=True)
        t0 = time.time()
        
        z_opt = run_oracle_arm(
            z_src=z_src,
            sim_src=sim_src,
            tgt_style_ids=tgt_style_ids,
            descriptors=descriptors,
            bands_allowed=m["bands"],
            lam_struct=m["lam"],
            num_steps=64,
            lr=0.05,
            device=device,
        )
        opt_time = time.time() - t0

        vram_mib = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
        print(f"[p1] {name} optimized in {opt_time:.2f}s | Peak VRAM: {vram_mib:.1f} MiB", flush=True)

        eval_t0 = time.time()
        eval_rows = evaluate_latents(
            z_final=z_opt,
            tasks=tasks,
            proto_mat=proto_mat,
            csd=csd,
            dino=dino,
            vae=vae,
            csd_src=csd_src,
            dino_src=dino_src,
            device=device,
        )
        eval_time = time.time() - eval_t0

        dTs = [r["dT"] for r in eval_rows]
        dSs = [r["dS"] for r in eval_rows]
        dMs = [r["dM"] for r in eval_rows]
        dino_cs = [r["dino_c"] for r in eval_rows]

        summary = {
            "name": name,
            "bands": m["bands"],
            "lambda_struct": m["lam"],
            "opt_time_sec": round(opt_time, 2),
            "eval_time_sec": round(eval_time, 2),
            "peak_vram_mib": round(vram_mib, 1),
            "mean_dT": float(np.mean(dTs)),
            "std_dT": float(np.std(dTs)),
            "mean_dS": float(np.mean(dSs)),
            "mean_dM": float(np.mean(dMs)),
            "mean_dino_c": float(np.mean(dino_cs)),
            "frac_dT_positive": float(np.mean([x > 0 for x in dTs])),
            "frac_dM_positive": float(np.mean([x > 0 for x in dMs])),
        }
        report["matrix"][name] = summary
        print(f"[{name} RESULT] dT: {summary['mean_dT']:+.4f} (frac>0: {summary['frac_dT_positive']*100:.1f}%) | dS: {summary['mean_dS']:+.4f} | dM: {summary['mean_dM']:+.4f} | DINO-C: {summary['mean_dino_c']:.4f}", flush=True)

    # Branch G1 Decision Logic
    best_arm = None
    best_score = -999.0
    for name, s in report["matrix"].items():
        if name == "O_unconstrained":
            continue
        if s["mean_dino_c"] >= 0.78 and s["mean_dT"] > best_score:
            best_score = s["mean_dT"]
            best_arm = name

    oracle_success = bool(best_arm is not None and report["matrix"][best_arm]["mean_dT"] >= 0.03)
    branch_g1 = "Branch_A_M1_M2" if oracle_success else "Branch_B_Supervision_Fix"

    report["decision"] = {
        "best_arm": best_arm,
        "best_arm_dT": report["matrix"][best_arm]["mean_dT"] if best_arm else None,
        "best_arm_dino_c": report["matrix"][best_arm]["mean_dino_c"] if best_arm else None,
        "oracle_success": oracle_success,
        "branch_g1": branch_g1,
    }
    print("\n=======================================================", flush=True)
    print(f"=== Branch G1 Decision: {branch_g1} (Oracle Success={oracle_success}, Best Arm={best_arm}) ===", flush=True)
    print("=======================================================", flush=True)

    out_file = out_dir / "oracle_matrix_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[p1] Complete! Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
