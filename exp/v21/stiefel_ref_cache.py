"""Stiefel Polar Quotient (0% 2nd-Order Gram Leak!) & Arbitrary-Image 1-Shot Reference Token Cache."""
from __future__ import annotations

import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from . import OUT, ROOT, STYLES, save_json, seed_all
from exp.v18 import tasks_and_sources
from exp.v18.structure import _train_linear_mlp_probe, local_contrast_norm
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import encode_dino, load_dino_model
from src.utils.protocol import canonical_pid
from src.utils.structure import extract_color_fiber_27d, extract_whitened_dino_layout, gaussian_blur_3x3, rgb_to_gray

SG_5_STEMS = [
    "Early_Renaissance__jacopo-bellini_sketch-of-figures-and-scenes-from-the-antique-age",
    "Impressionism__berthe-morisot_young-woman-holding-a-dog-in-her-arms",
    "Minimalism__agnes-martin_untitled-11-2002",
    "Rococo__maurice-quentin-de-la-tour_self-portrait",
    "Ukiyo_e__katsushika-hokusai_tea-house-at-koishikawa-the-morning-after-a-snowfall",
]


def stiefel_polar_quotient(p16: torch.Tensor, W_k: torch.Tensor) -> torch.Tensor:
    """Project (B, 384, 16, 16) -> (B, k, 16, 16) on Stiefel frame: mean=0 AND Gram=(1/256)Q Q^T == I_k!"""
    w = W_k.to(device=p16.device, dtype=torch.float32)
    fl = torch.einsum("dk,bdp->bkp", w, p16.float().flatten(2))  # (B, k, 256)
    fl = fl - fl.mean(dim=-1, keepdim=True)
    # Run SVD in float32 for exact orthogonality U @ U.T == I_k
    U, _, Vh = torch.linalg.svd(fl, full_matrices=False)
    q = torch.bmm(U, Vh) * 16.0  # sqrt(256) = 16.0; Vh rows already sum to 0!
    return q.reshape(p16.shape[0], w.shape[1], 16, 16)


def smooth_y32_shading(y64: torch.Tensor) -> torch.Tensor:
    """Brushstroke-free 32x32 smooth luminance envelope in [-1, 1]."""
    y32 = F.avg_pool2d(y64.float(), kernel_size=2, stride=2)
    yb = gaussian_blur_3x3(gaussian_blur_3x3(y32))
    mu = yb.mean(dim=(2, 3), keepdim=True)
    sd = yb.std(dim=(2, 3), keepdim=True, unbiased=False)
    return ((yb - mu) / (sd + 0.04)).clamp(-2.5, 2.5) / 2.5


def coarse_y16_lcn(y64: torch.Tensor) -> torch.Tensor:
    y16 = F.avg_pool2d(y64.float(), kernel_size=4, stride=4)
    return local_contrast_norm(y16, kernel=3, eps=0.04)


@torch.no_grad()
def encode_rgb_dino_patches(dino_model, dec_s10, z_all: torch.Tensor, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode z_all via S10 and extract RGB DINOv2 (N, 256, 384) patches + 795-D global fiber + y64."""
    patches_list, fib_list, y64_list = [], [], []
    bs = 32
    for st in range(0, len(z_all), bs):
        zb = z_all[st : st + bs].to(device).float()
        rgb = decode_s10(dec_s10, zb, batch=16).to(device).clamp(0.0, 1.0)
        y512 = rgb_to_gray(rgb)
        y64 = F.avg_pool2d(gaussian_blur_3x3(F.avg_pool2d(y512, 4, 4)), 2, 2)
        y64_list.append(y64.cpu().half())

        col27 = extract_color_fiber_27d(rgb)
        u8 = (rgb.cpu() * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        imgs = [Image.fromarray(a) for a in u8]
        _, patch = encode_dino(dino_model, imgs, batch=32, device=device)  # (B, 256, 384)
        pf = patch.float().to(device)
        mu = pf.mean(dim=1)
        sd = pf.std(dim=1, unbiased=False)
        fib_list.append(torch.cat([col27, mu, sd], dim=-1).cpu())
        patches_list.append(pf.cpu().half())
    return torch.cat(patches_list, 0), torch.cat(fib_list, 0), torch.cat(y64_list, 0)


@torch.no_grad()
def build_or_load_v21_cache(pool_dir: str | Path, k_stiefel: int = 32, d_rgb_pca: int = 128, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p = OUT / "cache/v21_stiefel_ref_cache.pt"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_or_load_v19_foundation(pool_dir, device=device)
    if cache_p.exists():
        print("[V21 Cache] Reusing v21_stiefel_ref_cache.pt", flush=True)
        c21 = torch.load(cache_p, map_location="cpu", weights_only=False)
        return {"bundle": bundle, "c21": c21}

    t0 = time.time()
    seed_all(20261121)
    tr = bundle["v18_cache"]["train"]
    ho = bundle["v18_cache"]["holdout"]
    ts = bundle["test_struct"]
    fnd = bundle["fnd"]
    _, z_src_750, _ = tasks_and_sources()

    # 0. Mask out StyleGallery's 5 exact 1-shot reference stems from train_4400 so they are 100% unseen!
    sg_pids = {canonical_pid(s) for s in SG_5_STEMS}
    tr_ids = [str(x) for x in tr.get("ids", [f"tr_{i}" for i in range(len(tr["z"]))])]
    train_valid_mask = torch.tensor([canonical_pid(x) not in sg_pids for x in tr_ids], dtype=torch.bool)

    # 1. Fit universal k=32 PCA basis for Stiefel Polar Quotient on valid train_4400 grayscale DINO tokens
    X_gray = tr["dino_16"][train_valid_mask].float().permute(0, 2, 3, 1).reshape(-1, 384)[::12]
    X_gray = X_gray - X_gray.mean(0, keepdim=True)
    _, _, Vh_g = torch.linalg.svd(X_gray.to(device), full_matrices=False)
    W_stiefel = Vh_g[:k_stiefel].T.cpu()  # (384, 32)

    q32_tr = stiefel_polar_quotient(tr["dino_16"], W_stiefel).half()
    q32_ho = stiefel_polar_quotient(ho["dino_16"], W_stiefel).half()
    q32_750 = stiefel_polar_quotient(ts["dino_16"], W_stiefel).half()

    # Audit old 2nd-order Gram leak vs Stiefel Polar Quotient Gram leak!
    def _gram_feat(t4d):
        fl = t4d.float().flatten(2)  # (N, C, 256)
        if fl.shape[1] > 48:
            fl = fl[:, :48, :]
        G = torch.bmm(fl, fl.transpose(1, 2)) / 256.0
        ii = torch.triu_indices(fl.shape[1], fl.shape[1])
        return G[:, ii[0], ii[1]]

    with torch.enable_grad():
        acc_old_gram = _train_linear_mlp_probe(_gram_feat(tr["dino_16"]), tr["sid"], _gram_feat(ho["dino_16"]), ho["sid"])
        acc_stiefel_gram = _train_linear_mlp_probe(_gram_feat(q32_tr), tr["sid"], _gram_feat(q32_ho), ho["sid"])
    gram_ident_err = float((torch.bmm(q32_ho.float().flatten(2), q32_ho.float().flatten(2).transpose(1, 2)) / 256.0 - torch.eye(k_stiefel)).abs().max().item())
    audit = {
        "old_p16_gram_5way_leak": round(acc_old_gram, 4),
        "stiefel_q32_gram_5way_leak": round(acc_stiefel_gram, 4),
        "max_gram_identity_err": gram_ident_err,
    }
    save_json(OUT / "stiefel_gram_leak_audit.json", audit)
    print(f"[V21 Audit] Old P_16 Gram leak={acc_old_gram*100:.2f}% -> Stiefel Q_32 Gram leak={acc_stiefel_gram*100:.2f}% (err={gram_ident_err:.2e})!", flush=True)

    # 2. Extract RGB DINOv2 16x16 patches for train_4400, holdout_300, and StyleGallery's exact 5 1-shot references!
    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)

    tr_rgb_p, _, _ = encode_rgb_dino_patches(dino, dec, tr["z"], device=device)
    ho_rgb_p, _, _ = encode_rgb_dino_patches(dino, dec, ho["z"], device=device)

    # Load exact 5 StyleGallery 1-shot reference latents from pool_dir
    sg5_z = []
    for s, st_name in enumerate(SG_5_STEMS):
        d = torch.load(Path(pool_dir) / f"{s:02d}_{STYLES[s]}.pt", map_location="cpu", weights_only=False)
        cp = canonical_pid(st_name)
        hit = next(i for i, x in enumerate(d["ids"]) if canonical_pid(str(x)) == cp)
        sg5_z.append(d["latents"][hit].float())
    sg5_z_t = torch.stack(sg5_z, dim=0)
    sg5_rgb_p, sg5_fib, sg5_y64 = encode_rgb_dino_patches(dino, dec, sg5_z_t, device=device)
    sg5_rgb = decode_s10(dec, sg5_z_t.to(device), batch=5).to(device).clamp(0.0, 1.0)
    sg5_d16 = extract_whitened_dino_layout(dino, sg5_rgb, batch=5, device=device).cpu()
    sg5_q32 = stiefel_polar_quotient(sg5_d16, W_stiefel).half()

    # Also compute y64 for test750 if not in test_struct
    if "y_64" in ts:
        y64_750 = ts["y_64"].float()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y64_750 = ((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)

    del dec, dino
    torch.cuda.empty_cache()

    # 3. Fit universal un-centered PCA-128 on RGB DINO patches (strictly on valid train_4400) so V_ref preserves full palette+texture!
    X_rgb = tr_rgb_p[train_valid_mask].float().reshape(-1, 384)[::16].to(device)
    rgb_mu = X_rgb.mean(0, keepdim=True)
    _, _, Vh_r = torch.linalg.svd(X_rgb - rgb_mu, full_matrices=False)
    W_rgb128 = Vh_r[:d_rgb_pca].T.cpu()  # (384, 128)
    rgb_mu_cpu = rgb_mu.cpu()

    def _pack_vref(rgb_patches: torch.Tensor, z_lat: torch.Tensor) -> torch.Tensor:
        # (N, 256, 128) RGB DINO appearance + (N, 256, 64) 4x4 latent brushstroke patch = (N, 256, 192)
        vd = ((rgb_patches.float() - rgb_mu_cpu) @ W_rgb128) / 8.0
        vl = F.unfold(z_lat.float(), kernel_size=4, stride=4).transpose(1, 2)
        return torch.cat([vd, vl], dim=-1).half()

    vref_tr = _pack_vref(tr_rgb_p, tr["z"])
    vref_ho = _pack_vref(ho_rgb_p, ho["z"])
    vref_sg5 = _pack_vref(sg5_rgb_p, sg5_z_t)

    # 4. Global continuous fibers u_ref & Intra-Train Core-kNN distinct-neighbor graph!
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_tr = fiber_enc.encode(fnd["tr_fib"].to(device).float()).cpu()
    u_ho = fiber_enc.encode(fnd["ho_fib"].to(device).float()).cpu()
    u_sg5 = fiber_enc.encode(sg5_fib.to(device).float()).cpu()
    logits_tr = fiber_enc.logits(fnd["tr_fib"].to(device).float()).cpu()
    del fiber_enc

    tr_sid = tr["sid"].long()
    tr_dmu = F.normalize(fnd["tr_fib"][:, 27:27 + 384].float(), dim=-1)
    tr_dsd = fnd["tr_fib"][:, 27 + 384:27 + 768].float().mean(1)
    ho_dmu = F.normalize(fnd["ho_fib"][:, 27:27 + 384].float(), dim=-1)
    ho_dsd = fnd["ho_fib"][:, 27 + 384:27 + 768].float().mean(1)
    sg5_dsd = sg5_fib[:, 27 + 384:27 + 768].float().mean(1)

    core_idx_per_style: list[torch.Tensor] = []
    nbr_idx = torch.zeros(len(tr["z"]), 12, dtype=torch.long)
    tr_pids = [canonical_pid(x) for x in tr_ids]

    for s in range(5):
        ms = ((tr_sid == s) & train_valid_mask).nonzero(as_tuple=True)[0]
        dm = tr_dmu[ms]
        sim = dm @ dm.T
        knn30 = sim.topk(min(31, len(ms)), dim=1).values[:, 1:].mean(1)
        oh = F.one_hot(torch.full((len(ms),), s), 5).bool()
        pur = (logits_tr[ms][:, s] - logits_tr[ms].masked_fill(oh, -1e9).max(1).values).clamp(-2.0, 8.0)
        ok = (tr_dsd[ms] > tr_dsd[ms].quantile(0.22)) & (pur > 1.0)
        core_local = (knn30 + 0.01 * pur - (~ok).float() * 10.0).topk(min(200, len(ms))).indices
        ms_core = ms[core_local]
        core_idx_per_style.append(ms_core)

        # For EVERY sample in style s (even excluded ones), match to top-12 DISTINCT core paintings (pid_j != pid_i)!
        all_s = (tr_sid == s).nonzero(as_tuple=True)[0]
        sim_to_core = tr_dmu[all_s] @ tr_dmu[ms_core].T  # (N_s, 200)
        for loc_i, orig_i in enumerate(all_s.tolist()):
            row_sim = sim_to_core[loc_i].clone()
            for cj, core_j in enumerate(ms_core.tolist()):
                if core_j == orig_i or tr_pids[core_j] == tr_pids[orig_i]:
                    row_sim[cj] = -1e9
            nbr_idx[orig_i] = ms_core[row_sim.topk(12).indices]

    # Also pick 1 single unseen holdout medoid per style (and top-12 holdout core per style for layout-matched unseen 1-shot!)
    ho_sid = ho["sid"].long()
    ho_single_idx = []
    ho_core_per_style = []
    for s in range(5):
        ms = (ho_sid == s).nonzero(as_tuple=True)[0]
        dm = ho_dmu[ms]
        knn_h = (dm @ dm.T).topk(min(15, len(ms)), dim=1).values[:, 1:].mean(1)
        ok_h = ho_dsd[ms] > ho_dsd[ms].quantile(0.20)
        sc_h = knn_h - (~ok_h).float() * 10.0
        ho_single_idx.append(int(ms[sc_h.argmax()].item()))
        ho_core_per_style.append(ms[sc_h.topk(min(12, len(ms))).indices])

    mid_dsd = 0.5 * (
        min(float(tr_dsd[ci].mean().item()) for ci in core_idx_per_style)
        + max(float(tr_dsd[ci].mean().item()) for ci in core_idx_per_style)
    )

    c21 = {
        "W_stiefel": W_stiefel,
        "W_rgb128": W_rgb128,
        "rgb_mu": rgb_mu_cpu,
        "mid_dsd": mid_dsd,
        "audit": audit,
        "train": {
            "valid_mask": train_valid_mask,
            "q32": q32_tr,
            "y16": coarse_y16_lcn(tr["y_64"]).half(),
            "y32": smooth_y32_shading(tr["y_64"]).half(),
            "vref": vref_tr,
            "u": u_tr.half(),
            "dsd": tr_dsd,
            "core_idx_per_style": core_idx_per_style,
            "nbr_idx": nbr_idx,
        },
        "holdout": {
            "q32": q32_ho,
            "vref": vref_ho,
            "u": u_ho.half(),
            "dsd": ho_dsd,
            "single_idx": torch.tensor(ho_single_idx, dtype=torch.long),
            "core_per_style": ho_core_per_style,
        },
        "sg5": {
            "stems": SG_5_STEMS,
            "z": sg5_z_t.half(),
            "q32": sg5_q32,
            "vref": vref_sg5,
            "u": u_sg5.half(),
            "dsd": sg5_dsd,
        },
        "test750": {
            "q32": q32_750,
            "y16": coarse_y16_lcn(y64_750).half(),
            "y32": smooth_y32_shading(y64_750).half(),
        },
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(c21, cache_p)
    print(f"[V21 Cache] Built v21_stiefel_ref_cache.pt in {c21['wall_sec']}s", flush=True)
    return {"bundle": bundle, "c21": c21}
