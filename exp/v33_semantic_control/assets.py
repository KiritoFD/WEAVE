"""CLEAVE V33 Cached Assets (100% Zero-DINO):
1. v33_routing_and_energy.pt: Train-only CSLS density penalty r_k (5x8) + Haar/grad texture energy sub_E_tgt (5x8) + E_src_750.
2. v33_q64_hi32_cache.pt: W_b32 (PCA 33-64), Q64+ complement Q_b, partial-whitening residual (alpha=0.5),
   and 512px ConvNeXt-Stage3 32x32 Stiefel quotient Q_hi32 for train_4400 & test750.
"""
from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v33_semantic_control import OUT, SEED_V33, texture_energy_rgb
from exp.v33_semantic_control.models import (
    ConvNeXtMultiScaleExtractor,
    extract_cnx_q32_hi,
    partial_whitening_residual,
    stiefel_q64_plus,
)
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc


@torch.no_grad()
def match_cnx_k8_csls(
    p16_gray_src: torch.Tensor,
    tid: torch.Tensor,
    cnx_bank: dict,
    csls_r: torch.Tensor,
    *,
    mode: str = "csls",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match source scene 4x4 grayscale layout to K=8 sub-school centroid.

    mode: 'argmax' (R0), 'csls' (R1: 2*cos - r_k), or 'second' (R2: 2nd-nearest cosine).
    Returns (u_style (B, 417), best_k (B,), raw_sims (B, 8)).
    """
    dev = p16_gray_src.device
    sc_src = F.normalize(F.avg_pool2d(p16_gray_src.float(), 4).flatten(1), dim=-1)
    sub_sc = cnx_bank["sub_scene"].to(dev).float()[tid]
    sims = torch.einsum("bd,bkd->bk", sc_src, sub_sc)
    if mode == "argmax":
        pick = sims.argmax(dim=1)
    elif mode == "csls":
        rk = csls_r.to(dev).float()[tid]
        pick = (2.0 * sims - rk).argmax(dim=1)
    elif mode == "second":
        pick = sims.topk(2, dim=1).indices[:, 1]
    else:
        raise ValueError(f"Unknown routing mode: {mode!r}")
    u = cnx_bank["sub_raw417"].to(dev).float()[tid, pick]
    return u, pick, sims


@torch.no_grad()
def build_or_load_routing_and_energy(device: str = "cuda", *, k_csls: int = 50) -> dict:
    """Build train-only CSLS density penalty r_k and Haar+grad texture energy assets."""
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "v33_routing_and_energy.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)

    sanitize_cuda_env()
    t0 = time.monotonic()
    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    sid_tr = tr["sid"].long()
    cnx_gray_tr = stiefel["train"]["cnx_gray"].float()
    sc_tr = F.normalize(F.avg_pool2d(cnx_gray_tr, 4).flatten(1), dim=-1)  # (4400, 6144)

    sub_sc = cnx_bank["sub_scene"].float()  # (5, 8, 6144)
    csls_r = torch.zeros(5, 8, dtype=torch.float32)
    for s in range(5):
        cross_mask = sid_tr != s
        sc_cross = sc_tr[cross_mask].to(device)
        cent = F.normalize(sub_sc[s].to(device), dim=-1)
        sim_cross = sc_cross @ cent.T  # (N_cross, 8)
        top_sim = sim_cross.topk(min(k_csls, sim_cross.shape[0]), dim=0).values
        csls_r[s] = top_sim.mean(dim=0).cpu()

    # Compute cluster texture energy E_tgt strictly on train_4400 core members
    dec = load_s10_decoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    core_idx_per_style = [ci.cpu().long() for ci in cnx_bank["core_idx_per_style"]]
    norm_417 = cnx_bank["norm_417_all"].float()
    sub_raw = cnx_bank["sub_raw417"].float()

    sub_E_tgt = torch.zeros(5, 8, dtype=torch.float32)
    for s in range(5):
        ci = core_idx_per_style[s]
        sim_s = F.normalize(norm_417[ci], dim=-1) @ F.normalize(sub_raw[s], dim=-1).T
        lbl = sim_s.argmax(dim=1)
        e_list = []
        for st in range(0, len(ci), 32):
            sl = ci[st:min(st + 32, len(ci))]
            rgb = decode_s10(dec, tr["z"][sl].to(device).float(), dtype=torch.bfloat16, batch=len(sl), output_cpu=False).clamp(0.0, 1.0)
            e_list.append(texture_energy_rgb(rgb).cpu())
        e_ci = torch.cat(e_list, 0)
        for k in range(8):
            m = lbl == k
            if bool(m.any()):
                sub_E_tgt[s, k] = e_ci[m].mean()
            else:
                rgb_ex = decode_s10(dec, cnx_bank["sub_exemplar_z"][s, k:k + 1].to(device).float(), dtype=torch.bfloat16, batch=1, output_cpu=False).clamp(0.0, 1.0)
                sub_E_tgt[s, k] = float(texture_energy_rgb(rgb_ex).item())

    # Source texture energy on test750 (deduplicated by unique source latent for speed)
    tasks, z_src_750, _ = tasks_and_sources()
    e_src_list = []
    for st in range(0, len(z_src_750), 32):
        sl = slice(st, min(st + 32, len(z_src_750)))
        rgb = decode_s10(dec, z_src_750[sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
        e_src_list.append(texture_energy_rgb(rgb).cpu())
    E_src_750 = torch.cat(e_src_list, 0)

    del dec
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "updated_utc": utc(),
        "k_csls": k_csls,
        "csls_r": csls_r,
        "sub_E_tgt": sub_E_tgt,
        "E_src_750": E_src_750,
    }
    torch.save(payload, p)
    save_json(OUT / "v33_routing_energy_summary.json", {
        "wall_sec": round(time.monotonic() - t0, 2),
        "k_csls": k_csls,
        "csls_r_rococo": [round(float(x), 4) for x in csls_r[3].tolist()],
        "sub_E_tgt_mean_per_style": [round(float(sub_E_tgt[s].mean()), 5) for s in range(5)],
        "E_src_750_mean": round(float(E_src_750.mean()), 5),
    })
    print(f"[V33 Assets] Built routing & texture energy in {time.monotonic() - t0:.1f}s | Rococo r_k={[round(float(x), 3) for x in csls_r[3].tolist()]}", flush=True)
    return payload


@torch.no_grad()
def build_or_load_q64_hi32_cache(device: str = "cuda") -> dict:
    """Build/load Q64+ complement, partial-whitening residual, and 512px 32x32 Q_hi32 cache."""
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "v33_q64_hi32_cache.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)

    sanitize_cuda_env()
    seed_all(SEED_V33)
    t0 = time.monotonic()

    assets = build_or_load_cnx_assets(device=device)
    stiefel = assets["stiefel"]
    w32 = stiefel["W_stiefel_cnx"].float()  # (384, 32)
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr, ho = v18["train"], v18["holdout"]
    _, z_src_750, _ = tasks_and_sources()

    # 1. Fit PCA 33-64 complement W_b32 strictly on train_4400 256px grayscale ConvNeXt Stage-3
    cnx_gray_tr = stiefel["train"]["cnx_gray"].float()
    X_gray = cnx_gray_tr.permute(0, 2, 3, 1).reshape(-1, 384)[::12]
    X_gray = X_gray - X_gray.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X_gray.to(device), full_matrices=False)
    wb32 = Vh[32:64].T.cpu()

    # 2. Extract missing holdout cnx_gray (256px) AND 512px 32x32 layouts for train_4400 & test_750 (deduped by source)
    dec = load_s10_decoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    ext = ConvNeXtMultiScaleExtractor(device=device).eval()

    ho_g16_list = []
    for st in range(0, len(ho["z"]), 32):
        sl = slice(st, min(st + 32, len(ho["z"])))
        rgb = decode_s10(dec, ho["z"][sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
        ho_g16_list.append(ext.extract_gray_layout(rgb).cpu().half())
    cnx_gray_ho = torch.cat(ho_g16_list, 0)

    def _compute_q64_and_pw(g16: torch.Tensor, q32_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qb_l, pw_l = [], []
        for st in range(0, len(g16), 256):
            sl = slice(st, min(st + 256, len(g16)))
            gb = g16[sl].to(device).float()
            qref = q32_ref[sl].to(device).float()
            q64 = stiefel_q64_plus(gb, w32, wb32, q32=qref)
            pw = partial_whitening_residual(gb, w32, q32=qref, alpha=0.5)
            qb_l.append(q64[:, 32:].cpu().half())
            pw_l.append(pw.cpu().half())
        return torch.cat(qb_l, 0), torch.cat(pw_l, 0)

    qb_tr, pw_tr = _compute_q64_and_pw(stiefel["train"]["cnx_gray"], stiefel["train"]["q32"])
    qb_ho, pw_ho = _compute_q64_and_pw(cnx_gray_ho, stiefel["holdout"]["q32"])
    qb_750, pw_750 = _compute_q64_and_pw(stiefel["test750"]["cnx_gray"], stiefel["test750"]["q32"])

    # 3. Extract 512px -> 32x32 grayscale ConvNeXt Stage-3 features (train_4400 + unique test sources)
    print("[V33 Assets] Extracting 512px ConvNeXt Stage-3 32x32 layouts...", flush=True)
    tr_g32_chunks = []
    for st in range(0, len(tr["z"]), 16):
        sl = slice(st, min(st + 16, len(tr["z"])))
        rgb = decode_s10(dec, tr["z"][sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
        tr_g32_chunks.append(ext.extract_gray_layout_32(rgb).cpu().half())
    cnx_g32_tr = torch.cat(tr_g32_chunks, 0)

    # Deduplicate test750 by unique z_src rows (150 unique sources repeated 5x)
    tasks, _, _ = tasks_and_sources()
    stem_to_first: dict[str, int] = {}
    unique_idx, inv_idx = [], []
    for i, t in enumerate(tasks):
        st_name = t["source_stem"]
        if st_name not in stem_to_first:
            stem_to_first[st_name] = len(unique_idx)
            unique_idx.append(i)
        inv_idx.append(stem_to_first[st_name])
    z_uniq = z_src_750[unique_idx]
    uniq_g32_chunks = []
    for st in range(0, len(z_uniq), 16):
        sl = slice(st, min(st + 16, len(z_uniq)))
        rgb = decode_s10(dec, z_uniq[sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
        uniq_g32_chunks.append(ext.extract_gray_layout_32(rgb).cpu().half())
    cnx_g32_uniq = torch.cat(uniq_g32_chunks, 0)

    del dec, ext
    gc.collect()
    torch.cuda.empty_cache()

    # Fit 512px Stiefel basis on train_4400
    X_hi = cnx_g32_tr.float().permute(0, 2, 3, 1).reshape(-1, 384)[::48]
    X_hi = X_hi - X_hi.mean(0, keepdim=True)
    _, _, Vh_hi = torch.linalg.svd(X_hi.to(device), full_matrices=False)
    w_hi32 = Vh_hi[:32].T.cpu()

    q32_hi_tr_l = []
    for st in range(0, len(cnx_g32_tr), 128):
        sl = slice(st, min(st + 128, len(cnx_g32_tr)))
        q32_hi_tr_l.append(extract_cnx_q32_hi(cnx_g32_tr[sl].to(device), w_hi32).cpu().half())
    q32_hi_tr = torch.cat(q32_hi_tr_l, 0)

    q32_hi_uniq = extract_cnx_q32_hi(cnx_g32_uniq.to(device), w_hi32).cpu().half()
    q32_hi_750 = q32_hi_uniq[torch.tensor(inv_idx, dtype=torch.long)]

    del cnx_g32_tr, cnx_g32_uniq
    gc.collect()

    payload = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "W_b32": wb32,
        "W_hi32": w_hi32,
        "train": {"q_b": qb_tr, "pw_res": pw_tr, "q32_hi": q32_hi_tr},
        "holdout": {"cnx_gray": cnx_gray_ho, "q_b": qb_ho, "pw_res": pw_ho},
        "test750": {"q_b": qb_750, "pw_res": pw_750, "q32_hi": q32_hi_750},
    }
    torch.save(payload, p)
    print(f"[V33 Assets] Built Q64+ & 512px Q32_hi cache in {payload['wall_sec']:.1f}s -> {p}", flush=True)
    return payload


@torch.no_grad()
def apply_style_perturbation_rgb(rgb01: torch.Tensor) -> torch.Tensor:
    """Apply T_style(x): random brightness curve, local contrast/shading tilt, and surface texture smoothing/noise."""
    B, _, H, W = rgb01.shape
    dev = rgb01.device
    x = rgb01.float().clamp(1e-4, 1.0)

    gamma = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.60, 1.65)
    x = x.pow(gamma)
    gain = torch.empty(B, 1, 1, 1, device=dev).uniform_(1.2, 3.2)
    s_curve = 0.5 + 0.5 * torch.tanh(gain * (x - 0.5)) / torch.tanh(0.5 * gain)
    mix = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, 0.6)
    x = (1.0 - mix) * x + mix * s_curve

    yy = torch.linspace(-1.0, 1.0, H, device=dev).view(1, 1, H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=dev).view(1, 1, 1, W)
    ax = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.25, 0.25)
    ay = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.25, 0.25)
    c_scale = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.78, 1.22)
    x = (x - 0.5) * c_scale + 0.5 + 0.15 * (ax * xx + ay * yy)

    from src.utils.structure import gaussian_blur_3x3
    sm_mask = (torch.rand(B, device=dev) < 0.5).view(B, 1, 1, 1)
    x_blur = gaussian_blur_3x3(gaussian_blur_3x3(x.clamp(0.0, 1.0)))
    x_grain = (x + 0.035 * torch.randn_like(x)).clamp(0.0, 1.0)
    return torch.where(sm_mask, x_blur, x_grain).clamp(0.0, 1.0)


@torch.no_grad()
def build_or_load_s2_pert_cache(n_variants: int = 2, device: str = "cuda") -> dict:
    """Precompute n_variants random T_style(x) condition tensors (q16, q_b, q32_hi, y16) for train_4400."""
    from exp.v32_nodino.cnx_layout import extract_cnx_q32
    from src.utils.structure import local_contrast_norm, rgb_to_gray

    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"v33_s2_pert_v{n_variants}.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)

    sanitize_cuda_env()
    seed_all(SEED_V33 + 333)
    t0 = time.monotonic()

    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    w32_st = assets["stiefel"]["W_stiefel_cnx"].to(device).float()
    wb32_st = q64_cache["W_b32"].to(device).float()
    whi32_st = q64_cache["W_hi32"].to(device).float()

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    z_tr = v18["train"]["z"]

    dec = load_s10_decoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    ext = ConvNeXtMultiScaleExtractor(device=device).eval()

    variants = []
    for v_i in range(n_variants):
        q16_l, qb_l, q32_l, y16_l = [], [], [], []
        for st in range(0, len(z_tr), 32):
            sl = slice(st, min(st + 32, len(z_tr)))
            rgb_sub = decode_s10(dec, z_tr[sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
            rgb_pert = apply_style_perturbation_rgb(rgb_sub)
            g16 = ext.extract_gray_layout(rgb_pert)
            g32 = ext.extract_gray_layout_32(rgb_pert)
            q64 = stiefel_q64_plus(g16, w32_st, wb32_st)
            q16_l.append(q64[:, :32].cpu().half())
            qb_l.append(q64[:, 32:].cpu().half())
            q32_l.append(extract_cnx_q32_hi(g32, whi32_st).cpu().half())
            y16_p = local_contrast_norm(F.avg_pool2d(rgb_to_gray(rgb_pert), 32, 32), kernel=3, eps=0.04)
            y16_l.append(y16_p.cpu().half())
        variants.append({
            "q16": torch.cat(q16_l, 0),
            "q_b": torch.cat(qb_l, 0),
            "q32_hi": torch.cat(q32_l, 0),
            "y16": torch.cat(y16_l, 0),
        })
        print(f"[V33 Assets] Built S2 T_style variant {v_i+1}/{n_variants} ({time.monotonic() - t0:.1f}s)", flush=True)

    del dec, ext
    gc.collect()
    torch.cuda.empty_cache()

    payload = {"updated_utc": utc(), "n_variants": n_variants, "variants": variants}
    torch.save(payload, p)
    return payload
