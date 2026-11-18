"""Stage 1: Vivid Scene-Matched Sub-School Bank & Multi-Scale ConvNeXt+VGG Patch-SWD Refiner.

Why V19 plateaued at DINO-S = 0.462 despite 59% CSD Top-1:
  1. Centroid Collapse on Hypersphere: DINO-S = max_{r in 30} cos(x_hat, x_r). Pushing toward the
     mean of 880 diverse paintings lands in the interior average rather than on a vivid artistic
     sub-school (e.g., Hokusai Prussian-blue landscape vs Sharaku kabuki portrait).
  2. Global-Moment Averaging: Spatial mean/Gram over 256x256 never enforces sharp local 7x7/16x16
     brushstroke distributions.

Fix (100% strictly train-split only, 0 referee overlap):
  - Cluster each style's 880 clean training paintings into K=8 vivid sub-schools in Pi_parallel space,
    retaining their coarse scene layout signature so portraits match portrait exemplars and landscapes
    match landscape exemplars!
  - Extract local patch tokens from ImageNet ConvNeXt-Tiny (7x7 depthwise patches, cached offline) +
    VGG-16 + RGB 3x3 patches, optimizing Sliced Wasserstein Distance (Patch-SWD) + balanced margin
    under our anisotropic Luminance-LCN structural lock.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from . import OUT, ROOT, STYLES, RawWriterV20, pick_wrong_cross_tid, read_json, save_json, seed_all
from exp.v19.aniso_teacher import AnisotropicS10Teacher, aniso_rms_direction, latent_lcn_cost, project_anisotropic_trust, _smooth_grad_3x3
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v18 import tasks_and_sources


def load_convnext_tiny_offline(device: str = "cuda") -> nn.Module:
    import paths
    fname = "convnext_tiny-983f1562.pth"
    candidates = [
        Path(torch.hub.get_dir()) / "checkpoints" / fname,
        paths.eval_cache_root("torch") / "hub/checkpoints" / fname,
        paths.weights_root(fname),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100_000_000:
            net = tvm.convnext_tiny(weights=None)
            net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            feat = net.features.to(device).eval()
            for param in feat.parameters():
                param.requires_grad_(False)
            return feat
    feat = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.to(device).eval()
    for param in feat.parameters():
        param.requires_grad_(False)
    return feat


def sliced_wasserstein_1d(a: torch.Tensor, b: torch.Tensor, projs: torch.Tensor) -> torch.Tensor:
    """Fast batched Sliced 1-Wasserstein distance.
    a: (B, N, C), b: (B, M, C), projs: (C, P) unit-norm -> (B,) per-sample SWD.
    """
    pa, _ = (a @ projs).sort(dim=1)  # (B, N, P)
    pb, _ = (b @ projs).sort(dim=1)  # (B, M, P)
    if pa.shape[1] != pb.shape[1]:
        # Interpolate along token count axis to match N
        pb_t = F.interpolate(pb.permute(0, 2, 1), size=pa.shape[1], mode="linear", align_corners=True).permute(0, 2, 1)
    else:
        pb_t = pb
    return (pa - pb_t).abs().mean(dim=(1, 2))


class MultiScalePatchSWDExtractor(nn.Module):
    """Frozen ImageNet ConvNeXt-Tiny + RGB 3x3 patch extractor for sharp brushstroke SWD."""

    def __init__(self, n_slices: int = 96, seed: int = 20261120):
        super().__init__()
        cnx = load_convnext_tiny_offline("cpu")
        # ConvNeXt-Tiny stages: [0..1]=96ch (64x64 @256px), [2..3]=192ch (32x32), [4..5]=384ch (16x16)
        self.s1 = cnx[:2]
        self.s2 = cnx[2:4]
        self.s3 = cnx[4:6]
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        g = torch.Generator().manual_seed(seed)
        for name, c_dim in [("p_rgb", 27), ("p_s1", 96), ("p_s2", 192), ("p_s3", 384)]:
            w = torch.randn(c_dim, n_slices, generator=g)
            w = w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.register_buffer(name, w)

    def extract_tokens(self, rgb01: torch.Tensor) -> dict[str, torch.Tensor]:
        x256 = F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False)
        # 3x3 RGB patches on stride 4 -> (B, 27, 64, 64) -> (B, 4096, 27) -> subsample 1024
        rgb_unf = F.unfold(x256, kernel_size=3, padding=1, stride=4).transpose(1, 2)  # (B, 4096, 27)
        rgb_tok = rgb_unf[:, ::4, :]  # (B, 1024, 27)

        xn = (x256 - self.im_mean) / self.im_std
        f1 = self.s1(xn)  # (B, 96, 64, 64)
        f2 = self.s2(f1)  # (B, 192, 32, 32)
        f3 = self.s3(f2)  # (B, 384, 16, 16)
        return {
            "rgb": rgb_tok,
            "s1": f1.flatten(2).transpose(1, 2)[:, ::4, :],  # (B, 1024, 96)
            "s2": f2.flatten(2).transpose(1, 2),             # (B, 1024, 192)
            "s3": f3.flatten(2).transpose(1, 2),             # (B, 256, 384)
        }

    def swd_loss(self, rgb_pred: torch.Tensor, ref_tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample multi-scale Patch-SWD (B,)."""
        cur = self.extract_tokens(rgb_pred)
        l_rgb = sliced_wasserstein_1d(cur["rgb"], ref_tokens["rgb"], self.p_rgb)
        l_s1 = sliced_wasserstein_1d(cur["s1"], ref_tokens["s1"], self.p_s1)
        l_s2 = sliced_wasserstein_1d(cur["s2"], ref_tokens["s2"], self.p_s2)
        l_s3 = sliced_wasserstein_1d(cur["s3"], ref_tokens["s3"], self.p_s3)
        return 1.2 * l_rgb + 0.35 * l_s1 + 0.45 * l_s2 + 0.55 * l_s3


def _kmeans_unit(X: torch.Tensor, k: int = 8, iters: int = 35, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=X.device).manual_seed(seed)
    perm = torch.randperm(len(X), generator=g, device=X.device)[:k]
    centers = F.normalize(X[perm].clone(), dim=-1)
    labels = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for _ in range(iters):
        sim = X @ centers.T
        labels = sim.argmax(dim=1)
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = F.normalize(X[m].mean(dim=0), dim=-1)
    return centers, labels


@torch.no_grad()
def build_or_load_vivid_submode_bank(bundle: dict, k_sub: int = 8, device: str = "cuda") -> dict:
    """Cluster 4,400 clean train paintings into K=8 vivid sub-schools per style + pre-extract exemplar tokens."""
    OUT.mkdir(parents=True, exist_ok=True)
    bank_p = OUT / "cache/vivid_submode_bank.pt"
    bank_p.parent.mkdir(parents=True, exist_ok=True)
    if bank_p.exists():
        print("[V20 Bank] Reusing vivid_submode_bank.pt", flush=True)
        return torch.load(bank_p, map_location="cpu", weights_only=False)

    t0 = time.time()
    from src.utils import load_s10_decoder, decode_s10

    v18c = bundle["v18_cache"]
    fnd = bundle["fnd"]
    tr = v18c["train"]
    tr_sid = tr["sid"].long()
    tr_fib = fnd["tr_fib"].float().to(device)

    fiber_enc = StyleFiberEncoder(in_dim=tr_fib.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_all = fiber_enc.encode(tr_fib)            # (4400, 128) on unit sphere
    logits_all = fiber_enc.logits(tr_fib)       # (4400, 5)

    # Coarse 4x4 scene layout descriptor from de-styled whitened DINO P_16 (0 style leak, pure scene type: portrait vs landscape!)
    d16_tr = tr["dino_16"].float()              # (4400, 384, 16, 16)
    scene_desc_tr = F.normalize(F.avg_pool2d(d16_tr, 4).flatten(1), dim=-1)  # (4400, 6144)

    sub_u = torch.zeros(5, k_sub, 128)
    sub_scene = torch.zeros(5, k_sub, scene_desc_tr.shape[1])
    sub_exemplar_z = torch.zeros(5, k_sub, 4, 64, 64)
    sample_sub_u = torch.zeros(len(u_all), 128)

    for s in range(5):
        ms = (tr_sid == s).nonzero(as_tuple=True)[0]
        Us = u_all[ms]
        # Margin purity score of each training painting
        ls = logits_all[ms]
        oh = F.one_hot(torch.full((len(ms),), s, device=device), 5).bool()
        purity = ls[:, s] - ls.masked_fill(oh, -1e9).max(dim=1).values

        centers, labels = _kmeans_unit(Us, k=k_sub, iters=40, seed=500 + s)
        for c in range(k_sub):
            cm = (labels == c).nonzero(as_tuple=True)[0]
            if len(cm) == 0:
                cm = torch.arange(len(ms), device=device)
            # Pick top-purity real training painting in this sub-school as vivid anchor!
            top_k = min(5, len(cm))
            best_in_c = cm[purity[cm].topk(top_k).indices]
            vivid_u = F.normalize(Us[best_in_c].mean(dim=0), dim=-1)
            pick_one = int(ms[int(best_in_c[0].item())].item())
            cm_cpu = cm.cpu()

            sub_u[s, c] = vivid_u.cpu()
            sub_scene[s, c] = F.normalize(scene_desc_tr[ms[cm_cpu]].mean(dim=0), dim=-1)
            sub_exemplar_z[s, c] = tr["z"][pick_one].float()
            sample_sub_u[ms[cm_cpu]] = vivid_u.cpu()

    # Pre-decode the 5*8 = 40 vivid training exemplars & cache their multi-scale ConvNeXt+RGB tokens!
    dec = load_s10_decoder(device)
    swd_ext = MultiScalePatchSWDExtractor().to(device).eval()
    ex_flat_z = sub_exemplar_z.reshape(5 * k_sub, 4, 64, 64).to(device)
    ex_rgb = decode_s10(dec, ex_flat_z, batch=10).to(device).clamp(0.0, 1.0)
    ex_toks = {k: v.cpu().half() for k, v in swd_ext.extract_tokens(ex_rgb).items()}
    del dec, swd_ext
    torch.cuda.empty_cache()

    bank = {
        "k_sub": k_sub,
        "sub_u": sub_u,                  # (5, K, 128)
        "sub_scene": sub_scene.half(),   # (5, K, 6144)
        "sub_exemplar_z": sub_exemplar_z.half(),
        "sample_sub_u": sample_sub_u.half(),  # (4400, 128) vivid target per train sample!
        "ex_tokens": ex_toks,            # each (40, N_tok, C)
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(bank, bank_p)
    print(f"[V20 Bank] Built vivid submode bank (5x{k_sub}) in {bank['wall_sec']}s", flush=True)
    return bank


def match_scene_submode(d16_src: torch.Tensor, tid: torch.Tensor, bank: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Match source coarse de-styled scene layout to compatible vivid sub-school c* in target style tid!
    Returns (flat_idx in [0..5*K-1], vivid_u (B, 128)).
    """
    dev = d16_src.device
    k_sub = int(bank["k_sub"])
    sc_src = F.normalize(F.avg_pool2d(d16_src.float(), 4).flatten(1), dim=-1)  # (B, 6144)
    cand_sc = bank["sub_scene"].to(dev).float()[tid]  # (B, K, 6144)
    sim = torch.einsum("bd,bkd->bk", sc_src, cand_sc)
    best_c = sim.argmax(dim=1)  # (B,)
    flat_idx = tid * k_sub + best_c
    u_viv = bank["sub_u"].to(dev).float()[tid, best_c]
    return flat_idx, u_viv


def load_raw_latents_750(raw_arm_dir: Path | str) -> torch.Tensor:
    raw_p = Path(raw_arm_dir)
    meta = read_json(raw_p / "meta.json")
    zs = []
    for b in meta["batches"]:
        d = torch.load(raw_p / b["file"], map_location="cpu", weights_only=False)
        zs.append(d["z_final"].float())
    return torch.cat(zs, dim=0)


@torch.no_grad()
def build_or_load_core_knn_bank(
    bundle: dict, k_sub: int = 8, core_per_style: int = 180, device: str = "cuda"
) -> dict:
    """Blind Intra-Train kNN-30 Density Core Bank (strictly within train_4400, 0 test/ref overlap!).

    Filters out boundary WikiArt noise (3D installation photos, blank scans) that artificially
    inflate classifier hyperplane margin while lying off the 2D painting manifold.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    bank_p = OUT / "cache/core_knn_bank.pt"
    bank_p.parent.mkdir(parents=True, exist_ok=True)
    if bank_p.exists():
        print("[V20 CoreBank] Reusing core_knn_bank.pt", flush=True)
        return torch.load(bank_p, map_location="cpu", weights_only=False)

    t0 = time.time()
    from src.utils import decode_s10, load_s10_decoder

    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    tr_sid = tr["sid"].long().to(device)
    tr_fib = fnd["tr_fib"].float().to(device)
    tr_dmu = F.normalize(tr_fib[:, 27:27 + 384], dim=-1)
    tr_dsd = tr_fib[:, 27 + 384:27 + 768].mean(dim=1)

    fiber_enc = StyleFiberEncoder(in_dim=tr_fib.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_all = fiber_enc.encode(tr_fib)
    logits_all = fiber_enc.logits(tr_fib)

    d16_tr = tr["dino_16"].float()
    scene_desc_tr = F.normalize(F.avg_pool2d(d16_tr, 4).flatten(1), dim=-1)

    sub_u = torch.zeros(5, k_sub, 128)
    sub_scene = torch.zeros(5, k_sub, scene_desc_tr.shape[1])
    sub_exemplar_z = torch.zeros(5, k_sub, 4, 64, 64)
    sub_dsd = torch.zeros(5, k_sub)
    core_idx_per_style: list[torch.Tensor] = []

    for s in range(5):
        ms = (tr_sid == s).nonzero(as_tuple=True)[0]
        dm = tr_dmu[ms]
        sim = dm @ dm.T
        knn30 = sim.topk(31, dim=1).values[:, 1:].mean(dim=1)
        ls = logits_all[ms]
        oh = F.one_hot(torch.full((len(ms),), s, device=device), 5).bool()
        purity = (ls[:, s] - ls.masked_fill(oh, -1e9).max(dim=1).values).clamp(-2.0, 8.0)
        non_blank = (tr_dsd[ms] > tr_dsd[ms].quantile(0.22)) & (purity > 1.0)
        score = knn30 + 0.01 * purity - (~non_blank).float() * 10.0

        n_core = min(core_per_style, len(ms))
        top_core = score.topk(n_core).indices
        ms_core = ms[top_core]
        core_idx_per_style.append(ms_core.cpu())

        _, labels = _kmeans_unit(dm[top_core], k=k_sub, iters=40, seed=700 + s)
        for c in range(k_sub):
            cm = (labels == c).nonzero(as_tuple=True)[0]
            if len(cm) == 0:
                cm = torch.arange(n_core, device=device)
            best_local = cm[knn30[top_core[cm]].argmax()]
            pick_idx = int(ms_core[best_local].item())
            sub_u[s, c] = u_all[pick_idx].cpu()
            sub_scene[s, c] = F.normalize(scene_desc_tr[ms_core[cm].cpu()].mean(dim=0), dim=-1)
            sub_exemplar_z[s, c] = tr["z"][pick_idx].float()
            sub_dsd[s, c] = tr_dsd[pick_idx].cpu()

    dec = load_s10_decoder(device)
    swd_ext = MultiScalePatchSWDExtractor().to(device).eval()
    ex_flat_z = sub_exemplar_z.reshape(5 * k_sub, 4, 64, 64).to(device)
    ex_rgb = decode_s10(dec, ex_flat_z, batch=10).to(device).clamp(0.0, 1.0)
    ex_toks = {k: v.cpu().half() for k, v in swd_ext.extract_tokens(ex_rgb).items()}
    del dec, swd_ext
    torch.cuda.empty_cache()

    min_dsd = float(sub_dsd.mean(dim=1).min().item())
    max_dsd = float(sub_dsd.mean(dim=1).max().item())
    mid_dsd = 0.5 * (min_dsd + max_dsd)

    bank = {
        "k_sub": k_sub,
        "sub_u": sub_u,
        "sub_scene": sub_scene.half(),
        "sub_exemplar_z": sub_exemplar_z.half(),
        "sub_dsd": sub_dsd,
        "mid_dsd": mid_dsd,
        "core_idx_per_style": core_idx_per_style,
        "u_exact_all": u_all.cpu().half(),
        "ex_tokens": ex_toks,
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(bank, bank_p)
    print(f"[V20 CoreBank] Built kNN-30 core bank (5x{k_sub}, mid_dsd={mid_dsd:.4f}) in {bank['wall_sec']}s", flush=True)
    return bank

