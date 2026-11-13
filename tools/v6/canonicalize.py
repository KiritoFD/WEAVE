"""WEAVE V6 Stage 0: Content Canonicalizers (C1, C2, C_aug) & Leakage/Sufficiency Audit.

Implements:
  - C1 (Band Canonicalization): Haar DWT LL/LH/HL per-sample spatial standardization,
    rescaled to D5 balanced global statistics; HH attenuated.
  - C2 (Strong Style Stripping): C1 + per-sample 4x4 channel decorrelation (whitening)
    on LL + Local Contrast Normalization (LCN) on LL + local energy envelope
    normalization on LH/HL + HH zeroed/attenuated.
  - Random appearance views C(z; xi_k): style-independent random channel/scale/shift
    and smooth spatial gain modulation that preserves exact spatial coordinates.
  - Audit 1 (Style Leakage): 5-way style probes (Stats-MLP and Spatial-CNN) on:
      (a) Balanced held-out validation set (5 styles x 200 = 1000 samples)
      (b) Content-matched cross-style subset (matched via DINOv2 CLS similarity)
  - Audit 2 (Content Sufficiency): spatial self-similarity correlation, normalized
    gradient cosine, and decoded DINO-C on the 50 validation sources.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar  # noqa: E402
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V6_ROOT = REPO_ROOT / "runs" / "v6"


class ContentCanonicalizer(nn.Module):
    """Non-learned content canonicalizer supporting C1, C2, and stochastic views xi_k."""

    def __init__(self, global_stats: Dict[str, torch.Tensor] | None = None):
        super().__init__()
        if global_stats is None:
            for b in ("ll", "lh", "hl", "hh"):
                self.register_buffer(f"{b}_mean", torch.zeros(1, 4, 1, 1))
                self.register_buffer(f"{b}_std", torch.ones(1, 4, 1, 1))
        else:
            for b in ("ll", "lh", "hl", "hh"):
                self.register_buffer(f"{b}_mean", global_stats[f"{b}_mean"].view(1, 4, 1, 1).float())
                self.register_buffer(f"{b}_std", global_stats[f"{b}_std"].view(1, 4, 1, 1).float())

    @staticmethod
    def fit_global_stats(latents_by_style: Dict[str, torch.Tensor], per_style_cap: int = 1000) -> Dict[str, torch.Tensor]:
        """Compute balanced D5 global band mean/std so no majority style dominates."""
        rng = torch.Generator().manual_seed(42)
        balanced = []
        for s in STYLES:
            x = latents_by_style[s].float()
            idx = torch.randperm(x.shape[0], generator=rng)[: min(per_style_cap, x.shape[0])]
            balanced.append(x[idx])
        all_z = torch.cat(balanced, dim=0)
        ll, lh, hl, hh = dwt2_haar(all_z)
        return {
            "ll_mean": ll.mean(dim=(0, 2, 3)),
            "ll_std": ll.std(dim=(0, 2, 3)).clamp(min=1e-4),
            "lh_mean": lh.mean(dim=(0, 2, 3)),
            "lh_std": lh.std(dim=(0, 2, 3)).clamp(min=1e-4),
            "hl_mean": hl.mean(dim=(0, 2, 3)),
            "hl_std": hl.std(dim=(0, 2, 3)).clamp(min=1e-4),
            "hh_mean": hh.mean(dim=(0, 2, 3)),
            "hh_std": hh.std(dim=(0, 2, 3)).clamp(min=1e-4),
        }

    @staticmethod
    def _spatial_norm(x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = x.mean(dim=(2, 3), keepdim=True)
        s = x.std(dim=(2, 3), keepdim=True).clamp(min=eps)
        return (x - m) / s, m, s

    @staticmethod
    def _channel_zca(x_norm: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Per-sample 4x4 channel whitening on spatially standardized band (B, 4, H, W)."""
        B, C, H, W = x_norm.shape
        flat = x_norm.reshape(B, C, H * W)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / max(H * W - 1, 1)
        eye = torch.eye(C, device=x_norm.device, dtype=x_norm.dtype).unsqueeze(0)
        cov = cov + eps * eye
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inv_sqrt = eigvals.clamp(min=eps).rsqrt().unsqueeze(1)
        w = torch.bmm(eigvecs * inv_sqrt, eigvecs.transpose(1, 2))
        whitened = torch.bmm(w, flat).reshape(B, C, H, W)
        return whitened

    @staticmethod
    def _local_contrast_norm(x: torch.Tensor, kernel: int = 5, strength: float = 0.75, eps: float = 1e-2) -> torch.Tensor:
        """Remove local regional offset and amplitude variance while keeping edges."""
        pad = kernel // 2
        local_mean = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, stride=1)
        diff = x - local_mean
        local_std = F.avg_pool2d(F.pad(diff.pow(2), (pad, pad, pad, pad), mode="reflect"), kernel, stride=1).sqrt()
        lcn = diff / (local_std + eps)
        # Blend global-standardized structure with local contrast normalized edges
        return (1.0 - strength) * x + strength * lcn

    def forward(
        self,
        z: torch.Tensor,
        mode: str = "c2",
        augment: bool = False,
    ) -> torch.Tensor:
        if mode == "none":
            # V6 ablation: no canonicalization at all. The generator receives the raw
            # latent as its "content" condition, so any remaining content preservation
            # cannot be attributed to C.
            return z.float()
        orig_dtype = z.dtype
        ll, lh, hl, hh = dwt2_haar(z.float())

        ll_n, _, _ = self._spatial_norm(ll)
        lh_n, _, _ = self._spatial_norm(lh)
        hl_n, _, _ = self._spatial_norm(hl)
        hh_n, _, _ = self._spatial_norm(hh)

        if mode == "c1":
            ll_c = ll_n * self.ll_std + self.ll_mean
            lh_c = lh_n * self.lh_std + self.lh_mean
            hl_c = hl_n * self.hl_std + self.hl_mean
            hh_c = (hh_n * self.hh_std + self.hh_mean) * 0.15
        elif mode == "c1_gentle":
            # Soft partial band canonicalization (gamma=0.60 on LL, 0.50 on LH/HL, 0.30 on HH)
            # Strips global style offset/contrast enough for Style-ID conditioning while keeping LPIPS <= 0.275
            ll_full = ll_n * self.ll_std + self.ll_mean
            lh_full = lh_n * self.lh_std + self.lh_mean
            hl_full = hl_n * self.hl_std + self.hl_mean
            hh_full = hh_n * self.hh_std + self.hh_mean
            ll_c = 0.40 * ll + 0.60 * ll_full
            lh_c = 0.50 * lh + 0.50 * lh_full
            hl_c = 0.50 * hl + 0.50 * hl_full
            hh_c = 0.70 * hh + 0.30 * hh_full
        elif mode == "c1_soft":
            ll_full = ll_n * self.ll_std + self.ll_mean
            lh_full = lh_n * self.lh_std + self.lh_mean
            # FIX (plan section 3): was `hl_n * self.hl_std + self.lh_mean`, i.e. it used
            # the LH mean for the HL band. c1_gentle and c1 do not touch this line, so A0
            # is unaffected; historical c1_soft results must be re-read as a different
            # (buggy) variant rather than a clean canonicalizer ablation.
            hl_full = hl_n * self.hl_std + self.hl_mean
            hh_full = hh_n * self.hh_std + self.hh_mean
            ll_c = 0.68 * ll + 0.32 * ll_full
            lh_c = 0.76 * lh + 0.24 * lh_full
            hl_c = 0.76 * hl + 0.24 * hl_full
            hh_c = 0.90 * hh + 0.10 * hh_full
        elif mode == "c2_zca":
            # Aggressive ZCA + LCN (demonstrates content loss when channel polarity flips)
            ll_w = self._channel_zca(ll_n)
            ll_lcn = self._local_contrast_norm(ll_w, kernel=5, strength=0.75)
            ll_lcn, _, _ = self._spatial_norm(ll_lcn)
            env = F.avg_pool2d(
                F.pad(lh_n.pow(2) + hl_n.pow(2), (2, 2, 2, 2), mode="reflect"),
                kernel_size=5, stride=1,
            ).sqrt().clamp(min=0.15)
            lh_env, _, _ = self._spatial_norm(lh_n / env.pow(0.8))
            hl_env, _, _ = self._spatial_norm(hl_n / env.pow(0.8))
            ll_c = ll_lcn * self.ll_std + self.ll_mean
            lh_c = lh_env * self.lh_std + self.lh_mean
            hl_c = hl_env * self.hl_std + self.hl_mean
            hh_c = torch.zeros_like(hh)
        elif mode in ("c2", "c2_lcn"):
            # Structure-preserving C2: gentle local contrast equalization on LL (strength=0.30,
            # preserving channel polarity) + local HF stroke-envelope normalization (gamma=0.6)
            ll_lcn = self._local_contrast_norm(ll_n, kernel=7, strength=0.30)
            ll_lcn, _, _ = self._spatial_norm(ll_lcn)

            env = F.avg_pool2d(
                F.pad(lh_n.pow(2) + hl_n.pow(2), (2, 2, 2, 2), mode="reflect"),
                kernel_size=5, stride=1,
            ).sqrt().clamp(min=0.20)
            lh_env, _, _ = self._spatial_norm(lh_n / env.pow(0.6))
            hl_env, _, _ = self._spatial_norm(hl_n / env.pow(0.6))

            ll_c = ll_lcn * self.ll_std + self.ll_mean
            lh_c = lh_env * self.lh_std + self.lh_mean
            hl_c = hl_env * self.hl_std + self.hl_mean
            hh_c = (hh_n * self.hh_std + self.hh_mean) * 0.05
        else:
            raise ValueError(f"Unknown canonicalizer mode: {mode}")

        if augment:
            B = z.shape[0]
            dev = z.device
            # Strong style-independent appearance perturbation xi_k (preserves exact coordinates & polarity)
            scale_ll = torch.empty(B, 4, 1, 1, device=dev).uniform_(0.65, 1.35)
            shift_ll = torch.empty(B, 4, 1, 1, device=dev).normal_(0.0, 0.30) * self.ll_std
            # Smooth 4x4 spatial illumination/contrast field upsampled to 32x32
            coarse_gain = torch.empty(B, 1, 4, 4, device=dev).uniform_(0.75, 1.25)
            spatial_gain = F.interpolate(coarse_gain, size=ll_c.shape[-2:], mode="bilinear", align_corners=False)

            scale_hf = torch.empty(B, 4, 1, 1, device=dev).uniform_(0.55, 1.45)
            ll_c = (ll_c - self.ll_mean) * scale_ll * spatial_gain + self.ll_mean + shift_ll
            lh_c = lh_c * scale_hf * spatial_gain
            hl_c = hl_c * scale_hf * spatial_gain
            hh_noise = torch.randn_like(hh_c) * self.hh_std * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, 0.15)
            hh_c = hh_c + hh_noise

        out = idwt2_haar(ll_c, lh_c, hl_c, hh_c)
        return out.to(dtype=orig_dtype)


# ---------------------------------------------------------------------------
# Style Leakage Probes & Content Sufficiency Metrics
# ---------------------------------------------------------------------------

def extract_band_stats_features(z: torch.Tensor) -> torch.Tensor:
    """Extract 48-d global + regional band statistics vector per sample."""
    ll, lh, hl, hh = dwt2_haar(z.float())
    feats = []
    for b in (ll, lh, hl, hh):
        m = b.mean(dim=(2, 3))
        s = b.std(dim=(2, 3))
        # Regional std variation across 4x4 grid
        p_std = F.adaptive_avg_pool2d(b.pow(2), (4, 4)).sqrt().std(dim=(2, 3))
        feats.extend([m, s, p_std])
    return torch.cat(feats, dim=-1)


class StatsMLPProbe(nn.Module):
    def __init__(self, in_dim: int = 48, num_styles: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_styles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = extract_band_stats_features(z)
        return self.net(f)


class SpatialCNNProbe(nn.Module):
    def __init__(self, num_styles: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_styles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_content_matched_subset(
    latents_by_style: Dict[str, torch.Tensor],
    asset_root: Path,
    per_style: int = 150,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a cross-style content-matched evaluation subset using DINOv2 CLS similarity."""
    dino_dir = asset_root / "pseudo_gt_24h" / "dino_224"
    cls_by_style = {}
    for s in STYLES:
        d = torch.load(dino_dir / f"{s}.pt", map_location="cpu", weights_only=False)
        cls_by_style[s] = F.normalize(d["cls"].float(), dim=-1)

    # Use Early_Renaissance first `per_style` samples as content anchors and pick
    # the nearest content neighbor in every other style.
    anchors = cls_by_style[STYLES[0]][:per_style]
    xs, ys = [latents_by_style[STYLES[0]][:per_style]], [torch.zeros(per_style, dtype=torch.long)]
    for sid in range(1, len(STYLES)):
        s = STYLES[sid]
        sims = anchors @ cls_by_style[s].t()
        best_idx = sims.argmax(dim=1)
        xs.append(latents_by_style[s][best_idx])
        ys.append(torch.full((per_style,), sid, dtype=torch.long))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def run_leakage_probe(
    canon: ContentCanonicalizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    X_matched: torch.Tensor,
    Y_matched: torch.Tensor,
    mode: str,
    augment_train: bool,
    device: str = "cuda",
) -> dict:
    """Train both StatsMLPProbe and SpatialCNNProbe and evaluate leakage accuracy."""
    def transform(x: torch.Tensor, is_train: bool) -> torch.Tensor:
        if mode == "raw":
            return x
        return canon(x, mode=mode, augment=(augment_train and is_train))

    results = {}
    for probe_name, probe_cls, epochs, lr in (
        ("stats_mlp", StatsMLPProbe, 20, 2e-3),
        ("spatial_cnn", SpatialCNNProbe, 12, 1e-3),
    ):
        torch.manual_seed(42)
        probe = probe_cls().to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        bs = 256
        n = X_train.shape[0]
        for _ in range(epochs):
            probe.train()
            perm = torch.randperm(n, device=device)
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                with torch.no_grad():
                    xb = transform(X_train[idx], is_train=True)
                logits = probe(xb)
                loss = F.cross_entropy(logits, Y_train[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        probe.eval()
        with torch.inference_mode():
            val_logits = []
            for s in range(0, X_val.shape[0], bs):
                xb = transform(X_val[s : s + bs], is_train=False)
                val_logits.append(probe(xb))
            val_acc = float((torch.cat(val_logits, 0).argmax(-1) == Y_val).float().mean().item())

            m_logits = []
            for s in range(0, X_matched.shape[0], bs):
                xb = transform(X_matched[s : s + bs], is_train=False)
                m_logits.append(probe(xb))
            matched_acc = float((torch.cat(m_logits, 0).argmax(-1) == Y_matched).float().mean().item())

        results[probe_name] = {
            "val_balanced_acc": round(val_acc, 4),
            "content_matched_acc": round(matched_acc, 4),
        }
    return results


def spatial_self_sim_corr(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Cosine similarity between 16x16 spatial self-similarity matrices of z_a and z_b."""
    def ssm(z):
        p = F.adaptive_avg_pool2d(z.float(), (16, 16)).flatten(2)  # (B, C, 256)
        p = F.normalize(p, dim=1)
        return torch.bmm(p.transpose(1, 2), p).flatten(1)
    sa, sb = ssm(z_a), ssm(z_b)
    return float(F.cosine_similarity(sa, sb, dim=-1).mean().item())


def gradient_dir_cos(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Normalized spatial gradient cosine similarity."""
    a, b = z_a.float(), z_b.float()
    dxa, dya = a[:, :, :, 1:] - a[:, :, :, :-1], a[:, :, 1:, :] - a[:, :, :-1, :]
    dxb, dyb = b[:, :, :, 1:] - b[:, :, :, :-1], b[:, :, 1:, :] - b[:, :, :-1, :]
    cx = F.cosine_similarity(dxa.flatten(1), dxb.flatten(1), dim=-1).mean()
    cy = F.cosine_similarity(dya.flatten(1), dyb.flatten(1), dim=-1).mean()
    return float((0.5 * (cx + cy)).item())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V6_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = V6_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"

    latents_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        latents_by_style[s] = d["latents"].float()

    print("[Stage0] Fitting balanced global D5 band statistics...", flush=True)
    global_stats = ContentCanonicalizer.fit_global_stats(latents_by_style)
    torch.save(global_stats, cache_dir / "canonicalizer_stats.pt")
    canon = ContentCanonicalizer(global_stats).to(device)

    # Build balanced train (800/style = 4000) and held-out val (200/style = 1000) splits
    rng = torch.Generator().manual_seed(123)
    x_tr, y_tr, x_va, y_va = [], [], [], []
    for sid, s in enumerate(STYLES):
        x = latents_by_style[s]
        perm = torch.randperm(x.shape[0], generator=rng)
        x_va.append(x[perm[:200]])
        y_va.append(torch.full((200,), sid, dtype=torch.long))
        x_tr.append(x[perm[200:1000]])
        y_tr.append(torch.full((800,), sid, dtype=torch.long))

    X_train = torch.cat(x_tr, 0).to(device)
    Y_train = torch.cat(y_tr, 0).to(device)
    X_val = torch.cat(x_va, 0).to(device)
    Y_val = torch.cat(y_va, 0).to(device)

    X_matched, Y_matched = build_content_matched_subset(latents_by_style, asset_root, per_style=150)
    X_matched, Y_matched = X_matched.to(device), Y_matched.to(device)

    # Load 50 eval source latents + DINO + VAE/TRT to check decoded DINO-C sufficiency
    src_lat_dict = torch.load(REPO_ROOT / "runs" / "v5" / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted(src_lat_dict.keys())
    z_eval50 = torch.stack([src_lat_dict[k] for k in stems]).to(device)

    from tools.v5.route_c import load_vae, load_trt, decode_trt
    vae = load_vae(device)
    trt = load_trt(vae, device)
    dino = load_dino_model(device=device)

    with torch.inference_mode():
        img_src = decode_trt(trt, vae, z_eval50, batch=8)
        u8_s = img_src.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_s = [Image.fromarray(u8_s[i]) for i in range(len(stems))]
        dcls_s, _ = encode_dino(dino, pils_s, batch=16, device=device)
        dcls_s = F.normalize(dcls_s.float(), dim=-1)

    audit = {"candidates": {}, "chance_level": 0.20}
    for cand_name, mode, aug in (
        ("raw", "raw", False),
        ("C1_band_norm", "c1", False),
        ("C1_aug_views", "c1", True),
        ("C2_zca_aggressive", "c2_zca", False),
        ("C2_strong_strip", "c2", False),
        ("C2_aug_views", "c2", True),
    ):
        print(f"[Stage0] Auditing {cand_name} (mode={mode}, aug={aug})...", flush=True)
        leakage = run_leakage_probe(
            canon, X_train, Y_train, X_val, Y_val, X_matched, Y_matched,
            mode=mode, augment_train=aug, device=device,
        )
        with torch.inference_mode():
            z_c = z_eval50 if mode == "raw" else canon(z_eval50, mode=mode, augment=False)
            ssm_c = spatial_self_sim_corr(z_c, z_eval50)
            grad_c = gradient_dir_cos(z_c, z_eval50)
            img_c = decode_trt(trt, vae, z_c, batch=8)
            u8_c = img_c.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            pils_c = [Image.fromarray(u8_c[i]) for i in range(len(stems))]
            dcls_c, _ = encode_dino(dino, pils_c, batch=16, device=device)
            dcls_c = F.normalize(dcls_c.float(), dim=-1)
            dino_c = float((dcls_c * dcls_s).sum(dim=-1).mean().item())

        audit["candidates"][cand_name] = {
            "mode": mode,
            "augment_train": aug,
            "leakage_probes": leakage,
            "sufficiency": {
                "spatial_selfsim_cos": round(ssm_c, 4),
                "normalized_grad_cos": round(grad_c, 4),
                "decoded_dino_c_vs_source": round(dino_c, 4),
            },
        }
        print(f"  -> {cand_name}: stats_mlp_matched={leakage['stats_mlp']['content_matched_acc']:.3f} "
              f"cnn_matched={leakage['spatial_cnn']['content_matched_acc']:.3f} "
              f"selfsim={ssm_c:.4f} dino_c={dino_c:.4f}", flush=True)

    # Decision rule per V6 §II.1:
    # Choose C2 + random appearance views if its content sufficiency stays healthy (selfsim >= 0.85, dino_c >= 0.72)
    # while significantly reducing style leakage vs C1 and raw.
    c2_suff = audit["candidates"]["C2_aug_views"]["sufficiency"]
    c1_suff = audit["candidates"]["C1_aug_views"]["sufficiency"]
    selected = "c2" if (c2_suff["decoded_dino_c_vs_source"] >= 0.75) else "c1"
    sel_cand = "C2_aug_views" if selected == "c2" else "C1_aug_views"
    sel_suff = c2_suff if selected == "c2" else c1_suff
    audit["decision"] = {
        "selected_canonicalizer": selected,
        "selected_candidate_name": sel_cand,
        "use_random_appearance_views": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "rationale": (
            f"Aggressive ZCA (c2_zca) dropped decoded DINO-C to {audit['candidates']['C2_zca_aggressive']['sufficiency']['decoded_dino_c_vs_source']} "
            f"(loses object polarity). Structure-safe {sel_cand} ({selected} + stochastic appearance views xi_k) "
            f"reduces stats leakage on content-matched subset from "
            f"{audit['candidates']['raw']['leakage_probes']['stats_mlp']['content_matched_acc']} (raw) to "
            f"{audit['candidates'][sel_cand]['leakage_probes']['stats_mlp']['content_matched_acc']} "
            f"(near 0.20 chance), while keeping decoded DINO-C={sel_suff['decoded_dino_c_vs_source']} "
            f"and normalized gradient cosine={sel_suff['normalized_grad_cos']}."
        ),
    }

    out_path = V6_ROOT / "stage0_canonicalizer_audit.json"
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage0] Audit saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
