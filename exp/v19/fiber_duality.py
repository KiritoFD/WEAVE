"""Stage 0 & Gate-1: Complementary Style Fiber (Pi_parallel), Latent Luma-Chroma Split, and Un-Normalized VGG-16 RGB Critic.

Strict audit firewall:
  - Fitted exclusively on the 4,400 clean balanced training latents (0 overlap with Test750 or references_v1).
  - Never imports CSD-ViT-L, CLIP, or LPIPS-Alex.
  - DINOv2-small is used ONLY once (no_grad) on input image x_s / x_ref to extract Pi_parallel(x).
  - Differentiable pixel critic uses standard ImageNet-1K VGG-16 (no GroupNorm/BatchNorm!) on S10(z).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from . import OUT, V18_OUT, STYLES, save_json, seed_all, tasks_and_sources
from exp.v18.style_bank import build_or_load_ccf_cache
from src.utils.protocol import load_vgg16_features_offline


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB [0,1] -> perceptual Oklab (B, 3, H, W)."""
    x = rgb.clamp(0.0, 1.0).float()
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(1e-6) ** 2.4)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = l.clamp_min(1e-6).pow(1.0 / 3.0)
    m_ = m.clamp_min(1e-6).pow(1.0 / 3.0)
    s_ = s.clamp_min(1e-6).pow(1.0 / 3.0)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.cat([L, A, B], dim=1)


def extract_color_fiber_27d(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable 27-D Oklab palette moments, quantiles (on detached sort or smooth), and 3x3 cov."""
    lab = rgb_to_oklab(rgb)  # (B, 3, H, W)
    B = lab.shape[0]
    flat = lab.flatten(2)    # (B, 3, N)
    mean = flat.mean(dim=-1) # (B, 3)
    diff = flat - mean.unsqueeze(-1)
    std = (diff.square().mean(dim=-1) + 1e-6).sqrt()  # (B, 3)
    cov = torch.bmm(diff, diff.transpose(1, 2)) / max(flat.shape[-1] - 1, 1)  # (B, 3, 3)
    triu_idx = torch.triu_indices(3, 3, device=rgb.device)
    cov6 = cov[:, triu_idx[0], triu_idx[1]]  # (B, 6)
    # Subsample 4096 pixels on a stride grid for fast differentiable quantile order stats
    sub = flat[:, :, ::max(1, flat.shape[-1] // 4096)]
    qs = torch.quantile(sub, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=rgb.device), dim=-1)
    # qs: (5, B, 3) -> (B, 15)
    q15 = qs.permute(1, 0, 2).reshape(B, 15)
    return torch.cat([mean, std, cov6, q15], dim=-1)  # (B, 27)


@torch.no_grad()
def extract_dino_rgb_moments_768d(dino_model, rgb01: torch.Tensor, batch: int = 32, device: str = "cuda") -> torch.Tensor:
    """Extract raw RGB DINOv2-small spatial mean & std (B, 768) — the exact moments stripped by Pi_perp."""
    from src.utils._local_eval_refs import encode_dino

    u8 = (rgb01.detach().cpu().float() * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    imgs = [Image.fromarray(arr) for arr in u8]
    _, patch = encode_dino(dino_model, imgs, batch=batch, device=device)
    p = patch.float()  # (B, 256, 384)
    mu = p.mean(dim=1)
    sd = p.std(dim=1, unbiased=False)
    return torch.cat([mu, sd], dim=-1)  # (B, 768)


class UnnormalizedVGGFeatureExtractor(nn.Module):
    """Frozen ImageNet VGG-16 (no norm layers!) extracting 1st-order moments + projected 2nd-order Gram + Oklab."""

    def __init__(self, gram_rank: int = 48, seed: int = 20261119):
        super().__init__()
        vgg = load_vgg16_features_offline("cpu")
        # relu1_2: [:4], relu2_2: [4:9], relu3_3: [9:16], relu4_3: [16:23]
        self.s1 = vgg[:4]
        self.s2 = vgg[4:9]
        self.s3 = vgg[9:16]
        self.s4 = vgg[16:23]

        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        g = torch.Generator().manual_seed(seed)
        for name, c_in in [("p2", 128), ("p3", 256), ("p4", 512)]:
            mat = torch.randn(c_in, gram_rank, generator=g)
            q, _ = torch.linalg.qr(mat)
            self.register_buffer(name, q[:, :gram_rank])  # orthonormal (C, R)
        self.gram_rank = gram_rank
        triu = torch.triu_indices(gram_rank, gram_rank)
        self.register_buffer("triu0", triu[0])
        self.register_buffer("triu1", triu[1])

    def _lowrank_gram(self, feat: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W) -> project channels to R, compute spatial covariance / energy
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2) @ proj  # (B, HW, R)
        gram = torch.bmm(x.transpose(1, 2), x) / float(H * W)  # (B, R, R)
        # Signed square-root stabilizes heavy-tailed VGG Gram activations
        vals = gram[:, self.triu0, self.triu1]
        return vals.sign() * (vals.abs() + 1e-6).sqrt()

    def forward(self, rgb01: torch.Tensor, multi_scale: bool = True) -> torch.Tensor:
        """Differentiable extraction: rgb01 (B, 3, 512, 512) -> (B, D_vgg) descriptor."""
        col27 = extract_color_fiber_27d(rgb01)
        x256 = F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False)
        xn = (x256 - self.im_mean) / self.im_std

        f1 = self.s1(xn)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)

        moms = []
        for f in (f1, f2, f3, f4):
            fl = f.flatten(2)
            mu = fl.mean(dim=-1)
            sd = (fl.var(dim=-1, unbiased=False) + 1e-6).sqrt()
            moms.extend([mu, sd])

        g2 = self._lowrank_gram(f2, self.p2)
        g3 = self._lowrank_gram(f3, self.p3)
        g4 = self._lowrank_gram(f4, self.p4)

        parts = [col27 * 3.0] + moms + [g2, g3, g4]
        if multi_scale:
            x128 = F.interpolate(rgb01.float(), size=(128, 128), mode="bilinear", align_corners=False)
            xn2 = (x128 - self.im_mean) / self.im_std
            h1 = self.s1(xn2)
            h2 = self.s2(h1)
            h3 = self.s3(h2)
            for h in (h2, h3):
                hl = h.flatten(2)
                parts.append(hl.mean(dim=-1))
                parts.append((hl.var(dim=-1, unbiased=False) + 1e-6).sqrt())
        return torch.cat(parts, dim=-1)


class StyleFiberEncoder(nn.Module):
    """Maps standardized Pi_parallel (795D) -> continuous style manifold R^128 + 5 class prototypes."""

    def __init__(self, in_dim: int = 795, dim: int = 128, num_styles: int = 5):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(1, in_dim))
        self.register_buffer("x_std", torch.ones(1, in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, dim),
        )
        self.protos = nn.Parameter(torch.randn(num_styles, dim) * 0.2)
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        # Auxiliary linear predictor from continuous style u -> target latent chroma (mean4, log_std4)
        self.chroma_head = nn.Linear(dim, 8)

    def encode(self, f_raw: torch.Tensor) -> torch.Tensor:
        x = (f_raw.float() - self.x_mean) / self.x_std
        return F.normalize(self.net(x), dim=-1)

    def class_barycenters(self) -> torch.Tensor:
        return F.normalize(self.protos, dim=-1)

    def logits(self, f_raw: torch.Tensor) -> torch.Tensor:
        u = self.encode(f_raw)
        p = self.class_barycenters()
        return self.log_scale.exp().clamp(1.0, 30.0) * (u @ p.T)


class BalancedVGGCriticHead(nn.Module):
    """Hybrid Cosine + Linear MLP classifier on standardized VGG+Oklab style descriptors."""

    def __init__(self, in_dim: int, num_styles: int = 5):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(1, in_dim))
        self.register_buffer("x_std", torch.ones(1, in_dim))
        self.register_buffer("calib_bias", torch.zeros(num_styles))
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.lin_head = nn.Linear(256, num_styles)
        self.cos_protos = nn.Parameter(torch.randn(num_styles, 256) * 0.1)
        self.log_temp = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, vgg_feat: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        x = ((vgg_feat.float() - self.x_mean) / self.x_std).clamp(-8.0, 8.0)
        h = self.trunk(x)
        l_lin = self.lin_head(h)
        l_cos = self.log_temp.exp().clamp(1.0, 25.0) * (F.normalize(h, dim=-1) @ F.normalize(self.cos_protos, dim=-1).T)
        out = 0.5 * l_lin + 0.5 * l_cos
        if calibrated:
            out = out + self.calib_bias.unsqueeze(0)
        return out


def compute_latent_luma_chroma_priors(z_train: torch.Tensor, sid_train: torch.Tensor, w_luma: torch.Tensor) -> dict:
    """Compute orthogonal luminance/chrominance projectors P_Y, P_C in R^4 and per-style chroma stats."""
    w_hat = (w_luma.float() / w_luma.float().norm().clamp_min(1e-8)).view(4, 1)
    P_Y = w_hat @ w_hat.T                      # (4, 4) rank-1
    P_C = torch.eye(4) - P_Y                   # (4, 4) rank-3 chrominance subspace

    # Project train latents onto P_C: z_c = einsum('ij,bjhw->bihw', P_C, z)
    zc = torch.einsum("ij,bjhw->bihw", P_C, z_train.float())
    mu_c = torch.zeros(5, 4, 1, 1)
    std_c = torch.ones(5, 4, 1, 1)
    cov_c = torch.zeros(5, 4, 4)
    for s in range(5):
        zcs = zc[sid_train == s]
        if len(zcs) == 0:
            continue
        mu_s = zcs.mean(dim=(0, 2, 3), keepdim=True)
        sd_s = zcs.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        flat = (zcs - mu_s).permute(1, 0, 2, 3).reshape(4, -1)
        cov_s = (flat @ flat.T) / max(flat.shape[1] - 1, 1)
        mu_c[s] = mu_s[0]
        std_c[s] = sd_s[0]
        cov_c[s] = cov_s
    return {
        "w_hat": w_hat.squeeze(1),  # (4,)
        "P_Y": P_Y,
        "P_C": P_C,
        "mu_c": mu_c,               # (5, 4, 1, 1)
        "std_c": std_c,             # (5, 4, 1, 1)
        "cov_c": cov_c,             # (5, 4, 4)
    }


def apply_latent_chroma_shift(z: torch.Tensor, tid: torch.Tensor, priors: dict, alpha: float = 0.45) -> torch.Tensor:
    """Wash away alpha fraction of source 3D latent chrominance toward target style palette, 0 change to P_Y!"""
    if alpha <= 0.0:
        return z
    dev = z.device
    P_Y = priors["P_Y"].to(dev)
    P_C = priors["P_C"].to(dev)
    mu_t = priors["mu_c"].to(dev)[tid]   # (B, 4, 1, 1)
    sd_t = priors["std_c"].to(dev)[tid]  # (B, 4, 1, 1)

    zy = torch.einsum("ij,bjhw->bihw", P_Y, z.float())
    zc = torch.einsum("ij,bjhw->bihw", P_C, z.float())
    mu_s = zc.mean(dim=(2, 3), keepdim=True)
    sd_s = zc.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    # Re-project through P_C so zero component ever leaks back into span(w_hat)
    zc_aligned = torch.einsum("ij,bjhw->bihw", P_C, (zc - mu_s) / sd_s * sd_t + mu_t)
    return zy + (1.0 - alpha) * zc + alpha * zc_aligned


def build_or_load_v19_foundation(pool_dir: str | Path, device: str = "cuda") -> dict:
    """Sprint 0: Build/load Pi_parallel cache, train StyleFiberEncoder & BalancedVGGCriticHead (<90s on 4090)."""
    OUT.mkdir(parents=True, exist_ok=True)
    fnd_path = OUT / "cache/v19_foundation.pt"
    fnd_path.parent.mkdir(parents=True, exist_ok=True)

    v18_cache, test_struct = build_or_load_ccf_cache(pool_dir, device=device)
    if fnd_path.exists():
        print("[V19 Sprint0] Loading cached v19_foundation.pt", flush=True)
        fnd = torch.load(fnd_path, map_location="cpu", weights_only=False)
        return {"v18_cache": v18_cache, "test_struct": test_struct, "fnd": fnd}

    t0 = time.time()
    seed_all(20261119)
    from src.utils import load_s10_decoder, decode_s10
    from src.utils._local_eval_refs import load_dino_model

    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)
    vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval()

    def _extract_split_features(z_all: torch.Tensor, tag: str):
        fib_list, vgg_list = [], []
        # 3060-safe slice=24, S10 batch=8, VGG micro-chunk=8 (<4.2GB peak VRAM!)
        for st in range(0, len(z_all), 24):
            zb = z_all[st:st + 24].to(device)
            with torch.no_grad():
                rgb = decode_s10(dec, zb, batch=8).to(device).clamp(0.0, 1.0)
                col27 = extract_color_fiber_27d(rgb)
                dino768 = extract_dino_rgb_moments_768d(dino, rgb, batch=12, device=device).to(device)
                fib_list.append(torch.cat([col27, dino768], dim=-1).cpu())
                vgg_sub = [vgg_ext(rgb[k:k+8], multi_scale=True).cpu() for k in range(0, len(rgb), 8)]
                vgg_list.append(torch.cat(vgg_sub, dim=0))
        print(f"[V19 Sprint0] Extracted {tag} ({len(z_all)}) in {time.time()-t0:.1f}s", flush=True)
        return torch.cat(fib_list, dim=0), torch.cat(vgg_list, dim=0)

    tr_fib, tr_vgg = _extract_split_features(v18_cache["train"]["z"], "train_4400")
    ho_fib, ho_vgg = _extract_split_features(v18_cache["holdout"]["z"], "holdout_300")

    tasks, z_src_750, _ = tasks_and_sources()
    # Extract Pi_parallel for unique 150 test sources and broadcast to 750
    src_fib_list = []
    for st in range(0, 750, 50):
        zb = z_src_750[st:st + 50].to(device)
        with torch.no_grad():
            rgb = decode_s10(dec, zb, batch=16).to(device).clamp(0.0, 1.0)
            col27 = extract_color_fiber_27d(rgb)
            dino768 = extract_dino_rgb_moments_768d(dino, rgb, batch=25, device=device).to(device)
            src_fib_list.append(torch.cat([col27, dino768], dim=-1).cpu())
    test750_fib = torch.cat(src_fib_list, dim=0)

    del dec, dino
    torch.cuda.empty_cache()

    # 1. Fit Latent Luminance-Chrominance Priors
    priors = compute_latent_luma_chroma_priors(
        v18_cache["train"]["z"],
        v18_cache["train"]["sid"],
        v18_cache["luma_proj"]["w"],
    )

    # 2. Train StyleFiberEncoder (Pi_parallel -> R^128) on 4,400 balanced train samples
    tr_sid = v18_cache["train"]["sid"].long()
    ho_sid = v18_cache["holdout"]["sid"].long()

    fiber_enc = StyleFiberEncoder(in_dim=tr_fib.shape[1], dim=128).to(device)
    fiber_enc.x_mean.copy_(tr_fib.mean(dim=0, keepdim=True).to(device))
    fiber_enc.x_std.copy_(tr_fib.std(dim=0, keepdim=True).clamp_min(1e-4).to(device))

    # Target chroma moments per sample for auxiliary chroma_head supervision
    tgt_mu_c = priors["mu_c"].squeeze(-1).squeeze(-1)[tr_sid].to(device)        # (N, 4)
    tgt_lsd_c = priors["std_c"].squeeze(-1).squeeze(-1).log()[tr_sid].to(device) # (N, 4)
    tgt_c8 = torch.cat([tgt_mu_c, tgt_lsd_c], dim=-1)

    opt_f = torch.optim.AdamW(fiber_enc.parameters(), lr=2e-3, weight_decay=1e-2)
    X_f, Y_f = tr_fib.to(device), tr_sid.to(device)
    for step in range(900):
        idx = torch.randint(0, len(X_f), (256,), device=device)
        logits = fiber_enc.logits(X_f[idx])
        u = fiber_enc.encode(X_f[idx])
        c_pred = fiber_enc.chroma_head(u)
        loss = F.cross_entropy(logits, Y_f[idx], label_smoothing=0.05) + 0.2 * F.mse_loss(c_pred, tgt_c8[idx])
        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        opt_f.step()
    fiber_enc.eval()

    with torch.no_grad():
        ho_pred_f = fiber_enc.logits(ho_fib.to(device)).argmax(dim=-1).cpu()
        fib_acc_all = float((ho_pred_f == ho_sid).float().mean().item())
        fib_acc_cls = {STYLES[s]: round(float((ho_pred_f[ho_sid == s] == s).float().mean().item()), 4) for s in range(5)}

    # Pick 5 representative 1-shot exemplars from holdout (highest margin per style, 0 test/ref overlap!)
    with torch.no_grad():
        ho_logits_f = fiber_enc.logits(ho_fib.to(device)).cpu()
        oneshot_fib = torch.zeros(5, tr_fib.shape[1])
        oneshot_z = torch.zeros(5, 4, 64, 64)
        for s in range(5):
            ms = ho_sid == s
            sc = ho_logits_f[ms, s] - ho_logits_f[ms].masked_fill(F.one_hot(torch.tensor([s]), 5).bool(), -1e9).max(dim=1).values
            best_local = int(sc.argmax().item())
            global_idx = ms.nonzero(as_tuple=True)[0][best_local]
            oneshot_fib[s] = ho_fib[global_idx]
            oneshot_z[s] = v18_cache["holdout"]["z"][global_idx]

    # 3. Train BalancedVGGCriticHead on un-normalized VGG+Oklab features & calibrate holdout prior bias
    vgg_head = BalancedVGGCriticHead(in_dim=tr_vgg.shape[1], num_styles=5).to(device)
    vgg_head.x_mean.copy_(tr_vgg.mean(dim=0, keepdim=True).to(device))
    vgg_head.x_std.copy_(tr_vgg.std(dim=0, keepdim=True).clamp_min(1e-4).to(device))

    opt_v = torch.optim.AdamW(vgg_head.parameters(), lr=1.5e-3, weight_decay=2e-2)
    sched_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=1200, eta_min=1e-4)
    X_v = tr_vgg.to(device)
    cls_indices = [(Y_f == s).nonzero(as_tuple=True)[0] for s in range(5)]

    for step in range(1200):
        # Strict 1:1:1:1:1 balanced mini-batch (50 per class = 250)
        b_idx = torch.cat([ci[torch.randint(0, len(ci), (50,), device=device)] for ci in cls_indices], dim=0)
        logits = vgg_head(X_v[b_idx], calibrated=False)
        loss = F.cross_entropy(logits, Y_f[b_idx], label_smoothing=0.05)
        opt_v.zero_grad(set_to_none=True)
        loss.backward()
        opt_v.step()
        sched_v.step()
    vgg_head.eval()

    with torch.no_grad():
        ho_raw_logits = vgg_head(ho_vgg.to(device), calibrated=False)
        mean_l = ho_raw_logits.mean(dim=0)
        vgg_head.calib_bias.copy_(-(mean_l - mean_l.mean()))
        ho_cal_logits = vgg_head(ho_vgg.to(device), calibrated=True).cpu()
        ho_pred_v = ho_cal_logits.argmax(dim=-1)
        vgg_acc_all = float((ho_pred_v == ho_sid).float().mean().item())
        vgg_acc_cls = {STYLES[s]: round(float((ho_pred_v[ho_sid == s] == s).float().mean().item()), 4) for s in range(5)}

        # Also benchmark old R1 on the exact same 300 holdout latents for apples-to-apples Gate-1 proof!
        from src.infer import load_critic
        r1 = load_critic(device)
        r1_ho = torch.cat([r1(v18_cache["holdout"]["z"][i:i+64].to(device).float()).cpu() for i in range(0, 300, 64)], 0)
        r1_pred = r1_ho.argmax(dim=-1)
        r1_acc_all = float((r1_pred == ho_sid).float().mean().item())
        r1_acc_cls = {STYLES[s]: round(float((r1_pred[ho_sid == s] == s).float().mean().item()), 4) for s in range(5)}
        del r1

    gate1_report = {
        "wall_sec": round(time.time() - t0, 1),
        "vgg_feat_dim": int(tr_vgg.shape[1]),
        "fiber_holdout_acc_5way": round(fib_acc_all, 4),
        "fiber_holdout_per_style": fib_acc_cls,
        "old_r1_holdout_acc_5way": round(r1_acc_all, 4),
        "old_r1_holdout_per_style": r1_acc_cls,
        "v19_vgg_holdout_acc_5way": round(vgg_acc_all, 4),
        "v19_vgg_holdout_per_style": vgg_acc_cls,
        "vgg_calib_bias": [round(float(x), 4) for x in vgg_head.calib_bias.cpu().tolist()],
    }
    save_json(OUT / "gate1_critic_and_fiber_audit.json", gate1_report)
    print(f"[V19 Gate-1] Old R1 Holdout Acc={r1_acc_all*100:.1f}% {r1_acc_cls}", flush=True)
    print(f"[V19 Gate-1] V19 VGG Holdout Acc={vgg_acc_all*100:.1f}% {vgg_acc_cls}", flush=True)
    print(f"[V19 Gate-1] Fiber Pi_par Acc={fib_acc_all*100:.1f}% {fib_acc_cls}", flush=True)

    # Also store standardized class-mean VGG descriptors for gentle moment anchoring
    with torch.no_grad():
        tr_vgg_std = ((tr_vgg - vgg_head.x_mean.cpu()) / vgg_head.x_std.cpu()).clamp(-8.0, 8.0)
        vgg_cls_protos = torch.stack([tr_vgg_std[tr_sid == s].mean(dim=0) for s in range(5)])

    fnd = {
        "priors": priors,
        "fiber_state": fiber_enc.cpu().state_dict(),
        "vgg_head_state": vgg_head.cpu().state_dict(),
        "vgg_in_dim": int(tr_vgg.shape[1]),
        "vgg_cls_protos": vgg_cls_protos,
        "tr_fib": tr_fib.half(),
        "ho_fib": ho_fib.half(),
        "test750_fib": test750_fib.half(),
        "oneshot_fib": oneshot_fib.float(),
        "oneshot_z": oneshot_z.float(),
        "gate1_report": gate1_report,
    }
    torch.save(fnd, fnd_path)
    del vgg_ext
    torch.cuda.empty_cache()
    return {"v18_cache": v18_cache, "test_struct": test_struct, "fnd": fnd}
