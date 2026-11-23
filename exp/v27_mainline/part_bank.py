"""阶段 0 & 阶段 2/3 核心模块: Q32 通用语义部件词典审计 + 去几何 102D 经验贝叶斯部件银行 + PartSpatialResBlock.

设计要点:
  1) 通用无泄漏部件词典 (M=32): 对 train_4400 的 1,126,400 个 Q32 token 做球面 K-Means;
     硬审计每个簇的 5 画派归一化香农熵 (>= 0.85, 无单画派 > 55%).
  2) 102D 格内置换不变局部笔触描述子 v_patch (零边缘方向泄露):
     - Oklab 12D (mean 3, std 3, q15/q85 6)
     - 2 级 Haar 笔触频谱 12D (6 log-energies + 6 variances across L1/L2 LH/HL/HH)
     - 潜空间 4x4 纹理矩 14D (mean 4, std 4, 上三角协方差非对角 6)
     - 无中心化 RGB DINO PCA-64 (64D)
  3) 经验贝叶斯收缩构建 PartStyleBank B in R^{5 x 32 x 102} (1-Shot 零改动支持).
  4) PartSpatialResBlock: 零初始化空间 AdaGN 仅注入 dec32 / dec64, 零触碰 enc/mid16.
  5) part_transport_plan + part_barycentric_swd: 零 Sinkhorn 迭代的部件重心 SWD.

用法 (阶段 0 审计 + 构建部件银行):
  python -m exp.v27_mainline.part_bank --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v20.slf_flow import sinusoidal_time_emb
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27_mainline import OUT, ROOT
from exp.v27_mainline.dir1_reflow import SLFUNetGamma
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import encode_dino, load_dino_model
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc
from src.utils.structure import dwt2_haar, rgb_to_gray, rgb_to_oklab

PART_CACHE = OUT / "cache/part_bank_m32_d102.pt"
AUDIT_JSON = OUT / "stage0_part_audit.json"


# =============================================================================
# 1. Pure Modules: PartStyleBank, PartSpatialResBlock, SLFUNetPart, Bary-SWD
# =============================================================================

class PartStyleBank(nn.Module):
    def __init__(self, centers: torch.Tensor, bank: torch.Tensor, temp: float = 0.12):
        super().__init__()
        self.register_buffer("centers", F.normalize(centers.float(), dim=-1))  # (M, 32)
        self.register_buffer("bank", bank.float())                             # (5, M, D)
        self.temp = float(temp)

    def assign(self, q32: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q32.float().flatten(2).transpose(1, 2), dim=-1)        # (B, 256, 32)
        return F.softmax((q @ self.centers.T) / self.temp, dim=-1)             # (B, 256, M)

    def lookup_class(self, q32: torch.Tensor, tid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi = self.assign(q32)
        b_t = self.bank[tid]                                                   # (B, M, D)
        s16 = torch.bmm(pi, b_t).transpose(1, 2).reshape(q32.shape[0], -1, 16, 16)
        return s16, pi

    def lookup_exemplar(
        self, q32_src: torch.Tensor, q32_ref: torch.Tensor, v_ref_tok: torch.Tensor, kappa: float = 8.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pi_s = self.assign(q32_src)
        pi_r = self.assign(q32_ref)                                            # (B, 256, M)
        mass = pi_r.sum(dim=1, keepdim=True).transpose(1, 2)                   # (B, M, 1)
        v_part = torch.bmm(pi_r.transpose(1, 2), v_ref_tok.float())            # (B, M, D)
        v_glob = v_ref_tok.float().mean(dim=1, keepdim=True)                   # (B, 1, D)
        b_ref = (v_part + kappa * v_glob) / (mass + kappa)                     # (B, M, D)
        s16 = torch.bmm(pi_s, b_ref).transpose(1, 2).reshape(q32_src.shape[0], -1, 16, 16)
        return s16, pi_s


class PartSpatialResBlock(nn.Module):
    """Wraps PureStyleResBlock without losing pre-trained weights; adds zero-init spatial AdaGN."""

    def __init__(self, base_block: nn.Module, part_dim: int):
        super().__init__()
        self.base = base_block
        ch = base_block.conv1.in_channels
        self.sp_film1 = nn.Conv2d(part_dim, ch * 2, 1)
        self.sp_film2 = nn.Conv2d(part_dim, ch * 2, 1)
        nn.init.zeros_(self.sp_film1.weight)
        nn.init.zeros_(self.sp_film1.bias)
        nn.init.zeros_(self.sp_film2.weight)
        nn.init.zeros_(self.sp_film2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
        if s_map.shape[-2:] != x.shape[-2:]:
            s_map = F.interpolate(s_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        g1, b1 = self.base.film1(cond).chunk(2, dim=-1)
        dg1, db1 = self.sp_film1(s_map).chunk(2, dim=1)
        h = self.base.norm1(x) * (1.0 + g1[:, :, None, None] + torch.tanh(dg1)) + (b1[:, :, None, None] + db1)
        h = self.base.conv1(F.silu(h))
        g2, b2 = self.base.film2(cond).chunk(2, dim=-1)
        dg2, db2 = self.sp_film2(s_map).chunk(2, dim=1)
        h = self.base.norm2(h) * (1.0 + g2[:, :, None, None] + torch.tanh(dg2)) + (b2[:, :, None, None] + db2)
        return x + self.base.conv2(F.silu(h))


class SLFUNetPart(nn.Module):
    """Dir1 SLFUNetGamma + shared PartStyleProjector + dec32/dec64 PartSpatialResBlock (zero-init hot-start)."""

    def __init__(self, base_model: SLFUNetGamma, part_in_dim: int = 102, part_dim: int = 64):
        super().__init__()
        self.dino_proj = base_model.dino_proj
        self.cond_mlp = base_model.cond_mlp
        self.in_conv = base_model.in_conv
        self.enc64 = base_model.enc64
        self.down32 = base_model.down32
        self.enc32 = base_model.enc32
        self.down16 = base_model.down16
        self.mid1 = base_model.mid1
        self.mid_attn = base_model.mid_attn
        self.mid2 = base_model.mid2
        self.up32 = base_model.up32
        self.skip32 = base_model.skip32
        self.up64 = base_model.up64
        self.skip64 = base_model.skip64
        self.out_norm = base_model.out_norm
        self.out_conv = base_model.out_conv
        self.N_STYLE = base_model.N_STYLE

        self.part_proj = nn.Sequential(
            nn.Conv2d(part_in_dim, part_dim, 1),
            nn.GroupNorm(8, part_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(part_dim, part_dim, 1),
        )
        self.dec32 = nn.ModuleList([PartSpatialResBlock(blk, part_dim) for blk in base_model.dec32])
        self.dec64 = nn.ModuleList([PartSpatialResBlock(blk, part_dim) for blk in base_model.dec64])

    def part_parameters(self) -> list[nn.Parameter]:
        ps = list(self.part_proj.parameters())
        for blk in list(self.dec32) + list(self.dec64):
            ps.extend(blk.sp_film1.parameters())
            ps.extend(blk.sp_film2.parameters())
        return ps

    def forward(
        self,
        z_tau: torch.Tensor,
        y16_lcn: torch.Tensor,
        q32_16: torch.Tensor,
        style_id: torch.Tensor,
        u_vivid: torch.Tensor,
        tau: torch.Tensor,
        *,
        s16_part: torch.Tensor,
        drop_style: torch.Tensor | None = None,
        drop_content: torch.Tensor | None = None,
        layout_scale: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(q32_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1), self.N_STYLE).float() * smask2
        uv = u_vivid * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        g_val = torch.ones(B, device=dev, dtype=torch.float32) if gamma is None else gamma.to(dev).float().view(B)
        g_emb = sinusoidal_time_emb(g_val, 16)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags, g_emb], dim=-1))

        s_map = self.part_proj(s16_part.to(dtype=z_tau.dtype) * cmask4 * lscale4)

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond, s_map)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond, s_map)

        return self.out_conv(F.silu(self.out_norm(u64)))


def part_transport_plan(pi_src: torch.Tensor, pi_ref: torch.Tensor, kappa: float = 4.0) -> torch.Tensor:
    mass = pi_ref.sum(dim=1, keepdim=True)                                     # (B, 1, M)
    alpha = mass / (mass + kappa)                                              # (B, 1, M)
    p_cond = pi_ref / mass.clamp_min(1e-6)                                     # (B, 256, M)
    unif = 1.0 / float(pi_ref.shape[1])
    target_dist = alpha * p_cond + (1.0 - alpha) * unif                        # (B, 256, M)
    return torch.bmm(pi_src, target_dist.transpose(1, 2))                      # (B, 256, 256)


def part_barycentric_swd(
    f_out: torch.Tensor, f_ref: torch.Tensor, gamma_part: torch.Tensor, grid: int = 16, n_proj: int = 64
) -> torch.Tensor:
    B, C, H, W = f_out.shape
    if H > grid * 8:
        f_out = F.avg_pool2d(f_out, H // (grid * 8))
        f_ref = F.avg_pool2d(f_ref, H // (grid * 8))
        H, W = f_out.shape[-2:]
    ph, pw = H // grid, W // grid
    proj = F.normalize(torch.randn(C, n_proj, device=f_out.device, dtype=f_out.dtype), dim=0)

    def _quantiles(f: torch.Tensor) -> torch.Tensor:
        x = f.view(B, C, grid, ph, grid, pw).permute(0, 2, 4, 3, 5, 1).reshape(B, grid * grid, ph * pw, C)
        return torch.sort(x @ proj, dim=2).values

    q_out = _quantiles(f_out)
    q_ref = _quantiles(f_ref.detach())
    q_tgt = torch.einsum("bij,bjpk->bipk", gamma_part.to(q_ref.dtype), q_ref)
    return (q_out - q_tgt).abs().mean()


# =============================================================================
# 2. 102D Permutation-Invariant Patch Appearance Descriptor & K-Means Audit
# =============================================================================

@torch.no_grad()
def extract_patch_102d(
    rgb01: torch.Tensor,
    z_lat: torch.Tensor,
    dino_model,
    pca_basis: torch.Tensor,
    pca_mean: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract 16x16=256 tokens of 102D permutation-invariant local appearance (B, 256, 102)."""
    from PIL import Image

    B = len(rgb01)
    # 1) Oklab 12D per 32x32 cell -> (B, 256, 12)
    lab = rgb_to_oklab(rgb01.to(device).float())                               # (B, 3, 512, 512)
    c_lab = lab.view(B, 3, 16, 32, 16, 32).permute(0, 2, 4, 1, 3, 5).reshape(B, 256, 3, 1024)
    mu_ok = c_lab.mean(dim=-1)                                                 # (B, 256, 3)
    sd_ok = c_lab.std(dim=-1, unbiased=False)                                  # (B, 256, 3)
    q_ok = torch.quantile(c_lab[:, :, :, ::4], torch.tensor([0.15, 0.85], device=device), dim=-1)  # (2, B, 256, 3)
    ok12 = torch.cat([mu_ok, sd_ok, q_ok[0], q_ok[1]], dim=-1)                 # (B, 256, 12)

    # 2) 2-level Haar 12D per 32x32 cell -> (B, 256, 12)
    y = rgb_to_gray(rgb01.to(device).float())
    ll1, lh1, hl1, hh1 = dwt2_haar(y)                                          # (B, 1, 256, 256) -> cell 16x16
    _, lh2, hl2, hh2 = dwt2_haar(ll1)                                          # (B, 1, 128, 128) -> cell 8x8
    haar_feats = []
    for sub, sz in ((lh1, 16), (hl1, 16), (hh1, 16), (lh2, 8), (hl2, 8), (hh2, 8)):
        cells = sub.view(B, 16, sz, 16, sz).permute(0, 1, 3, 2, 4).reshape(B, 256, sz * sz)
        e = torch.log(cells.square().mean(dim=-1) + 1e-6)
        v = torch.log(cells.var(dim=-1, unbiased=False) + 1e-6)
        haar_feats.extend([e, v])
    haar12 = torch.stack(haar_feats, dim=-1)                                   # (B, 256, 12)

    # 3) Latent 4x4 14D per cell -> (B, 256, 14)
    zf = z_lat.to(device).float()
    c_z = zf.view(B, 4, 16, 4, 16, 4).permute(0, 2, 4, 1, 3, 5).reshape(B, 256, 4, 16)
    mu_z = c_z.mean(dim=-1)                                                    # (B, 256, 4)
    dz = c_z - mu_z.unsqueeze(-1)
    sd_z = (dz.square().mean(dim=-1) + 1e-6).sqrt()                            # (B, 256, 4)
    cov_z = torch.matmul(dz, dz.transpose(-1, -2)) / 16.0                      # (B, 256, 4, 4)
    i0, i1 = torch.triu_indices(4, 4, offset=1, device=device)
    cov6 = cov_z[:, :, i0, i1]                                                 # (B, 256, 6)
    lat14 = torch.cat([mu_z, sd_z, cov6], dim=-1)                              # (B, 256, 14)

    # 4) Un-centered RGB DINO PCA-64 -> (B, 256, 64)
    u8 = (rgb01.detach().cpu().float() * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    imgs = [Image.fromarray(arr) for arr in u8]
    _, patch = encode_dino(dino_model, imgs, batch=len(imgs), device=device)   # (B, 256, 384)
    d64 = (patch.to(device).float() - pca_mean.to(device)) @ pca_basis.to(device)

    return torch.cat([ok12, haar12, lat14, d64], dim=-1)                       # (B, 256, 102)


@torch.no_grad()
def spherical_kmeans_q32(q32_tr: torch.Tensor, sid_tr: torch.Tensor, m_parts: int = 32,
                         smooth_2x2: bool = False, iters: int = 40, device: str = "cuda") -> tuple[torch.Tensor, dict]:
    """Cluster all 4400 x 256 Q32 tokens into M=32 prototypes & run style-entropy audit."""
    seed_all(20268500)
    x = q32_tr.float()  # (4400, 32, 16, 16)
    if smooth_2x2:
        x = F.avg_pool2d(F.pad(x, (0, 1, 0, 1), mode="replicate"), 2, stride=1)
    tokens = F.normalize(x.flatten(2).transpose(1, 2).reshape(-1, 32), dim=-1).to(device)  # (1.126M, 32)

    # Subsample 220k tokens for fast spherical K-Means, then assign all 1.126M tokens for exact audit
    g = torch.Generator(device=device).manual_seed(20268501)
    sub_idx = torch.randperm(len(tokens), generator=g, device=device)[:220_000]
    sub = tokens[sub_idx]
    c_idx = torch.randperm(len(sub), generator=g, device=device)[:m_parts]
    centers = sub[c_idx].clone()

    for _ in range(iters):
        assign = (sub @ centers.T).argmax(dim=1)
        new_c = torch.zeros_like(centers)
        new_c.index_add_(0, assign, sub)
        empty = new_c.norm(dim=-1) < 1e-6
        if empty.any():
            new_c[empty] = sub[torch.randint(0, len(sub), (int(empty.sum()),), device=device)]
        centers = F.normalize(new_c, dim=-1)

    # Exact audit over ALL 4400 x 256 tokens
    N = len(q32_tr)
    counts = torch.zeros((m_parts, 5), dtype=torch.float64)
    sid_cpu = sid_tr.long().cpu()
    for st in range(0, N, 400):
        sl = slice(st, min(st + 400, N))
        tok_b = tokens[st * 256:min(st + 400, N) * 256]
        lbl = (tok_b @ centers.T).argmax(dim=1).cpu().view(-1, 256)
        for s_id in range(5):
            mask = sid_cpu[sl] == s_id
            if mask.any():
                bc = torch.bincount(lbl[mask].reshape(-1), minlength=m_parts).double()
                counts[:, s_id] += bc

    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    log5 = math.log(5.0)
    ent = -(probs * (probs + 1e-12).log()).sum(dim=1) / log5
    max_dom = probs.max(dim=1).values
    audit = {
        "m_parts": m_parts,
        "smooth_2x2": smooth_2x2,
        "mean_norm_entropy": round(float(ent.mean().item()), 4),
        "min_norm_entropy": round(float(ent.min().item()), 4),
        "max_single_style_share": round(float(max_dom.max().item()), 4),
        "cluster_sizes_min_max": [int(counts.sum(1).min().item()), int(counts.sum(1).max().item())],
        "pass_gate": bool(ent.mean().item() >= 0.85 and max_dom.max().item() <= 0.55),
    }
    return centers.cpu(), audit


@torch.no_grad()
def build_or_load_part_bank(pool_dir: str, m_parts: int = 32, kappa_cls: float = 200.0, device: str = "cuda") -> dict:
    if PART_CACHE.exists():
        print(f"[PartBank] Reusing {PART_CACHE}", flush=True)
        return torch.load(PART_CACHE, map_location="cpu", weights_only=False)

    OUT.mkdir(parents=True, exist_ok=True)
    PART_CACHE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pack = build_or_load_v21_cache(pool_dir, device=device)
    bundle, c21 = pack["bundle"], pack["c21"]
    tr = bundle["v18_cache"]["train"]
    z_tr = tr["z"].float()
    sid_tr = tr["sid"].long()
    q32_tr = c21["train"]["q32"].float()
    N = len(z_tr)

    # 1) Audit Q32 clusters
    centers, audit = spherical_kmeans_q32(q32_tr, sid_tr, m_parts=m_parts, smooth_2x2=False, device=device)
    print(f"[Stage0 PartAudit raw] {audit}", flush=True)
    if not audit["pass_gate"]:
        centers_sm, audit_sm = spherical_kmeans_q32(q32_tr, sid_tr, m_parts=m_parts, smooth_2x2=True, device=device)
        print(f"[Stage0 PartAudit smooth2x2] {audit_sm}", flush=True)
        if audit_sm["mean_norm_entropy"] > audit["mean_norm_entropy"]:
            centers, audit = centers_sm, audit_sm
    save_json(AUDIT_JSON, {"updated_utc": utc(), "audit": audit})
    if not audit["pass_gate"]:
        raise RuntimeError(f"Part dictionary failed entropy gate: {audit}")

    # 2) Decode train_4400 via S10 & fit PCA-64 on un-centered RGB DINO tokens (1100 subsample)
    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)
    from PIL import Image

    sample_dino = []
    for st in range(0, N, 64):
        if (st // 64) % 4 != 0:
            continue
        sl = slice(st, min(st + 64, N))
        rgb = decode_s10(dec, z_tr[sl].to(device), batch=16, output_cpu=False).clamp(0.0, 1.0)
        u8 = (rgb.cpu() * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        imgs = [Image.fromarray(arr) for arr in u8]
        _, patch = encode_dino(dino, imgs, batch=32, device=device)
        sample_dino.append(patch[:, ::8, :].reshape(-1, 384).cpu().float())

    sd_all = torch.cat(sample_dino, dim=0)
    pca_mean = sd_all.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd((sd_all - pca_mean), full_matrices=False)
    pca_basis = vh[:64].T.contiguous()  # (384, 64)
    print(f"[PartBank] Fitted RGB-DINO PCA-64 from {len(sd_all)} tokens in {time.time()-t0:.1f}s", flush=True)

    # 3) Accumulate Empirical Bayes Part Style Bank across train_4400
    centers_gpu = F.normalize(centers.to(device).float(), dim=-1)
    mass_sm = torch.zeros((5, m_parts), device=device, dtype=torch.float64)
    sum_sm = torch.zeros((5, m_parts, 102), device=device, dtype=torch.float64)
    glob_cnt = torch.zeros(5, device=device, dtype=torch.float64)
    glob_sum = torch.zeros((5, 102), device=device, dtype=torch.float64)

    # Also compute mean & std for v_patch standardization so projector starts well-conditioned
    v_sum = torch.zeros(102, device=device, dtype=torch.float64)
    v_sq = torch.zeros(102, device=device, dtype=torch.float64)
    v_n = 0

    for st in range(0, N, 32):
        sl = slice(st, min(st + 32, N))
        zb = z_tr[sl].to(device)
        rgb = decode_s10(dec, zb, batch=16, output_cpu=False).clamp(0.0, 1.0)
        v_tok = extract_patch_102d(rgb, zb, dino, pca_basis, pca_mean, device=device)  # (B, 256, 102)

        q_tok = F.normalize(q32_tr[sl].to(device).flatten(2).transpose(1, 2), dim=-1)   # (B, 256, 32)
        pi = F.softmax((q_tok @ centers_gpu.T) / 0.12, dim=-1)                         # (B, 256, M)

        v_flat = v_tok.reshape(-1, 102).double()
        v_sum += v_flat.sum(dim=0)
        v_sq += v_flat.square().sum(dim=0)
        v_n += len(v_flat)

        sb = sid_tr[sl].to(device)
        for s_id in range(5):
            m = sb == s_id
            if not m.any():
                continue
            pi_s = pi[m].double()     # (Bs, 256, M)
            vt_s = v_tok[m].double()  # (Bs, 256, 102)
            mass_sm[s_id] += pi_s.sum(dim=(0, 1))
            sum_sm[s_id] += torch.einsum("bjm,bjd->md", pi_s, vt_s)
            glob_cnt[s_id] += float(m.sum().item() * 256)
            glob_sum[s_id] += vt_s.sum(dim=(0, 1))

    v_mean = (v_sum / v_n).float()
    v_std = ((v_sq / v_n - (v_sum / v_n).square()).clamp_min(1e-6).sqrt()).float()

    # Standardize v_patch before Empirical Bayes shrinkage
    bar_glob = ((glob_sum / glob_cnt.unsqueeze(-1)).float() - v_mean) / v_std
    bar_part = ((sum_sm / mass_sm.unsqueeze(-1).clamp_min(1e-6)).float() - v_mean) / v_std
    m_f = mass_sm.float().unsqueeze(-1)
    bank_b = (m_f * bar_part + kappa_cls * bar_glob.unsqueeze(1)) / (m_f + kappa_cls)

    del dec, dino
    torch.cuda.empty_cache()

    payload = {
        "centers": centers.cpu(),
        "bank": bank_b.cpu(),
        "v_mean": v_mean.cpu(),
        "v_std": v_std.cpu(),
        "pca_basis": pca_basis.cpu(),
        "pca_mean": pca_mean.cpu(),
        "mass_sm": mass_sm.cpu().float(),
        "audit": audit,
        "kappa_cls": kappa_cls,
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(payload, PART_CACHE)
    print(f"[PartBank] Saved {PART_CACHE} in {payload['wall_sec']}s (min mass={mass_sm.min().item():.1f})", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    build_or_load_part_bank(args.pool, device=args.device)


if __name__ == "__main__":
    main()
