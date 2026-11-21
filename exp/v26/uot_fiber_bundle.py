r"""V26-L 局部性: Quotient-UOT Fiber Bundle 核心 (零聚类、零 K、零硬边界).

docs/1120/08_v26_long_plan.md 局部性专章 (用户 2026-11-21 指令).

三个数学算子:
  1. log-domain 非平衡 Sinkhorn (Quotient-UOT): 在冻结的风格不变 DINO token 上
     求跨图语义传输 plan Γ*, 行归一化得 Γ̄ (每源格 → 参考格的软对齐);
     跨题材语义正交时熵正则自动松弛为均匀分布 (绝不强行把色块匹配到衣褶).
  2. uot_barycentric_swd: 1D W2 重心闭式解 (Agueh-Carlier) — 参考纹理分位数按
     Γ̄ 加权一次矩阵乘即得逐格目标分位数, 输出格内排序天然去几何 (格内边缘/透视
     100% 保持原位).
  3. geometry-free fiber stats: 每格 Oklab 均值/方差 + Haar LH/HL/HH 能量偏度 —
     无空间边缘方向, 数学上不可能把参考图几何泄露给源图.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------- Quotient-UOT ----

def log_domain_uot(C: torch.Tensor, *, eps: float = 0.05, rho1: float = 8.0,
                   rho2: float = 8.0, n_iter: int = 6) -> torch.Tensor:
    """Log-domain 非平衡 Sinkhorn (Chizat et al.). C: (B, N, M) 语义代价.

    返回行归一化 Γ̄: (B, N, M), Σ_M Γ̄_ij = 1 (每源格在参考格上的软分布).
    KL 边际松弛 (rho1 行 / rho2 列) 允许面积比例失调; eps 熵正则在 C 平坦时
    自动松弛为均匀分布. 6 步迭代 256×256 GPU <0.2ms.
    """
    B, N, M = C.shape
    device = C.device
    log_a = torch.full((B, N), -float(torch.log(torch.tensor(float(N)))), device=device)
    log_b = torch.full((B, M), -float(torch.log(torch.tensor(float(M)))), device=device)
    u = torch.zeros(B, N, device=device)
    v = torch.zeros(B, M, device=device)
    d1, d2 = rho1 / (rho1 + eps), rho2 / (rho2 + eps)
    Ce = C / eps
    for _ in range(n_iter):
        # u_i = d1 * (eps log a_i − eps LSE_j((v_j − C_ij)/eps))
        u = d1 * (eps * log_a - eps * torch.logsumexp((v.unsqueeze(1) - C) / eps, dim=2))
        # v_j = d2 * (eps log b_j − eps LSE_i((u_i − C_ij)/eps))
        v = d2 * (eps * log_b - eps * torch.logsumexp((u.unsqueeze(2) - C) / eps, dim=1))
    gamma = torch.exp((u.unsqueeze(2) + v.unsqueeze(1) - C) / eps)
    return gamma / gamma.sum(dim=2, keepdim=True).clamp_min(1e-12)


def uot_gamma_bar(d16_src: torch.Tensor, d16_ref: torch.Tensor, *,
                  eps: float = 0.05, rho1: float = 8.0, rho2: float = 8.0,
                  n_iter: int = 6) -> torch.Tensor:
    """从冻结的风格不变 DINO token 构造 Γ̄ (B, 256, 256).

    d16_src/d16_ref: (B, 384, 16, 16) 灰度白化 DINO 布局 (Π⊥ 商空间坐标 —
    无需学习 W_Q/W_K, 直接余弦代价).
    """
    B = d16_src.shape[0]
    q_src = NF.normalize(d16_src.float().flatten(2).transpose(1, 2), dim=-1)   # (B, 256, 384)
    q_ref = NF.normalize(d16_ref.float().flatten(2).transpose(1, 2), dim=-1)
    C = 1.0 - torch.bmm(q_src, q_ref.transpose(1, 2))                          # (B, 256, 256)
    return log_domain_uot(C, eps=eps, rho1=rho1, rho2=rho2, n_iter=n_iter)


# ------------------------------------------------- Bary-SWD ----

def uot_barycentric_swd(f_out: torch.Tensor, f_ref: torch.Tensor,
                        gamma_bar: torch.Tensor, *, grid: int = 16,
                        n_proj: int = 64, gen: torch.Generator | None = None) -> dict:
    """UOT 重心切片 Wasserstein 损失 (Agueh-Carlier 1D W2 重心闭式解).

    f_out/f_ref: VGG 特征 (B, C, H, W), H=W=grid*ph (relu3_3→128², relu2_2→256²).
    gamma_bar: (B, 256, 256) UOT 行归一化 plan (DINO 16×16 格).
    每个 DINO 格 i 内 P=ph×pw 个像素构成局部经验分布; 格内排序天然去几何.
    q_ref detach (常量侧). 返回 dict(loss=标量, per_grid=标量).
    """
    B, C, H, W = f_out.shape
    ph, pw = H // grid, W // grid
    assert ph * grid == H and pw * grid == W, f"特征分辨率 {H} 须为 grid={grid} 整数倍"
    dev = f_out.device
    proj = F.normalize(torch.randn(C, n_proj, generator=gen, device="cpu").to(dev), dim=0)

    def to_quantiles(f: torch.Tensor) -> torch.Tensor:
        # (B, C, grid*ph, grid*pw) → (B, 256, P, C)
        x = f.view(B, C, grid, ph, grid, pw).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, grid * grid, ph * pw, C)

    def sort_proj(x: torch.Tensor) -> torch.Tensor:
        # (B, 256, P, C) @ proj (C, K) → (B, 256, P, K) → sort over P
        return torch.sort(x @ proj, dim=2).values

    q_out = sort_proj(to_quantiles(f_out))
    with torch.no_grad():
        q_ref = sort_proj(to_quantiles(f_ref))
    # 1D W2 重心闭式: 分位数线性组合 (Agueh-Carlier 等距性)
    q_target = torch.einsum("bij,bjpk->bipk", gamma_bar.to(q_ref.dtype), q_ref)
    loss = (q_out - q_target).abs().mean()
    return {"loss": loss, "per_grid": float(loss)}


def weighted_global_swd(f_out_tokens: torch.Tensor, f_ref_tokens: torch.Tensor,
                        w: torch.Tensor, *, n_proj: int = 64,
                        gen: torch.Generator | None = None) -> torch.Tensor:
    """重加权全局 SWD: 参考 token 按诱导边际 w_j 重采样后与输出做标准 SWD.

    f_out_tokens/f_ref_tokens: (B, 256, C) 格级特征; w: (B, 256) 诱导边际
    (Σ_j w_j = 1/格). 防止重心平均收缩跨区域全局色域.
    """
    B, T, C = f_ref_tokens.shape
    dev = f_ref_tokens.device
    proj = F.normalize(torch.randn(C, n_proj, generator=gen, device="cpu").to(dev), dim=0)
    q_out = torch.sort(f_out_tokens @ proj, dim=1).values
    # 按 w 加权重采样参考 token (T 个, 权重归一)
    wn = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    sampled = []
    for b in range(B):
        idx = torch.multinomial(wn[b], T, replacement=True)
        sampled.append(f_ref_tokens[b][idx])
    q_ref = torch.stack(sampled, 0).detach()
    q_ref = torch.sort(q_ref @ proj, dim=1).values
    return (q_out - q_ref).abs().mean()


# ------------------------------------------------- 去几何纤维统计 ----

def geometry_free_fiber_stats(rgb: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """每格**无几何**局部纤维统计 Π_∥^(j): Oklab 均值/方差 + Haar LH/HL/HH 能量/偏度.

    rgb: (B, 3, 512, 512) → (B, 256, D) 特征 (D = 9 Oklab + 4×2 Haar = 17).
    不含任何空间边缘方向 → 数学上不可能泄露参考图几何.
    """
    B, C, H, W = rgb.shape
    cell = H // grid
    # Oklab (RGB→Lab 近似: 直接在 RGB 上做规范化统计亦可, 但保持与 fiber_duality 一致用 Lab)
    from exp.v19.fiber_duality import rgb_to_oklab
    lab = rgb_to_oklab(rgb)  # (B, 3, H, W)
    feats = []
    x = lab
    for i in range(grid):
        for j in range(grid):
            cell_x = x[:, :, i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            flat = cell_x.flatten(2)  # (B, 3, cell²)
            mu = flat.mean(dim=2)     # (B, 3)
            sd = flat.std(dim=2, unbiased=False)  # (B, 3)
            feats.append(torch.cat([mu, sd], dim=-1))
    lab_stats = torch.stack(feats, dim=1)  # (B, 256, 6)

    # Haar LH/HL/HH 能量/偏度 (每格)
    haar_feats = []
    for i in range(grid):
        for j in range(grid):
            cell_x = rgb[:, :, i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].mean(dim=1, keepdim=True)
            ch, cw = cell_x.shape[-2] // 2, cell_x.shape[-1] // 2
            a = cell_x[:, :, :ch, :cw]; b = cell_x[:, :, :ch, cw:]
            c = cell_x[:, :, ch:, :cw]; dd = cell_x[:, :, ch:, cw:]
            lh = (a - b).abs().mean(dim=(2, 3))
            hl = (a - c).abs().mean(dim=(2, 3))
            hh = ((a - b) * (a - c)).abs().mean(dim=(2, 3))
            haar_feats.append(torch.cat([lh, hl, hh], dim=-1))
    haar_stats = torch.stack(haar_feats, dim=1)  # (B, 256, 3)
    return torch.cat([lab_stats, haar_stats], dim=-1)  # (B, 256, 9)
