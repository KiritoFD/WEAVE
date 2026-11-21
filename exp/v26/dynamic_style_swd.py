r"""V26 共享: 动态投影多尺度 SWD 风格距离 (docs/1120 §P2-A, 调整版: 单轮受控试验).

冻结 VGG-16 的 relu1_2 / relu2_2 / relu3_3 三层特征, 每次距离计算动态生成
`n_proj` 个单位随机投影方向, 对输出/参考/源图使用**同一组方向与同一组空间
token 抽样索引** (契约 7); 每层独立归一化 (契约: 防浅层大幅值支配).

距离: 对每层每方向计算 1-D Wasserstein 距离 (排序经验分布 L1), 层内求均值后
按层做方差归一, 三层求和.

供 K2 审计 (判别力/梯度) 与 3060 单轮 SWD 受控训练试验共用.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.protocol import load_vgg16_features_offline

SWD_LAYERS = (3, 8, 15)      # relu1_2(64ch) / relu2_2(128ch) / relu3_3(256ch) in vgg index
DEFAULT_N_PROJ = 64
DEFAULT_N_TOKEN = 512
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DynamicSWD:
    """冻结 VGG-16 多尺度动态投影 SWD 距离 (训练专用, 非评测裁判)."""

    def __init__(self, device: str = "cuda", n_proj: int = DEFAULT_N_PROJ,
                 n_token: int = DEFAULT_N_TOKEN, dtype: torch.dtype = torch.float32):
        vgg = load_vgg16_features_offline(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        if dtype != torch.float32:
            vgg = vgg.to(dtype)  # bf16 权重: 激活/梯度显存减半 (3060 单轮 SWD 试验需要)
        self.vgg = vgg.to(device)
        self.device = device
        self.n_proj = n_proj
        self.n_token = n_token
        self.dtype = dtype
        self._mean = _MEAN.to(device, dtype)
        self._std = _STD.to(device, dtype)
        self._layer_scales = None  # 首次调用时按参考特征幅度初始化层归一
        self._layer_scales_bary = None  # Bary-SWD 独立层归一 (与全局 SWD 尺度不可混)

    def _feats(self, rgb01: torch.Tensor) -> list[torch.Tensor]:
        x = (rgb01.to(self.device).to(self.dtype) - self._mean) / self._std
        outs = []
        h = x
        for i, m in enumerate(self.vgg):
            h = m(h)
            if i in SWD_LAYERS:
                outs.append(h)
        return outs

    def _token_sample(self, f: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) + 全局共享 token 索引 → (B, T, C)."""
        B, C, H, W = f.shape
        flat = f.flatten(2).transpose(1, 2)          # (B, HW, C)
        return flat[:, idx, :]

    def loss(self, rgb_out: torch.Tensor, rgb_ref: torch.Tensor,
             rgb_src: torch.Tensor | None = None, *, gen: torch.Generator | None = None) -> dict:
        """动态 SWD 风格距离.

        返回 dict(loss=标量, per_layer=List[标量]); rgb_out 带梯度路径,
        rgb_ref/rgb_src 为常量侧 (同一次调用内共享投影方向与 token 索引 — 契约 7).
        """
        g = gen if gen is not None else torch.Generator(device="cpu")
        imgs = torch.cat([rgb_out, rgb_ref] + ([rgb_src] if rgb_src is not None else []), 0)
        feats = self._feats(imgs)
        k = rgb_out.shape[0]
        losses_per_layer = []
        for f in feats:
            B, C, H, W = f.shape
            # 共享投影方向: (C, n_proj) 每次调用重采样 (契约 6: 逐 update 刷新)
            proj = torch.randn(C, self.n_proj, generator=g).to(self.device, self.dtype)
            proj = F.normalize(proj.float(), dim=0).to(self.dtype)
            # 共享 token 索引
            n_hw = H * W
            t = min(self.n_token, n_hw)
            idx = torch.randperm(n_hw, generator=g)[:t].to(self.device)
            # 投影后的 1-D 边际分布: (B, T, n_proj)
            p_all = self._token_sample(f, idx) @ proj            # (B, T, n_proj)
            p_out, p_ref = p_all[:k], p_all[k:2 * k]
            # 逐方向排序经验分布 L1 (fp32 排序求差, 数值稳定)
            q_out, _ = torch.sort(p_out.float(), dim=1)
            q_ref, _ = torch.sort(p_ref.float(), dim=1)
            w1 = (q_out - q_ref).abs().mean(dim=(1, 2))          # (k,)
            losses_per_layer.append(w1)
        # 层归一 (首次: 以参考侧幅度为准, 之后固定)
        if self._layer_scales is None:
            self._layer_scales = [max(float(l.detach().mean()), 1e-4) for l in losses_per_layer]
        scaled = [l / s for l, s in zip(losses_per_layer, self._layer_scales)]
        loss = torch.stack(scaled).sum()
        return {"loss": loss, "per_layer": [float(l.detach().mean()) for l in losses_per_layer]}

    def loss_bary(self, rgb_out: torch.Tensor, rgb_ref: torch.Tensor,
                  gamma_bar: torch.Tensor, *, gen: torch.Generator | None = None) -> dict:
        """UOT Bary-SWD (V26-L Exp2 / L3-S4): 摊销式局部性训练目标.

        与 loss() 同一 VGG 管线/动态投影/层归一, 仅目标不同: 全局 1-D 边际 →
        **逐 DINO 格 (16×16) 分位数**, 参考侧按 Γ̄ (uot_gamma_bar, 行归一) 做
        1D W2 重心线性组合 (Agueh-Carlier 闭式). 格内排序天然去几何 (边缘/透视
        100% 保持原位); gamma_bar 为常量侧 (无梯度).
        """
        g = gen if gen is not None else torch.Generator(device="cpu")
        imgs = torch.cat([rgb_out, rgb_ref], 0)
        feats = self._feats(imgs)
        k = rgb_out.shape[0]
        losses_per_layer = []
        for f in feats:
            B, C, H, W = f.shape
            ph, pw = H // 16, W // 16
            assert ph * 16 == H and pw * 16 == W, f"特征分辨率 {H} 须为 16 整数倍"
            proj = torch.randn(C, self.n_proj, generator=g).to(self.device, self.dtype)
            proj = F.normalize(proj.float(), dim=0).to(self.dtype)
            # 逐格分位数: (2k, 256, P, C) @ proj → (2k, 256, P, K), P = ph*pw
            x = f.view(B, C, 16, ph, 16, pw).permute(0, 2, 4, 3, 5, 1).reshape(B, 256, ph * pw, C)
            p_all = (x @ proj).float()
            q = torch.sort(p_all, dim=2).values
            q_out, q_ref = q[:k], q[k:]
            # 1D W2 重心闭式: Γ̄ 加权参考格分位数 (常量侧, detach)
            q_tgt = torch.einsum("bij,bjpk->bipk", gamma_bar.to(q_ref.dtype).detach(), q_ref)
            w1 = (q_out - q_tgt).abs().mean(dim=(1, 2, 3))  # (k,)
            losses_per_layer.append(w1)
        if self._layer_scales_bary is None:
            self._layer_scales_bary = [max(float(l.detach().mean()), 1e-4) for l in losses_per_layer]
        scaled = [l / s for l, s in zip(losses_per_layer, self._layer_scales_bary)]
        return {"loss": torch.stack(scaled).sum(),
                "per_layer": [float(l.detach().mean()) for l in losses_per_layer]}


def swd_distance_matrix(imgs: torch.Tensor, extractor, n_pairs_round: int = 256,
                        gen: torch.Generator | None = None) -> torch.Tensor:
    """成对 SWD 距离矩阵 (N, N) — K2-A 判别力审计用 (对称, 对角 0).

    imgs: (N, 3, H, W) 已在 [0,1]; extractor: DynamicSWD (loss 忽略梯度).
    为控制耗时, 对每对独立调用 (方向重采样), N≤200 时可承受.
    """
    n = imgs.shape[0]
    d = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            r = extractor.loss(imgs[i:i + 1], imgs[j:j + 1], gen=gen)
            v = float(r["loss"].detach())
            d[i, j] = d[j, i] = v
    return d
