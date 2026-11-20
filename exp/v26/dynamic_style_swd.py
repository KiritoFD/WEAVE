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
        self.vgg = vgg.to(device)
        self.device = device
        self.n_proj = n_proj
        self.n_token = n_token
        self.dtype = dtype
        self._mean = _MEAN.to(device)
        self._std = _STD.to(device)
        self._layer_scales = None  # 首次调用时按参考特征幅度初始化层归一

    def _feats(self, rgb01: torch.Tensor) -> list[torch.Tensor]:
        x = (rgb01.float().to(self.device) - self._mean) / self._std
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
            proj = torch.randn(C, self.n_proj, generator=g, device="cpu").to(self.device)
            proj = F.normalize(proj, dim=0)
            # 共享 token 索引
            n_hw = H * W
            t = min(self.n_token, n_hw)
            idx = torch.randperm(n_hw, generator=g)[:t].to(self.device)
            # 投影后的 1-D 边际分布: (B, T, n_proj)
            p_all = self._token_sample(f, idx) @ proj            # (B*k?, no: imgs batch)
            p_out, p_ref = p_all[:k], p_all[k:2 * k]
            # 逐方向排序经验分布 L1
            q_out, _ = torch.sort(p_out, dim=1)
            q_ref, _ = torch.sort(p_ref, dim=1)
            w1 = (q_out - q_ref).abs().mean(dim=(1, 2))          # (k,)
            losses_per_layer.append(w1)
        # 层归一 (首次: 以参考侧幅度为准, 之后固定)
        if self._layer_scales is None:
            self._layer_scales = [max(float(l.detach().mean()), 1e-4) for l in losses_per_layer]
        scaled = [l / s for l, s in zip(losses_per_layer, self._layer_scales)]
        loss = torch.stack(scaled).sum()
        return {"loss": loss, "per_layer": [float(l.detach().mean()) for l in losses_per_layer]}


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
