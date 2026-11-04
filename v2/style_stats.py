"""Style-ID-only 目标风格统计（离线提取）。

每个风格统计量 = Haar 四个子带 (LL/LH/HL/HH) 的逐通道 mean/std，共
4×4×2 = 32 维。训练时作为 ``style_stats`` bank 传入生成器；bank 缺失时
v2 主干退化为纯 style-id embedding，旧 WEAVE 固定参考图协议不受影响。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch

from wavelet import dwt2_haar


STATS_DIM = 32


def compute_style_stats(latents: torch.Tensor | Iterable[torch.Tensor], *, max_images: int | None = None) -> torch.Tensor:
    """``latents``: (N,4,H,W) 或可迭代的 latent 张量；返回 (32,) float32。"""
    if isinstance(latents, torch.Tensor):
        bank = latents.float()
    else:
        chunks = [t.float().reshape(-1, *t.shape[-3:]) for t in latents]
        bank = torch.cat(chunks, dim=0) if chunks else torch.zeros(1, 4, 8, 8)
    if max_images is not None and bank.shape[0] > max_images:
        bank = bank[:max_images]
    means = []
    stds = []
    ll, lh, hl, hh = dwt2_haar(bank)
    for sub in (ll, lh, hl, hh):
        sub = sub.float()
        means.append(sub.mean(dim=(0, 2, 3)))
        stds.append(sub.std(dim=(0, 2, 3), unbiased=False))
    return torch.cat(means + stds, dim=0).float()  # 4 subbands × 4ch × (mean,std)


def build_style_stats_from_tensors(
    tensors_by_style: Mapping[int, torch.Tensor],
    *,
    num_styles: int | None = None,
    max_images_per_style: int | None = 64,
) -> torch.Tensor:
    n = int(num_styles if num_styles is not None else max(tensors_by_style.keys()) + 1)
    bank = torch.zeros(n, STATS_DIM)
    for sid, tensors in tensors_by_style.items():
        if sid >= n:
            continue
        bank[int(sid)] = compute_style_stats(tensors, max_images=max_images_per_style)
    return bank


def build_style_stats_from_dataset(
    dataset,
    *,
    max_images_per_style: int | None = 64,
) -> torch.Tensor:
    """从 ``AdaCUTLatentDataset`` 的 ``style_tensors`` 离线统计，不依赖测试集首图。"""
    tensors = getattr(dataset, "style_tensors", None)
    if not isinstance(tensors, Mapping):
        raise AttributeError("dataset has no style_tensors mapping; cannot build style stats")
    return build_style_stats_from_tensors(
        {int(k): v for k, v in tensors.items()},
        num_styles=len(getattr(dataset, "style_subdirs", tensors)),
        max_images_per_style=max_images_per_style,
    )


def save_style_stats(bank: torch.Tensor, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stats": bank.cpu().float(), "dim": int(bank.shape[-1])}, str(path))
    return str(path)


def load_style_stats(path: str | Path, *, device: torch.device | str = "cpu") -> torch.Tensor:
    payload = torch.load(str(path), map_location=device)
    bank = payload["stats"] if isinstance(payload, Mapping) else payload
    if bank.dim() != 2 or bank.shape[-1] != STATS_DIM:
        raise ValueError(f"style stats must be (num_styles, {STATS_DIM}), got {tuple(bank.shape)}")
    return bank.float().to(device)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from config_schema import load_experiment_config
    from .train import _build_real_dataset

    parser = argparse.ArgumentParser(description="离线提取 Style-ID-only 风格统计")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-images-per-style", type=int, default=64)
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    dataset = _build_real_dataset(config, torch.device("cpu"))
    bank = build_style_stats_from_dataset(dataset, max_images_per_style=args.max_images_per_style)
    path = save_style_stats(bank, args.out)
    print(f"saved style stats: {path} shape={tuple(bank.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "STATS_DIM",
    "compute_style_stats",
    "build_style_stats_from_tensors",
    "build_style_stats_from_dataset",
    "save_style_stats",
    "load_style_stats",
]
