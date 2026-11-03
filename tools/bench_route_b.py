"""路线 B1 算子 T_phi 的容量 / FLOPs / 显存 / 步时预估（仅 CPU 测量 + 解析估算）。

用法：
    python tools/bench_route_b.py --batch 8 --budget-gb 10

测量内容：
  * 参数量与参数+优化器常驻显存；
  * 单样本前向 FLOPs（FlopCounterMode）；
  * CPU 前向/反向墙钟（只用于 sanity check，不外推到 GPU）；
  * 基于显式假设的 3060 12G 步时与 30k 墙钟。

注意：GPU 数字是从 FLOPs 与假设的有效算力推出来的**估算**，不是实测。
本脚本不导入 CUDA，也不访问 GPU/远端。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b1_style_operator import B1OperatorConfig, B1StyleOperator  # noqa: E402


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_flops(op: torch.nn.Module, z: torch.Tensor, r: torch.Tensor) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            op(z, r)
    return int(counter.get_total_flops())


def time_call(fn, warmup: int = 1, repeat: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def activation_bytes_estimate(cfg: B1OperatorConfig, *, latent: int = 64, batch: int = 8) -> float:
    """按特征图尺寸粗估单步训练激活（前向保存 + 反向临时），单位 GB。

    假设：
      * 每个 ResBlock 保存 ~4 个同尺寸张量（norm/act/conv 输出）；
      * 注意力保存每个头的 256x256 矩阵与 FFN 中间层（4x 宽度）。
    """
    w0, w1, w2 = cfg.widths
    n0, n1, n2 = cfg.encoder_blocks
    # 编码器（内容 + 参考共用同一份权重，但激活各一份）
    lv0 = latent
    lv1 = latent // 2
    lv2 = latent // 4
    per_branch = (
        w0 * lv0 * lv0 * 4 * (2 + 4 * n0)
        + w1 * lv1 * lv1 * 4 * (2 + 4 * n1)
        + w2 * lv2 * lv2 * 4 * (2 + 4 * n2)
    )
    tokens = cfg.token_side ** 2
    attn_per_level = 0
    for w, heads, layers in zip(cfg.widths, cfg.attention_heads, cfg.attention_layers):
        attn_per_level += layers * (
            tokens * tokens * heads * 4.0  # 注意力权重
            + tokens * w * 4.0 * 6.0        # qkv + ffn 中间层
        )
    acts = 2.0 * per_branch + attn_per_level + (w0 * lv0 * lv0) * 4.0 * 6.0
    # 反向按 ~2x 前向激活计
    total_bytes = batch * acts * 3.0
    return total_bytes / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--budget-gb", type=float, default=10.0)
    parser.add_argument("--tflops-fp32", type=float, default=8.0)
    parser.add_argument("--tflops-bf16", type=float, default=14.0)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()

    torch.manual_seed(0)
    op = B1StyleOperator(B1OperatorConfig())
    n = count_params(op)
    z = torch.randn(1, 4, args.latent, args.latent)
    r = torch.randn(1, 4, args.latent, args.latent)

    flops = measure_flops(op, z, r)
    fwd_cpu = time_call(lambda: op(z, r))
    params_mem = n * 16 / (1024 ** 3)  # fp32 参数 + 梯度 + Adam 两个状态

    print(f"params={n/1e6:.3f}M  param+optimizer≈{params_mem:.2f} GB")
    print(f"forward FLOPs/sample={flops/1e9:.2f} GFLOPs (latent {args.latent})")
    print(f"train FLOPs/sample≈{3*flops/1e9:.2f} GFLOPs (fwd+bwd≈3x)")
    print(f"CPU forward latency={fwd_cpu*1000:.1f} ms/sample (reference only)")

    for tag, tf in (("fp32", args.tflops_fp32), ("bf16", args.tflops_bf16)):
        eff = tf * 1e12
        step_s = 3 * flops * args.batch / eff
        wall_h = step_s * args.steps / 3600.0
        print(
            f"[{tag}] est step={step_s*1000:.0f} ms @ batch={args.batch}, "
            f"effective={tf:.1f} TFLOPS -> {args.steps} steps ≈ {wall_h:.2f} h"
        )

    for b in (1, 8, 16):
        act = activation_bytes_estimate(B1OperatorConfig(), batch=b)
        ok = "OK" if params_mem + act <= args.budget_gb else "OVER"
        print(
            f"activation est batch={b}: {act:.2f} GB "
            f"(params {params_mem:.2f} + acts {act:.2f} <= {args.budget_gb:.1f} GB) [{ok}]"
        )

    print(
        "note: GPU numbers are analytic estimates from FLOPs; "
        "run the real measurement when the training line releases the GPU."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
