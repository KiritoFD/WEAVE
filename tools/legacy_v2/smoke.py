"""CPU smoke：小分辨率、几十步、生成器/判别器交替更新。

输出参数量、FLOPs（torch FlopCounterMode）、autograd 保存量（显存估算的代理）
与峰值 RSS。默认不碰 GPU。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from .config import V2Config
from .discriminator import ConditionalPatchDiscriminator
from .model import V2Generator, count_parameters
from .objective import V2DistributionObjective


def _flops(module: torch.nn.Module, args: tuple, kwargs: dict) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    was_training = module.training
    module.eval()
    try:
        with FlopCounterMode(display=False) as counter:
            with torch.no_grad():
                module(*args, **kwargs)
        return int(counter.get_total_flops())
    finally:
        module.train(was_training)


def _measure_activations(compute_loss) -> int:
    """saved_tensors_hooks 必须包住前向+反向，才能统计 autograd 保存量。"""
    saved: list[int] = []

    def pack(t: torch.Tensor):
        saved.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = compute_loss()
        loss.backward(retain_graph=False)
    return int(sum(saved))


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return float("nan")


def run(steps: int, size: int, batch: int, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = V2Config(enabled=True, variant="V3", batch_size=batch)
    # smoke 用缩小通道，保证 CPU 也能几十步跑完；参数/FLOPs 另用默认配置报告
    smoke_cfg = replace(cfg, channels=(32, 64, 96, 96), blocks_per_stage=(1, 1, 1, 1), decoder_blocks_per_stage=(1, 1, 1, 1))
    gen = V2Generator(smoke_cfg, latent_channels=4, num_styles=5)
    objective = V2DistributionObjective(smoke_cfg, latent_channels=4, num_styles=5)
    opt_g = torch.optim.Adam(
        list(gen.parameters()) + list(objective.nce_encoder.parameters()) + list(objective.nce.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(objective.discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))

    content = torch.randn(batch, 4, size, size) * 0.2
    target = torch.randn(batch, 4, size, size) * 0.2
    ids = torch.arange(batch) % 5
    src_ids = (ids + 1) % 5
    identity = torch.zeros(batch, dtype=torch.bool)

    out = gen(content, ids)
    assert out["z_hat"].shape == content.shape, "generator must preserve latent shape"
    assert out["delta_ll"].shape[1] == 4 and out["delta_lh"].shape[1] == 4
    assert out["delta_hh"].shape[1] == 4, "HH must stay open"
    d_logits = objective.discriminator(out["z_hat"], ids)
    assert d_logits and all(item.dim() == 4 for item in d_logits)

    t0 = time.time()
    for step in range(1, steps + 1):
        with torch.no_grad():
            fake_d = gen(content, ids)["z_hat"]
        d = objective.compute_discriminator_loss(
            content, target, ids, fake_d, source_style_id=src_ids, conditioning={"identity_mask": identity}
        )
        opt_d.zero_grad(set_to_none=True)
        d["loss"].backward()
        opt_d.step()

        g = objective.compute_generator_loss(
            gen, content, target, ids, source_style_id=src_ids, conditioning={"identity_mask": identity}
        )
        opt_g.zero_grad(set_to_none=True)
        g["loss"].backward()
        opt_g.step()
        if step in {1, 10, 20, 30, steps}:
            print(
                f"step {step:3d} | G {float(g['loss'].detach()):.4f} adv {float(g['loss_adv']):.4f} "
                f"nce {float(g['loss_patchnce']):.4f} geo {float(g['loss_geometry']):.4f} "
                f"id {float(g['loss_identity']):.4f} | D {float(d['loss'].detach()):.4f}",
                flush=True,
            )

    # 默认配置与 10M 档的参数/FLOPs
    report: dict[str, float | int] = {}
    for label, rc in (("default", cfg), ("wide_10m", replace(cfg, channels=(96, 192, 256, 256), blocks_per_stage=(1, 1, 2, 1), decoder_blocks_per_stage=(1, 1, 1, 1)))):
        g_full = V2Generator(rc, latent_channels=4, num_styles=5)
        d_full = ConditionalPatchDiscriminator(rc, latent_channels=4, num_styles=5)
        obj_full = V2DistributionObjective(rc, latent_channels=4, num_styles=5)
        report[f"{label}_gen_params"] = count_parameters(g_full)
        report[f"{label}_disc_params"] = count_parameters(d_full)
        report[f"{label}_nce_params"] = count_parameters(obj_full.nce_encoder) + count_parameters(obj_full.nce)
        x64 = torch.randn(1, 4, 64, 64)
        report[f"{label}_gen_flops_64"] = _flops(g_full, (x64, torch.zeros(1, dtype=torch.long)), {})
        report[f"{label}_disc_flops_64"] = _flops(d_full, (x64, torch.zeros(1, dtype=torch.long)), {})

    # autograd 保存量（fp32）作为训练显存代理；默认配置 batch=2，按 batch 线性外推
    b_mem = 2
    g_mem = V2Generator(cfg, latent_channels=4, num_styles=5)
    obj_mem = V2DistributionObjective(cfg, latent_channels=4, num_styles=5)
    xb = torch.randn(b_mem, 4, 64, 64) * 0.2
    tb = torch.randn(b_mem, 4, 64, 64) * 0.2
    ids8 = torch.arange(b_mem) % 5
    saved = _measure_activations(
        lambda: obj_mem.compute_generator_loss(
            g_mem, xb, tb, ids8, source_style_id=(ids8 + 1) % 5
        )["loss"]
    )
    params_bytes = (
        count_parameters(g_mem) + count_parameters(obj_mem.nce_encoder) + count_parameters(obj_mem.nce)
        + count_parameters(obj_mem.discriminator)
    ) * 4
    per_image_mb = saved / 1024 ** 2 / b_mem
    report["activation_mb_per_image_64"] = per_image_mb
    report["params_total_mb_fp32"] = params_bytes / 1024 ** 2
    # 权重+梯度+Adam 状态 ≈ 3×参数；激活按 batch 线性
    report["est_b8_train_mb_fp32"] = params_bytes * 3 / 1024 ** 2 + per_image_mb * 8
    report["est_b14_train_mb_fp32"] = params_bytes * 3 / 1024 ** 2 + per_image_mb * 14
    report["smoke_rss_mb"] = _rss_mb()
    report["smoke_seconds"] = time.time() - t0
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WEAVE-v2 CPU smoke")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    rep = run(a.steps, a.size, a.batch, a.seed)
    print("\n=== report ===")
    for k, v in rep.items():
        if isinstance(v, float):
            print(f"{k:32s} {v:.3f}")
        else:
            print(f"{k:32s} {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
