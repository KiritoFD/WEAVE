"""路线 B 的 CPU smoke：合成数据上跑完阶段 1 -> 筛选 -> 阶段 2 端点 -> 阶段 3 结构测试。

用法：
    python tools/smoke_route_b.py --steps 40 --device cpu            # small preset
    python tools/smoke_route_b.py --preset full --steps 0            # 只打印参数量/前向

该脚本只使用合成 latent，不加载数据集 / VAE / DINO / VGG，也不访问 GPU/远端。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b1_style_operator import B1OperatorConfig, B1StyleOperator  # noqa: E402
from b1_style_operator import reference_permutation_probe  # noqa: E402
from b2_destyle import DestyleConfig, destyle_latent  # noqa: E402
from b_pseudo_targets import BPseudoTargetProvider, ProviderConfig  # noqa: E402


SHAPE = (4, 4, 16, 16)
STYLES = [
    (2.0, 1.0),
    (-0.8, 0.3),
]


def small_config() -> B1OperatorConfig:
    return B1OperatorConfig(
        latent_channels=4,
        widths=(32, 64, 96),
        encoder_blocks=(1, 1, 1),
        attention_heads=(2, 4, 4),
        attention_layers=(1, 1, 1),
        token_side=8,
        decoder_hidden=32,
        num_directions=4,
        num_brush_scales=2,
        direction_kernel=3,
        block_kernel=3,
        style_gate_init=0.3,
        patch_size=4,
        patch_stride=4,
    )


def full_config() -> B1OperatorConfig:
    return B1OperatorConfig()


def _style_latent(base: torch.Tensor, style_id: int) -> torch.Tensor:
    scale, shift = STYLES[style_id]
    return base * scale + shift


def _make_paired(style_id: int, gen: torch.Generator) -> dict:
    base_target = torch.randn(SHAPE, generator=gen)
    base_ref = torch.randn(SHAPE, generator=gen)
    target = _style_latent(base_target, style_id)
    reference = _style_latent(base_ref, style_id)
    content = destyle_latent(
        target,
        DestyleConfig(clamp_range=None),
        torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen))),
    )
    return {
        "content_latent": content,
        "reference_latent": reference,
        "target_latent": target,
    }


def _make_unpaired(style_id: int, gen: torch.Generator) -> dict:
    base_content = torch.randn(SHAPE, generator=gen) * 2.5
    base_ref = torch.randn(SHAPE, generator=gen)
    return {
        "content_latent": base_content,
        "reference_latent": _style_latent(base_ref, style_id),
    }


def run_full_preset(device: torch.device) -> int:
    torch.manual_seed(0)
    op = B1StyleOperator(full_config()).to(device).eval()
    n = sum(p.numel() for p in op.parameters() if p.requires_grad)
    z = torch.randn(1, 4, 64, 64, device=device)
    r = torch.randn(1, 4, 64, 64, device=device)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = op(z, r, return_attention=True)
        dt = time.perf_counter() - t0
    print(f"[full] params={n/1e6:.3f}M")
    print(f"[full] forward {dt*1000:.1f} ms | z_star={tuple(out['z_star'].shape)}")
    print(
        "[full] region_params "
        + ", ".join(f"{k}={tuple(v.shape)}" for k, v in out["region_params"].items())
    )
    print(f"[full] attention={tuple(out['attention'].shape)} tokens={out['reference_tokens']}")
    print(f"[full] ll_delta_abs={float(out['ll_delta'].abs().mean()):.4f} "
          f"hf_delta_abs={float(out['hf_delta'].abs().mean()):.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--preset", choices=("small", "full"), default="small")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        raise SystemExit("smoke 默认只允许 CPU；本次任务明确要求不占用 GPU。")

    device = torch.device(args.device)
    if args.preset == "full":
        return run_full_preset(device)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)

    op = B1StyleOperator(small_config()).to(device)
    provider = BPseudoTargetProvider(
        op,
        cfg=ProviderConfig(min_swap_delta_rel=1e-4, max_permutation_delta=1e-4),
    )
    opt = torch.optim.Adam(op.parameters(), lr=args.lr)

    print(f"[stage 1] {args.steps} steps on synthetic pairs (CPU, small preset)")
    op.train()
    for step in range(1, args.steps + 1):
        style_id = step % len(STYLES)
        paired = _make_paired(style_id, gen)
        unpaired = _make_unpaired(style_id, gen)
        logs = provider.stage1_step(
            paired,
            unpaired,
            optimizer=opt,
            grad_clip=1.0,
            unpaired_weight=0.5,
        )
        if step % 10 == 0 or step == 1:
            print(
                f"  step {step:03d} loss={float(logs['loss'].detach()):.4f} "
                f"paired_swd={float(logs['paired/loss_style_swd']):.4f} "
                f"unpaired_swd={float(logs['unpaired/loss_style_swd']):.4f} "
                f"gate={float(op.repaint.renderer.gate_value()):.3f}"
            )

    op.eval()
    with torch.no_grad():
        print("[screen] evaluate z* on both synthetic styles")
        reports = []
        for style_id in range(len(STYLES)):
            paired = _make_paired(style_id, gen)
            repo = provider.evaluate_z_star(
                paired["content_latent"].to(device), paired["reference_latent"].to(device)
            )
            reports.append(repo)
            print(
                f"  style {style_id}: gain={repo['style_gain']:.4f} "
                f"content_keep={repo['content_keep']:.4f} "
                f"ref_copy={repo['reference_copy_corr']:.4f} passed={repo['passed']}"
            )
        provider.mark_screened({"passed": all(r["passed"] for r in reports)})

        print("[stage 2] freeze T_phi and build endpoint")
        provider.freeze_operator()
        paired = _make_paired(0, gen)
        ep = provider.build_fm_endpoint(
            paired["content_latent"].to(device),
            paired["reference_latent"].to(device),
            allow_unscreened=not provider.screen_passed,
        )
        if ep is None:
            print("  endpoint = None（没有参考/离线 style_key；trainer 应回退 target_hf）")
        else:
            print(
                f"  z1 shape={tuple(ep['z1'].shape)} "
                f"velocity_abs_mean={float(ep['velocity'].abs().mean()):.5f} "
                f"stopgrad={not ep['z1'].requires_grad}"
            )

        print("[stage 3] swap / permutation structural checks")
        content = _make_paired(0, gen)["content_latent"]
        ref_a = _make_paired(0, gen)["reference_latent"]
        ref_b = _make_paired(1, gen)["reference_latent"]
        checks = provider.stage3_checks(content.to(device), ref_a.to(device), ref_b.to(device))
        print(
            f"  swap delta_rel={checks['swap']['output_delta_rel']:.5f} "
            f"passed={checks['swap_passed']}"
        )
        print(
            f"  permutation max_delta={checks['permutation']['output_delta_max']:.2e} "
            f"passed={checks['permutation_passed']}"
        )
        probe = reference_permutation_probe(op, content.to(device), ref_b.to(device))
        print(
            f"  standalone permutation max_delta={probe['output_delta_max']:.2e}"
        )

    ok = checks["swap_passed"] and checks["permutation_passed"]
    print(f"[result] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
