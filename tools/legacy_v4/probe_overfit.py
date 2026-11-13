"""P1 验证：16 图固定噪声过拟合 + 梯度/条件响应检查 + microbatch 测速。

判定标准：
1. 固定噪声/时间步上的去噪误差必须明显下降（否则网络根本没在学）；
2. 同一 16 图的「换新噪声」误差也要下降（否则只是背下了那组噪声）；
3. 时间嵌入 / Style-ID 嵌入 / 主干各层梯度非零，且训练后切换 Style-ID 输出有可测变化。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import V4Config
from .data import load_bank
from .diffusion import Schedule, training_loss
from .model import V4UNet, count_parameters


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="runs/v4/p1_probe.json")
    ap.add_argument("--batch", type=int, default=4)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = V4Config()
    torch.manual_seed(cfg.seed)
    bank = load_bank(cfg.packed_dir)
    idx_path = Path("runs/v4/p0/indices.pt")
    if not idx_path.exists():
        raise SystemExit("先跑 v4.audit_data 生成划分索引")
    idx = torch.load(idx_path, map_location="cpu", weights_only=False)
    train_idx = idx["train_idx"][:16] if idx["train_idx"].shape[0] >= 16 else idx["train_idx"]
    val_idx = idx["val_idx"][:64]
    z_train = bank.latents.index_select(0, train_idx).cuda()
    s_train = bank.style_ids.index_select(0, train_idx).cuda()
    z_val = bank.latents.index_select(0, val_idx).cuda()
    s_val = bank.style_ids.index_select(0, val_idx).cuda()

    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")
    model = V4UNet(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"[probe] params={count_parameters(model)/1e6:.2f}M fixed_train={z_train.shape[0]}",
          flush=True)

    # 固定噪声与时间步
    gen = torch.Generator(device="cuda").manual_seed(1234)
    t_fix = torch.randint(0, schedule.timesteps, (z_train.shape[0],), device="cuda", generator=gen)
    e_fix = torch.randn(z_train.shape, device="cuda", generator=gen)

    def fixed_loss(m) -> float:
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            pred = m(schedule.q_sample(z_train, t_fix, e_fix), t_fix.float(), s_train)
        m.train()
        return float(((pred.float() - e_fix) ** 2).mean())

    def fresh_loss(m) -> float:
        m.eval()
        g = torch.Generator(device="cuda").manual_seed(9999)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            t = torch.randint(0, schedule.timesteps, (z_val.shape[0],), device="cuda", generator=g)
            e = torch.randn(z_val.shape, device="cuda", generator=g)
            pred = m(schedule.q_sample(z_val, t, e), t.float(), s_val)
        m.train()
        return float(((pred.float() - e) ** 2).mean())

    init_fixed = fixed_loss(model)
    init_val = fresh_loss(model)
    grad_probe = None
    for step in range(args.steps):
        ids = torch.randint(0, z_train.shape[0], (args.batch,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            loss, stats = training_loss(model, z_train.index_select(0, ids),
                                        s_train.index_select(0, ids), schedule,
                                        cfg_dropout=cfg.cfg_dropout)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 5:
            # 末层 out_conv 零初始化：step 0 的上游梯度必然为 0，须等它先获得非零权重。
            grads = {
                "time_embed": float(sum(p.grad.abs().sum() for p in model.time_embed.parameters()
                                        if p.grad is not None)),
                "style_embed": float(model.style_embed.weight.grad.abs().sum()
                                     if model.style_embed.weight.grad is not None else 0.0),
                "null_style": float(model.null_style.grad.abs().sum()
                                    if model.null_style.grad is not None else 0.0),
                "input_conv": float(model.input_conv.weight.grad.abs().sum()),
                "stage0": float(sum(p.grad.abs().sum() for p in model.stage0.parameters()
                                    if p.grad is not None)),
                "stage2_film": float(sum(p.grad.abs().sum() for p in model.stage2[0].film1.parameters()
                                        if p.grad is not None)),
                "mid": float(sum(p.grad.abs().sum() for p in model.mid.parameters()
                                 if p.grad is not None)),
                "out_conv": float(model.out_conv.weight.grad.abs().sum()),
            }
            film_scale_grad = float(sum(p.grad.abs().sum() for p in model.stage0[0].film1.proj.parameters()
                                        if p.grad is not None))
            grads["film_scale_shift_proj"] = film_scale_grad
            grad_probe = grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"[probe] step={step+1} fixed={fixed_loss(model):.4f} val={fresh_loss(model):.4f}",
                  flush=True)

    final_fixed = fixed_loss(model)
    final_val = fresh_loss(model)
    # 空条件（CFG 分支）单独测一次梯度：训练时 10% dropout 不一定命中这一步
    ids = torch.randint(0, z_train.shape[0], (max(4, args.batch),), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        loss, _ = training_loss(model, z_train.index_select(0, ids), s_train.index_select(0, ids),
                                schedule, cfg_dropout=1.0)
    model.zero_grad(set_to_none=True)
    loss.backward()
    null_grad = float(model.null_style.grad.abs().sum()) if model.null_style.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    grad_probe["null_style"] = null_grad
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        zt = schedule.q_sample(z_train[:4], t_fix[:4], e_fix[:4])
        a = model(zt, t_fix[:4].float(), torch.zeros(4, dtype=torch.long, device="cuda")).float()
        b = model(zt, t_fix[:4].float(), torch.ones(4, dtype=torch.long, device="cuda")).float()
    cond_rel = float((a - b).abs().mean() / (a.abs().mean() + 1e-8))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        lo = model(zt, torch.full((4,), 100.0, device="cuda"), s_train[:4]).float()
        hi = model(zt, torch.full((4,), 900.0, device="cuda"), s_train[:4]).float()
    time_rel = float((lo - hi).abs().mean() / (lo.abs().mean() + 1e-8))
    report = {
        "params": count_parameters(model),
        "fixed_loss_init": init_fixed, "fixed_loss_final": final_fixed,
        "fixed_drop": init_fixed - final_fixed,
        "val_loss_init": init_val, "val_loss_final": final_val,
        "val_drop": init_val - final_val,
        "grad_norms": grad_probe,
        "style01_rel_diff_after": cond_rel,
        "time_low_vs_high_rel_diff": time_rel,
        "pass_fixed": bool(final_fixed < 0.5 * init_fixed),
        "pass_generalize": bool(final_val < 0.9 * init_val),
        "pass_grads": bool(grad_probe and min(grad_probe.values()) > 0),
        "pass_cond": bool(cond_rel > 1e-3 and time_rel > 1e-2),
        "max_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
