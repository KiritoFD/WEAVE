"""小样本过拟合：真实 8 图/风格 vs 人工可区分 5 类 latent。

目的：如果人工数据能学会 Style-ID、真实数据学不会 -> 问题在数据覆盖/标签可辨识性，
而不是网络容量或条件注入。

  python3 -m v4.small_overfit --mode real  --steps 800 --out runs/v4/diag/overfit_real.json
  python3 -m v4.small_overfit --mode synth --steps 800 --out runs/v4/diag/overfit_synth.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import V4Config
from .data import STYLE_NAMES, load_bank
from .diffusion import Schedule, training_loss
from .model import V4UNet, count_parameters


def synth_bank(cfg: V4Config, per_style: int, seed: int = 5):
    """5 类人工可区分分布：每类一个固定均值方向（2σ）+ 类内高斯。"""
    g = torch.Generator().manual_seed(seed)
    shape = (cfg.latent_channels, 64, 64)
    dirs = torch.randn(len(STYLE_NAMES), *shape, generator=g)
    dirs = torch.nn.functional.normalize(dirs.view(len(STYLE_NAMES), -1), dim=1).view_as(dirs)
    lat, sid = [], []
    for s in range(len(STYLE_NAMES)):
        noise = torch.randn(per_style, *shape, generator=g) * 0.88
        lat.append(noise + 2.0 * dirs[s].unsqueeze(0))
        sid.append(torch.full((per_style,), s, dtype=torch.long))
    return torch.cat(lat), torch.cat(sid)


def real_subset(cfg: V4Config, per_style: int) -> tuple[torch.Tensor, torch.Tensor]:
    bank = load_bank(cfg.packed_dir)
    idx = torch.load("runs/v4/prior/indices.pt", map_location="cpu", weights_only=False)
    tr = idx["train_idx"].tolist()
    lat, sid = [], []
    for s in range(len(STYLE_NAMES)):
        take = [i for i in tr if int(bank.style_ids[i]) == s][:per_style]
        lat.append(bank.latents[take])
        sid.append(bank.style_ids[take])
    return torch.cat(lat), torch.cat(sid)


@torch.no_grad()
def evaluate(model, cfg, sched, lat, sid, *, reps=8, seed=99):
    model.eval()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = lat.shape[0]
    wrong = (sid + 1) % len(STYLE_NAMES)
    neg = torch.full_like(sid, -1)
    t = torch.randint(0, cfg.timesteps, (reps, n), device="cuda", generator=gen)
    noise = torch.randn((reps, n, *lat.shape[1:]), device="cuda", dtype=lat.dtype, generator=gen)
    err = {k: [] for k in ("true", "null", "wrong")}
    for r in range(reps):
        zt = sched.q_sample(lat, t[r], noise[r])
        for key, cond in (("true", sid), ("null", neg), ("wrong", wrong)):
            pred = model(zt, t[r].float(), cond)
            err[key].append(((pred.float() - noise[r]) ** 2).flatten(1).mean(1))
    err = {k: torch.stack(v, 0) for k, v in err.items()}
    mid = (t[0] >= 333) & (t[0] < 666)
    out = {"overall": {k: round(float(v.mean()), 5) for k, v in err.items()}}
    for key in ("null", "wrong"):
        d = (err[key] - err["true"]).reshape(-1)
        out[f"d_{key}"] = round(float(d.mean()), 5)
        out[f"d_{key}_pct"] = round(100 * float(d.mean()) / float(err["true"].mean()), 3)
    d_mid = (err["null"][:, mid] - err["true"][:, mid]).reshape(-1)
    out["mid_snr_d_null"] = round(float(d_mid.mean()), 5)
    out["mid_snr_d_null_pct"] = round(100 * float(d_mid.mean()) / float(err["true"][:, mid].mean()), 3)
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth"], required=True)
    ap.add_argument("--per-style", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--film-init-std", type=float, default=0.0)
    ap.add_argument("--cfg-dropout", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = V4Config(batch_size=args.batch, grad_accum=1, lr=args.lr, max_steps=args.steps,
                   cfg_dropout=args.cfg_dropout, film_init_std=args.film_init_std,
                   warmup_steps=100, milestones=(), early_stop_after=10**9)
    cfg.validate()
    torch.manual_seed(cfg.seed)
    sched = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")

    if args.mode == "real":
        lat, sid = real_subset(cfg, args.per_style)
    else:
        lat, sid = synth_bank(cfg, args.per_style)
    lat, sid = lat.cuda(), sid.cuda()
    print(f"[overfit] mode={args.mode} n={lat.shape[0]} per_style={args.per_style} "
          f"params={count_parameters(V4UNet(cfg))/1e6:.2f}M", flush=True)

    model = V4UNet(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)
    out = {"mode": args.mode, "n": int(lat.shape[0]), "per_style": args.per_style,
           "steps": args.steps, "film_init_std": args.film_init_std,
           "cfg_dropout": args.cfg_dropout, "curve": []}
    t0 = __import__("time").perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, lat.shape[0], (cfg.batch_size,), device="cuda", generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            loss, _ = training_loss(model, lat.index_select(0, idx), sid.index_select(0, idx),
                                    sched, cfg_dropout=cfg.cfg_dropout, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            ev = evaluate(model, cfg, sched, lat, sid)
            out["curve"].append({"step": step, "loss": round(float(loss.detach()), 4), **ev})
            print(f"[overfit] step={step} loss={float(loss.detach()):.4f} "
                  f"L_true={ev['overall']['true']:.4f} d_null={ev['d_null']:+.5f} "
                  f"({ev['d_null_pct']:+.2f}%) d_wrong={ev['d_wrong']:+.5f} "
                  f"({ev['d_wrong_pct']:+.2f}%) mid_d_null={ev['mid_snr_d_null_pct']:+.2f}%", flush=True)

    grad_style = float(model.style_embed.weight.grad.norm()) if model.style_embed.weight.grad is not None else 0.0
    final = evaluate(model, cfg, sched, lat, sid, reps=16)
    out["final"] = final
    out["params"] = count_parameters(model)
    out["elapsed_s"] = round(__import__("time").perf_counter() - t0, 1)
    out["style_embed_norm"] = round(float(model.style_embed.weight.norm()), 4)
    out["null_style_norm"] = round(float(model.null_style.norm()), 4)
    out["film_proj_norm_mean"] = round(float(np.mean([
        float(p.norm()) for n, p in model.named_parameters() if "film1.proj" in n or "film2.proj" in n])), 4)
    out["grad_style_embed_last"] = round(grad_style, 6)
    out_path = Path(args.out) if args.out else Path(f"runs/v4/diag/overfit_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("mode", "n", "final", "style_embed_norm",
                                         "film_proj_norm_mean")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

