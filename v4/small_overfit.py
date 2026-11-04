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
from .diffusion import Schedule, pred_to_x0_eps, training_loss, v_target
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


def real_subset(cfg: V4Config, per_style: int,
                indices: str = "runs/v4/arms/B/indices.pt") -> tuple[torch.Tensor, torch.Tensor]:
    bank = load_bank(cfg.packed_dir)
    idx = torch.load(indices, map_location="cpu", weights_only=False)
    tr = idx["train_idx"].tolist()
    lat, sid = [], []
    for s in range(len(STYLE_NAMES)):
        take = [i for i in tr if int(bank.style_ids[i]) == s][:per_style]
        lat.append(bank.latents[take])
        sid.append(bank.style_ids[take])
    return torch.cat(lat), torch.cat(sid)


@torch.no_grad()
def evaluate(model, cfg, sched, lat, sid, *, reps=8, seed=99, device="cuda",
             bins=6, swap_strength=0.45):
    """分桶比较"真 / 空 / 错 ID"恢复 x0 的误差，并做固定噪声换 ID 的幅度检查。

    误差统一换算到 x0 空间，eps 与 v 臂可横比（"恢复真实目标"的语义就是 x0）。
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    n = lat.shape[0]
    wrong = (sid + 1) % len(STYLE_NAMES)
    neg = torch.full_like(sid, -1)
    t = torch.randint(0, cfg.timesteps, (reps, n), device=device, generator=gen)
    noise = torch.randn((reps, n, *lat.shape[1:]), device=device, dtype=lat.dtype, generator=gen)
    err = {k: [] for k in ("true", "null", "wrong")}
    out_rel = {"null": [], "wrong": []}
    for r in range(reps):
        zt = sched.q_sample(lat, t[r], noise)
        x0 = {"true": None, "null": None, "wrong": None}
        for key, cond in (("true", sid), ("null", neg), ("wrong", wrong)):
            pred = model(zt, t[r].float(), cond)
            x0[key], _ = pred_to_x0_eps(pred.float(), zt, t[r], sched, cfg.prediction_type)
            err[key].append(((x0[key] - lat) ** 2).flatten(1).mean(1))
        for key in ("null", "wrong"):
            out_rel[key].append(((x0["true"] - x0[key]) ** 2).flatten(1).mean(1).sqrt()
                                / x0["true"].flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8))
    err = {k: torch.stack(v, 0) for k, v in err.items()}
    rel = {k: torch.stack(v, 0) for k, v in out_rel.items()}
    t_np = t.cpu().numpy()
    edges = [k * cfg.timesteps // bins for k in range(bins + 1)]
    buckets = []
    for k in range(bins):
        mask_np = (t_np >= edges[k]) & (t_np < edges[k + 1])
        if not mask_np.any():
            continue
        mask = torch.from_numpy(mask_np).to(device)
        row = {"bucket": k, "t_lo": edges[k], "t_hi": edges[k + 1] - 1,
               "L_true": round(float(err["true"][mask].mean()), 5),
               "out_rel_wrong": round(float(rel["wrong"][mask].mean()), 5)}
        for key in ("null", "wrong"):
            d = (err[key][mask] - err["true"][mask]).reshape(-1)
            row[f"d_{key}"] = round(float(d.mean()), 6)
            row[f"rel_{key}_pct"] = round(100 * float(d.mean()) /
                                           max(float(err["true"][mask].mean()), 1e-8), 3)
            row[f"win_{key}"] = round(float((d > 0).float().mean()), 3)
            row[f"rep_means_{key}"] = [round(float((err[key][i][mask[i]] -
                                                     err["true"][i][mask[i]]).mean()), 6)
                                       for i in range(reps) if bool(mask[i].any())]
        buckets.append(row)

    # 换 ID 幅度：固定 z0/噪声/t，只换目标 ID（此处用中噪声 t=500 与 SDEdit 强度两种口径）
    g2 = torch.Generator(device=device).manual_seed(seed + 1)
    t_swap = torch.full((n,), 500, device=device, dtype=torch.long)
    noise2 = torch.randn(lat.shape, device=device, dtype=lat.dtype, generator=g2)
    zt = sched.q_sample(lat, t_swap, noise2)
    xs = []
    for s in range(cfg.num_styles):
        pred = model(zt, t_swap.float(), torch.full_like(sid, s))
        x0, _ = pred_to_x0_eps(pred.float(), zt, t_swap, sched, cfg.prediction_type)
        xs.append(x0)
    xs = torch.stack(xs, 0)
    pair = []
    for a in range(cfg.num_styles):
        for b in range(a + 1, cfg.num_styles):
            pair.append(float((xs[a] - xs[b]).pow(2).mean().sqrt()))
    swap = {"t": 500, "pairwise_x0_rms": round(float(np.mean(pair)), 5),
            "to_true_x0_rms": round(float(np.mean(
                [(xs[s] - lat).pow(2).mean().sqrt() for s in range(cfg.num_styles)])), 5)}

    eff_buckets = [r["bucket"] for r in buckets
                   if r["rel_wrong_pct"] >= 1.0 and r["win_wrong"] >= 0.75
                   and all(v > 0 for v in r["rep_means_wrong"])]
    model.train()
    return {"buckets": buckets, "swap": swap,
            "verdict": {"effective_buckets": eff_buckets,
                        "style_id_effective": bool(eff_buckets) or swap["pairwise_x0_rms"] >= 0.02,
                        "criterion": "ΔL_wrong 相对≥1%、胜率≥0.75、所有 rep 同号；或换 ID 的 x0 RMS≥0.02"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth"], required=True)
    ap.add_argument("--per-style", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--film-init-std", type=float, default=0.0)
    ap.add_argument("--cfg-dropout", type=float, default=0.0)
    ap.add_argument("--prediction-type", choices=["eps", "v"], default="eps")
    ap.add_argument("--cond-mode", choices=["sum", "split"], default="sum")
    ap.add_argument("--style-film-init-std", type=float, default=0.05)
    ap.add_argument("--indices", default="runs/v4/arms/B/indices.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = V4Config(batch_size=args.batch, grad_accum=1, lr=args.lr, max_steps=args.steps,
                   cfg_dropout=args.cfg_dropout, film_init_std=args.film_init_std,
                   prediction_type=args.prediction_type, cond_mode=args.cond_mode,
                   style_film_init_std=args.style_film_init_std,
                   warmup_steps=100, milestones=(), early_stop_after=10**9)
    cfg.validate()
    torch.manual_seed(cfg.seed)
    sched = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=args.device)

    if args.mode == "real":
        lat, sid = real_subset(cfg, args.per_style, args.indices)
    else:
        lat, sid = synth_bank(cfg, args.per_style)
    lat, sid = lat.to(args.device), sid.to(args.device)
    print(f"[overfit] mode={args.mode} n={lat.shape[0]} per_style={args.per_style} "
          f"pred={cfg.prediction_type} cond={cfg.cond_mode} "
          f"params={count_parameters(V4UNet(cfg))/1e6:.2f}M", flush=True)

    model = V4UNet(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    gen = torch.Generator(device=args.device).manual_seed(cfg.seed)
    out = {"mode": args.mode, "n": int(lat.shape[0]), "per_style": args.per_style,
           "steps": args.steps, "film_init_std": args.film_init_std,
           "prediction_type": cfg.prediction_type, "cond_mode": cfg.cond_mode,
           "cfg_dropout": args.cfg_dropout, "curve": []}
    t0 = __import__("time").perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, lat.shape[0], (cfg.batch_size,), device=args.device, generator=gen)
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=cfg.use_bf16):
            loss, _ = training_loss(model, lat.index_select(0, idx), sid.index_select(0, idx),
                                    sched, cfg_dropout=cfg.cfg_dropout,
                                    prediction_type=cfg.prediction_type, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            ev = evaluate(model, cfg, sched, lat, sid, device=args.device)
            mid = next((r for r in ev["buckets"] if r["bucket"] == 3), ev["buckets"][len(ev["buckets"]) // 2])
            out["curve"].append({"step": step, "loss": round(float(loss.detach()), 4), **ev})
            print(f"[overfit] step={step} loss={float(loss.detach()):.4f} "
                  f"L_true={mid['L_true']:.4f} d_null={mid['d_null']:+.5f} "
                  f"({mid['rel_null_pct']:+.2f}%) d_wrong={mid['d_wrong']:+.5f} "
                  f"({mid['rel_wrong_pct']:+.2f}%) swap_rms={ev['swap']['pairwise_x0_rms']:.4f} "
                  f"pass={ev['verdict']['style_id_effective']}", flush=True)

    grad_style = float(model.style_embed.weight.grad.norm()) if model.style_embed.weight.grad is not None else 0.0
    final = evaluate(model, cfg, sched, lat, sid, reps=16, device=args.device)
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

