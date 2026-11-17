"""V15 post-training on 4070 Laptop (drop-in for missing ZIP script).

Modes (docs/v15/WEAVE_V15_POSTTRAIN_4070LP.md §1):
  balance : balanced 5-style flow-matching continuation from --source
  band    : balance + Haar-band weighted velocity error (--w-h2/--w-h1)
  guided  : critic-guided cross-style edit finetune (R1 frozen)
            unrolls k-steps from --tau0 to 1; losses:
              style   = -target_margin(critic, z_pred, t)      * lambda_style
              content = MSE(z_pred, z_src)                     * lambda_content
              delta   = MSE(z_pred, z_pred_frozen_src)         * lambda_delta
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, TensorDataset

from exp.v15.model import V15UNet
from exp.v15.train import load_training_latents, fm_loss, STYLES
from exp.v8.guidance import target_margin
from exp.v11b.bands import haar2d

REPO = Path(__file__).resolve().parents[2]


def band_velocity_loss(model, z_real, sid, tau, w_h2, w_h1, p_drop=0.15):
    """FM loss + extra Haar-band weighted error on (v_pred - v*)."""
    device = z_real.device
    B = z_real.shape[0]
    eps = torch.randn_like(z_real)
    ta = tau.view(-1, 1, 1, 1)
    z_tau = (1 - ta) * eps + ta * z_real
    v_star = z_real - eps
    drop = torch.rand(B, device=device) < p_drop
    with torch.autocast(device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        v = model(z_tau, sid, tau, unconditional=False).float()
    if drop.any():
        idx = drop.nonzero(as_tuple=True)[0]
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            v[idx] = model(z_tau[idx], sid[idx], tau[idx],
                           unconditional=True).float()
    e = v - v_star
    base = e.square().mean()
    ll1, h1 = haar2d(e)                    # h1: (B,4,3,32,32)
    ll2, h2 = haar2d(ll1)                  # h2: (B,4,3,16,16)
    h1f = h1.flatten(1, 2)                  # (B,12,32,32)
    h2f = h2.flatten(1, 2)                  # (B,12,16,16)
    band = w_h2 * h2f.square().mean() + w_h1 * h1f.square().mean()
    return base + band


def guided_loss(model, frozen, z_src, sid_src, tid, tau0, k_steps, critic,
                lam_s, lam_c, lam_d):
    """Unroll k Euler steps tau0->1 on partial-noise latents; critic-guided."""
    dev = z_src.device
    B = z_src.shape[0]
    eps = torch.randn_like(z_src)
    z = (1 - tau0) * eps + tau0 * z_src

    def unroll(m):
        zz = z
        span = (1 - tau0) / k_steps
        for j in range(k_steps):
            t = tau0 + span * j
            tt = torch.full((B,), t, device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16,
                                enabled=dev.type == "cuda"):
                v = m(zz, tid, tt).float()
            zz = zz + v * span
        return zz

    z_pred = unroll(model)
    with torch.no_grad():
        z_frozen = unroll(frozen)
    l_style = -target_margin(critic, z_pred.float(), tid).mean()
    l_content = (z_pred - z_src).square().mean()
    l_delta = (z_pred - z_frozen).square().mean()
    return (lam_s * l_style + lam_c * l_content + lam_d * l_delta,
            {"style": l_style.item(), "content": l_content.item(),
             "delta": l_delta.item()})


def balanced_batches(z, y, micro, rng):
    """Yield indices: micro/5 per style, cycling styles each batch."""
    by_style = {s: (y == s).nonzero(as_tuple=True)[0] for s in range(5)}
    ptr = {s: 0 for s in range(5)}
    pos = {s: torch.randperm(len(by_style[s]), generator=rng).tolist()
           for s in range(5)}
    ptr = {s: 0 for s in range(5)}
    while True:
        idx = []
        for s in range(5):
            take = max(1, micro // 5)
            ids = by_style[s][torch.tensor(
                [pos[s][(ptr[s] + j) % len(pos[s])] for j in range(take)])]
            ptr[s] += take
            idx.append(ids)
        yield torch.cat(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["balance", "band", "guided"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--critic", default=str(REPO / "runs/v6_next/probes/r1_r1.pt"))
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--w-h2", type=float, default=0.75)
    ap.add_argument("--w-h1", type=float, default=0.25)
    ap.add_argument("--k-steps", type=int, default=2)
    ap.add_argument("--tau0", type=float, default=0.7)
    ap.add_argument("--lambda-style", type=float, default=0.10)
    ap.add_argument("--lambda-content", type=float, default=0.20)
    ap.add_argument("--lambda-delta", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.source, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = V15UNet(base_dim=int(cfg["base_dim"]),
                    use_attention=bool(cfg["use_attention"]),
                    style_mode=str(cfg.get("style_mode", "adaln")),
                    n_experts=int(cfg.get("n_experts", 4))).to(dev)
    model.load_state_dict(ck["model"])
    frozen = None
    critic = None
    if a.mode == "guided":
        frozen = V15UNet(base_dim=int(cfg["base_dim"]),
                         use_attention=bool(cfg["use_attention"]),
                         style_mode=str(cfg.get("style_mode", "adaln")),
                         n_experts=int(cfg.get("n_experts", 4))).to(dev)
        frozen.load_state_dict(ck["model"])
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        from src.infer import load_critic
        critic = load_critic(dev)
        for p in critic.parameters():
            p.requires_grad_(False)

    z_all, y_all, counts = load_training_latents(a.pool, "cpu")
    print(f"pool {z_all.shape[0]} {counts}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.updates)
    rng = torch.Generator().manual_seed(a.seed)
    batch_iter = balanced_batches(z_all, y_all, a.micro, rng)

    curve = out / "train_curves.jsonl"
    opt_step = micro_in = 0
    run_loss = 0.0
    t0 = time.time()
    peak = 0
    while opt_step < a.updates:
        idx = next(batch_iter)
        z_real = z_all[idx].to(dev)
        sid = y_all[idx].to(dev)
        tau = torch.rand(len(idx), device=dev)
        if a.mode == "guided":
            # random cross-style target != source
            tid = (sid + 1 + torch.randint(0, 4, (len(idx),),
                                           device=dev)) % 5
            loss, parts = guided_loss(model, frozen, z_real, sid, tid,
                                      a.tau0, a.k_steps, critic,
                                      a.lambda_style, a.lambda_content,
                                      a.lambda_delta)
            loss = loss / a.accum
        elif a.mode == "band":
            loss = band_velocity_loss(model, z_real, sid, tau,
                                      a.w_h2, a.w_h1) / a.accum
        else:
            loss = fm_loss(model, z_real, sid, tau) / a.accum
        loss.backward()
        run_loss += loss.item() * a.accum
        micro_in += 1
        if micro_in % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            peak = max(peak, torch.cuda.memory_allocated() // 2**20)
            if opt_step % 100 == 0:
                rec = {"opt_step": opt_step,
                       "loss": round(run_loss / (a.accum * 100), 5),
                       "sec": round(time.time() - t0, 1),
                       "peak_mb": peak}
                if a.mode == "guided":
                    rec.update({k: round(v, 4) for k, v in parts.items()})
                with curve.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                print(f"[{a.mode}] {opt_step}/{a.updates} "
                      f"loss={rec['loss']} {rec['sec']}s peak={peak}MB",
                      flush=True)
                run_loss = 0.0
        if opt_step >= a.updates:
            break
    torch.save({"model": model.state_dict(), "opt_step": opt_step,
                "config": {**cfg, "posttrain_mode": a.mode,
                           "source": str(a.source), "lr": a.lr,
                           "updates": a.updates}},
               out / "final.pt")
    print(f"DONE {opt_step} updates {time.time()-t0:.0f}s peak={peak}MB",
          flush=True)


if __name__ == "__main__":
    main()
