"""V15 training: variable-capacity U-Net with configurable context and where
supervision. Supports all 16 main-matrix arms (plan §4).

Each training run:
  - 12,000 real optimizer updates, effective batch 128
  - Flow matching on real painting latents (V6 objective)
  - + optional Where self-supervision losses
  - 2 seeds per arm

Usage:
  python -m exp.v15.train --base-dim 64 --use-attention --use-where --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, TensorDataset

from .model import V15UNet
from .where_losses import where_losses

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v15"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
POOL = REPO / "weights_local/sd_pool_orig"


def _h10(pid):
    import hashlib
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10


def load_training_latents(device="cpu", n_per_style=None):
    """Load all train-split latents, balanced sampling."""
    latents, labels, pids = [], [], []
    g = torch.Generator().manual_seed(0)
    for si, s in enumerate(STYLES):
        d = torch.load(POOL / f"{si:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        tr = [i for i, pid in enumerate(d["ids"]) if _h10(str(pid)) < 9]
        take = tr
        if n_per_style:
            take = torch.randperm(len(tr), generator=g)[:n_per_style].tolist()
        latents.append(d["latents"][torch.tensor(tr, dtype=torch.long)[take]])
        labels.extend([si] * len(take))
        pids.extend([str(d["ids"][i]) for i in take])
    z = torch.cat(latents).float()
    y = torch.tensor(labels)
    return z, y


def fm_loss(model, z_real, style_id, tau, p_drop=0.15):
    """Flow matching: h_tau = (1-tau)*eps + tau*h_real, v* = h_real - eps."""
    B = z_real.shape[0]
    eps = torch.randn_like(z_real)
    ta = tau.view(-1, 1, 1, 1)
    z_tau = (1 - ta) * eps + ta * z_real
    v_star = z_real - eps
    drop = torch.rand(B, device=device) < p_drop
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        v_cond = model(z_tau, style_id, tau, unconditional=False).float()
    if drop.any():
        idx = drop.nonzero(as_tuple=True)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            v_unc = model(z_tau[idx], style_id[idx], tau[idx],
                          unconditional=True).float()
        v_cond[idx] = v_unc
    return Fn.mse_loss(v_cond, v_star)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dim", type=int, default=64)
    ap.add_argument("--use-attention", action="store_true")
    ap.add_argument("--use-where", action="store_true")
    ap.add_argument("--style-mode", default="adaln",
                    choices=["adaln", "spatial_film", "router"])
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=12000)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)  # eff batch 128
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-m", type=float, default=0.1)
    ap.add_argument("--lambda-a", type=float, default=0.05)
    ap.add_argument("--lambda-i", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    torch.manual_seed(args.seed)
    arm_name = f"W_base{args.base_dim}_att{int(args.use_attention)}_where{int(args.use_when if hasattr(args,'use_when') else int(args.use_where))}_seed{args.seed}"
    out_dir = OUT / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config = vars(args).copy()
    config["arm_name"] = arm_name
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))

    model = V15UNet(base_dim=args.base_dim, use_attention=args.use_attention,
                    style_mode=args.style_mode, n_experts=args.n_experts).to(device)
    n_params = model.n_params()
    print(f"[v15] {arm_name}: {n_params/1e6:.2f}M params", flush=True)
    (out_dir / "params.txt").write_text(str(n_params))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.updates, pct_start=0.05)

    # load training data (all train-split latents)
    z_all, y_all = load_training_latents()
    dataset = TensorDataset(z_all, y_all)
    loader = DataLoader(dataset, batch_size=args.micro, shuffle=True,
                        drop_last=True, num_workers=0, pin_memory=True)

    # save fixed noise for reproducible eval
    eval_noise = torch.randn(64, 4, 64, 64, generator=torch.Generator().manual_seed(999))
    torch.save(eval_noise, out_dir / "eval_noise.pt")

    curve = out_dir / "train_curves.jsonl"
    opt_step = 0
    micro_in_accum = 0
    running_loss = 0.0
    t0 = time.time()
    done = False

    while not done:
        for z_real, sid in loader:
            z_real = z_real.to(device)
            sid = sid.to(device)
            tau = torch.rand(z_real.shape[0], device=device)

            # V6 flow matching objective
            loss_fm = fm_loss(model, z_real, sid, tau)

            # Where self-supervision (if enabled)
            if args.use_where:
                loss_where, w_parts = where_losses(
                    model, z_real, sid, tau,
                    lambda_m=args.lambda_m, lambda_a=args.lambda_a,
                    lambda_i=args.lambda_i)
                loss = loss_fm + loss_where
            else:
                loss = loss_fm

            (loss / args.accum).backward()
            running_loss += loss.item()
            micro_in_accum += 1

            if micro_in_accum % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % 200 == 0:
                    rec = {"opt_step": opt_step,
                           "loss": round(running_loss / (args.accum * 200), 4),
                           "sec": round(time.time() - t0, 1)}
                    with curve.open("a") as fp:
                        fp.write(json.dumps(rec) + "\n")
                    print(f"[{arm_name}] {opt_step}/{args.updates} "
                          f"loss={rec['loss']} {rec['sec']}s", flush=True)
                    running_loss = 0.0

                if opt_step % 3000 == 0 or opt_step == args.updates:
                    torch.save({"model": model.state_dict(),
                                "opt": opt.state_dict(),
                                "sched": sched.state_dict(),
                                "opt_step": opt_step,
                                "config": config},
                               out_dir / f"ckpt_{opt_step}.pt")
                    print(f"[ckpt] {opt_step} saved", flush=True)

                if opt_step >= args.updates:
                    done = True
                    break

        if done:
            break

    # save final model
    torch.save({"model": model.state_dict(), "opt_step": opt_step,
                "config": config}, out_dir / "final.pt")
    print(f"[{arm_name}] DONE {opt_step} updates, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
