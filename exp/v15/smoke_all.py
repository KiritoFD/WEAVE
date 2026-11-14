"""V15 smoke matrix: 2 training steps for every variant class, one process.

Run on the 3060: python -m exp.v15.smoke_all --pool <pool>
Covers capacity (16/32/64/96) x context (conv/att) x where (base/where)
plus the Group-B style modes (spatial_film, router).  Reports params, final
loss, wall time and peak CUDA memory per config; exit code != 0 on any FAIL.
"""
import argparse
import time

import torch

from .model import V15UNet
from .train import fm_loss, load_training_latents
from .where_losses import where_losses


def run_one(name, z_all, y_all, device, base_dim, use_attention, use_where,
            style_mode, updates=2, micro=32):
    torch.manual_seed(0)
    try:
        model = V15UNet(base_dim=base_dim, use_attention=use_attention,
                        style_mode=style_mode).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        t0 = time.time()
        loss_val = float("nan")
        for _ in range(updates):
            idx = torch.randint(0, z_all.shape[0], (micro,))
            z = z_all[idx].to(device)
            y = y_all[idx].to(device)
            tau = torch.rand(z.shape[0], device=device)
            loss = fm_loss(model, z, y, tau)
            if use_where:
                lw, _ = where_losses(model, z, y, tau)
                loss = loss + lw
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_val = loss.item()
        assert loss_val == loss_val and abs(loss_val) < 1e6, "non-finite loss"
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"PASS {name:26s} params={model.n_params()/1e6:6.3f}M "
              f"loss={loss_val:9.4f} {time.time()-t0:6.1f}s peak={peak:.2f}GB",
              flush=True)
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"FAIL {name:26s} {type(e).__name__}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    z_all, y_all, counts = load_training_latents(args.pool)
    print(f"pool: {z_all.shape[0]} latents {counts}", flush=True)
    z_all = z_all.to(device)   # pool is small (<2k latents), keep resident
    y_all = y_all.to(device)

    ok = True
    for dim in (16, 32, 64, 96):
        for att in (False, True):
            for wh in (False, True):
                name = f"dim{dim}_{'att' if att else 'conv'}_{'where' if wh else 'base'}"
                ok &= run_one(name, z_all, y_all, device, dim, att, wh, "adaln")
    ok &= run_one("dim64_conv_spatial_film", z_all, y_all, device, 64, False,
                  False, "spatial_film")
    ok &= run_one("dim64_conv_router", z_all, y_all, device, 64, False,
                  False, "router")
    print("SMOKE_DONE" if ok else "SMOKE_FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
