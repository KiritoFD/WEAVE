"""Scaling smoke: real n_params, 2-step peak memory and per-update time for
the S1-S5 configs at the user-specified micro/accum.

Run on the 3060: python -m exp.v15.smoke_scaling --pool <pool>
"""
import argparse
import time

import torch

from .model import V15UNet
from .train import fm_loss, load_training_latents

CONFIGS = [(128, 64, 2), (160, 32, 4), (192, 16, 8), (256, 8, 16), (320, 4, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    z_all, y_all, _ = load_training_latents(a.pool)
    z_all = z_all.to(dev)
    y_all = y_all.to(dev)

    ok = True
    for dim, micro, accum in CONFIGS:
        torch.manual_seed(0)
        try:
            model = V15UNet(base_dim=dim).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
            t0 = time.time()
            loss = float("nan")
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                idx = torch.randint(0, z_all.shape[0], (micro,))
                z = z_all[idx]
                y = y_all[idx]
                tau = torch.rand(z.shape[0], device=dev)
                loss = fm_loss(model, z, y, tau)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            dt = (time.time() - t0) / 2
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"dim{dim:3d} micro{micro:3d}/accum{accum:2d} "
                  f"params={model.n_params()/1e6:6.2f}M loss={loss.item():.4f} "
                  f"s/update={dt:6.3f} peak={peak:5.2f}GB "
                  f"-> 4k ETA {dt*4000/3600:5.1f}h", flush=True)
            del model, opt
            torch.cuda.empty_cache()
        except Exception as e:
            ok = False
            print(f"dim{dim:3d} FAIL {type(e).__name__}: {e}", flush=True)
    print("SCALING_SMOKE_DONE" if ok else "SCALING_SMOKE_FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
