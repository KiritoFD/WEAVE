"""G4 engineering cost: forward/backward time + peak VRAM + grad-norm per latent loss.

Usage: python3 tools/audit/f_cost.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C
import latent_losses as L


def bench(loss, kind: str, z: torch.Tensor, s: int, repeat: int = 5) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    zb = z.clone().requires_grad_(True)
    for _ in range(2):  # warmup
        v = loss(kind, zb, s)
        (g,) = torch.autograd.grad(v, zb, retain_graph=True)
    torch.cuda.synchronize()
    fwd = []
    bwd = []
    gnorm = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = loss(kind, zb, s)
        torch.cuda.synchronize()
        fwd.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        (g,) = torch.autograd.grad(v, zb, retain_graph=True)
        torch.cuda.synchronize()
        bwd.append(time.perf_counter() - t1)
        gnorm.append(float(g.flatten(1).norm(dim=1).mean()))
    return {
        "fwd_ms": float(torch.tensor(fwd).mean() * 1e3),
        "bwd_ms": float(torch.tensor(bwd).mean() * 1e3),
        "grad_norm": float(torch.tensor(gnorm).mean()),
        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2**20),
    }


def main() -> None:
    torch.manual_seed(0)
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])
    z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to("cuda")
    tg = L.Targets.build(z_ref, device="cuda")
    loss = L.LatentStyleLoss(tg)
    out = {"device": torch.cuda.get_device_name(0), "per_loss": {}}
    for b in (8, 32):
        z = z_q[1, :b].detach()
        for kind in L.KINDS:
            r = bench(loss, kind, z, 1)
            r["batch"] = b
            out["per_loss"][f"{kind}_b{b}"] = r
            print(f"[COST] {kind:8s} b{b:2d} fwd={r['fwd_ms']:.2f}ms bwd={r['bwd_ms']:.2f}ms "
                  f"peak={r['peak_mem_mb']:.0f}MB |g|={r['grad_norm']:.2e}")
    # full training-step proxy: content+style loss on b96 (current training batch)
    z = z_q[1, :96].detach()
    t0 = time.perf_counter()
    r = bench(loss, "swd", z, 1, repeat=3)
    r["batch"] = 96
    out["per_loss"]["swd_b96"] = r
    print(f"[COST] swd b96 fwd={r['fwd_ms']:.2f}ms bwd={r['bwd_ms']:.2f}ms "
          f"peak={r['peak_mem_mb']:.0f}MB")
    C.save_json("cost.json", out)


if __name__ == "__main__":
    main()
