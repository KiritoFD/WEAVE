"""P1 gate: finite-difference + wrong-style checks for local style direction.

1. L_loc(z - eta*grad L_loc) < L_loc(z) for small eta (descent sanity).
2. On dev N2C outputs: mean E_t < mean E_wrong (direction points at target).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from exp.v9_local.local_style import LocalStyleEnergy
from exp.v9_local.patch_bank import STYLES

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "runs/v9_local_4070/cache/n2c_z.pt"
BANK = REPO / "runs/v9_local_4070/bank/style_bank.pt"
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]


def main():
    dev = "cuda"
    lse = LocalStyleEnergy(str(BANK), dev)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    # use first 16 cross tasks
    idx = [i for i, t in enumerate(TASKS)
           if t["source_style_id"] != t["target_style_id"]][:16]
    z = cache["z_n2"][idx].to(dev)
    tid = torch.tensor([TASKS[i]["target_style_id"] for i in idx], device=dev)

    # 1) finite-difference descent check on one sample
    z1 = z[:1].clone().requires_grad_(True)
    l0 = lse.loss(z1, tid[:1])
    g = torch.autograd.grad(l0, z1)[0]
    for eta in (1e-3, 1e-2, 5e-2):
        with torch.no_grad():
            l1 = lse.loss(z1.detach() - eta * g, tid[:1])
        print(f"eta={eta}: L {l0.item():.4f} -> {l1.item():.4f} "
              f"{'OK' if l1 < l0 else 'FAIL'}")

    # 2) wrong-style energy comparison (batch)
    E_t, E_n = lse.energies(z, tid)
    print("E_t mean", E_t.mean().item(), "E_neg mean", E_n.mean().item(),
          "margin", (E_n - E_t).mean().item())


if __name__ == "__main__":
    main()
