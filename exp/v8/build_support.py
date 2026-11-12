"""Build the C-route real-painting support bank from FROZEN R1 embeddings.

Prototypes come from training-pool paintings only. Evaluation sources and the
independent reference set are excluded by construction: this reads the packed
per-style training pools and drops the first 100 rows, which is the same split
the rest of the repository uses.

Output: runs/v8/real_support.pt  {prototypes: (5,K,D), sigma: (5,)}
Run before any C arm:  python3 -m exp.v8.build_support
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v8/real_support.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32, help="prototypes per style")
    ap.add_argument("--max-per-style", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from src.infer import load_critic
    from exp.v8.energy import build_real_support
    from tools.v7r.v7r_common import SD_POOL

    critic = load_critic(a.device)
    support = build_real_support(critic, SD_POOL, k=a.k, seed=a.seed,
                                device=a.device, max_per_style=a.max_per_style)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"prototypes": support.prototypes.cpu(), "sigma": support.sigma.cpu(),
                "k": a.k, "seed": a.seed, "source": "training pool [100:] only",
                "tau": support.tau}, tmp)
    tmp.replace(out)
    print(json.dumps({"out": str(out), "k": a.k,
                      "sigma": [round(float(s), 5) for s in support.sigma],
                      "prototypes_shape": list(support.prototypes.shape)}), flush=True)


if __name__ == "__main__":
    main()
