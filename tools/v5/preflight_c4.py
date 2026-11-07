"""Preflight C4 dependency check (read-only, bounded)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))
sys.path.insert(0, str(ROOT / "tools" / "trt"))


def main():
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_mib = free_b / 2**20
    print(f"[preflight] VRAM free={free_mib:.0f} MiB / {total_b/2**20:.0f} MiB", flush=True)
    if free_mib < 4500:
        print("[preflight] SKIP Q.Metrics load: GPU is occupied by C1 training", flush=True)
        return 0
    from tools.pseudo_gt import q0_quality as Q
    print("[preflight] loading Q.Metrics (C4 v4_eval750 dependency)...", flush=True)
    m = Q.Metrics()
    print("[preflight] Q.Metrics OK; VAE/CSD/DINO/LPIPS/TRT wired", flush=True)
    proto = Q.build_prototypes(m, Q.load_latents(), set())
    print(f"[preflight] prototypes OK: {len(proto)} styles", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
