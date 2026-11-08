"""Benchmark 100 V8 optimizer updates to determine a common update count N."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v8.bridge import sample_bridge  # noqa: E402
from tools.v8.common import REPO_ROOT, STYLES, V8_ROOT, ensure_dirs, load_d5_latents, save_json  # noqa: E402
from tools.v8.models import VelocityUNet  # noqa: E402
from tools.v8.pair_sampler import ot_sample_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--pairing", choices=["random", "ot"], default="random")
    ap.add_argument("--updates", type=int, default=100)
    args = ap.parse_args()
    ensure_dirs()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_by_style, _, data_meta = load_d5_latents(device=device, heldout=100)
    model = VelocityUNet(base_dim=48, num_styles=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    canon = None
    if args.pairing == "ot":
        stats = torch.load(REPO_ROOT / "runs" / "v6" / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
        canon = ContentCanonicalizer(stats).to(device).eval()
    per_style = max(1, args.microbatch // 5)
    B = per_style * 5
    s = torch.cat([torch.full((per_style,), i, device=device, dtype=torch.long) for i in range(5)])
    t0 = time.time()
    for step in range(1, args.updates + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            z0 = torch.cat([gpu_by_style[k][torch.randint(0, gpu_by_style[k].shape[0], (per_style,), device=device)] for k in STYLES])
            tgt = (s + torch.randint(1, 5, (B,), device=device)) % 5
            z1 = torch.empty_like(z0)
            for tid in range(5):
                m = tgt == tid
                if m.any():
                    pool = gpu_by_style[STYLES[tid]]
                    z1[m] = pool[torch.randint(0, pool.shape[0], (int(m.sum()),), device=device)]
            if args.pairing == "ot":
                z1_ot = torch.empty_like(z1)
                for tid in range(5):
                    m = tgt == tid
                    if not m.any():
                        continue
                    pool = gpu_by_style[STYLES[tid]]
                    ot_batch = min(128, pool.shape[0])
                    oidx = torch.randint(0, pool.shape[0], (ot_batch,), device=device)
                    z1_ot[m] = ot_sample_targets(z0[m], pool[oidx], canon, reg=0.08)[0]
                z1 = z1_ot
            tau = 0.01 + 0.98 * torch.rand(B, device=device)
            zt, vt = sample_bridge(z0, z1, tau, 0.0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = F.mse_loss(model(zt, tau, s, tgt).float(), vt.float()) / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (1, 50, args.updates):
            print(f"[bench] update {step}/{args.updates} elapsed={time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    out = {
        "pairing": args.pairing,
        "updates": args.updates,
        "microbatch": B,
        "accum": args.accum,
        "effective_batch": B * args.accum,
        "wall_sec": round(wall, 1),
        "sec_per_update": round(wall / args.updates, 4),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "data_meta": data_meta,
    }
    save_json(out, V8_ROOT / f"benchmark_100_updates_{args.pairing}.json")
    print(f"[bench] Saved {V8_ROOT / f'benchmark_100_updates_{args.pairing}.json'}", flush=True)


if __name__ == "__main__":
    main()
