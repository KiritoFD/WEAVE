"""Decode v9 latent outputs to rgb_u8 gen-format, then run src.eval.measure.

Writes runs/v9_local_4070/<split>/<arm>/gen/{batch_*.pt,meta.json} then
eval/ via the standard WEAVE evaluator (same pipeline as run_arm.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.utils import decode_s10, load_s10_decoder
from src.eval.measure import measure

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v9_local_4070"
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    dev = a.device
    src_dir = OUT / a.split / a.arm
    gen_dir = src_dir / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)

    if not a.skip_decode:
        s10 = load_s10_decoder(dev)
        batches = []
        files = sorted(src_dir.glob("batch_*.pt"))
        t0 = time.time()
        n = 0
        for p in files:
            d = torch.load(p, map_location="cpu", weights_only=False)
            ids = d["ids"]
            if a.limit and n >= a.limit:
                break
            take = min(len(ids), max(0, a.limit - n)) if a.limit else len(ids)
            ids, z = ids[:take], d["z_final"][:take].to(dev)
            rgb = decode_s10(s10, z, batch=a.batch)
            rgb_u8 = rgb.mul(255).round().clamp(0, 255).byte().cpu()
            gp = gen_dir / p.name
            torch.save({"ids": ids, "z_src": d["z_src"][:take],
                        "z0": d["z0"][:take], "z_final": z.cpu(),
                        "rgb_u8": rgb_u8}, gp.with_suffix(".tmp"))
            gp.with_suffix(".tmp").replace(gp)
            batches.append({"offset": n, "ids": ids, "file": gp.name})
            n += take
            print(f"  decode {n} {time.time()-t0:.0f}s", flush=True)
        meta = {"kind": "v9_local", "arm": a.arm, "split": a.split,
                "decode": "s10_p5_lowlr",
                "n_tasks": n, "batches": batches,
                "wall_sec": round(time.time() - t0, 1)}
        (gen_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    if not a.skip_eval:
        eval_dir = src_dir / "eval"
        rows = measure(gen_dir, eval_dir, device=dev, write_png=True,
                       limit=a.limit)
        print("EVAL_DONE", a.arm, len(rows))


if __name__ == "__main__":
    main()
