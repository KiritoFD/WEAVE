"""V25 R14b (3060 / 4090): PRISMRef 双接口端到端 GPU 冒烟测试."""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

from . import EVAL_SEED, OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from src.model.prism_v25 import PRISMRef, RESULT_KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    seed_all(EVAL_SEED)
    t0 = time.perf_counter()
    ref = PRISMRef().prepare(pool=args.pool, device=args.device)
    prep_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out1 = ref.transfer(None, None, strength=0.5)
    run_s = time.perf_counter() - t1

    for k in RESULT_KEYS:
        assert k in out1, f"missing {k}"
    z, rgb = out1["z"], out1["rgb01"]
    assert tuple(z.shape) == (750, 4, 64, 64)
    assert tuple(rgb.shape) == (750, 3, 512, 512)
    assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(rgb).all())
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

    # Check NotImplementedError on arbitrary single RGB reference (honest capability boundary)
    arb_ok = False
    try:
        ref.transfer(None, torch.rand(1, 3, 512, 512), strength=0.5)
    except NotImplementedError:
        arb_ok = True
    assert arb_ok, "expected NotImplementedError for arbitrary single RGB reference"

    res = {
        "task": "V25-R14b PRISMRef E2E GPU smoke",
        "updated_utc": utc(),
        "gpu": torch.cuda.get_device_name(args.device),
        "status": "PASS",
        "engine": ref.engine,
        "config_hash": ref.config_hash[:12],
        "knob": out1["knob"],
        "knob_value": round(float(out1["knob_value"]), 5),
        "prep_s": round(prep_s, 2),
        "wall_s_750": round(run_s, 2),
        "arbitrary_rgb_ref_gate": "PASS (NotImplementedError raised cleanly)",
    }
    save_json(OUT / "r14_ref_smoke.json", res)
    print(f"[R14b SMOKE] PRISMRef PASS: knob={res['knob']}={res['knob_value']} wall={res['wall_s_750']}s", flush=True)


if __name__ == "__main__":
    main()
