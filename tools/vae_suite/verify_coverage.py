"""Verify full-coverage epoch sampler: every image exposed >= 1 per epoch."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch

from tools.vae_suite.style_rank_data import EpochCoverageSampler, verify_epoch_coverage

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")


def main():
    idx = torch.load(OUT / "index.pt", weights_only=False)
    y = idx["y"].numpy()
    a = idx["a"].numpy()
    n_styles = int(y.max()) + 1

    style_to_items = defaultdict(list)
    for i in range(len(y)):
        style_to_items[int(y[i])].append((i, int(a[i])))

    print(f"[Pool] {len(y)} images | {n_styles} styles", flush=True)
    for s in sorted(style_to_items):
        items = style_to_items[s]
        arts = len({ar for _, ar in items})
        print(f"   style {s:>2}: {len(items):>6} images, {arts:>3} artists", flush=True)

    res, counts = verify_epoch_coverage(style_to_items, seed=42)

    # multi-epoch check
    smp = EpochCoverageSampler(style_to_items, seed=42)
    it = iter(smp)
    tot = np.zeros(len(y), dtype=np.int64)
    per_epoch = []
    n_micro = smp.n_micro
    for ep in range(3):
        before = tot.copy()
        for _ in range(n_micro):
            b = next(it)
            for p in b:
                tot[p] += 1
        d = tot - before
        per_epoch.append({"epoch": ep + 1, "covered": int((d >= 1).sum()),
                          "min": int(d.min()), "max": int(d.max()),
                          "mean": round(float(d.mean()), 3)})
        print(f"[Epoch {ep+1}] covered={per_epoch[-1]['covered']}/{len(y)} "
              f"min={d.min()} max={d.max()} mean={d.mean():.2f}", flush=True)

    out = {"single_epoch": res, "three_epochs": per_epoch,
           "n_micro_per_epoch": n_micro,
           "updates_per_epoch": n_micro // 8,
           "pool": len(y)}
    (OUT / "coverage_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    ok = res["n_zero"] == 0 and res["count_min"] >= 1
    print("\n" + "=" * 70)
    print(f"COVERAGE GATE: {'PASS' if ok else 'FAIL'} "
          f"(zero-exposure images = {res['n_zero']})")
    print(f"  1 epoch = {res['microbatches']} microbatches = {res['updates']} updates")
    print(f"  11h @ 11.8s/update = ~3335 updates = {3335/res['updates']:.2f} epochs")
    print("=" * 70)
    print(f"[Saved] {OUT/'coverage_check.json'}")


if __name__ == "__main__":
    main()
