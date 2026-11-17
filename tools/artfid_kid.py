"""KID (Kernel Inception Distance) companion to tools/artfid_750.py.

Computes unbiased MMD^2 with polynomial kernel k(x,y)=(x.y/d + 1)^3,
d=2048, between each arm's cached gen features (runs/artfid/feats_*.npz)
and real style references (datasets/distinct5_512/train/<style>).

Saves per-arm 'kid_style' (mean over 5 target styles) into
runs/artfid/artfid_results.json alongside the ArtFID fields.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_orig_load = torch.load


def _patched_load(f, *a, **k):
    if isinstance(f, (str, Path)) and k.get("mmap") is None \
            and Path(f).stat().st_size > 64 * 1024 * 1024:
        k["mmap"] = True
    try:
        return _orig_load(f, *a, **k)
    except RuntimeError as exc:
        if "mmap" in str(exc).lower():
            k.pop("mmap", None)
            return _orig_load(f, *a, **k)
        raise


torch.load = _patched_load

from tools.artfid_750 import (ARTFID_CKPT, OUT, REF_DIR, STYLES,  # noqa: E402
                              collect_feats_threaded, ref_paths)

REF_FEATS = OUT / "ref_feats.npz"
D = 2048


def kid_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased MMD^2, polynomial kernel (x.y/d + 1)^3."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    kxx = (x @ x.T) / D + 1.0
    kyy = (y @ y.T) / D + 1.0
    kxy = (x @ y.T) / D + 1.0
    kxx, kyy, kxy = kxx ** 3, kyy ** 3, kxy ** 3
    n, m = len(x), len(y)
    # unbiased: exclude diagonal terms
    sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    sxy = kxy.mean()
    return float(sxx + syy - 2 * sxy)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tools.legacy_utils.artfid_metric import ArtFIDFeatureExtractor
    model = ArtFIDFeatureExtractor(ARTFID_CKPT, device=device)
    model.eval()

    if REF_FEATS.is_file():
        z = np.load(REF_FEATS)
        ref_feats = {s: z[s] for s in STYLES}
    else:
        ref_feats = {}
        for s in STYLES:
            paths = [str(p) for p in ref_paths(s)]
            ref_feats[s] = collect_feats_threaded(
                paths, model=model, batch_size=64, device=device)
            print(f"ref {s}: {ref_feats[s].shape}", flush=True)
        np.savez(REF_FEATS, **ref_feats)

    results_path = OUT / "artfid_results.json"
    results = json.loads(results_path.read_text())
    for f in sorted(OUT.glob("feats_*.npz")):
        arm_key = f.stem[len("feats_"):]
        # map filename back to results key
        key = next((k for k in results
                    if k.replace(".", "_").replace("-", "_")
                    == arm_key or
                    "".join(c if c.isalnum() or c in "_.-" else "_"
                            for c in k) == arm_key), None)
        # feats_ filename used re.sub(r'[^A-Za-z0-9_.-]','_',name)
        key = next((k for k in results
                    if __import__('re').sub(r'[^A-Za-z0-9_.-]', '_', k)
                    == arm_key), None)
        if key is None or results[key].get("status") != "done":
            continue
        z = np.load(f)
        kids = {}
        for s in STYLES:
            if s not in z.files:
                continue
            kids[s] = kid_unbiased(z[s], ref_feats[s])
        results[key]["kid_style"] = {k: float(v) for k, v in kids.items()}
        results[key]["kid_mean"] = float(np.mean(list(kids.values())))
        print(f"{key}: KID={results[key]['kid_mean']:.2f}", flush=True)
    results_path.write_text(json.dumps(results, indent=1))
    print("KID_DONE")


if __name__ == "__main__":
    main()
