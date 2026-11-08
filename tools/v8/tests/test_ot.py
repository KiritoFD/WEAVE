"""V8 OT endpoint-pairing audit: matched C1 content cost vs random and target coverage."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v8.common import STYLES, V8_ROOT, ensure_dirs, load_d5_latents, save_json  # noqa: E402
from tools.v8.pair_sampler import c1_content_feature, sinkhorn_log, ot_sample_targets  # noqa: E402


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, _, data_meta = load_d5_latents(device=device, heldout=100)
    stats = torch.load(Path(_REPO_ROOT) / "runs" / "v6" / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(stats).to(device).eval()
    out = {"data": {"scaling_factor": data_meta["scaling_factor"], "vae": data_meta["vae"]}, "pairs": {}}
    for sid, s in enumerate(STYLES):
        tid = (sid + 1) % len(STYLES)
        t = STYLES[tid]
        z0 = train[s][:64]
        pool = train[t][:128]
        _, info = ot_sample_targets(z0, pool, canon, reg=0.08)
        f0 = c1_content_feature(z0, canon)
        f1 = c1_content_feature(pool, canon)
        cost = torch.cdist(f0, f1).pow(2)
        out["pairs"][f"{s}->{t}"] = info | {"random_cost_exact": float(cost.mean().item())}
    save_json(out, V8_ROOT / "p0_ot_pairing_audit.json")
    print(f"[V8 OT Audit] Saved {V8_ROOT / 'p0_ot_pairing_audit.json'}")
    for k, v in out["pairs"].items():
        print(f"  {k}: matched={v['matched_cost']:.4f} random={v['random_cost']:.4f} coverage={v['coverage_ratio']:.3f}")


if __name__ == "__main__":
    main()
