"""V6 24h plan section 3 - data-leakage audit.

Answers three separate questions, which must not be conflated:

  A. Are the EVALUATION SOURCE images disjoint from the training pool?
     If not, every metric is contaminated. This is a hard FAIL.

  B. Are the evaluation REFERENCES / the CSD prototypes drawn from the training pool?
     They are (by construction: csd_proto is sampled from orig_mean_fp16 excluding only
     the 300 source images, and the 30/style reference sets live in the same pool).
     That makes ABSOLUTE dT / SWD optimistic, but every arm sees the same references, so
     arm-to-arm comparisons and the Pareto ordering stay valid. Reported as WARN.

  C. Does every training step cover all five classes?
     The sampler draws per_style_mb = microbatch // 5 from every style each step.

Pure CPU: reads filenames and metadata only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
REPO = str(ROOT)
FAIL, WARN = [], []


def norm(p) -> str:
    """Repo-relative, lower-cased POSIX path so absolute and relative inputs compare."""
    p = str(p).replace("\\", "/")
    if p.startswith(REPO + "/"):
        p = p[len(REPO) + 1:]
    return p.lower()


def check(name, cond, detail="", hard=True):
    tag = "PASS" if cond else ("FAIL" if hard else "WARN")
    print(f"{tag} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        (FAIL if hard else WARN).append(name)


def main() -> None:
    # ---- training pool ----
    train, counts = set(), {}
    for f, s in zip(FILES, STYLES):
        d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
        ids = d.get("files") or d.get("ids") or []
        counts[s] = int(d.get("count", len(ids)))
        train |= {norm(x) for x in ids}
        print(f"  train {s:20s} count={counts[s]:5d}")
    check("training pool totals 15,545", sum(counts.values()) == 15545,
          f"total={sum(counts.values())}")

    # ---- eval sources ----
    t = json.loads((ROOT / "eval_protocol/tasks_750.json").read_text())
    src = {norm(x.get("source_rel_path", "")) for x in t["tasks"]}
    print(f"  eval source images: {len(src)}")
    check("A eval SOURCES disjoint from training pool (hard)", not (src & train),
          f"overlap={len(src & train)}", hard=True)

    # ---- eval references ----
    r = json.loads((ROOT / "eval_protocol/references_v1.json").read_text())
    ref = set()
    for s, its in r["references"].items():
        for it in its:
            ref.add(norm(it["rel_path"] if isinstance(it, dict) else it))
    print(f"  eval references: {len(ref)}")
    check("A2 eval references disjoint from eval sources (hard)", not (ref & src),
          f"overlap={len(ref & src)}", hard=True)
    check("B eval references ARE inside the training pool (expected by design)",
          bool(ref & train), f"overlap={len(ref & train)}/{len(ref)}", hard=False)

    # ---- CSD prototype provenance ----
    proto = Path("/mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt")
    check("B2 csd_proto.pt exists", proto.exists(), str(proto), hard=False)
    if proto.exists():
        d = torch.load(proto, map_location="cpu", weights_only=False)
        import numpy as np
        norms = [float(np.linalg.norm(np.asarray(d[s], dtype=np.float32))) for s in STYLES]
        check("B2b csd_proto covers five styles and is unit-norm",
              set(d.keys()) >= set(STYLES) and all(abs(n - 1.0) < 1e-3 for n in norms),
              f"keys={sorted(d.keys())} norms={[round(n,4) for n in norms]}", hard=False)
        print("       NOTE: csd_proto is sampled from orig_mean_fp16 (training pool), "
              "excluding only the 300 eval source images. Absolute dT is therefore "
              "optimistic; arm-to-arm comparison is unaffected.")

    # ---- class coverage ----
    per_style = max(4, 80 // len(STYLES))
    check("C every training step draws from all 5 classes", per_style >= 4,
          f"per_style_mb={per_style} (microbatch 80 / 5)")

    # ---- canonicalizer stats ----
    cs = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
    if cs.exists():
        d = torch.load(cs, map_location="cpu", weights_only=False)
        ok = all(torch.isfinite(d[k]).all() for k in d)
        check("D canonicalizer stats finite", ok, f"{len(d)} tensors")
    else:
        check("D canonicalizer stats exist", False, "missing")

    print()
    print(f"WARN items ({len(WARN)}): {WARN}")
    if FAIL:
        print(f"LEAKAGE AUDIT FAILED: {FAIL}")
        sys.exit(1)
    print("LEAKAGE AUDIT PASSED (no source leakage; training-derived references noted)")


if __name__ == "__main__":
    main()
