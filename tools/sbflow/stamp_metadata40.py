"""Stamp suite40 state with the authoritative local Git SHA and fixed VAE metadata.

The 3060 copy under /mnt/i is not a git repository, so the SHA is passed in from
the local G:\\WEAVE checkout (the git origin of truth). Run on 3060:

    python3 sbflow/stamp_metadata40.py --sha <short-sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
STATE = ROOT / "runs/blog_sbflow/suite40/suite_state.json"
VAE = {"vae": "stabilityai/sd-vae-ft-ema", "scaling_factor": 0.18215,
       "latent_shape": [4, 64, 64], "dataset_styles": 5,
       "dataset_images": 15545}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    args = ap.parse_args()
    if not STATE.exists():
        raise SystemExit(f"missing {STATE}")
    state = json.loads(STATE.read_text())
    for aid, rec in state.items():
        if not rec.get("git_sha"):
            rec["git_sha"] = args.sha
        rec.update(VAE)
        cfg = rec.get("config", {})
        rec.setdefault("batch", cfg.get("batch"))
        rec.setdefault("patch", cfg.get("patch"))
        rec.setdefault("updates", cfg.get("steps"))
        rec.setdefault("seed", cfg.get("seed", 42))
        rec.setdefault("per_pair_exposure",
                       (cfg.get("steps", 0) * cfg.get("batch", 0) * 2) // 10)
        print(f"  {aid}: sha={rec['git_sha']} vae={rec['vae']} "
              f"batch={rec.get('batch')} patch={rec.get('patch')} seed={rec.get('seed')}")
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    from tools.sbflow.suite40 import write_results
    write_results(state, ROOT / "runs/blog_sbflow/suite40")
    print(f"stamped {len(state)} arms")


if __name__ == "__main__":
    main()
