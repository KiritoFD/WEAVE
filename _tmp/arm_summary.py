"""Print per-arm training facts: early-stop summary, steps used, checkpoints, wall time."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE/runs/pgt")
ARMS = ["S0_base", "S1_positive", "S2_reverse"]

for arm in ARMS:
    d = ROOT / arm
    print(f"=== {arm} ===")
    es_p = d / "internal_dynamics_summary.json"
    if es_p.exists():
        es = json.loads(es_p.read_text(encoding="utf-8"))
        for k, v in es.items():
            if isinstance(v, list) and len(v) > 8:
                print(f"  {k}: [{v[0]} ... {v[-3:]}] (n={len(v)})")
            else:
                print(f"  {k}: {v}")
    else:
        print("  no internal_dynamics_summary.json")
    cks = sorted(d.glob("epoch_*.pt"))
    if cks:
        import torch
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        print(f"  last_ckpt={cks[-1].name} global_step={ck.get('global_step')} "
              f"epoch={ck.get('epoch')} keys={sorted(ck.keys())}")
    log = d / "train.log"
    if log.exists():
        txt = log.read_text(encoding="utf-8", errors="ignore")
        times = re.findall(r"2026-11-06 (\d\d:\d\d:\d\d)", txt)
        first = times[0] if times else "?"
        last = times[-1] if times else "?"
        print(f"  log_first={first} log_last={last}")
        print("  tail:")
        for line in txt.strip().splitlines()[-8:]:
            print("   ", line)
    print()
