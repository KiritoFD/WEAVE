"""P3-long config: does the failure come from the budget or from capacity?

P3 at 4000 equivalent steps could not generate ER / Rococo / Ukiyo_e at all, but two
confounds make that uninterpretable on its own:
  * pivot mode has 5 pairs, so per-pair exposure is HALF the cross-domain arms';
  * 4000 updates is orders of magnitude below what unconditional image generation needs.

This config trains the SAME pivot model for 40,000 steps (10x), which gives ~4.08M
endpoints per pair -- 10x the cross-domain per-pair budget. If the three hard styles stay
at style Top-1 = 0, the limit is capacity/representation. If they rise, the earlier
cross-domain failures were largely a budget artefact.

Checkpoints at 10k/20k/30k/40k let us read the trend rather than a single endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "v11"
base = json.loads((OUT / "p3.json").read_text(encoding="utf-8"))

cfg = dict(base)
cfg.update(
    experiment_id="p3long",
    output="runs/blog_sbflow/v11/p3long",
    steps=40000,
    lr_cycle_steps=40000,
    save_interval=10000,
    save_equiv="10000,20000,30000,40000",
    full_equiv=40000,
    log_interval=1000,
    equiv_steps=40000,
    note="P3 extended 10x to separate training budget from model capacity "
         "(pivot mode, 5 pairs, ~4.08M endpoints per pair)",
    p3long=True,
)
(OUT / "p3long.json").write_text(
    json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {OUT / 'p3long.json'}")
print(f"  steps={cfg['steps']} batch={cfg['batch']} patch={cfg['patch']} "
      f"save_equiv={cfg['save_equiv']}")
print(f"  est time = {cfg['steps'] * 0.176 / 60:.0f} min at the measured 0.176 s/update")
