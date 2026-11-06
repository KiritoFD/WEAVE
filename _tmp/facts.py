"""Collect facts for the A-E accounting table: Q0 coverage, fields, dataset pool sizes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
q0 = json.loads((ROOT / "runs/p2/pgt_q0.json").read_text(encoding="utf-8"))
print("q0 top keys:", sorted(q0.keys()))
cfg = q0.get("configs") or q0.get("per_config") or {}
print("n_configs:", len(cfg))
print("config names:", sorted(cfg.keys()))
if cfg:
    k = sorted(cfg.keys())[0]
    print(f"sample config {k} keys:", sorted(cfg[k].keys()))
    print(f"sample config {k}:", json.dumps({kk: vv for kk, vv in cfg[k].items() if not isinstance(vv, (list, dict))}, ensure_ascii=False)[:400])
print("errors:", q0.get("errors"))
print("shared match:", {k: q0[k] for k in ("n_sources", "n_matched", "match_pass_rate") if k in q0})

sel = json.loads((ROOT / "runs/p2/pgt_selection.json").read_text(encoding="utf-8"))
print("selection keys:", sorted(sel.keys()))

for name, p in (("train_manifest", ROOT / "data/wikiart_new_packed/manifest.json"),
                ("test_manifest", ROOT / "data/test/manifest.json")):
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        n = len(m.get("records", m.get("items", []))) if isinstance(m, dict) else len(m)
        print(f"{name}: {p} records={n} keys={sorted(m.keys())[:6] if isinstance(m, dict) else 'list'}")
    else:
        print(f"{name}: missing {p}")
