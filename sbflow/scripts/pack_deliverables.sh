#!/usr/bin/env bash
# Build a delivery bundle of the V6 next-phase results and emit it as base64 on stdout.
#   compact/  aggregate metrics only (no per-task rows)
#   full/     the original JSONs including rows, for re-analysis
#
# Naming: every arm ships a file literally called report.json and every tier ships eval.json, so the
# output name must come from the PARENT directory for those two, otherwise ten arms collapse into one
# file and nine silently vanish.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
OUT=runs/v6_next/deliver
rm -rf "$OUT"; mkdir -p "$OUT/compact" "$OUT/full"

python3 - <<'PY'
import json
from pathlib import Path
OUT = Path("runs/v6_next/deliver")

def strip_rows(d):
    if isinstance(d, dict):
        return {k: strip_rows(v) for k, v in d.items() if k != "rows"}
    if isinstance(d, list):
        return [strip_rows(x) for x in d]
    return d

srcs = []
srcs += list(Path("runs/v6_next/amortize").glob("*.json"))
srcs += list(Path("runs/v6_next").glob("eval_750_*/eval.json"))
srcs += list(Path("runs/v6_next").glob("eval_heldout400_*/eval.json"))
srcs += list(Path("runs/v6_next").glob("eval_same150_*/eval.json"))
srcs += list(Path("runs/v6_next/probes").glob("*.json"))
for a in ("Q2a_embed_style", "P2_G0_bf16", "P2_G1a_bf16", "P2_G1b_bf16",
          "P1_B0_ctrl218", "P1_B1_distill218", "P1_B2c_ctrl845",
          "P1_B2_distill845", "P1_B3_distill20", "P1_B4_distill50"):
    p = Path(f"runs/v6/{a}/report.json")
    if p.exists():
        srcs.append(p)

def out_name(p: Path) -> str:
    if p.name in ("eval.json", "report.json"):
        return f"{p.parent.name}.json"
    return f"{p.stem}.json"

seen = {}
n_c = n_f = 0
for p in srcs:
    if not p.exists():
        continue
    name = out_name(p)
    if name in seen:
        print(f"  COLLISION {name}: {p} vs {seen[name]}")
        continue
    seen[name] = p
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    (OUT / "compact" / name).write_text(
        json.dumps(strip_rows(d), ensure_ascii=False, indent=1), encoding="utf-8")
    n_c += 1
    (OUT / "full" / name).write_bytes(p.read_bytes())
    n_f += 1
print(f"compact={n_c} full={n_f} unique_names={len(seen)}")
PY

tar czf - -C "$OUT" compact full | base64 -w0
