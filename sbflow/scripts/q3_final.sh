#!/usr/bin/env bash
# Q3: independent evaluation of at most two Val200-selected candidates.
#
# Model selection happens on Val200 ONLY; Heldout400 is used here for confirmation and its use is
# logged. Same150 is included so a style-identity regression on same-style input cannot hide.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source sbflow/scripts/q_lib.sh

# pick the two best arms on Val200 by (Top-1, then dT) among everything trained in this queue
read -r A1 A2 <<< "$(python3 - <<'PY'
import json
from pathlib import Path
cands = ["G0_ctrl", "G1a_rank_weak", "G1b_rank_strong", "Q1a_rank_00003", "Q1b_rank_0002",
         "Q1c_G0_s43", "Q1d_G1_s43", "Q2a_embed_style", "Q2b_trust", "Q2c_embed_trust"]
rows = []
for a in cands:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values()) / len(pt) if pt else 0.0
    dino = e.get("mean_dino_c") or 0.0
    # acceptance requires DINO-C >= 0.85; arms below that are strong-style tier only
    if dino < 0.85:
        continue
    rows.append((t1, e.get("mean_dT") or 0.0, a))
rows.sort(reverse=True)
names = [r[2] for r in rows[:2]] + ["", ""]
print(names[0], names[1])
PY
)"

qlog "Q3: Val200-selected candidates: A1=$A1 A2=$A2"
if [ -z "$A1" ]; then
  qlog "Q3: no candidate met DINO-C >= 0.85; nothing to confirm"
  exit 0
fi

confirm () {
  local name=$1
  [ -z "$name" ] && return 0
  local ck="runs/v6/${name}/${name}.pt"
  [ -s "$ck" ] || { qlog "Q3: $name checkpoint missing"; QFAIL=$((QFAIL+1)); return 1; }
  for tier in heldout400 same150; do
    run_step "Q3_${name}_${tier}" "runs/v6_next/eval_${tier}_${name}/eval.json" \
      python3 -u -m tools.v6_24h.eval_tiers --ckpt "$ck" \
        --tasks "runs/v6_24h/${tier}_tasks.json" \
        --out "runs/v6_next/eval_${tier}_${name}"
  done
}

confirm "$A1"
confirm "$A2"

qlog "Q3 done, failures so far: $QFAIL"
