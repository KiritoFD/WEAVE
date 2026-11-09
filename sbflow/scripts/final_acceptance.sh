#!/usr/bin/env bash
# Final acceptance check for the V6 24h plan execution.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}

echo "================ 1. arms completed ================"
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("runs/v6_24h/state.json").read_text())
done = [k for k, v in s.items() if v.get("status") == "done"]
bad = {k: v.get("status") for k, v in s.items() if v.get("status") != "done"}
print(f"  {len(done)} arms done")
print(f"  non-done: {bad if bad else 'NONE'}")
PY

echo
echo "================ 2. artefacts present ================"
for f in runs/v6_24h/heldout400_tasks.json \
         runs/v6_24h/same150_tasks.json \
         runs/v6_24h/screen50_tasks.json \
         runs/v6_24h/p5_final/p5_final.json \
         runs/v6_24h/m0_main/m0_calibration.json \
         runs/v6_24h/m0_extrap/m0_calibration.json \
         runs/v6_24h/csd_manifold_best.json \
         runs/v6_24h/style_condition_efficacy.json \
         runs/v6_24h/grid/manifest.json \
         runs/v6_24h/folded/A0_folded.pt; do
  [ -f "$f" ] && echo "  OK   $f" || echo "  MISS $f"
done
echo "  grids: $(ls runs/v6_24h/grid/*.png 2>/dev/null | wc -l) PNG"

echo
echo "================ 3. no compute processes ================"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || echo "  none"
ps -eo args --no-headers | grep -E "train_eval_v6|v6_24h.queue|m0_alpha|eval_tiers|make_grid" \
  | grep -v grep | head -3 || echo "  no v6 processes"

echo
echo "================ 4. pipeline exit codes ================"
grep -a "PASS3_EXIT" runs/v6_24h/pass3.log 2>/dev/null || echo "  no pass3 exit"
grep -a "POSTQUEUE_EXIT" runs/v6_24h/postqueue.log 2>/dev/null | tail -1 || echo "  no postq exit"

echo
echo "================ 5. regression suite ================"
bash sbflow/scripts/regression_v6_all.sh 2>&1 | tail -4
