#!/usr/bin/env bash
# Serial V10 suite launcher (SD-VAE latent space, canonical 15,545-image dataset).
#
# Stage 1 gate: arm 04 trained from scratch, full 600 cross-style generation.
# Then the remaining implemented arms on Quick200, strictly serial.
# CPU scoring runs in the background so the GPU never waits for metrics.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=runs/blog_sbflow/v10
mkdir -p "$ROOT/logs"

echo "[launch] $(date +%H:%M:%S) === stage-1 gate: arm 04, full600 ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 04 --eval-subset full600 --resume \
  2>&1 | tee -a "$ROOT/logs/gate.log"
GATE_RC=${PIPESTATUS[0]}
echo "[launch] $(date +%H:%M:%S) gate exit=$GATE_RC"
if [ "$GATE_RC" -ne 0 ]; then
  echo "[launch] gate failed; not starting mechanism arms"
  exit 1
fi

for GROUP in "01-03:groupA" "05-10:groupB" "11,12,14,15:groupC" "37,38,39,40:groupGH" \
             "16-23:groupD" "24-31:groupE" "32-35:groupF"; do
  ARMS=${GROUP%%:*}; NAME=${GROUP##*:}
  echo "[launch] $(date +%H:%M:%S) === $NAME (arms $ARMS) ==="
  python3 -u -m sbflow.run_suite --root "$ROOT" --arms "$ARMS" --eval-subset quick200 --resume \
    2>&1 | tee -a "$ROOT/logs/$NAME.log"
done

echo "[launch] $(date +%H:%M:%S) === ALL GROUPS DONE ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v10/suite_state.json")
s = json.loads(p.read_text())
done = {k: v for k, v in s.items() if v.get("status") == "done"}
print(f"[launch] completed {len(done)}/{len(s)} arms")
for k in sorted(done):
    e = done[k].get("eval") or {}
    if "target_top1" in e:
        print(f"  {k}: top1={e['target_top1']:.4f} margin={e['target_margin']:+.4f} "
              f"dT={e['csd_dT']:+.4f} dS={e['csd_dS']:+.4f} swd={e['csd_swd']:.4f}")
PY
