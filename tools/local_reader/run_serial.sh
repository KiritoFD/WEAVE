#!/bin/bash
# Local Style Reader pre-experiment: full serial pipeline.
#   bash tools/local_reader/run_serial.sh
#   nohup bash tools/local_reader/run_serial.sh > runs/local_reader/serial.log 2>&1 &
set -euo pipefail

R=/root/Workspace/WEAVE_VAE
cd "$R"
P=/opt/conda/envs/cu121/bin/python
OUT="$R/runs/local_reader"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo; echo "================================================================"; echo "[SERIAL] $*"; echo "================================================================"; }

log "0. compile"
$P -m py_compile tools/local_reader/*.py

log "1. component + data-alignment selftests"
$P -u tools/local_reader/selftest.py 2>&1 | tee "$OUT/step_1_selftest.log"

log "2. smoke (60 steps, both readers)"
$P -u -m tools.local_reader.run --steps 60 --readers local,cnn \
    > "$OUT/step_2_smoke.log" 2>&1
grep -E "GATE R:|EXIT|Error|Traceback" "$OUT/step_2_smoke.log" | tail -n 3 || true
# smoke artifacts must not be mistaken for the real checkpoints
rm -f "$OUT/local_best.pt" "$OUT/cnn_best.pt" \
      "$OUT/eval_local.json" "$OUT/eval_cnn.json" "$OUT/gate_R.json"

log "3. full run (1500 steps, both readers, Gate R)"
$P -u -m tools.local_reader.run --steps 1500 --readers local,cnn \
    2>&1 | tee "$OUT/step_3_full.log"

log "4. summary"
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
g = json.loads((out / "gate_R.json").read_text())
print(json.dumps(g["gate_R"], indent=2))
print("\n--- reader comparison ---")
for k, v in g["readers"].items():
    if isinstance(v, dict) and "auroc" in v:
        print(f"  {k:<8} AUROC={v['auroc']:.4f} (PD {v['auroc_painter_disjoint']:.4f}) "
              f"retr5={v['retrieval_5shot']:.4f} flat={v['retrieval_5shot_flat']:.4f} "
              f"unseen={v['unseen_retrieval']:.4f} params={v['param_count']:,}")
print(f"\n  fixed-readout AUROC (this run) = {g['protocol']['fixed_reader_auroc']}")
print(f"  GATE R = {'PASSED' if g['gate_R']['passed'] else 'FAILED'}")
PY

log "SERIAL PIPELINE COMPLETE"
