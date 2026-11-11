#!/bin/bash
# P5 补跑脚本：训练全部完成后手动执行，不改正在运行的 run_sd0_s10.sh。
#   bash tools/vae_suite/run_p5_final.sh
set -euo pipefail
R=/root/Workspace/WEAVE_VAE
cd "$R"
P=/opt/conda/envs/cu121/bin/python
OUT="$R/runs/sd0_s10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo; echo "================================================================"; echo "[P5] $*"; echo "================================================================"; }
FAILS=""
mark_fail() { FAILS="$FAILS $1"; echo "[P5][FAILED STEP] $1"; }
mark_ok()   { echo "[P5][OK] $1"; }

# 检查训练产物是否齐备（C1 可缺；C0 必须有）
[ -f "$OUT/c0_best.pt" ] || { echo "c0_best.pt missing; P5 aborted"; exit 2; }

MODELS="official,old_s10,c0"
[ -f "$OUT/c1_best.pt" ] && MODELS="$MODELS,c1"

log "0. full unit tests (main + extra)"
if $P -u tools/vae_suite/test_sd0_s10.py > "$OUT/p5_selftest_main.log" 2>&1; then mark_ok main_selftest; else mark_fail main_selftest; fi
if $P -u tools/vae_suite/test_sd0_s10_extra.py > "$OUT/p5_selftest_extra.log" 2>&1; then mark_ok extra_selftest; else mark_fail extra_selftest; fi
if CUDA_VISIBLE_DEVICES="" $P -u tools/vae_suite/test_c1_graph.py > "$OUT/p5_selftest_c1graph.log" 2>&1; then mark_ok c1_graph; else mark_fail c1_graph; fi

log "1. four-layer acceptance eval (L1 recon / L2 Jacobian / L4 infra)"
if $P -u -m tools.vae_suite.eval_sd0_s10 --models "$MODELS" --tag p5_final --n 256 > "$OUT/p5_layer124.log" 2>&1; then mark_ok layer124; else mark_fail layer124; fi

log "2. layer 3: five-generator decoder swap"
# canonicalizer stats must exist (used by WEAVE generators)
if [ ! -f "$R/runs/v6/cache/canonicalizer_stats.pt" ]; then
  echo "[WARN] canonicalizer_stats.pt missing -> building from SD0 pool"
  $P -u -c "
import sys, torch; sys.path.insert(0,'$R')
from tools.v6_next.vae_swap_full import build_canon_stats, SDVAE
from diffusers import AutoencoderKL
sd = AutoencoderKL.from_pretrained(SDVAE, torch_dtype=torch.float32).to('cuda').eval()
build_canon_stats(sd, 'cuda')
" || mark_fail canon_stats
fi
if $P -u -m tools.vae_suite.eval_generator_swap_full \
    --variants "$MODELS" \
    --arms "Oracle1,B4,B3,ScoreDiff_N12,ScoreDiff_N24" \
    > "$OUT/p5_layer3.log" 2>&1; then mark_ok layer3; else mark_fail layer3; fi

log "3. export standalone Diffusers-compatible wrappers"
CKS="c0_best.pt"
[ -f "$OUT/c1_best.pt" ] && CKS="$CKS,c1_best.pt"
if $P -u -m tools.vae_suite.export_sd0_s10 --checkpoints "$CKS" \
    > "$OUT/p5_export.log" 2>&1; then mark_ok export; else mark_fail export; fi

log "3b. post-hoc Gate audit (CSD / colour / gradient guards)"
if $P -u -m tools.vae_suite.audit_gates > "$OUT/p5_gate_audit.log" 2>&1; then mark_ok gate_audit; else mark_fail gate_audit; fi

log "4. assemble final report"
P5_FAILS="$FAILS" $P - "$OUT" <<'PY'
import json, os, sys
from pathlib import Path
out = Path(sys.argv[1])
fails = os.environ.get("P5_FAILS", "").strip()
r = {"layers": {}, "artifacts": [],
     "failed_steps": fails.split() if fails else [],
     "status": "complete" if not fails else "incomplete"}
for f in ("eval_p5_final.json", "generator_swap_results.json",
         "gate_audit.json"):
    p = out / f
    if p.exists():
        r["layers"][f] = json.loads(p.read_text())
for g in ("p1_gate.json", "c0_gate.json", "c1_gate.json",
          "NEGATIVE_REPORT.json"):
    p = out / g
    if p.exists():
        r[g] = json.loads(p.read_text())
cs = out / ".chain_status"
if cs.exists():
    r["chain_status"] = cs.read_text().strip()
    if r["chain_status"] == "2":
        r["status"] = "degraded_chain_incomplete_c0_delivered"
r["artifacts"] = [str(p) for p in sorted(out.glob("sd0_s10_best_*.pt"))]
(out / "P5_FINAL_REPORT.json").write_text(json.dumps(r, indent=2))
print("artifacts:", r["artifacts"])
PY

if [ -n "$FAILS" ]; then
  log "P5 INCOMPLETE — failed steps:$FAILS  (weights preserved; see p5_*.log)"
else
  log "P5 COMPLETE — all steps passed"
fi
