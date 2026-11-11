#!/bin/bash
# Art-VAE encoder: full serial pipeline.
#   bash tools/art_vae/run_serial.sh            (foreground, tee'd logs)
#   nohup bash tools/art_vae/run_serial.sh > runs/art_vae/serial.log 2>&1 &
#
# Everything is implemented and self-tested BEFORE training starts. Each step
# checks its exit code; a failure stops the chain instead of continuing with a
# broken artifact. Completed steps are skipped so the run is resumable.
set -euo pipefail

R=/root/Workspace/WEAVE_VAE
cd "$R"
P=/opt/conda/envs/cu121/bin/python
OUT="$R/runs/art_vae"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo; echo "================================================================"; echo "[SERIAL] $*"; echo "================================================================"; }
run() { local tag="$1"; shift; log "$tag"; "$@" 2>&1 | tee "$OUT/step_${tag}.log"; }

# ------------------------------------------------------------------ 0. compile
log "0. compile check"
$P -m py_compile tools/art_vae/*.py

# ------------------------------------------------------------------ 1. component tests
if [ ! -f "$OUT/.selftest_ok" ]; then
  run "1_selftest_components" $P -u tools/art_vae/selftest.py --device cpu --skip-data
  $P -u tools/art_vae/selftest.py --device cuda --skip-data > "$OUT/step_1b_selftest_gpu.log" 2>&1 \
    || { tail -n 30 "$OUT/step_1b_selftest_gpu.log"; exit 1; }
  touch "$OUT/.selftest_ok"
else
  log "1. component selftests already passed (skip)"
fi

# ------------------------------------------------------------------ 2. data alignment
run "2_data_alignment" $P -u tools/art_vae/data.py --selftest

# ------------------------------------------------------------------ 3. offline DINO
if [ ! -f "$OUT/dino_train.pt" ]; then
  run "3_prepare_dino" $P -u tools/art_vae/data.py --prepare-dino --device cuda
else
  log "3. DINO cache exists (skip)"
fi

# ------------------------------------------------------------------ 4. content-matched negatives
if [ ! -f "$OUT/neg_candidates.npy" ]; then
  run "4_prepare_neg" $P -u tools/art_vae/data.py --prepare-neg --device cuda
else
  log "4. negative index exists (skip)"
fi

# ------------------------------------------------------------------ 5. full data selftest (caches aligned)
run "5_data_full_selftest" $P -u tools/art_vae/data.py --selftest
if [ ! -f "$OUT/.selftest_ok_data" ]; then
  $P -u tools/art_vae/selftest.py --device cuda > "$OUT/step_5b_selftest_all.log" 2>&1 \
    || { tail -n 30 "$OUT/step_5b_selftest_all.log"; exit 1; }
  touch "$OUT/.selftest_ok_data"
fi

# ------------------------------------------------------------------ 6. smoke train (OOM / throughput)
if [ ! -f "$OUT/.smoke_ok" ]; then
  run "6_smoke_train" $P -u tools/art_vae/train.py --stage A --updates 5 --seed 0 --device cuda
  # smoke artifacts must not be mistaken for a real stage-A checkpoint
  rm -f "$OUT/A_best.pt" "$OUT/A_last.pt" "$OUT/A_metrics.jsonl" \
        "$OUT/train_A.log" "$OUT/train_A_summary.json"
  touch "$OUT/.smoke_ok"
else
  log "6. smoke train already passed (skip)"
fi

# ------------------------------------------------------------------ 7. stage A
if [ ! -f "$OUT/A_best.pt" ]; then
  run "7_train_A" $P -u tools/art_vae/train.py --stage A --updates 500 --seed 0 --device cuda
else
  log "7. stage A checkpoint exists (skip)"
fi
run "8_eval_A" $P -u tools/art_vae/eval.py --ckpt "$OUT/A_best.pt" --tag A --baselines a0,s10 --device cuda

# ------------------------------------------------------------------ 9. gate A
log "9. stage-A gate (does the objective shape the latent?)"
set +e
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
import re
lines = [l for l in (out / "train_A.log").read_text().splitlines() if l.strip()]
# Use the FROZEN-reference-batch loss (ref_style), not the per-update SupCon
# value: the latter swings ~0.1 with batch composition and a first-vs-last
# difference would be a coin flip. Segment means over the first/last quarter
# are robust to that noise.
refs = [float(re.search(r"ref_style=([0-9.]+)", l).group(1)) for l in lines
        if "ref_style=" in l]
if not refs:
    raise RuntimeError("no ref_style values in train_A.log")
q = max(1, len(refs) // 4)
early = sum(refs[:q]) / q
late = sum(refs[-q:]) / q
d_style = early - late
ev = json.loads((out / "eval_A.json").read_text())
a = ev.get("new", ev.get("A", {}))
a0 = ev.get("a0", {})
s10 = ev.get("s10", {})
d_auroc = a.get("auroc", 0) - a0.get("auroc", 0)
d_lpips = a.get("recon", {}).get("lpips", 9) - s10.get("recon", {}).get("lpips", 9)
ok = (d_style > 0.01) and (d_auroc > 0.005) and (d_lpips < 0.02)
res = {"ref_style_early": round(early, 5), "ref_style_late": round(late, 5),
       "ref_style_drop": round(d_style, 5), "auroc_gain_vs_a0": round(d_auroc, 4),
       "lpips_delta_vs_s10": round(d_lpips, 5), "passed": bool(ok),
       "criterion": "ref_style quarter-mean drop>0.01 (noise-free), "
                    "AUROC gain>0.005, LPIPS regression<0.02"}
(out / "gate_A.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
if not ok:
    print("GATE A FAILED: stop the chain; do not run B/C on a failed objective.")
    sys.exit(2)
PY
GATE=$?
set -e

# ------------------------------------------------------------------ 10/11. stage B
if [ "$GATE" -eq 0 ]; then
  if [ ! -f "$OUT/B_best.pt" ]; then
    run "10_train_B" $P -u tools/art_vae/train.py --stage B --updates 3000 \
        --init-from "$OUT/A_best.pt" --seed 1 --device cuda
  else
    log "10. stage B checkpoint exists (skip)"
  fi
  run "11_eval_B" $P -u tools/art_vae/eval.py --ckpt "$OUT/B_best.pt" --tag B \
      --baselines a0,s10 --device cuda
else
  log "stage B/C SKIPPED because gate A failed"
fi

# ------------------------------------------------------------------ 12/13. stage C
if [ "$GATE" -eq 0 ] && [ -f "$OUT/B_best.pt" ]; then
  if [ ! -f "$OUT/C_best.pt" ]; then
    run "12_train_C" $P -u tools/art_vae/train.py --stage C --updates 1500 \
        --init-from "$OUT/B_best.pt" --seed 2 --device cuda
  else
    log "12. stage C checkpoint exists (skip)"
  fi
  run "13_eval_C" $P -u tools/art_vae/eval.py --ckpt "$OUT/C_best.pt" --tag C \
      --baselines a0,s10 --device cuda
fi

# ------------------------------------------------------------------ 14. final report
log "14. final report"
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
report = {"stages": {}, "gate_A": None, "verdict": None}
g = out / "gate_A.json"
if g.exists():
    report["gate_A"] = json.loads(g.read_text())
for t in ("A", "B", "C"):
    f = out / f"eval_{t}.json"
    if f.exists():
        d = json.loads(f.read_text())
        report["stages"][t] = {k: v for k, v in d.items() if k in ("a0", "s10", "new", t)}
(out / "FINAL_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("Saved", out / "FINAL_REPORT.json")
for t, d in report["stages"].items():
    for k in ("a0", "s10", "new", t):
        if k in d and isinstance(d[k], dict) and "auroc" in d[k]:
            r = d[k]
            print(f"  {t:<3}{k:<6} AUROC={r['auroc']:.4f} retr={r['retrieval_5shot']*100:.1f}% "
                  f"unseen={r.get('unseen_retrieval')} LPIPS={r['recon']['lpips']:.5f}")
PY

log "SERIAL PIPELINE COMPLETE"
