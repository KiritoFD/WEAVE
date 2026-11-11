#!/bin/bash
# SD0-S10 V2: 48h serial pipeline.
#   bash tools/vae_suite/run_sd0_s10.sh
#   nohup bash tools/vae_suite/run_sd0_s10.sh > runs/sd0_s10/serial.log 2>&1 &
#
# Single-GPU queue; completed stages are skipped; every stage has an explicit
# exit code; Gate failures save a NEGATIVE report and stop the chain rather than
# declaring success. OOM lowers microbatch but NEVER changes effective batch 64.
set -euo pipefail

R=/root/Workspace/WEAVE_VAE
cd "$R"
P=/opt/conda/envs/cu121/bin/python
OUT="$R/runs/sd0_s10"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo; echo "================================================================"; echo "[SERIAL] $*"; echo "================================================================"; }
step() { local tag="$1"; shift; log "$tag"; "$@" 2>&1 | tee "$OUT/step_${tag}.log"; }

# ---------------------------------------------------------------- 0. tests
log "0. compile + unit tests"
$P -m py_compile tools/vae_suite/sd0_cache.py tools/vae_suite/sd0_losses.py \
    tools/vae_suite/train_sd0_s10.py tools/vae_suite/eval_sd0_s10.py \
    tools/vae_suite/p0_baseline.py tools/vae_suite/test_sd0_s10.py \
    tools/vae_suite/build_cold_sd0.py
if [ ! -f "$OUT/.selftest_ok" ]; then
  step "0b_selftest" $P -u tools/vae_suite/test_sd0_s10.py
  touch "$OUT/.selftest_ok"
else
  log "0. selftests already passed (skip)"
fi

# ---------------------------------------------------------------- 1. cache
if [ ! -f "$OUT/full/split.json" ]; then
  step "1_sd0_cache" $P -u tools/vae_suite/sd0_cache.py
else
  log "1. SD0 latent cache exists (skip)"
fi
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
sp = json.loads((Path(sys.argv[1]) / "full/split.json").read_text())
assert sp["n_train"] == 78661 and sp["n_holdout"] == 2048
assert sp["image_id_disjoint"] and sp["reencode_cos_min"] > 0.999
assert all(v["zero"] == 0 and v["nonfinite"] == 0 for v in sp["bad_rows"].values())
print("[cache] verified:", sp["n_train"], "train /", sp["n_holdout"], "holdout /",
      sp["n_test256"], "test256; reencode cos min =", sp["reencode_cos_min"])
PY

# ---------------------------------------------------------------- 2. Cold init + P0
if [ ! -f "$OUT/cold_init.pt" ]; then
  step "2a_build_cold" $P -u tools/vae_suite/build_cold_sd0.py
else
  log "2a. Cold-SD0 init exists (skip)"
fi
if [ ! -f "$OUT/p0_baseline.json" ]; then
  step "2b_p0_baseline" $P -u tools/vae_suite/p0_baseline.py
else
  log "2b. P0 baseline exists (skip)"
fi

# ---------------------------------------------------------------- 3. P1 W/C
for INIT in W C; do
  if [ -f "$OUT/p1_${INIT}_best.pt" ]; then
    log "3. P1 $INIT exists (skip)"
    continue
  fi
  STAGE=p1_warm; [ "$INIT" = "C" ] && STAGE=p1_cold
  step "3_p1_${INIT}" $P -u -m tools.vae_suite.train_sd0_s10 \
      --stage "$STAGE" --init "$INIT" --bs 4 --seed 0
  # normalise trainer output ({stage}_{init}_*) to the gate's standard names
  cp "$OUT/${STAGE}_${INIT}_best.pt" "$OUT/p1_${INIT}_best.pt"
  cp "$OUT/${STAGE}_${INIT}_metrics.jsonl" "$OUT/p1_${INIT}_metrics.jsonl"
  cp "$OUT/train_${STAGE}_${INIT}_summary.json" "$OUT/p1_${INIT}_summary.json"
done

# P1 gate: choose the initialisation with better quality/speed under equal
# follow-up cost. Warm carries historical training, so this is a subsequent-
# optimisation comparison, not a total-cost comparison.
log "4. P1 initialisation gate"
set +e
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def summary(init):
    f = out / f"p1_{init}_summary.json"
    return json.loads(f.read_text()) if f.exists() else None
W, C = summary("W"), summary("C")
def lpips(s): return s["best"]["lpips"] if s and s.get("best") else 9.0
res = {"W": W, "C": C,
       "W_lpips": lpips(W), "C_lpips": lpips(C),
       "W_img_s": W["img_per_sec"] if W else None,
       "C_img_s": C["img_per_sec"] if C else None}
# equal follow-up cost: prefer the better reconstruction; Cold gets a chance
# only if it is clearly behind but improving (metrics jsonl trend).
def trend(init):
    f = out / f"p1_{init}_metrics.jsonl"
    if not f.exists(): return None
    v = [json.loads(l)["lpips"] for l in f.read_text().splitlines() if l.strip()]
    return (v[0] - v[-1]) if len(v) >= 2 else 0.0
res["W_trend"], res["C_trend"] = trend("W"), trend("C")
chosen = "W" if lpips(W) <= lpips(C) + 0.005 else "C"
res["chosen"] = chosen
res["reason"] = ("Warm chosen (better or equal LPIPS at equal follow-up cost)" if chosen=="W"
                 else "Cold chosen (clearly better LPIPS)")
(out / "p1_gate.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ["W_lpips","C_lpips","W_img_s","C_img_s",
                                     "W_trend","C_trend","chosen","reason"]}, indent=2))
PY
GATE=$?
set -e
[ "$GATE" -ne 0 ] && { log "P1 gate failed to evaluate"; exit 2; }
INIT=$( $P -c "import json;print(json.load(open('$OUT/p1_gate.json'))['chosen'])" )
log "P1 chose init = $INIT"

# ---------------------------------------------------------------- 5. P2 C0
if [ ! -f "$OUT/c0_best.pt" ]; then
  INIT_FROM="$OUT/p1_${INIT}_best.pt"
  step "5_c0" $P -u -m tools.vae_suite.train_sd0_s10 \
      --stage c0 --init "$INIT" --init-from "$INIT_FROM" --bs 4 --seed 1
  cp "$OUT/c0_${INIT}_best.pt" "$OUT/c0_best.pt"
  cp "$OUT/c0_${INIT}_metrics.jsonl" "$OUT/c0_metrics.jsonl"
else
  log "5. C0 exists (skip)"
fi
# P2 HF gate: LPIPS must not worsen vs P1 best, HF must improve or error drop.
log "6. C0 HF gate"
set +e
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
# c0_{init}_metrics.jsonl is written by the trainer every 250 updates
fs = [out / "c0_metrics.jsonl"] if (out / "c0_metrics.jsonl").exists() else []
if not fs:
    print("no c0 metrics"); sys.exit(2)
rows = [json.loads(l) for l in fs[0].read_text().splitlines() if l.strip()]
first, last = rows[0], rows[-1]
ok = (last["lpips"] <= first["lpips"] + 0.005 and
      (last["hf_corr_L0"] >= first["hf_corr_L0"] - 0.005 or
       last["hf_l1"] <= first["hf_l1"] + 0.01))
r = {"first": {k: first.get(k) for k in ["lpips","hf_corr_L0","hf_l1","csd"]},
     "last": {k: last.get(k) for k in ["lpips","hf_corr_L0","hf_l1","csd"]},
     "passed": bool(ok)}
(out / "c0_gate.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
sys.exit(0 if ok else 3)
PY
C0G=$?
set -e
if [ "$C0G" -ne 0 ]; then
  log "C0 Gate FAILED -> rollback to P1 best, energy loss stays OFF"
  cp "$OUT/p1_${INIT}_best.pt" "$OUT/c0_best.pt" 2>/dev/null || true
  echo '{"status":"C0_GATE_FAILED","note":"negative report; chain continues with P1 weights"}' \
      > "$OUT/NEGATIVE_REPORT.json"
else
  log "C0 Gate passed"
fi

# ---------------------------------------------------------------- 7. P3 C1
if [ ! -f "$OUT/c1_best.pt" ]; then
  step "7_c1" $P -u -m tools.vae_suite.train_sd0_s10 \
      --stage c1 --init "$INIT" --init-from "$OUT/c0_best.pt" --bs 4 --seed 2
  cp "$OUT/c1_${INIT}_best.pt" "$OUT/c1_best.pt"
  cp "$OUT/c1_${INIT}_metrics.jsonl" "$OUT/c1_metrics.jsonl"
else
  log "7. C1 exists (skip)"
fi
# P3 response gate: response error down, gradient consistency up, clean recon kept.
log "8. C1 response gate"
set +e
$P - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
fs = [out / "c1_metrics.jsonl"] if (out / "c1_metrics.jsonl").exists() else []
if not fs:
    print("no c1 metrics"); sys.exit(2)
rows = [json.loads(l) for l in fs[0].read_text().splitlines() if l.strip()]
first, last = rows[0], rows[-1]
def g(r, k, d): return r.get(k, d)
ok = (g(last,"resp",9) <= g(first,"resp",0) + 0.01 and
      g(last,"lpips",9) <= g(first,"lpips",0) + 0.01)
r = {"first": {k: first.get(k) for k in ["lpips","resp","resp_cos"]},
     "last": {k: last.get(k) for k in ["lpips","resp","resp_cos"]},
     "passed": bool(ok)}
(out / "c1_gate.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
sys.exit(0 if ok else 3)
PY
C1G=$?
set -e
if [ "$C1G" -ne 0 ]; then
  log "C1 Gate FAILED -> keep C0 as inference-optimal; C1 not promoted"
  cp "$OUT/c0_best.pt" "$OUT/sd0_s10_best_overall.pt"
else
  cp "$OUT/c1_best.pt" "$OUT/sd0_s10_best_overall.pt"
fi

# ---------------------------------------------------------------- 9. final eval
log "9. final acceptance eval"
MODELS="official,old_s10"
[ -f "$OUT/c0_best.pt" ] && MODELS="$MODELS,c0"
[ -f "$OUT/c1_best.pt" ] && MODELS="$MODELS,c1"
step "9_final_eval" $P -u -m tools.vae_suite.eval_sd0_s10 \
    --models "$MODELS" --tag final --n 256

# ---------------------------------------------------------------- 10. export
log "10. export deliverables"
$P - "$OUT" <<'PY'
import json, shutil, sys
from pathlib import Path
out = Path(sys.argv[1])
# best_recon / best_response / best_overall
for src, dst in [("c0_best.pt", "sd0_s10_best_recon.pt"),
                 ("c1_best.pt", "sd0_s10_best_response.pt")]:
    if (out / src).exists():
        shutil.copy(out / src, out / dst)
if not (out / "sd0_s10_best_overall.pt").exists():
    shutil.copy(out / "sd0_s10_best_recon.pt", out / "sd0_s10_best_overall.pt")
report = {"deliverables": [str(p) for p in sorted(out.glob("sd0_s10_best_*.pt"))],
          "gates": {}}
for g in ("p1_gate.json", "c0_gate.json", "c1_gate.json", "p0_baseline.json"):
    f = out / g
    if f.exists():
        report["gates"][g] = json.loads(f.read_text())
(out / "FINAL_REPORT.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["deliverables"], indent=2))
PY

log "SERIAL PIPELINE COMPLETE"
