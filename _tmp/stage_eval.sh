#!/usr/bin/env bash
# 分析阶段：T0 FM 探针 + supp 分析（pool1）+ 多样本池稳定性 + 作者均衡 + P4 汇总
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
PV=$(python3 -c "from tools.pseudo_gt.build_assets import pgt_root; print(pgt_root()/'pairs_val')")

CK0=$(ls -1v runs/pgt/S0_base_b256/step_*.pt 2>/dev/null | tail -1)
if [ -z "${CK0:-}" ]; then CK0=$(ls -1v runs/pgt/S0_base_b256/epoch_*.pt | tail -1); fi
log "T0 fm probe ckpt=$CK0"
python3 -u tools/pseudo_gt/fm_probe.py --checkpoint "$CK0" --arm S0_base_b256 \
  --probe-cache "R64_pos=$PV/pattern_R64_positive" --probe-cache "F64_pos=$PV/pattern_F64_positive" \
  --probe-cache "R64_rev=$PV/pattern_R64_reverse" --probe-cache "F64_rev=$PV/pattern_F64_reverse" \
  --out runs/p2/fm_S0_base_b256.json > runs/p2/fm_S0_base_b256.log 2>&1 \
  && log "T0 fm_probe done" || { log "FAILED fm T0"; tail -n 12 runs/p2/fm_S0_base_b256.log; }

log "supp analyze (pool1)"
python3 -u tools/pseudo_gt/supp_analyze.py --records runs/p2/supp_records.jsonl \
  --out runs/p2/supp_summary.md --json runs/p2/supp_summary.json > runs/p2/supp_analyze.log 2>&1 \
  && log "supp analyze done" || { log "FAILED supp analyze"; tail -n 12 runs/p2/supp_analyze.log; }

log "supp stability (pool1/2/3)"
python3 -u tools/pseudo_gt/supp_stability.py \
  --pool "pool1=runs/p2/supp_records.jsonl" --pool "pool2=runs/p2/supp_pool2.jsonl" \
  --pool "pool3=runs/p2/supp_pool3.jsonl" \
  --out runs/p2/supp_stability.md --json runs/p2/supp_stability.json > runs/p2/supp_stability.log 2>&1 \
  && log "supp stability done" || { log "FAILED supp stability"; tail -n 12 runs/p2/supp_stability.log; }

log "author balance check"
python3 -u tools/pseudo_gt/author_check.py --out runs/p2/proto_author_check.json \
  > runs/p2/author_check.log 2>&1 \
  && log "author check done" || { log "FAILED author check"; tail -n 12 runs/p2/author_check.log; }

log "p4 summary"
python3 -u tools/pseudo_gt/p4_summary.py > runs/p2/p4_summary.log 2>&1 \
  && log "p4 summary done" || { log "FAILED p4 summary"; tail -n 12 runs/p2/p4_summary.log; }

log "STAGE_EVAL ALL DONE"
