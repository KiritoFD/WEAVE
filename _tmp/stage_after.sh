#!/usr/bin/env bash
# 修旧缓存 manifest 的 effective_coverage（重算实测），然后串到 P4 训练阶段
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

pgrep -af "supp_eval|p0_audit|build_pattern|train.py|pgt_render|fm_probe" && { log "ABORT: 仍有残留进程"; exit 1; }
free -g | head -2

log "coverage fix start (base/positive/reverse)"
python3 -u tools/pseudo_gt/fix_effective_coverage.py > runs/p2/coverage_fix.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED coverage fix rc=$rc"; tail -n 20 runs/p2/coverage_fix.log; exit 1; fi
log "coverage fix done: $(tail -n 2 runs/p2/coverage_fix.log | head -1 | cut -c1-200)"

bash _tmp/stage_train.sh
rc=$?
[ $rc -ne 0 ] && log "STAGE_TRAIN FAILED rc=$rc" || log "STAGE_TRAIN OK"

bash _tmp/stage_eval.sh
rc2=$?
[ $rc2 -ne 0 ] && log "STAGE_EVAL FAILED rc=$rc2" || log "STAGE_EVAL OK"
log "STAGE_ALL_AFTER_SUPP done"
