#!/usr/bin/env bash
# Wait for Style-REPA 500-step training to finish, then run the two post-hoc diagnostics.
set -u
cd /root/Workspace/WEAVE_VAE
PY=/opt/conda/envs/cu121/bin/python
DIAG=runs_vae/v2_diagnosis

echo "[chain] waiting for train_repa.py to exit..."
while pgrep -f "tools/vae_suite/train_repa.py" > /dev/null 2>&1; do
    sleep 30
done
echo "[chain] training process exited at $(date)"

sleep 10

echo "[chain] === 1/2 positive/negative decomposition ==="
$PY -u tools/vae_suite/repa_pos_neg_decomp.py > $DIAG/repa_pos_neg.log 2>&1
echo "[chain] pos/neg done at $(date)"

echo "[chain] === 2/2 content-matched AUROC ==="
$PY -u tools/vae_suite/repa_content_matched.py > $DIAG/repa_content_matched.log 2>&1
echo "[chain] content-matched done at $(date)"

echo "[chain] ALL DONE"
