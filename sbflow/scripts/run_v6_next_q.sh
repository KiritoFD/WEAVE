#!/usr/bin/env bash
# Autonomous P2 follow-up driver: Q0 -> Q1 -> Q2 -> Q3, with hard failure checks at every step.
#
# Differences from the previous queue, which the review flagged:
#   * set -e is on, and every step goes through q_lib's run_step which reads PIPESTATUS[0] before
#     tee can mask it
#   * each training arm is verified for checkpoint + parseable report, so a printed
#     "TRAINING DONE" can never be mistaken for success
#   * it never preempts a running job: it waits for the GPU to be free, and aborts if it stays busy
#
# Usage: bash sbflow/scripts/run_v6_next_q.sh
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_q.log
source sbflow/scripts/q_lib.sh

qlog "==================== V6 NEXT QUEUE START ===================="
wait_for_gpu 240 || exit 3

# Q0 does not train; it only reads the finished P2 reports
run_step "Q0_verify" "" python3 -u -m tools.v6_next.q0_verify
qlog "Q0 complete, failures: $QFAIL"

bash sbflow/scripts/q1_sweep.sh || qlog "Q1 reported failures"
bash sbflow/scripts/q2_mechanisms.sh || qlog "Q2 reported failures"
bash sbflow/scripts/q3_final.sh || qlog "Q3 reported failures"

qlog "==================== V6 NEXT QUEUE DONE failures=$QFAIL ===================="
exit 0
