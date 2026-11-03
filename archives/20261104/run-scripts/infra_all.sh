#!/usr/bin/env bash
# Infra acceptance queue. Waits for the old P0 queue (cap32/cap128) to finish,
# then runs smoke -> fired demo -> closed loop -> equal-sample parity -> bench.
# No ACD scan, no 30k curve, no Route B training (user decision 2026-11-04).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
R=/mnt/i/Latent_Style/runs/p0_diag
mkdir -p "$R"
while pgrep -f 'bash runs/p0_all.sh' >/dev/null 2>&1; do
  echo "[wait] p0_all still running $(date '+%T')" | tee -a "$R/infra_all.log"
  sleep 20
done
echo "===== $(date '+%F %T') INFRA ALL START =====" | tee -a "$R/infra_all.log"
if ! bash /mnt/i/Latent_Style/WEAVE/runs/es_smoke.sh; then
  echo "SMOKE_FAILED: not queueing the rest" | tee -a "$R/infra_all.log"
  exit 1
fi
bash /mnt/i/Latent_Style/WEAVE/runs/es_fire_demo.sh
bash /mnt/i/Latent_Style/WEAVE/runs/infra_loop.sh
bash /mnt/i/Latent_Style/WEAVE/runs/parity_equal_samples.sh
bash /mnt/i/Latent_Style/WEAVE/runs/infra_bench.sh
echo INFRA_ALL_DONE | tee -a "$R/infra_all.log"
