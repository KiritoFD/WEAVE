#!/usr/bin/env bash
# 串行：P0 审计 -> 小批量基准（严格单进程，避免 15GB 内存 OOM）
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
bash _tmp/run_p0_audit.sh
rc=$?
if [ $rc -ne 0 ]; then echo "P0 failed"; exit 1; fi
bash _tmp/run_supp_bench.sh
