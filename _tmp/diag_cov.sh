#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- covfix proc ---"; ps -o pid,etime,time,%cpu,rss,cmd -p 269249 2>/dev/null || echo gone
echo "--- log size ---"; ls -la runs/p2/covfix_reverse.log
echo "--- py stack (best effort) ---"; command -v py-spy >/dev/null && py-spy dump --pid 269249 2>&1 | head -n 30 || echo "no py-spy"
echo "--- stage_a2 ---"; tail -n 8 runs/p2/stage_a2.out 2>/dev/null | cut -c1-250
