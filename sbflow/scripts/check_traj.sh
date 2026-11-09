#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== gpu procs ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo "=== python procs ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\." | grep -v grep | cut -c1-110 || echo "  none"
echo "=== traj logs ==="
for f in runs/blog_sbflow/v11/logs/traj_d1.log runs/blog_sbflow/v11/logs/traj_ctrl.log; do
  [ -f "$f" ] && { echo "  -- $f"; tail -n 3 "$f"; }
done
echo "=== traj results ==="
ls -1 runs/blog_sbflow/v11/traj/*/traj_check.md 2>/dev/null || echo "  none"
echo "=== chain pane tail ==="
tmux capture-pane -p -t v11chain 2>/dev/null | grep -v '^$' | tail -12 || echo "  no pane"
