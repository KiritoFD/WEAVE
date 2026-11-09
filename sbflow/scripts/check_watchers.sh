#!/usr/bin/env bash
set -uo pipefail
echo "=== session command lines ==="
for s in suite40 suite40_continue suite40_full750 suite40_trend; do
  echo "  -- $s --"
  tmux list-panes -t "$s" -F '    pane=#{pane_id} pid=#{pane_pid} cmd=#{pane_current_command}' 2>/dev/null || echo "    missing"
done
echo "=== continuation script on 3060 (tail) ==="
tail -n 8 /mnt/i/Latent_Style/WEAVE/sbflow/scripts/continue_suite40_rest.sh 2>/dev/null || true
echo "=== tmux start script (continuation) ==="
grep -n "tmux new-session" /mnt/i/Latent_Style/WEAVE/sbflow/scripts/start_continuation_tmux.sh 2>/dev/null || true
