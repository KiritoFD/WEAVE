#!/usr/bin/env bash
# Kill every sbflow process EXCEPT the ones owned by the tmux 'v10' session.
set -uo pipefail
TMUX_PID=$(tmux list-panes -t v10 -F '#{pane_pid}' 2>/dev/null | head -1)
echo "tmux v10 pane pid = ${TMUX_PID:-none}"

# collect the tmux session's process subtree so we never kill it
keep=""
if [ -n "${TMUX_PID:-}" ]; then
  keep=$(ps -eo pid,ppid --no-headers | awk -v root="$TMUX_PID" '
    {ppid[$1]=$2}
    END{
      for (p in ppid) {
        q=p; while (q!="" && q!=1) { if (q==root) { print p; break } q=ppid[q] }
      }
      print root
    }')
fi
keep=$(echo "$keep" | tr '\n' ' ')
echo "keeping pids: $keep"

for pid in $(ps -eo pid,args --no-headers | grep -E "sbflow\.(run_suite|gen5|train5|score_cpu)" | grep -v grep | awk '{print $1}'); do
  if echo " $keep " | tr -s ' ' | grep -q " $pid "; then
    echo "  keep  $pid"
  else
    echo "  kill  $pid  $(ps -p $pid -o args= | cut -c1-90)"
    kill -9 "$pid" 2>/dev/null || true
  fi
done
sleep 2
echo "=== remaining ==="
ps -eo pid,etime,args | grep -E "sbflow\." | grep -v grep | cut -c1-120 || echo "  none"
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
