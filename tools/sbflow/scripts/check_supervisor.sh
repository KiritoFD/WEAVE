#!/usr/bin/env bash
set -uo pipefail
echo "=== available supervisors ==="
for t in tmux screen systemd-run setsid nohup disown; do
  p=$(command -v "$t" 2>/dev/null || true)
  [ -n "$p" ] && echo "  $t -> $p" || echo "  $t -> MISSING"
done
echo "=== init ==="
ps -p 1 -o pid,comm --no-headers
echo "=== systemd user ==="
systemctl --user is-system-running 2>&1 | head -2 || true
echo "=== existing tmux ==="
tmux ls 2>&1 | head -5 || true
