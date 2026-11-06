#!/usr/bin/env bash
tmux kill-session -t p0audit 2>/dev/null || true
tmux new-session -d -s p0audit "bash /mnt/i/Latent_Style/WEAVE/_tmp/run_p0_audit.sh; sleep 5"
echo STARTED p0audit
