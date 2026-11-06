#!/usr/bin/env bash
# Append one line to the pipeline progress file.
set -u
PROG=/mnt/i/Latent_Style/WEAVE/runs/pgt/PROGRESS.md
printf '%s\n' "$*" >> "$PROG"
tail -n 3 "$PROG"
