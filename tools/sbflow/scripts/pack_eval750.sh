#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
mkdir -p /tmp/sync_eval750
tar czf /tmp/sync_eval750/eval750.tgz runs/v6_next/eval_750_*/eval.json
base64 -w0 /tmp/sync_eval750/eval750.tgz
