#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== eval_protocol contents ==="
ls -la eval_protocol/ 2>/dev/null
echo
echo "=== any 'ranking' eval set mentioned in docs/protocol? ==="
grep -rilE "排名|ranking|rank_eval|leaderboard" eval_protocol/ docs/v6/*.md 2>/dev/null | head -10
echo
echo "=== existing 750-protocol evals ==="
ls -d runs/v6_next/eval_750_* 2>/dev/null
echo
echo "=== does L11 have a 750 eval? ==="
ls -d runs/v6_next/eval_750_*L11* runs/v6_24h/eval_750_* 2>/dev/null || echo "  none"
