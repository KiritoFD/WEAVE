#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== candidate historical task manifests ==="
find . -path ./datasets -prune -o -name "*heldout*" -print 2>/dev/null | head -20
find . -path ./datasets -prune -o -name "*tasks*750*" -print 2>/dev/null | head
echo
echo "=== eval_protocol dir ==="
ls -la eval_protocol/ 2>/dev/null | head -15
echo
echo "=== sha of my heldout400 task file ==="
sha256sum runs/v6_24h/heldout400_tasks.json 2>/dev/null
echo
echo "=== does the historical report record a manifest sha? ==="
grep -inE "sha|manifest|task_sha" docs/v6/STYLE_DIAGNOSIS_AND_CALIBRATION_REPORT.md 2>/dev/null | head -10 || echo "  none"
