#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== all arm statuses (raw state) ==="
python3 - <<'PY'
import json
from pathlib import Path
s=json.loads(Path("runs/blog_sbflow/suite40/suite_state.json").read_text())
for k in sorted(s):
    v=s[k]
    print(f"  {k} status={v.get('status')} error={v.get('error','')!r} "
          f"ckpt={Path(v.get('checkpoint','')).name if v.get('checkpoint') else '-'} "
          f"eval={bool(v.get('eval'))}")
PY
echo "=== exp10 checkpoint + score ==="
ls -lh "$ROOT/exp10/model_step4000.pt" "$ROOT/score_10_quick200/style_scores.json" 2>/dev/null || true
echo "=== exp24 train tail ==="
tail -n 3 "$ROOT/logs/24_train.log" 2>/dev/null || true
echo "=== rest watcher ==="
ps -eo pid,etime,args --no-headers | grep -E "continue_suite40|--arms rest" | grep -v grep | cut -c1-120 || echo "  none"
