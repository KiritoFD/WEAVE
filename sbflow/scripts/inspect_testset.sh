#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== data/test tree ==="
for d in data/test/*/; do
  n=$(ls -1 "$d" 2>/dev/null | wc -l)
  s=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "  %-40s %5s files  %6s\n" "$d" "$n" "$s"
done
echo
echo "=== total ==="
du -sh data/test 2>/dev/null
echo
echo "=== images referenced by the 750 protocol ==="
python3 - <<'PY'
import json
from pathlib import Path
spec = json.loads(Path("eval_protocol/tasks_750.json").read_text())
tasks = spec["tasks"]
paths = {t["source_rel_path"] for t in tasks}
print(f"  unique source images referenced: {len(paths)}")
missing = [p for p in paths if not Path(p).exists()]
print(f"  missing on disk: {len(missing)}")
for p in missing[:5]:
    print(f"    {p}")
sizes = [Path(p).stat().st_size for p in paths if Path(p).exists()]
if sizes:
    print(f"  total size: {sum(sizes)/1024/1024:.1f} MB "
          f"(mean {sum(sizes)/len(sizes)/1024:.0f} KB)")
PY
echo
echo "=== reference sets used by the metrics ==="
ls -la runs/v5/cache/*.pt 2>/dev/null | awk '{printf "  %-50s %8s\n", $NF, $5}'
ls -la /mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt 2>/dev/null | awk '{printf "  %-50s %8s\n", $NF, $5}'
