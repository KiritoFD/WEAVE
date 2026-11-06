#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- procs ---"; pgrep -af 'build_train_cache|fix_effective|p2_reach' || echo none
echo "--- progress ---"; tail -n 4 runs/gt/PROGRESS.md | cut -c1-200
echo "--- e8 preview ---"; tail -n 4 runs/p2/pairs_reverse_e8_preview.log 2>/dev/null | cut -c1-200
python3 - <<'PY'
import json
from pathlib import Path
R=Path('/mnt/i/Latent_Style/pseudo_gt_24h/pairs')
for d in ('base','positive','reverse','reverse_e8'):
    p=R/d/'manifest.json'
    if not p.exists(): print(d,'MISSING'); continue
    m=json.loads(p.read_text()); recs=m['records']
    cov=[r.get('effective_coverage_measured') for r in recs]
    has=sum(1 for c in cov if c is not None)
    if has:
        import statistics
        hi=sum(1 for c in cov if c is not None and c>=0.22)
        mid=sum(1 for c in cov if c is not None and 0.18<=c<0.22)
        lo=sum(1 for c in cov if c is not None and c<0.18)
        print(f"{d}: n={len(recs)} measured={has} mean={statistics.mean([c for c in cov if c is not None]):.4f} "
              f">=22%:{hi}({hi/len(recs)*100:.1f}%) 18-22%:{mid}({mid/len(recs)*100:.1f}%) <18%:{lo}({lo/len(recs)*100:.1f}%)")
    else:
        print(f"{d}: n={len(recs)} no measured field")
PY
