import json
from pathlib import Path

STAGE = Path("/mnt/i/Latent_Style/WEAVE/_baseline_stage")
if not STAGE.exists():
    print("no stage")
    raise SystemExit
for ds in sorted(p for p in STAGE.iterdir() if p.is_dir()):
    print(f"\n=== {ds.name} ===")
    jroot = ds / "jpg"
    if jroot.is_dir():
        for m in sorted(p for p in jroot.iterdir() if p.is_dir()):
            n = len(list(m.glob("*.jpg")))
            mb = sum(p.stat().st_size for p in m.glob("*.jpg")) / 1048576
            print(f"  jpg/{m.name:12s} {n:4d} imgs {mb:7.1f} MB")
    sroot = ds / "sheets"
    if sroot.is_dir():
        print(f"  sheets: {len(list(sroot.glob('*.jpg')))}")
    mroot = ds / "metrics"
    if mroot.is_dir():
        print(f"  metrics: {sorted(p.name for p in mroot.glob('*.csv'))}")
