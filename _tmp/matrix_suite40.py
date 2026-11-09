"""Compact arm x target-style matrices for the interim document."""
import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE/runs/blog_sbflow/suite40")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SHORT = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
         "Rococo": "Roc", "Ukiyo_e": "Uki"}

for metric in ("target_top1", "csd_dT", "csd_swd"):
    print(f"\n### {metric}")
    print(f"{'id':4s}" + "".join(f"{SHORT[s]:>9s}" for s in STYLES))
    for aid in ("01", "04", "08", "10", "24", "27"):
        p = ROOT / f"score_{aid}_quick200" / "style_scores.json"
        if not p.exists():
            continue
        bt = json.loads(p.read_text())["aggregate"]["by_target"]
        row = "".join(f"{bt[s][metric]:9.4f}" if s in bt else f"{'-':>9s}" for s in STYLES)
        print(f"{aid:4s}{row}")

# direction-level positivity and worst/best directions for the baseline
p = ROOT / "score_01_quick200" / "style_scores.json"
bd = json.loads(p.read_text())["aggregate"]["by_direction"]
pos = sorted([(v["csd_dT"], k) for k, v in bd.items()], reverse=True)
print("\n### exp01 best 5 directions (dT)")
for v, k in pos[:5]:
    print(f"  {k:44s} dT={v:+.4f}")
print("### exp01 worst 5 directions (dT)")
for v, k in pos[-5:]:
    print(f"  {k:44s} dT={v:+.4f}")
