"""Read-only summary of all completed suite40 Quick200 results."""
import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE/runs/blog_sbflow/suite40")
HARD = ["Rococo", "Early_Renaissance", "Ukiyo_e"]
state = json.loads((ROOT / "suite_state.json").read_text())
print(f"{'id':4s} {'top1':>7s} {'margin':>8s} {'dT':>8s} {'dS':>8s} {'swd':>7s} "
      f"{'dino':>6s} {'lpips':>6s} {'posDir':>7s} {'train_s':>8s}")
for aid in sorted(state):
    rec = state[aid]
    e = rec.get("eval") or {}
    if "target_top1" not in e:
        continue
    f = ROOT / f"score_{aid}_quick200" / "style_scores.json"
    agg = json.loads(f.read_text())["aggregate"] if f.exists() else None
    pos = sum(1 for v in agg["by_direction"].values() if v["csd_dT"] > 0) if agg else -1
    print(f"{aid:4s} {e['target_top1']:7.4f} {e['target_margin']:+8.4f} "
          f"{e['csd_dT']:+8.4f} {e['csd_dS']:+8.4f} {e['csd_swd']:7.4f} "
          f"{e.get('dino_c', float('nan')):6.3f} "
          f"{agg['overall']['lpips_to_input'] if agg else float('nan'):6.3f} "
          f"{pos:3d}/20 {rec.get('train_seconds', 0):8.0f}")
    if agg:
        for s in HARD:
            v = agg["by_target"].get(s)
            if v:
                print(f"       {s:22s} top1={v['target_top1']:.3f} "
                      f"dT={v['csd_dT']:+.4f} swd={v['csd_swd']:.4f} "
                      f"mmd={v['csd_mmd2_unbiased']:.4f}")
