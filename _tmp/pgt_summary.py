"""汇总三臂：600 跨风格 / 150 同风格 / 750 历史口径 + 早停触发情况。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["S0_base_b256", "S1_positive_b256", "S2_reverse_b256", "S2_reverse_e6_b256",
         "S0_base", "S1_positive", "S2_reverse"]


def fmt(s):
    return (f"mean={s.get('mean', float('nan')):+.4f} med={s.get('median', float('nan')):+.4f} "
            f"frac>0={s.get('frac_positive', float('nan')):.3f} "
            f"CI95=[{s.get('ci95_low', float('nan')):+.4f},{s.get('ci95_high', float('nan')):+.4f}]")


print("# PGT summary\n")
for cell in CELLS:
    print(f"## {cell}\n")
    mp = ROOT / "runs" / "p2" / f"pgt_{cell}.json"
    if not mp.exists():
        print("- metrics json 缺失（未观测）\n")
        continue
    m = json.loads(mp.read_text(encoding="utf-8"))
    agg = m.get("metrics", {}).get("metrics", {})
    print("### 600 跨风格")
    for k in ("csd_dT", "csd_dS", "csd_dM", "dino_dT", "dino_dS", "dino_dM", "swd_gain"):
        if k in agg:
            print(f"- {k:9s} {fmt(agg[k])}")
    print(f"- n_pairs={m.get('n_cross_pairs')} n_sources={m.get('n_sources')}")
    same = m.get("same_style_150", {}).get("metrics", {})
    print("\n### 150 同风格")
    for k in ("csd_dStyle", "dino_dStyle", "dino_content_gen", "dino_content_base", "lpips_gen", "lpips_base"):
        if k in same:
            print(f"- {k:16s} {fmt(same[k])}")
    hist = m.get("historical_750", {}).get("new_render", {}).get(cell, {})
    print("\n### 750 历史口径")
    print(f"- n={hist.get('n')} dino_style_similarity={hist.get('dino_style_similarity_mean')} "
          f"lpips={hist.get('lpips_mean')}")
    quad = m.get("quadrants", {})
    print("\n### quadrants")
    for tag, d in quad.items():
        print(f"- {tag}: {d}")
    es_path = ROOT / "runs" / "pgt" / cell / "internal_dynamics_summary.json"
    print("\n### 早停")
    if es_path.exists():
        es = json.loads(es_path.read_text(encoding="utf-8"))
        for k in ("early_stop_armed", "fired", "fired_epoch", "stop_reason", "rho_series", "gate_mean_series"):
            print(f"- {k}: {es.get(k)}")
    else:
        print("- internal_dynamics_summary.json 缺失（未观测）")
    print()
