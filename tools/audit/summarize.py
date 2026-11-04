"""Print a compact summary of all audit JSON files (for report writing)."""
from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "runs/audit"


def j(name):
    return json.load(open(D / name, encoding="utf-8"))


print("=== S1 ===")
d = j("s1_repr.json")
for k, v in d["representations"].items():
    print(f"{k:11s} d={v['dim']:3d} inter/intra={v['inter_over_intra']:.3f} "
          f"accNc={v['acc_nearest_centroid']:.3f} accLP={v['acc_linear_probe']:.3f} "
          f"per={ {a: round(b,3) for a,b in v['per_style_acc'].items()} }")

print("=== S3 ===")
d = j("s3_distances.json")
for k, v in d["distances"].items():
    print(f"{k:20s} same={v['d_same_mean']:.3f} cross={v['d_cross_mean'] if 'd_cross_mean' in v else v['d_cross_min_mean']:.3f} "
          f"ratio={v['ratio_cross_over_same']:.2f} top1={v.get('top1_retrieval', 'NA')} hardest={v['hardest_pair']}")

print("=== S2 ===")
d = j("s2_confound.json")
for k, v in d.items():
    if k != "content_clusters":
        print(k, json.dumps(v)[:220])

print("=== S4 ===")
d = j("s4_perturb.json")
for k, v in d["perturbations"].items():
    m = v["metrics"]
    print(f"{k:14s} dL={ {a: round(b,4) for a,b in v['delta_loss'].items()} } dinoS={m['dino_s_target']:.4f} "
          f"dinoC={m['dino_c']:.4f} clipS={m['clip_s']:.4f} lpips={m['lpips']:.4f}")

print("=== S8 ===")
d = j("s8_directions.json")
print({k: round(v, 4) for k, v in d["grad_norm"].items()}, d["mean_offdiag_cos"], d["max_pair_cos"], d["effective_rank"])
print(d["subband_energy_frac"])

print("=== S9 ===")
d = j("s9_ascent.json")
for k, v in d["targets"].items():
    m = v["metrics"]
    print(f"{k}: L {v['loss_start']:.4f}->{v['loss_end']:.4f} rms={v['content_rms_change']:.3f} "
          f"dinoS={m['dino_s_target']:.4f} per={ {a: round(b,4) for a,b in m['dino_s_per_style'].items()} } "
          f"dinoC={m['dino_c']:.4f} clipS={m['clip_s']:.4f} lpips={m['lpips']:.3f} sel={v['dino_s_selectivity']:+.4f}")

print("=== S10 ===")
d = j("s10_conflict.json")
for k, v in d["per_style"].items():
    print(k, round(v["cos_s_c_mean"], 4), v["cos_per_chunk"], v["grad_norm_style_vs_content"][0])

print("=== S11 ===")
d = j("s11_stability.json")
print(d["variants"])
print(d["mixed_style"])

print("=== S12 ===")
d = j("s12_degenerate.json")
for k in d["per_loss"]:
    for s in ("ER", "Imp", "Min", "Roc", "Uki"):
        vals = d["per_loss"][k][s]
        best = min(vals, key=vals.get)
        print(f"{k:8s} {s}: floor={vals['real_target_16']:.4f} best={best} ({vals[best]:.4f})")

print("=== S13 ===")
d = j("s13_latent_opt.json")
for k, v in d["per_loss"].items():
    for off, g in v.items():
        print(f"{k:8s} {off} " + " ".join(
            f"{s}:dS{gv['dino_s_gain']:+.3f}/sel{gv['dino_s_selectivity']:+.3f}/dC{gv['dino_c']:.3f}/lp{gv['lpips']:.3f}"
            for s, gv in g.items()))
d = j("s13_sweep.json")
for eps, v in d["sweep"].items():
    print(f"eps={eps} " + " ".join(
        f"{off}:dS{sum(g['dino_s_gain'] for g in gg.values())/len(gg):+.3f}/dC{sum(g['dino_c'] for g in gg.values())/len(gg):.3f}/lp{sum(g['lpips'] for g in gg.values())/len(gg):.3f}"
        for off, gg in v.items()))

print("=== S14 ===")
d = j("s14_coverage.json")
for k, v in d["per_target_style"].items():
    print(k, {o: (round(x["mmd_opt_vs_real"], 3), round(x["mean_pairwise_cos_opt"], 3),
                  round(x["coverage_frac"], 3), round(x["control_real8"]["coverage_frac"], 3))
              for o, x in v.items()})

print("=== S15 ===")
print(json.dumps(j("s15_legacy_fm.json"), indent=1)[:1200])
print("=== S16 ===")
print(json.dumps(j("s16_destyle_residual.json"), indent=1)[:800])
print("=== S5/S6/S7 ===")
d = j("s5_content_parts.json")
for k, v in d["variants"].items():
    print(f"{k:12s} dinoC={v['dino_c']:.4f} dinoS={v['dino_s_target']:.4f} lpips={v['lpips']:.4f} rms={v['latent_rms']:.3f} sub={ {a: round(b,3) for a,b in v['subband_energy'].items()} }")
d = j("s6_content_metrics.json")
for k, v in d["metrics"].items():
    print(k, v["ranking_ok"], round(v["mild_mean"], 3), round(v["severe_mean"], 3), round(v["margin_ratio"], 2))
d = j("s7_pareto.json")
for p in d["points"]:
    print(p)
print("=== CODE ===")
print(json.dumps(j("code_checks.json"), indent=1)[:900])
