import json

d = json.load(open("/mnt/i/Latent_Style/WEAVE/runs/p2/pgt_p0_audit.json"))
print("== specs ==")
for s, v in d["specs"].items():
    print(f"{s}: n={v['n']} failed={v['failed']} nominal={v['nominal_coverage']:.4f} "
          f"cov_mean={v['coverage_mean']:.4f} med={v['coverage_median']:.4f} "
          f"min={v['coverage_min']:.4f} patches={v['patches_mean']:.2f}/{v['nominal_patches']} "
          f"drop_img={v['frac_images_with_drop']:.2f} q={v['q_mean']:.3f}")
print("== cache ==")
for a, v in d["cache"].items():
    print(a, {k: v.get(k) for k in ("records", "rows", "dup", "sources", "specs", "methods",
                                     "nominal_cov_mean", "effective_cov_mean", "effective_equals_nominal")})
    print("   recompute:", v.get("recompute"))
print("== reverse ==")
for k, v in d["reverse_algebra"].get("configs", {}).items():
    print(k, {m: round(v[m], 4) for m in ("dT_R_mean", "dS_R_mean", "dM_R_mean", "legacy_dT_mean", "legacy_dS_mean")},
          "dT_R_ci", [round(v["dT_R_ci"]["lo"], 4), round(v["dT_R_ci"]["hi"], 4)])
print("== bands ==")
for k, v in d["band_audit"].items():
    print(k, v)
