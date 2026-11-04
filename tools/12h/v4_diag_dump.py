import json

d = json.load(open("runs/v4/diag/best.json"))
m = d["mapping"]
print("spot_n", len(m["spot_checks"]), "all_equal", m["spot_all_equal"],
      "maxdiff", max(s["max_abs_diff"] for s in m["spot_checks"]))
print("per_style_mean", m["per_style_latent_mean"], "std", m["per_style_latent_std"])
for f in ("overfit_synth_lr1e3", "overfit_real_lr1e3"):
    j = json.load(open(f"runs/v4/diag/{f}.json"))
    print(f, "L_true", j["final"]["overall"]["true"], "d_null%", j["final"]["d_null_pct"],
          "d_wrong%", j["final"]["d_wrong_pct"], "mid_dnull%", j["final"]["mid_snr_d_null_pct"],
          "last_loss", j["curve"][-1]["loss"], "embed_norm", j["style_embed_norm"],
          "film_norm", j["film_proj_norm_mean"])

