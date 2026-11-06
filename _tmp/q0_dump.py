import json, sys
p = "/mnt/i/Latent_Style/WEAVE/runs/p2/pgt_q0.json"
d = json.load(open(p))["configs"]
print("E configs:", [k for k in d if k.startswith("E")])
for k in d:
    if not k.startswith("E"):
        continue
    print(k, "n=", d[k]["n"],
          {m: round(d[k]["csd_" + m]["mean"], 4) for m in ("dT", "dS", "dM")},
          "ci_dM", [round(v, 4) for v in (d[k]["csd_dM"]["ci"]["lo"], d[k]["csd_dM"]["ci"]["hi"])],
          "legacy_dT_ci", [round(v, 4) for v in (d[k]["csd_dT"]["ci"]["lo"], d[k]["csd_dT"]["ci"]["hi"])])
print()
for k in ("A_P5", "C_P5", "D_P5", "D_HF_P5", "A_P1", "A_P2", "A_P3"):
    print(k, {b: round(d[k]["band_" + b], 4) for b in ("LL", "LH", "HL", "HH")})
