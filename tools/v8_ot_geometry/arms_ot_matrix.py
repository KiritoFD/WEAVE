"""Fixed V8 ablation arms. One dict, so the runner never hard-codes a method.

energy : M0 legacy E3 | M1 centroid | M2 mixture | M3 per-image OT | M3s SWD
         M4 M2+bounded specificity | M5 M3+bounded specificity
geometry: uniform | correct | shuffle | other_source | half | balanced
content : rms | proxy | cg
"""
from __future__ import annotations

M_ENERGIES = {
    "M1": "centroid",
    "M2a": "mixture", "M2b": "mixture_warm",
    "M3": "ot", "M3s": "swd",
    "M4": "mixture_specific", "M5": "ot_specific",
    "M6": "mixture_u", "M7": "mixture_ll", "M8": "mixture_wrong_target",
}

# H1: energy comparison, single step and 4 steps
H1 = [{"id": f"{e}.step{s}", "energy": e, "steps": s, "step_rms": 0.08}
      for e in ("M0", "M1", "M2a", "M2b", "M3", "M3s", "M4", "M5", "M6", "M7", "M8")
      for s in (1, 4)]

# H2: geometry conditioning at fixed energy and budget
_G = ["uniform", "uniform", "correct", "correct", "shuffle", "shuffle",
      "other_source", "other_source", "correct", "correct", "half", "half", "balanced"]
H2 = [{"id": f"G{i}", "energy": "M2a" if (i % 2 == 0 or i == 12) else "M3",
       "geometry": _G[i], "wrong_target": i in (8, 9),
       "steps": 4, "step_rms": 0.08} for i in range(13)]

# H3: content protection crossed with style direction
def _h3():
    arms = []
    for i in range(18):
        energy = ("M0" if i < 3 else "M2a" if i < 6 else "M3" if i < 9 else
                  "R1" if i == 9 else "M2a" if i in (10, 12, 14) else "M3")
        content = (["rms", "proxy", "cg"][i % 3] if i < 9 else
                   "rms" if i in (9, 16, 17) else "proxy")
        cfg = {"energy": energy, "content": content,
               "steps": 8 if i in (12, 13) else 4,
               "step_rms": 0.04 if i in (12, 13) else 0.08,
               "weak_r1": i in (10, 11), "wrong_target": i == 14,
               "geometry": "shuffle" if i == 15 else "correct"}
        tiers = ("lo", "hi") if (i < 9 or i in (10, 11, 12, 13, 14, 15)) else (None,)
        for tier in tiers:
            c = dict(cfg, rho_tier=tier)
            c["id"] = f"T{i}" + (f".rho_{tier}" if tier else "")
            arms.append(c)
    return arms

H3 = _h3()

# H0: direction / protocol checks
H0 = ([{"id": n, "kind": n} for n in
       ("B3", "B4", "Q2aDirect", "Oracle", "R1Full", "R1TopK")]
      + [{"id": f"Probe.{o}.{d}", "kind": "probe", "energy": o, "direction": d,
          "steps": 1, "step_rms": 0.08}
         for o in ("R1", "S1", "S2") for d in ("fwd", "rev", "wrong")]
      + [{"id": n, "kind": n, "steps": 4, "step_rms": 0.08}
         for n in ("Legacy4", "True4")]
      + [{"id": f"Random.seed{i}", "kind": "random", "seed": i, "step_rms": 0.08}
         for i in range(5)])

# H5: robustness and deployment diagnostics
H5 = [{"id": n, "kind": n} for n in
      ("M2_bankseed", "M3_bankseed", "M2_painter", "M3_painter",
       "D2_same_noedit", "D4_same_noedit")]

GROUPS = {"H0": H0, "H1": H1, "H2": H2, "H3": H3, "H5": H5}


def all_arms():
    return [(phase, arm) for phase, arms in GROUPS.items() for arm in arms]


if __name__ == "__main__":
    import collections
    counts = collections.Counter(len(v) for k, v in GROUPS.items())
    print("arm counts:", {k: len(v) for k, v in GROUPS.items()})
    print("total deterministic arms:", sum(len(v) for v in GROUPS.values()))
