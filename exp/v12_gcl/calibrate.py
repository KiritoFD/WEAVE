"""V12-GCL budget calibration on TRAIN-split paintings (plan §4).

For each Q variant: B_Q = p90 of the appearance-perturbation cost distribution;
geometric perturbation must EXCEED appearance by >=3x (sensitivity check).
Relation tau rule: largest tau in the pre-registered sweep satisfying the 3x rule.
Run: python -m exp.v12_gcl.calibrate
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .constraints import calibrate_budget
from .query_haar import haar_gcl_loss
from .query_relation import relation_gcl_loss, TAU_SWEEP
from .query_relation import g2o_selfsim_loss

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "weights_local/sd_pool_orig"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
OUT = REPO / "runs/v12_gcl"
N_PAINT = 48


def _h10(pid: str) -> int:
    import hashlib
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10


def train_latents(n_per=12):
    g = torch.Generator().manual_seed(0)
    out = []
    for si, s in enumerate(STYLES):
        d = torch.load(POOL / f"{si:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        tr = [i for i, pid in enumerate(d["ids"]) if _h10(str(pid)) < 9]
        take = torch.randperm(len(tr), generator=g)[:n_per].tolist()
        out.append(d["latents"][torch.tensor(tr, dtype=torch.long)[take]])
    return torch.cat(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lat = train_latents()
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_train_paintings": int(lat.shape[0])}

    cost_haar = lambda a, b: haar_gcl_loss(a, b)
    res["haar"] = calibrate_budget(lat, cost_haar, device)

    for tau in TAU_SWEEP:
        res[f"rel_tau{tau}"] = calibrate_budget(
            lat, lambda a, b, _t=tau: relation_gcl_loss(a, b, _t), device)

    res["g2o"] = calibrate_budget(lat, g2o_selfsim_loss, device)

    # tau rule: largest tau with geometric >= 3x appearance (mean ratio)
    ok_taus = [t for t in TAU_SWEEP
               if res[f"rel_tau{t}"]["sensitivity_ratio_geo_over_app_mean"] >= 3.0]
    res["chosen_tau"] = max(ok_taus) if ok_taus else None
    res["tau_rule_note"] = ("largest tau with geo>=3x app; "
                            + json.dumps({t: res[f"rel_tau{t}"]["sensitivity_ratio_geo_over_app_mean"]
                                          for t in TAU_SWEEP}))
    if res["chosen_tau"] is None:
        res["chosen_tau"] = min(TAU_SWEEP)  # conservative fallback, flagged
        res["tau_fallback_flag"] = True

    # 2026-11-14 amendment (recorded): the appearance-p90 gate is ~78x below the
    # geometric distribution for the relation Q — it would reject every real
    # style edit and degenerate the arm. B_Q uses the LOG-MIDPOINT between the
    # appearance p90 ("free" change) and the geometric p10 ("structural damage"):
    #   B_Q = exp( (log app_p90 + log geo_p10) / 2 )
    # candidates beyond it are structurally damaged rather than re-styled.
    def _logmid(app_p90, geo_p10):
        return math.exp((math.log(max(app_p90, 1e-12))
                         + math.log(max(geo_p10, 1e-12))) / 2)

    budgets = {
        "haar": round(_logmid(res["haar"]["appearance"]["p90"],
                              res["haar"]["geometric"]["p10"]), 6),
        "rel": round(_logmid(res[f"rel_tau{res['chosen_tau']}"]["appearance"]["p90"],
                             res[f"rel_tau{res['chosen_tau']}"]["geometric"]["p10"]), 6),
        "g2o": round(_logmid(res["g2o"]["appearance"]["p90"],
                             res["g2o"]["geometric"]["p10"]), 6),
        "chosen_tau": res["chosen_tau"],
        "rule": "log-midpoint(app p90, geo p10); pre-registration amendment logged",
    }
    res["budgets"] = budgets
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budgets.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("saved ->", OUT / "budgets.json")


import time  # noqa: E402

if __name__ == "__main__":
    main()
