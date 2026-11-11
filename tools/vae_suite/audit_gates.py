"""Post-hoc Gate audit for SD0-S10 stages (run after training; read-only).

The in-flight run_sd0_s10.sh gates check LPIPS / HF for C0 and resp / lpips for
C1. The plan additionally requires, for C0: no sharpening artefacts, CSD Keep
and colour must not visibly degrade; and for C1: gradient-direction consistency
improvement and no new artefacts at large perturbations. Those need either the
full acceptance metrics (eval_sd0_s10.py) or a visual/gradient probe, so they
are audited here WITHOUT modifying the running chain.

    python -m tools.vae_suite.audit_gates
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/sd0_s10"


def load(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else None


def c0_audit():
    g = load("c0_gate.json")
    if not g:
        return {"status": "missing c0_gate.json"}
    f, l = g.get("first", {}), g.get("last", {})
    csd_delta = l.get("csd", 0) - f.get("csd", 0)
    # colour is not in c0_gate.json (trainer's evaluate writes it to metrics
    # jsonl, not the gate file) -> pull from c0_metrics.jsonl
    mf = OUT / "c0_metrics.jsonl"
    colour_delta = None
    if mf.exists():
        rows = [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
        if len(rows) >= 2 and "colour" in rows[0] and "colour" in rows[-1]:
            colour_delta = round(rows[-1]["colour"] - rows[0]["colour"], 5)
    return {
        "lpips_delta": round(l.get("lpips", 0) - f.get("lpips", 0), 5),
        "hf_corr_delta": round(l.get("hf_corr_L0", 0) - f.get("hf_corr_L0", 0), 5),
        "hf_l1_delta": round(l.get("hf_l1", 0) - f.get("hf_l1", 0), 5),
        "csd_delta": round(csd_delta, 5),
        "colour_delta": colour_delta,
        "csd_not_worse": csd_delta >= -0.01,
        "colour_not_worse": (colour_delta is None or colour_delta <= 0.005),
        "no_sharpening_artifact": None,   # requires visual check; see note
        "note": ("HF energy ratio + visual grid needed to judge sharpening; "
                 "this script only checks the numeric CSD/colour guards."),
    }


def c1_audit():
    g = load("c1_gate.json")
    if not g:
        return {"status": "missing c1_gate.json"}
    f, l = g.get("first", {}), g.get("last", {})
    # IMPORTANT: `resp` / `resp_cos` in c1_metrics.jsonl are CUMULATIVE training
    # means from update 1 to that VALID point (stage_sum/stage_cnt), NOT the mean
    # of the last 250-update segment. So a genuinely falling response loss can
    # still show last > first if the early updates were much higher. The Gate's
    # `last <= first + 0.01` is therefore CONSERVATIVE by design (plan: Gate is
    # an engineering threshold; on failure keep C0 and write a negative report).
    # Report the trajectory explicitly so the verdict is interpretable.
    mf = OUT / "c1_metrics.jsonl"
    traj = []
    if mf.exists():
        for x in mf.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                traj.append({"upd": r.get("upd"),
                             "resp": r.get("resp"), "resp_cos": r.get("resp_cos"),
                             "lpips": r.get("lpips")})
    out = {
        "resp_delta": round(l.get("resp", 0) - f.get("resp", 0), 5),
        "resp_cos_delta": round(l.get("resp_cos", 0) - f.get("resp_cos", 0), 5),
        "lpips_delta": round(l.get("lpips", 0) - f.get("lpips", 0), 5),
        "resp_trajectory": traj,
        "resp_is_cumulative_mean": True,
        "gradient_consistency_improved": None,  # needs eval_sd0_s10 L2 output
        "large_perturb_no_artifact": None,      # needs visual/medium-band probe
        "note": ("`resp` is a CUMULATIVE mean (upd1..VALID), so the Gate is "
                 "conservative. Read eval_p5_final.json 'perturb' medium band + "
                 "input_grad.jtv_cos_median for the two qualitative requirements; "
                 "a Gate failure here is not proof the response got worse."),
    }
    return out


def main():
    r = {"c0": c0_audit(), "c1": c1_audit()}
    p0 = load("p0_baseline.json")
    if p0:
        r["p0_official_lpips"] = p0["variants"]["official"]["lpips"]
        r["p0_official_csd"] = p0["variants"]["official"]["csd"]
    ev = load("eval_final.json") or load("eval_p5_final.json")
    if ev and "c1" in ev and "perturb" in ev["c1"]:
        pt = ev["c1"]["perturb"]
        r["c1"]["gradient_consistency_improved"] = (
            pt.get("input_grad", {}).get("jtv_cos_median"))
        r["c1"]["medium_band_cos"] = pt.get("medium", {}).get("cos")
    (OUT / "gate_audit.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps(r, indent=2), flush=True)
    print(f"[Saved] {OUT / 'gate_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
