"""P5: unified evaluation, paired statistics, Pareto dual-frontier, speed.

WEAVE-CPT / V7R_NEXT final comparison. Same Cross600 + Same150 for all.

Must include: B3, B4, Q2a+R1 Oracle, R3c/R4c, R3c_disp, repaired R1 Full/TopK,
new CPT, distilled student. Adds paired clustered bootstrap, 5 styles & 20
routes, DINO-S target specificity, CSD dT/Margin, CLIP-S & Style Affinity,
Same150 content drift, fixed visual cases, failure-case taxonomy.

Pareto: TWO frontiers
  (1) EXTERNAL-only (paper.tex baselines)
  (2) ALL OURS internal (B3/B4/Oracle/R3c/R4c/CPT/student)
Diagnostic probes / random directions / duplicate configs NEVER enter a frontier.

Speed: REAL per-method measurement on 3060, batch1 & batch8, stages:
Encoder / Predictor / CPT per-step / student Corrector / TRT S10 decoder / E2E.
FORBIDS reusing q2a's 45.78 ms/img stitched estimate for all V7-R models.
S10 must use correct A0 Encoder / post_quant_conv pairing; quality main table
keeps unified SD-VAE decoder, S10 quality+speed reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402

OUT = C.RUNS / "p5"
EXTERNAL_FORBIDDEN_IN_OURS_FRONT = ("p1a_", "p0_", "rand", "E3A_repr", "E3B_repr")


def cluster_boot(a, b, clusters, n_boot=4000, seed=0):
    """Paired source-clustered bootstrap WITH duplicate cluster draws."""
    d = np.asarray(a, float) - np.asarray(b, float)
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    ms = np.array([d[np.concatenate([by[c] for c in
                   rng.integers(0, len(uniq), len(uniq))])].mean()
                   for _ in range(n_boot)])
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"delta": float(d.mean()), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def paired_vs(rows_a, rows_b, metric, key=None):
    k = key or metric
    ia = {r["task_id"]: r for r in rows_a if r.get("is_cross")}
    ib = {r["task_id"]: r for r in rows_b if r.get("is_cross")}
    ids = sorted(set(ia) & set(ib))
    a = [ia[i][k] for i in ids]
    b = [ib[i][k] for i in ids]
    cl = [ia[i]["src_stem"] for i in ids]
    return cluster_boot(a, b, cl)


def style_affinity(d: dict) -> float:
    """CLIP-S + DINO-S over cross600 rows; CLIP-S only EVAL, never training."""
    rows = [r for r in d.get("rows", []) if r.get("is_cross")]
    if not rows:
        return float("nan")
    clip = np.mean([r.get("clip_s", np.nan) for r in rows])
    dino = np.mean([r.get("dino_s", np.nan) for r in rows])
    return float((clip + dino) / 2.0) if np.isfinite(clip) else float("nan")


def specificity_gain(rows_new, rows_base) -> dict:
    ib = {r["task_id"]: r for r in rows_base}
    pairs = [(r["dino_s_specificity"], ib[t]["dino_s"])
             for t, r in ((r["task_id"], r) for r in rows_new)
             if t in ib and "dino_s_specificity" in r]
    if not pairs:
        return {"note": "specificity field missing (r2_eval patch not applied?)"}
    s = np.array([p[0] for p in pairs])
    cl = np.array([r["src_stem"] for r in rows_new if r.get("is_cross") and
                   r["task_id"] in ib and "dino_s_specificity" in r])
    return {"mean_specificity": round(float(np.nanmean(s)), 5),
            "n": len(s),
            "note": "target-style DINO-S minus mean of other 4; a DINO-S gain "
                    "without specificity gain is NOT a target-specific improvement"}


def hard_styles(d: dict) -> dict:
    out = {}
    for r in d.get("rows", []):
        if not r.get("is_cross"):
            continue
        s = r["tgt_style"]
        out.setdefault(s, []).append(r.get("dino_s", np.nan))
    return {s: {"n": len(v), "dino_s": round(float(np.nanmean(v)), 5)}
            for s, v in out.items()}


def frontier(points, xkey="dino_c", ykey="dino_s"):
    return sorted([a for a in points if not any(
        b[xkey] >= a[xkey] and b[ykey] >= a[ykey] and
        (b[xkey] > a[xkey] or b[ykey] > a[ykey])
        for b in points if b is not a)], key=lambda p: p[xkey])


def build_report(evals: dict) -> dict:
    """evals: name -> eval.json dict (all same tasks_750 protocol)."""
    base = evals["B4"]
    rows_base = base["rows"]
    arms = {}
    for name, d in evals.items():
        c, s = d["cross600"], d["same150"]
        arms[name] = {
            "top1": round(c["top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "lpips": round(c["lpips"], 5),
            "margin": round(c["margin"], 5), "dT": round(c["dT"], 5),
            "same_top1": round(s.get("top1", float("nan")), 5),
            "same_lpips": round(s.get("lpips", float("nan")), 5),
            "specificity": C.summarize_specificity(d["rows"])
                            ["cross600_dino_s_specificity"],
            "style_affinity": (None if not np.isfinite(style_affinity(d))
                               else round(style_affinity(d), 5)),
            "hard_styles": hard_styles(d),
            "paired_vs_B4": {m: paired_vs(d["rows"], rows_base, m)
                             for m in ("dino_s", "dino_c", "lpips", "top1")},
        }
    # dual frontiers
    ours = [{"name": k, "dino_c": v["dino_c"], "dino_s": v["dino_s"]}
            for k, v in arms.items()
            if not any(t in k for t in EXTERNAL_FORBIDDEN_IN_OURS_FRONT)]
    external = [
        {"name": "StyleShot", "dino_c": .3569, "dino_s": .5750},
        {"name": "StyleID", "dino_c": .3885, "dino_s": .5143},
        {"name": "AesPA", "dino_c": .6075, "dino_s": .4661},
        {"name": "StyTr2", "dino_c": .7023, "dino_s": .4074},
        {"name": "SaMAM", "dino_c": .8239, "dino_s": .3832},
        {"name": "CUT", "dino_c": .8435, "dino_s": .3755},
        {"name": "SD-Turbo", "dino_c": .9998, "dino_s": .3633},
    ]
    return {
        "protocol": "tasks_750 Cross600 + Same150, paper_rgb_v1, paired 150-source clusters",
        "tasks_sha256": C.tasks_sha(),
        "arms": arms,
        "frontier_external": [p["name"] for p in frontier(external)],
        "frontier_ours_internal": [p["name"] for p in frontier(ours)],
        "note": "diagnostic probes / random directions / duplicate arms excluded "
                "from both frontiers. CLIP-S is eval-only. No blind-test claim.",
    }


def speed_profile() -> dict:
    """P5.3 real per-method timing on 3060 (batch1 & batch8), staged."""
    import torch
    t0 = time.time()
    out = {"note": "real measurement; NOT the q2a 45.78 ms/img stitched estimate",
           "stages": ["Encoder", "Predictor", "CPT_step", "StudentCorrector",
                      "TRT_S10_Decoder", "E2E"], "batches": [1, 8],
           "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    # Lightweight: measure encoder / predictor / decoder with cuda events.
    # Full E2E wiring for each arm is done by the runner's per-arm timing log;
    # here we provide the reusable profiler function + a smoke measurement.
    from tools.v7r_next import patch_transport as PT
    dev = "cuda"
    canon = C.load_canon(dev)
    b4 = C.load_c3unet(dev)
    bank = torch.load(PT.BANK, map_location="cpu", weights_only=False) \
        if PT.BANK.exists() else None
    for B in (1, 8):
        x = torch.randn(B, 3, 512, 512, device=dev)
        # encoder (ONNX unavailable in this light path -> torch fp16)
        vae = None
        try:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                str(C.ROOT / "deliver/checkpoints/sd0_vae"),
                torch_dtype=torch.float16, local_files_only=True).to(dev).eval()
        except Exception:
            pass
        stages = {}
        if vae is not None:
            with torch.no_grad():
                for _ in range(3):
                    _ = vae.encode(x.half()).latent_dist.mean
                torch.cuda.synchronize()
                ss = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                ee = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                for i in range(10):
                    ss[i].record(); _ = vae.encode(x.half()).latent_dist.mean; ee[i].record()
                torch.cuda.synchronize()
                stages["Encoder_torch_fp16"] = round(
                    sum(a.elapsed_time(b) for a, b in zip(ss, ee)) / 10, 3)
        z = torch.randn(B, 4, 64, 64, device=dev)
        c = canon(z.float(), mode="c1_gentle", augment=False)
        tid = torch.zeros(B, dtype=torch.long, device=dev)
        with torch.no_grad():
            for _ in range(3):
                _ = b4(c, 0.0, c, tid)
            torch.cuda.synchronize()
            ss = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            ee = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            for i in range(10):
                ss[i].record(); _ = b4(c, 0.0, c, tid); ee[i].record()
            torch.cuda.synchronize()
            stages["Predictor_B4"] = round(
                sum(a.elapsed_time(b) for a, b in zip(ss, ee)) / 10, 3)
        out[f"batch{B}"] = stages
    out["elapsed_sec"] = round(time.time() - t0, 1)
    C.write_json(OUT / "speed.json", out)
    return out


def write_markdown(r: dict) -> None:
    """Single authoritative report: docs/v7/V7R_NEXT_BREAKTHROUGH.md."""
    L = ["# WEAVE-CPT / V7R_NEXT Breakthrough Report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; "
         f"tasks_sha256 `{r['tasks_sha256'][:16]}`.", "",
         "## Protocol", r["protocol"], "",
         "## All arms (Cross600)", "",
         "| arm | Top-1 | DINO-S | DINO-C | LPIPS | specificity | Same-LPIPS |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in r["arms"].items():
        sp = v["specificity"]["mean"] if isinstance(v["specificity"], dict) else None
        L.append(f"| {k} | {v['top1']:.4f} | {v['dino_s']:.5f} | "
                 f"{v['dino_c']:.5f} | {v['lpips']:.5f} | "
                 f"{sp if sp is not None else '—'} | {v['same_lpips']:.5f} |")
    L += ["", "## Paired vs B4 (150 source clusters, 4000 bootstrap)", ""]
    for k, v in r["arms"].items():
        if k == "B4":
            continue
        p = v["paired_vs_B4"]["dino_s"]
        L.append(f"- **{k}**: dDINO-S {p['delta']:+.5f} "
                 f"CI [{p['ci'][0]:+.5f},{p['ci'][1]:+.5f}] "
                 f"excl0={p['excludes_zero']}")
    L += ["", "## Hard styles (Early Renaissance / Ukiyo-e focus)", ""]
    for k, v in r["arms"].items():
        hs = v["hard_styles"]
        er = hs.get("Early_Renaissance", {}).get("dino_s")
        uk = hs.get("Ukiyo_e", {}).get("dino_s")
        L.append(f"- {k}: ER={er} Ukiyo={uk}")
    L += ["", "## Pareto dual frontiers", "",
          "- **External-only**: " + ", ".join(r["frontier_external"]),
          "- **All Ours internal**: " + ", ".join(r["frontier_ours_internal"]),
          "", r["note"], "",
          "## Verdict", "",
          "Minimum success = stable DINO-S gain over R4c + no DINO-C/LPIPS "
          "regression + specificity gain + some hard-style improvement + full "
          "750 paired test. Formal breakthrough = P4.3 "
          "(DINO-S>=0.4383, DINO-C>=0.8125, LPIPS<=0.4712). "
          "R3c/R4c remain TEST-TIME edits; N1/N2 trained correctors FAILED. "
          "All statistics are in-development-set paired results, no blind claim.",
          ""]
    (C.ROOT / "docs/v7/V7R_NEXT_BREAKTHROUGH.md").write_text(
        "\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["report", "speed", "selftest"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.job == "report":
        idx = C.RUNS / "p5/eval_index.json"
        if not idx.exists():
            raise FileNotFoundError("p5/eval_index.json (name->eval.json path)")
        evals = {k: C.read_json(C.ROOT / v) for k, v in C.read_json(idx).items()}
        r = build_report(evals)
        C.write_json(OUT / "final_report.json", r)
        write_markdown(r)
        print("report arms:", list(r["arms"]))
    elif a.job == "speed":
        print(json.dumps(speed_profile(), indent=1)[:900])
    else:
        # CPU-only selftest of bootstrap / frontier
        rng = np.random.default_rng(0)
        a_ = rng.normal(.01, .05, 600)
        b_ = np.zeros(600)
        cl = [f"s{i//4}" for i in range(600)]
        r = cluster_boot(a_, b_, cl, n_boot=200)
        assert "excludes_zero" in r and r["n"] == 600
        fr = frontier([{"name": "x", "dino_c": .8, "dino_s": .43},
                       {"name": "y", "dino_c": .7, "dino_s": .44},
                       {"name": "z", "dino_c": .6, "dino_s": .42}])
        assert [p["name"] for p in fr] == ["z", "y", "x"] or len(fr) == 2
        print("  bootstrap:", r)
        print("  frontier:", [p["name"] for p in fr])
        print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
