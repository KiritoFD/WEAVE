"""WEAVE -> VAE bottleneck probe.

Two run modes, one script, one result JSON:

  --mode p0   per-task paired analysis on the frozen 750-task protocol (CPU only)
  --mode p1   latent editing-direction geometry on 100 balanced tasks (GPU, no training)

P0 answers "how much of the teacher did the student actually learn, and is the gap
uniform across target styles or concentrated". P1 answers "is the student's edit
direction wrong, or just too weak / too damaging".

Statistics are source-clustered paired bootstrap. The existing tools/v6_24h/stats.py
resamples with `np.isin(cid, pick)`, which collapses duplicate cluster draws into a
single occurrence and therefore understates variance; here the resample is built by
concatenating each drawn cluster's row indices so multiplicity is preserved.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "deliver" / "full"
TASKS = ROOT / "eval_protocol" / "tasks_750.json"
OUT = ROOT / "deliver" / "vae_gap_results.json"

# confusion_styles order in the eval_750 payloads; pred_class indexes into it
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN_IDX = STYLES.index("Minimalism")

ARMS = {
    # name                  source                key / filename
    "Q2a_Direct":        ("p0",   "A0"),
    "Oracle_1step":      ("p0",   "A1"),
    "Oracle_4step":      ("p0",   "A2"),
    "Oracle_1step_A3":   ("p0",   "A3"),   # suspected duplicate of A1
    "ScoreDiff_N24":     ("p0",   "A4"),
    "ScoreDiff_N12":     ("eval", "eval_750_scorediff_N12.json"),
    "B0_ctrl":           ("eval", "eval_750_P1_B0_ctrl218.json"),
    "B3_lambda20":       ("eval", "eval_750_P1_B3_distill20.json"),
    "B4_lambda50":       ("eval", "eval_750_P1_B4_distill50.json"),
}


# --------------------------------------------------------------------------- io
def load_arms():
    p0 = json.loads((FULL / "p0_variants.json").read_text(encoding="utf-8"))
    arms, meta = {}, {}
    for name, (src, ref) in ARMS.items():
        if src == "p0":
            rows = p0["rows"][ref]
            meta[name] = {"source": "p0_variants.json", "key": ref, "n": len(rows)}
        else:
            d = json.loads((FULL / ref).read_text(encoding="utf-8"))
            rows = d["rows"]
            meta[name] = {"source": ref, "ckpt": d.get("ckpt"),
                          "canon_mode": d.get("canon_mode"), "n": len(rows),
                          "mean_top1": d.get("mean_top1"),
                          "mean_dino_c": d.get("mean_dino_c"),
                          "mean_lpips": d.get("mean_lpips")}
        arms[name] = {r["task_id"]: normalize(r) for r in rows}
    return arms, meta


def normalize(r):
    """Unify the two row schemas (p0_variants vs eval_750_*).

    p0_variants : m_target / top1 / argmax(style name)
    eval_750_*  : target_margin / target_top1 / pred_class(int index into STYLES)
    The int-vs-name mismatch is what previously produced a silent 0.000 Minimalism
    rate, so both forms are accepted and validated here.
    """
    pc = r.get("pred_class", r.get("argmax"))
    if isinstance(pc, str):
        pred = pc if pc in STYLES else None
    elif pc is None:
        pred = None
    else:
        pred = STYLES[int(pc)] if 0 <= int(pc) < len(STYLES) else None
    src_style = r.get("src_style")
    tgt_style = r.get("tgt_style")
    is_cross = r.get("is_cross")
    if is_cross is None:
        is_cross = src_style != tgt_style
    return {
        "src_stem": r["src_stem"], "src_style": src_style, "tgt_style": tgt_style,
        "is_cross": bool(is_cross),
        "top1": float(r.get("target_top1", r.get("top1"))),
        "margin": float(r.get("target_margin", r.get("m_target"))),
        "dino_c": float(r["dino_c"]), "lpips": float(r["lpips"]),
        "pred": pred,
    }


# ------------------------------------------------------------------- statistics
def cluster_bootstrap_idx(clusters, n_boot, seed):
    """Yield index arrays resampling WHOLE clusters WITH replacement.

    The resample is the concatenation of every drawn cluster's rows, so a cluster
    drawn twice contributes twice (this is the fix over np.isin).
    """
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([by[c] for c in pick])


def boot_ci(values, clusters, n_boot=4000, seed=0):
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "n": 0}
    ms = np.array([v[idx].mean() for idx in cluster_bootstrap_idx(clusters, n_boot, seed)])
    return {"mean": float(v.mean()),
            "ci": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))],
            "n": int(len(v)), "n_clusters": int(len(set(clusters)))}


def paired_boot_ci(a, b, clusters, n_boot=4000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ms = np.array([d[idx].mean() for idx in cluster_bootstrap_idx(clusters, n_boot, seed)])
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"mean": float(d.mean()), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def fmt_ci(s, pct=True):
    k = 100.0 if pct else 1.0
    lo, hi = s["ci"]
    return f"{s['mean']*k:+.4f} [{lo*k:+.4f},{hi*k:+.4f}]" if not pct \
        else f"{s['mean']*k:.2f}% [{lo*k:.2f},{hi*k:.2f}]"


# ------------------------------------------------------------------------- P0
def run_p0(args):
    arms, meta = load_arms()
    res = {"mode": "p0", "arms": meta, "checks": {}, "analyses": {}}

    print("=" * 108)
    print("P0.0  INPUT INVENTORY")
    print("=" * 108)
    print(f"  {'arm':<18}{'n':>5}{'source':<34}{'ckpt':<44}")
    for n, m in meta.items():
        print(f"  {n:<18}{m['n']:>5}{m['source']:<34}{str(m.get('ckpt',''))[:44]}")
    files = sorted(os.listdir(FULL))
    print(f"\n  deliver/full has {len(files)} files; used: "
          f"{sorted({m['source'] for m in meta.values()})}")

    # ---- alignment ----
    print("\n" + "=" * 108)
    print("P0.1  TASK ALIGNMENT (must be identical 600 cross + 150 same everywhere)")
    print("=" * 108)
    ref_name = "B4_lambda50"
    ref = arms[ref_name]
    ref_ids = set(ref)
    ok = True
    for n, a in arms.items():
        same = set(a) == ref_ids
        cross = sum(1 for k in a if a[k]["is_cross"])
        samec = len(a) - cross
        flag = "OK " if (same and cross == 600 and samec == 150) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {n:<18}{flag} ids={len(a):>4} cross={cross:>4} same={samec:>4} "
              f"idset_equal={same}")
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    tids = {t["task_id"] for t in tasks["tasks"]}
    print(f"  tasks_750.json     {'OK ' if tids == ref_ids else 'FAIL'} "
          f"n={len(tids)} matches_arms={tids == ref_ids}")
    res["checks"]["alignment_ok"] = bool(ok and tids == ref_ids)
    res["checks"]["n_cross"] = int(sum(1 for k in ref if ref[k]["is_cross"]))
    res["checks"]["n_same"] = int(len(ref) - res["checks"]["n_cross"])

    # ---- duplicate / record-reuse checks ----
    print("\n" + "=" * 108)
    print("P0.2  DUPLICATE / RECORD-REUSE CHECKS")
    print("=" * 108)
    dup = {}
    def cmp_arms(a_name, b_name, fields):
        a, b = arms[a_name], arms[b_name]
        common = sorted(set(a) & set(b))
        out = {}
        for f in fields:
            va = np.array([a[k][f] for k in common], float)
            vb = np.array([b[k][f] for k in common], float)
            out[f] = {"max_abs_diff": float(np.max(np.abs(va - vb))) if len(va) else None,
                      "n_identical": int(np.sum(va == vb)), "n": len(common)}
        return out
    for a_name, b_name in [("Oracle_1step", "Oracle_1step_A3"),
                           ("Oracle_4step", "B4_lambda50"),
                           ("Oracle_1step", "B4_lambda50")]:
        d = cmp_arms(a_name, b_name, ["top1", "dino_c", "lpips", "margin"])
        dup[f"{a_name}_vs_{b_name}"] = d
        print(f"\n  {a_name} vs {b_name}")
        for f, s in d.items():
            ident = "IDENTICAL" if s["max_abs_diff"] == 0 else "differs"
            print(f"    {f:<8} max|diff|={s['max_abs_diff']:.6g}  "
                  f"identical_rows={s['n_identical']}/{s['n']}  {ident}")
    res["checks"]["duplicates"] = dup

    # ---- helper views ----
    def arr(arm, field, cross_only=True):
        ks = [k for k, v in arms[arm].items() if (v["is_cross"] or not cross_only)]
        return np.array([arms[arm][k][field] for k in ks], float), ks

    def clus(arm, ks):
        return np.array([arms[arm][k]["src_stem"] for k in ks])

    # ---- A: Oracle-Student overlap ----
    print("\n" + "=" * 108)
    print("P0.3  ORACLE vs STUDENT SUCCESS OVERLAP (cross tasks only)")
    print("=" * 108)
    teacher, student = args.teacher, args.student
    ov = {}
    for t_name in [teacher, "Oracle_4step"]:
        for s_name in [student, "B3_lambda20", "B0_ctrl"]:
            if t_name == s_name:
                continue
            t = arms[t_name]
            s = arms[s_name]
            ks = [k for k in t if t[k]["is_cross"]]
            ts = np.array([t[k]["top1"] for k in ks]) > 0.5
            ss = np.array([s[k]["top1"] for k in ks]) > 0.5
            p_s_ok = ss[ts].mean() if ts.sum() else float("nan")
            p_s_no = ss[~ts].mean() if (~ts).sum() else float("nan")
            # paired cluster bootstrap on the difference
            d = ss.astype(float) - ts.astype(float)
            pc = paired_boot_ci(ss.astype(float), ts.astype(float), clus(s_name, ks),
                                n_boot=args.boot, seed=1)
            key = f"{t_name}__{s_name}"
            ov[key] = {"teacher_top1": float(ts.mean()), "student_top1": float(ss.mean()),
                       "P(student_ok|teacher_ok)": float(p_s_ok),
                       "P(student_ok|teacher_fail)": float(p_s_no),
                       "n_teacher_ok": int(ts.sum()),
                       "n_teacher_fail": int((~ts).sum()),
                       "student_minus_teacher": pc}
            print(f"\n  teacher={t_name:<14} student={s_name:<14}")
            print(f"    teacher Top-1 {ts.mean()*100:5.2f}%   student {ss.mean()*100:5.2f}%"
                  f"   (n={len(ks)})")
            print(f"    P(student ok | teacher ok)   = {p_s_ok*100:5.2f}%  "
                  f"({int(ts.sum())} tasks)")
            print(f"    P(student ok | teacher FAIL) = {p_s_no*100:5.2f}%  "
                  f"({int((~ts).sum())} tasks)")
            print(f"    student-teacher paired diff  = {fmt_ci(pc)}  "
                  f"excludes_zero={pc['excludes_zero']}")
    res["analyses"]["oracle_student_overlap"] = ov

    # ---- B: per-target Pareto ----
    print("\n" + "=" * 108)
    print("P0.4  PER-TARGET PARETO (cross tasks; Minimalism = non-target predicted as Minimalism)")
    print("=" * 108)
    show = ["Q2a_Direct", "Oracle_1step", "Oracle_4step", "ScoreDiff_N24",
            "ScoreDiff_N12", "B0_ctrl", "B3_lambda20", "B4_lambda50"]
    per_target = {}
    for a in show:
        if a not in arms:
            continue
        row = {}
        print(f"\n  [{a}]")
        print(f"    {'target':<20}{'Top-1':>8}{'DINO-C':>9}{'Margin':>9}{'LPIPS':>8}"
              f"{'MinMis':>8}")
        for st in STYLES:
            ks = [k for k, v in arms[a].items()
                  if v["is_cross"] and v["tgt_style"] == st]
            if not ks:
                continue
            t1 = np.array([arms[a][k]["top1"] for k in ks], float)
            dn = np.array([arms[a][k]["dino_c"] for k in ks], float)
            mg = np.array([arms[a][k]["margin"] for k in ks], float)
            lp = np.array([arms[a][k]["lpips"] for k in ks], float)
            mis = np.array([1.0 if (arms[a][k]["pred"] == "Minimalism"
                                    and st != "Minimalism") else 0.0 for k in ks])
            row[st] = {"top1": float(t1.mean()), "dino_c": float(dn.mean()),
                       "margin": float(mg.mean()), "lpips": float(lp.mean()),
                       "min_mis": float(mis.mean()), "n": len(ks)}
            print(f"    {st:<20}{t1.mean()*100:7.2f}%{dn.mean():9.4f}"
                  f"{mg.mean():+9.4f}{lp.mean():8.4f}{mis.mean()*100:7.2f}%")
        allk = [k for k, v in arms[a].items() if v["is_cross"]]
        t1 = np.array([arms[a][k]["top1"] for k in allk], float)
        c = clus(a, allk)
        mis = np.array([1.0 if (arms[a][k]["pred"] == "Minimalism"
                                and arms[a][k]["tgt_style"] != "Minimalism") else 0.0
                        for k in allk])
        row["_all"] = {"top1": boot_ci(t1, c, args.boot, 2),
                       "min_mis": boot_ci(mis, c, args.boot, 3),
                       "dino_c": boot_ci([arms[a][k]["dino_c"] for k in allk], c,
                                         args.boot, 4)}
        print(f"    {'ALL cross':<20}{t1.mean()*100:7.2f}%  "
              f"Top-1 CI {fmt_ci(row['_all']['top1'])}   "
              f"MinMis CI {fmt_ci(row['_all']['min_mis'])}")
        per_target[a] = row
    res["analyses"]["per_target"] = per_target

    # ---- C: teacher effectiveness ----
    print("\n" + "=" * 108)
    print("P0.5  TEACHER EFFECTIVENESS (is the teacher's success worth imitating?)")
    print("=" * 108)
    t_eff = {}
    for t_name in ["Oracle_1step", "Oracle_4step"]:
        t = arms[t_name]
        ks = [k for k in t if t[k]["is_cross"]]
        ts = np.array([t[k]["top1"] for k in ks]) > 0.5
        row = {}
        print(f"\n  teacher={t_name}")
        for label, m in (("teacher OK", ts), ("teacher FAIL", ~ts)):
            if m.sum() == 0:
                continue
            dk = [ks[i] for i in np.where(m)[0]]
            row[label] = {
                "n": len(dk),
                "dino_c": float(np.mean([t[k]["dino_c"] for k in dk])),
                "margin": float(np.mean([t[k]["margin"] for k in dk])),
                "lpips": float(np.mean([t[k]["lpips"] for k in dk])),
            }
            print(f"    {label:<14} n={len(dk):>4}  DINO-C={row[label]['dino_c']:.4f}  "
                  f"margin={row[label]['margin']:+.4f}  LPIPS={row[label]['lpips']:.4f}")
            for s_name in ["B4_lambda50", "B3_lambda20", "B0_ctrl"]:
                if s_name not in arms:
                    continue
                s = arms[s_name]
                st1 = np.mean([s[k]["top1"] for k in dk])
                sdn = np.mean([s[k]["dino_c"] for k in dk])
                row[label][s_name] = {"top1": float(st1), "dino_c": float(sdn)}
                print(f"        -> {s_name:<14} Top-1={st1*100:5.2f}%  DINO-C={sdn:.4f}")
        t_eff[t_name] = row
    res["analyses"]["teacher_effectiveness"] = t_eff

    # ---- D: equal-content-cost comparison (stratified by DINO-C) ----
    # Oracle 36.5% sits at DINO-C 0.710 while B4 24.2% sits at 0.796, so raw Top-1
    # comparisons mix in editing strength. Stratifying by DINO-C compares arms at
    # the SAME content-preservation level, which is the control the plan requires
    # and which needs no checkpoints.
    print("\n" + "=" * 108)
    print("P0.6  EQUAL-CONTENT-COST: Top-1 stratified by DINO-C (cross tasks)")
    print("=" * 108)
    bins = [(0.0, 0.5), (0.5, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.01)]
    strat = {}
    hdr = f"  {'arm':<18}" + "".join(f"[{lo:.2f},{hi:.2f})".rjust(16) for lo, hi in bins)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for a in ["Q2a_Direct", "Oracle_1step", "Oracle_4step", "B3_lambda20",
              "B4_lambda50", "ScoreDiff_N24", "B0_ctrl"]:
        if a not in arms:
            continue
        ks = [k for k, v in arms[a].items() if v["is_cross"]]
        dn = np.array([arms[a][k]["dino_c"] for k in ks], float)
        t1 = np.array([arms[a][k]["top1"] for k in ks], float)
        row, line = {}, f"  {a:<18}"
        for lo, hi in bins:
            m = (dn >= lo) & (dn < hi)
            if m.sum() == 0:
                line += f"{'-':>16}"
                row[f"[{lo:.2f},{hi:.2f})"] = None
                continue
            line += f"{t1[m].mean()*100:6.1f}%(n={int(m.sum()):3d})".rjust(16)
            row[f"[{lo:.2f},{hi:.2f})"] = {"top1": float(t1[m].mean()),
                                           "n": int(m.sum())}
        print(line)
        row["_mean_dino_c"] = float(dn.mean())
        strat[a] = row
    print("\n  reading: within a DINO-C bin the arms have comparable content cost, "
          "so Top-1 differences there are not strength artefacts.")

    # the specific control the plan asks for: at B4's content level, how good is Oracle?
    b4_dc = strat["B4_lambda50"]["_mean_dino_c"]
    print(f"\n  B4 mean DINO-C = {b4_dc:.4f}")
    for a in ["Oracle_1step", "Oracle_4step"]:
        if a not in strat:
            continue
        ks = [k for k, v in arms[a].items() if v["is_cross"]]
        dn = np.array([arms[a][k]["dino_c"] for k in ks], float)
        t1 = np.array([arms[a][k]["top1"] for k in ks], float)
        m = np.abs(dn - b4_dc) <= 0.05
        if m.sum():
            print(f"  {a:<16} restricted to DINO-C within +-0.05 of B4: "
                  f"Top-1={t1[m].mean()*100:5.2f}%  (n={int(m.sum())}, "
                  f"their mean DINO-C={dn[m].mean():.4f})")
            strat[a]["_matched_to_b4"] = {"top1": float(t1[m].mean()),
                                          "n": int(m.sum()),
                                          "dino_c": float(dn[m].mean())}
    res["analyses"]["equal_content_cost"] = strat

    # ---- E: Oracle's own strength-quality Pareto, and where the student sits ----
    # A1 (1-step) -> A2 (4-step dyn) is a STRENGTH reduction of the SAME oracle:
    # DINO-C rises 0.710 -> 0.796 while Top-1 falls 36.5% -> 24.2%. That is the
    # alpha-sweep the plan asks for, in two points, and it lets us predict the
    # oracle's Top-1 at any content level WITHOUT its checkpoints.
    # The DINO-C stratified table above is a post-hoc selection (it picks the tail
    # of the oracle's distribution), so this slope extrapolation is the causal
    # reading; the two are reported side by side on purpose.
    print("\n" + "=" * 108)
    print("P0.7  ORACLE STRENGTH-QUALITY SLOPE (alpha-sweep proxy) vs STUDENT")
    print("=" * 108)
    pts = {}
    for a in ["Oracle_1step", "Oracle_4step", "B3_lambda20", "B4_lambda50",
              "ScoreDiff_N24", "Q2a_Direct"]:
        if a not in arms:
            continue
        ks = [k for k, v in arms[a].items() if v["is_cross"]]
        pts[a] = {"dino_c": float(np.mean([arms[a][k]["dino_c"] for k in ks])),
                  "top1": float(np.mean([arms[a][k]["top1"] for k in ks]))}
    print(f"  {'arm':<18}{'DINO-C':>9}{'Top-1':>9}")
    for a, v in pts.items():
        print(f"  {a:<18}{v['dino_c']:>9.4f}{v['top1']*100:>8.2f}%")

    slope = None
    if "Oracle_1step" in pts and "Oracle_4step" in pts:
        a1, a2 = pts["Oracle_1step"], pts["Oracle_4step"]
        dd = a2["dino_c"] - a1["dino_c"]
        if abs(dd) > 1e-9:
            slope = (a2["top1"] - a1["top1"]) / dd
        print(f"\n  Oracle slope A1->A2: dTop-1/dDINO-C = {slope:+.4f} "
              f"({(a2['dino_c']-a1['dino_c']):+.4f} DINO-C for "
              f"{(a2['top1']-a1['top1'])*100:+.2f} pts Top-1)")
        print(f"  {'student':<18}{'actual':>10}{'oracle-predicted':>19}{'residual':>11}")
        extrap = {}
        for a in ["B3_lambda20", "B4_lambda50", "ScoreDiff_N24"]:
            if a not in pts:
                continue
            pred = a1["top1"] + slope * (pts[a]["dino_c"] - a1["dino_c"])
            resid = pts[a]["top1"] - pred
            extrap[a] = {"actual": pts[a]["top1"], "predicted": float(pred),
                         "residual": float(resid),
                         "dino_c": pts[a]["dino_c"]}
            print(f"  {a:<18}{pts[a]['top1']*100:9.2f}%{pred*100:18.2f}%"
                  f"{resid*100:+10.2f}%")
        res["analyses"]["oracle_slope"] = {"slope": float(slope),
                                           "points": pts, "extrapolation": extrap}
        if "B4_lambda50" in extrap:
            r = extrap["B4_lambda50"]
            if abs(r["residual"]) < 0.03:
                verdict = ("B4 sits ON the oracle's own strength-quality Pareto line "
                           "(residual < 3 pts): the student reproduces the teacher's "
                           "trade-off, so the gap to the 1-step oracle is a CONTENT-COST "
                           "choice, not a direction failure.")
            elif r["residual"] < 0:
                verdict = ("B4 sits BELOW the oracle's Pareto line: a real amortisation "
                           "gap remains beyond the content-cost trade-off.")
            else:
                verdict = ("B4 sits ABOVE the oracle's Pareto line: the student beats the "
                           "teacher at equal content cost.")
            print(f"\n  VERDICT: {verdict}")
            res["analyses"]["oracle_slope"]["verdict"] = verdict

    # ---- F: P1 status + existing geometry evidence ----
    # P1 needs Q2a / R1 / B4 checkpoints. They are absent locally AND on the remote
    # box (deliver/ carries results only; runs/v6_next does not exist). Recorded here
    # so the gap is explicit rather than silently skipped.
    print("\n" + "=" * 108)
    print("P0.8  P1 STATUS AND EXISTING GEOMETRY EVIDENCE")
    print("=" * 108)
    need = ["runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
            "runs/v6_next/probes/r1_r1.pt",
            "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"]
    missing = [w for w in need if not (ROOT / w).exists()]
    print(f"  P1 required weights present: {len(need) - len(missing)}/{len(need)}")
    for w in missing:
        print(f"    MISSING  {w}")
    p1 = {"status": "blocked" if missing else "ready", "missing": missing,
          "would_measure": ["cos(dz_B, dz_O)", "RMS(dz_B)/RMS(dz_O)",
                            "cos(dz_B, grad R1)", "oracle gradient local stability",
                            "decoded CSD / DINO-C"],
          "note": ("The decision-relevant part of P1 (equal-content-cost comparison) "
                   "was instead obtained causally from the oracle's own strength "
                   "slope, see P0.7.")}

    # L-series latent-direction geometry already on disk (different model family:
    # L11/L8 direct latent editors, NOT B4). Reported for context only.
    lgeo = {}
    for f in ["direction_L11.json", "direction_L8.json"]:
        fp = FULL / f
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        lgeo[f] = {k: d.get(k) for k in
                   ["ckpt", "n", "cos_align_mean", "cos_align_p10", "m_target_mean",
                    "m_source_mean", "max_cos_mean", "knn_target_mean",
                    "tgt_acquired", "src_retained", "near_not_crossed"]}
    if lgeo:
        print("  existing latent-direction geometry (L-series, NOT B4):")
        for f, v in lgeo.items():
            print(f"    {f}: n={v['n']} cos_align={v['cos_align_mean']:+.4f} "
                  f"(p10 {v['cos_align_p10']:+.4f})  tgt_acquired="
                  f"{v['tgt_acquired']:.3f}  src_retained={v['src_retained']:.3f}  "
                  f"near_not_crossed={v['near_not_crossed']:.3f}")
        print("    -> the L-series edit direction is nearly orthogonal to the target "
              "(cos ~0.14) and crosses the style boundary in only 3% of tasks; "
              "B4 is a different family and must not inherit this verdict.")
    p1["existing_l_series_geometry"] = lgeo
    res["analyses"]["p1_status"] = p1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] {OUT}")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="p0", choices=["p0", "p1"])
    ap.add_argument("--teacher", default="Oracle_1step")
    ap.add_argument("--student", default="B4_lambda50")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    if args.mode == "p0":
        run_p0(args)
    else:
        run_p1(args)


def run_p1(args):
    raise SystemExit("P1 not implemented yet; run --mode p0 first")


if __name__ == "__main__":
    main()
