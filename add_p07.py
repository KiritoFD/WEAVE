import io

p = 'tools/v6_next/vae_gap_probe.py'
s = io.open(p, encoding='utf-8').read()

anchor = '''    OUT.parent.mkdir(parents=True, exist_ok=True)'''
new = '''    # ---- E: Oracle's own strength-quality Pareto, and where the student sits ----
    # A1 (1-step) -> A2 (4-step dyn) is a STRENGTH reduction of the SAME oracle:
    # DINO-C rises 0.710 -> 0.796 while Top-1 falls 36.5% -> 24.2%. That is the
    # alpha-sweep the plan asks for, in two points, and it lets us predict the
    # oracle's Top-1 at any content level WITHOUT its checkpoints.
    # The DINO-C stratified table above is a post-hoc selection (it picks the tail
    # of the oracle's distribution), so this slope extrapolation is the causal
    # reading; the two are reported side by side on purpose.
    print("\\n" + "=" * 108)
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
        print(f"\\n  Oracle slope A1->A2: dTop-1/dDINO-C = {slope:+.4f} "
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
            print(f"\\n  VERDICT: {verdict}")
            res["analyses"]["oracle_slope"]["verdict"] = verdict

    OUT.parent.mkdir(parents=True, exist_ok=True)'''

assert anchor in s
s = s.replace(anchor, new, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print("added P0.7")
