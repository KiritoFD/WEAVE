import json
from pathlib import Path

B = Path("/mnt/i/Latent_Style")


def head(p, n=40):
    print(f"\n===== {p} =====")
    q = B / p
    if not q.exists():
        print("MISSING")
        return None
    try:
        d = json.load(open(q, encoding="utf-8"))
    except Exception as e:
        print("ERR", e)
        return None
    s = json.dumps(d, ensure_ascii=False)
    print(s[: n * 120])
    return d


# P4 cells & K sweep
d = head("WEAVE/runs/p3/p3_p4.json", 6)
if d:
    print("cells:", list(d.get("cells", {})))
    for c, cv in d.get("cells", {}).items():
        if not isinstance(cv, dict):
            continue
        out = []
        for k in ("csd_dT", "csd_dS", "csd_dM", "dino_dM", "swd_gain"):
            v = cv.get(k)
            if isinstance(v, dict) and "mean" in v:
                out.append(f"{k}={v['mean']:.4f}[{v['ci95_low']:.4f},{v['ci95_high']:.4f}]/med{v['median']:.4f}/fp{v['frac_positive']:.2f}")
        print(c, " ".join(out))

# selection / patterns
head("WEAVE/runs/p2/pgt_selection.json", 10)
head("WEAVE/runs/p2/pgt_q0.json", 10)
d = head("WEAVE/runs/p2/pgt_S1_positive_b256.json", 8)
d2 = head("WEAVE/runs/p2/pgt_S2_reverse_b256.json", 8)
if d and d2:
    for nm, x in (("S1", d), ("S2", d2)):
        print(nm, json.dumps({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v)
                              for k, v in x.items() if not isinstance(v, list)})[:900])

head("WEAVE/runs/p2/p5_styleid_sensitivity.json", 9999) if False else None

# sensitivity tail
p = B / "WEAVE/runs/p2/p5_styleid_sensitivity.json"
if p.exists():
    d = json.load(open(p, encoding="utf-8"))
    for k, v in d.get("sensitivity", {}).items():
        print("SENS", k, json.dumps({kk: (round(vv.get("mean", 0), 4) if isinstance(vv, dict) and "mean" in vv else vv)
                                     for kk, vv in v.items() if not isinstance(vv, list)}))
    print("QUAD", json.dumps(d.get("quadrants", {}))[:600])

# eval 750 S1/S2
for c in ("eval_render/S1_A_P5_final", "eval_render/S2_E7_final", "bench_render/b64_t8"):
    p = B / "pseudo_gt_24h" / c / "v4_eval750.json"
    if p.exists():
        d = json.load(open(p, encoding="utf-8"))
        print("EVAL", c, json.dumps({g: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in gv.items() if not isinstance(v, list)}
                                     for g, gv in d.items() if isinstance(gv, dict) and "n" in gv})[:1200])

# bench text
for rel in ("WEAVE/runs/v4/batch_bench.txt", "WEAVE/runs/v4/batch_bench2.txt", "WEAVE/runs/v4/train_bench.txt"):
    p = B / rel
    print(f"\n== {rel} ==")
    print(p.read_text(encoding="utf-8")[:500] if p.exists() else "MISSING")
