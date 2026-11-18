"""docs/performance/build_metrics.py — Single-Source-of-Truth Master Ledger Builder.

Consolidates ALL WEAVE generations (V6/Q2A, V8, V9-LD, V10, V11B, V15 Group-A /
Scaling / Style-Bridge, V16, V17, V18) and ALL 19 external baselines (including
reproduced StyleGallery & Hist2Style), enriched with Same150 identity controls,
ArtFID, KID, latency (ms/img), and 4-axis Pareto frontier flags.

Outputs:
  docs/performance/metrics.csv                        — flat unified table
  docs/performance/metrics.json                       — structured JSON + ArtFID details
  docs/performance/figures/pareto_frontier_cross600.{png,svg}
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PERF = Path(__file__).resolve().parent
FIG_DIR = PERF / "figures"

COLS = [
    "arm", "group", "protocol", "decoder", "is_diagnostic", "pareto", "n_cross",
    "dino_s", "dino_c", "lpips", "clip_s", "style_affinity",
    "target_top1", "target_margin", "dT", "G",
    "same_dino_s", "same_dino_c", "same_top1",
    "ms_per_img", "artfid", "kid", "artfid_n_images", "artfid_n_missing",
    "eval_source",
]

EXT_SPEED_MS = {
    "StyleShot": 24480.0, "StyleID": 5040.0, "StyTr2": 3040.0,
    "Seedream 4.5": 2000.0, "AesPA": 1840.0, "SaMAM": 1408.0,
    "SaMST": 800.0, "CUT": 400.0, "SD-Turbo": 400.0, "WCT": 293.6,
}

D5_ALIAS = {
    "adain": "AdaIN", "aespa": "AesPA", "cut": "CUT", "identity": "Identity",
    "samam": "SaMAM", "samst": "SaMST", "sdturbo": "SD-Turbo",
    "seedream": "Seedream 4.5", "styleid": "StyleID",
    "styleshot": "StyleShot", "stytr2": "StyTr2", "wct": "WCT",
}


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def blank_row(arm: str, group: str, protocol: str = "q2a_oracle_cross600",
              decoder: str = "S10", is_diagnostic: bool = False,
              eval_source: str = "") -> dict:
    r = {c: None for c in COLS}
    r.update({
        "arm": arm, "group": group, "protocol": protocol,
        "decoder": decoder, "is_diagnostic": is_diagnostic,
        "pareto": False, "eval_source": eval_source,
        "artfid_detail": None,
    })
    return r


def eval_json_to_row(arm: str, group: str, path: Path, *,
                     protocol: str = "q2a_oracle_cross600",
                     decoder: str = "S10",
                     is_diagnostic: bool = False) -> dict | None:
    d = read_json(path)
    if not d:
        return None
    r = blank_row(arm, group, protocol, decoder, is_diagnostic,
                  path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path))
    if "summary" in d and "cross600" in d["summary"] and "means" in d["summary"]["cross600"]:
        cm = d["summary"]["cross600"]["means"]
        sm = d["summary"].get("same150", {}).get("means", {})
        r["n_cross"] = d["summary"]["cross600"].get("n", 600)
        for k in ("dino_s", "dino_c", "lpips", "clip_s", "style_affinity",
                  "target_top1", "target_margin", "dT", "G"):
            if cm.get(k) is not None:
                r[k] = round(float(cm[k]), 5)
        if sm.get("dino_s") is not None:
            r["same_dino_s"] = round(float(sm["dino_s"]), 5)
        if sm.get("dino_c") is not None:
            r["same_dino_c"] = round(float(sm["dino_c"]), 5)
        if sm.get("target_top1") is not None:
            r["same_top1"] = round(float(sm["target_top1"]), 5)
        return r

    rows = d.get("rows", [])
    cr = [x for x in rows if x.get("is_cross")]
    sr = [x for x in rows if not x.get("is_cross")]
    if not cr:
        return None
    r["n_cross"] = len(cr)
    for k in ("dino_s", "dino_c", "lpips", "target_top1", "target_margin", "dT", "G"):
        vals = [float(x[k]) for x in cr if x.get(k) is not None]
        if vals:
            r[k] = round(statistics.mean(vals), 5)
    clips = [float(x["clip_scores_5"][x["tgt_style_id"]]) for x in cr if "clip_scores_5" in x]
    if clips:
        r["clip_s"] = round(statistics.mean(clips), 5)
        if r["dino_s"] is not None:
            r["style_affinity"] = round((r["dino_s"] + r["clip_s"]) / 2.0, 5)
    if sr:
        for dst, src_k in (("same_dino_s", "dino_s"), ("same_dino_c", "dino_c"), ("same_top1", "target_top1")):
            vals = [float(x[src_k]) for x in sr if x.get(src_k) is not None]
            if vals:
                r[dst] = round(statistics.mean(vals), 5)
    return r


def load_artfid_map() -> dict[str, dict]:
    p = ROOT / "runs/artfid/artfid_results.json"
    res = read_json(p) or {}
    sys.path.insert(0, str(ROOT / "docs/v15"))
    try:
        import plot_v15_dino_artfid as artplot
        name_fn = artplot._af_plot_name
    except Exception:
        name_fn = lambda x: x
    by = {}
    for k, v in res.items():
        if v.get("status") != "done":
            continue
        pn = name_fn(k)
        cands = {k, pn}
        if pn.startswith("D5."):
            cands.add(D5_ALIAS.get(pn[3:], pn))
        if pn in {f"LD{i}" for i in range(5)}:
            cands.update({f"V9.{pn}", f"WEAVE.{pn}"})
        if pn.startswith("W_"):
            cands.add(f"V15.{pn}")
        for c in cands:
            by.setdefault(c, v)
    return by


def attach_artfid(row: dict, af_map: dict):
    arm = row["arm"]
    v = af_map.get(arm) or af_map.get(f"Baseline.{arm}") or af_map.get(arm.replace("V9.", "WEAVE."))
    if not v:
        return
    if v.get("art_fid_mean_of_target_means") is not None:
        row["artfid"] = round(float(v["art_fid_mean_of_target_means"]), 4)
    if v.get("kid_mean") is not None:
        row["kid"] = round(float(v["kid_mean"]), 6)
    row["artfid_n_images"] = v.get("n_images")
    row["artfid_n_missing"] = v.get("n_missing")
    row["artfid_detail"] = {
        "fid_style": v.get("fid_style"),
        "kid_style": v.get("kid_style"),
        "content_lpips": v.get("content_lpips"),
        "art_fid_per_style": v.get("art_fid_per_style"),
    }


def collect_all() -> list[dict]:
    af_map = load_artfid_map()
    rows: list[dict] = []
    seen: set[str] = set()

    def push(r: dict | None):
        if r is None or r["arm"] in seen:
            return
        seen.add(r["arm"])
        attach_artfid(r, af_map)
        rows.append(r)

    # 1. 32-method leaderboard (published external baselines + V6 Q2a/Oracle/B3/B4/ScoreDiff)
    lb = read_json(ROOT / "deliver/q2a_oracle_scores_baseline_protocol.json")
    if lb:
        for item in lb["leaderboard_baseline_methods"]:
            m = item["method"]
            cat = "Baseline" if item.get("category") == "Baseline" else "V6-Ours"
            is_diag = m in ("Identity", "SD-Turbo")
            r = blank_row(m, cat, "q2a_oracle_cross600", item.get("vae", "native / free"),
                          is_diag, "deliver/q2a_oracle_scores_baseline_protocol.json")
            r["n_cross"] = 600
            r["dino_s"] = item.get("dino_s")
            r["dino_c"] = item.get("dino_c")
            r["lpips"] = item.get("lpips")
            r["clip_s"] = item.get("clip_s")
            r["style_affinity"] = item.get("style_affinity")
            r["target_top1"] = round(float(item["csd_top1"]), 5) if item.get("csd_top1") is not None else None
            if m in EXT_SPEED_MS:
                r["ms_per_img"] = EXT_SPEED_MS[m]
            push(r)

    # 2. Reproduced external baselines (Hist2Style, StyleGallery)
    for name, rel in [
        ("Hist2Style", "docs/baseline/Hist2Style/eval/eval.json"),
        ("StyleGallery", "docs/baseline/StyleGallery/eval_stylegallery.json"),
    ]:
        push(eval_json_to_row(name, "Baseline (reproduced)", ROOT / rel, decoder="native / free"))

    # 3. V8 (41 arms) & V11B (6 arms) CSVs + speed
    speed = read_json(ROOT / "runs/v8_speed.json") or {}
    speed.update(read_json(ROOT / "docs/v15/v11b_speed.json") or {})
    for rel_csv, grp, dec in [
        ("docs/v8/results.csv", "V8", "SDVAE-historical"),
        ("docs/v15/v11b_arms.csv", "V11B", "S10"),
    ]:
        p = ROOT / rel_csv
        if p.is_file():
            with p.open(encoding="utf-8-sig") as f:
                for rec in csv.DictReader(f):
                    if rec.get("split") != "cross":
                        continue
                    aname = f"{grp}.{rec['arm']}"
                    proto = "v8_sdvae_cross600" if grp == "V8" else "q2a_oracle_cross600"
                    is_diag = "rand" in rec["arm"].lower() or "ident" in rec["arm"].lower()
                    row = blank_row(aname, grp, proto, dec, is_diag, rel_csv)
                    row["n_cross"] = int(float(rec["n"])) if rec.get("n") else 600
                    for k in ("dino_s", "dino_c", "lpips", "clip_s", "style_affinity",
                              "target_top1", "target_margin", "dT", "G"):
                        if rec.get(k) not in (None, ""):
                            row[k] = round(float(rec[k]), 5)
                    sp = speed.get(rec["arm"])
                    if sp and sp.get("ms_per_img"):
                        row["ms_per_img"] = round(float(sp["ms_per_img"]), 2)
                    push(row)

    # 4. V9-LD (merged dev+test 750) & V10 (full 750)
    for i in range(5):
        parts = []
        for sp_name in ("dev", "test"):
            ep = ROOT / f"runs/v9_local_4070/{sp_name}/LD{i}/eval/eval.json"
            d = read_json(ep)
            if d:
                parts.extend(d.get("rows", []))
        if parts:
            cr = [x for x in parts if x.get("is_cross")]
            sr = [x for x in parts if not x.get("is_cross")]
            row = blank_row(f"V9.LD{i}", "V9-LD", "q2a_oracle_cross600", "S10", False,
                            f"runs/v9_local_4070/{{dev,test}}/LD{i}/eval/eval.json")
            row["n_cross"] = len(cr)
            for k in ("dino_s", "dino_c", "lpips", "clip_s", "style_affinity",
                      "target_top1", "target_margin", "dT", "G"):
                vals = [float(x[k]) for x in cr if x.get(k) is not None]
                if vals:
                    row[k] = round(statistics.mean(vals), 5)
            for dst, sk in (("same_dino_s", "dino_s"), ("same_dino_c", "dino_c"), ("same_top1", "target_top1")):
                vals = [float(x[sk]) for x in sr if x.get(sk) is not None]
                if vals:
                    row[dst] = round(statistics.mean(vals), 5)
            push(row)

    v10_sum = read_json(ROOT / "runs/v10_local_4070/summary_750.json") or {}
    for k, rec in v10_sum.items():
        if k.startswith("LD"):
            continue
        short = k.split(" ")[0]
        ep = ROOT / f"runs/v10_local_4070/all/{short}/eval/eval.json"
        if ep.is_file():
            push(eval_json_to_row(f"V10.{short}", "V10", ep))
        else:
            row = blank_row(f"V10.{short}", "V10", "q2a_oracle_cross600", "S10", False,
                            "runs/v10_local_4070/summary_750.json")
            row["n_cross"] = rec.get("cross_n", 600)
            for m in ("dino_s", "dino_c", "lpips", "clip_s", "style_affinity", "target_top1", "target_margin"):
                if rec.get(m) is not None:
                    row[m] = round(float(rec[m]), 5)
            push(row)

    # 5. V15 Group A (16 arms x 3 levels; t0=0.0 is unconditional generation -> is_diagnostic=True for transfer Pareto)
    full_v15 = {}
    ef_root = ROOT / "runs/v15/eval_full"
    if ef_root.is_dir():
        for d in sorted(ef_root.iterdir()):
            if (d / "eval/eval.json").is_file():
                full_v15[d.name] = d / "eval/eval.json"
    for f in sorted((ROOT / "runs/v15/eval").glob("W_*.json")):
        data = read_json(f)
        if not data or "rows" not in data:
            continue
        for lev in sorted({x["level"] for x in data["rows"]}):
            tag = {0.0: "t00", 0.4: "t04", 0.7: "t07"}.get(lev, f"t{lev}")
            aname = f"V15.{f.stem}.{tag}"
            is_diag = (lev == 0.0)  # pure noise generation has no source content anchor
            fk = f"{f.stem}_t0_{lev}"
            if fk in full_v15:
                push(eval_json_to_row(aname, "V15-GroupA", full_v15[fk], is_diagnostic=is_diag))
            else:
                cr = [x for x in data["rows"] if x["level"] == lev and x["cross"]]
                sr = [x for x in data["rows"] if x["level"] == lev and not x["cross"]]
                row = blank_row(aname, "V15-GroupA", "q2a_oracle_cross600", "S10", is_diag,
                                f.relative_to(ROOT).as_posix())
                row["n_cross"] = len(cr)
                row["dino_s"] = round(statistics.mean([x["dino_s"] for x in cr]), 5)
                row["dino_c"] = round(statistics.mean([x["dino_c"] for x in cr]), 5)
                if sr:
                    row["same_dino_s"] = round(statistics.mean([x["dino_s"] for x in sr]), 5)
                    row["same_dino_c"] = round(statistics.mean([x["dino_c"] for x in sr]), 5)
                push(row)

    # 6. V15 Scaling (S1..S6) & Style-Bridge (PT2)
    sc = read_json(ROOT / "docs/v15/scaling_results_summary.json")
    if sc and "cross600" in sc:
        for arm_k, lvls in sc["cross600"].items():
            if arm_k == "W_dim96_conv_base":
                continue
            for lk, tag in (("t0_0.0", "t00"), ("t0_0.4", "t04"), ("t0_0.7", "t07")):
                if lk not in lvls:
                    continue
                v = lvls[lk]
                row = blank_row(f"V15.{arm_k}.{tag}", "V15-Scaling", "q2a_oracle_cross600", "S10",
                                lk == "t0_0.0", "docs/v15/scaling_results_summary.json")
                row["n_cross"] = 600
                row.update({
                    "dino_s": v.get("dino_s"), "dino_c": v.get("dino_c"), "lpips": v.get("lpips"),
                    "target_top1": v.get("top1"), "target_margin": v.get("margin"), "G": v.get("G"),
                    "style_affinity": v.get("affinity"), "clip_s": v.get("clip"),
                })
                push(row)

    for nm, ds, dc, lp, t1, mar, g_val, is_d in [
        ("V15.PT2_P2_s0.t00", 0.5707, 0.3010, 0.8556, 0.4567, -0.0489, 0.0324, True),
        ("V15.PT2_P2_s0.t04", 0.5218, 0.4428, 0.5062, 0.1783, -0.1666, -0.0224, False),
        ("V15.PT2_P2_s0.t07", 0.4590, 0.7499, 0.2749, 0.1100, -0.2350, -0.0435, False),
        ("V15.PT2_P2_s1.t04", 0.5219, 0.4433, 0.5071, 0.1783, -0.1659, -0.0225, False),
        ("V15.PT2_P2_s1.t07", 0.4592, 0.7501, 0.2751, 0.1167, -0.2353, -0.0435, False),
        ("V15.PT2_P3_s0.t07", 0.4207, 0.8812, 0.1741, 0.0183, -0.3232, -0.0526, True),
    ]:
        row = blank_row(nm, "V15-Bridge", "q2a_oracle_cross600", "S10", is_d,
                        "docs/v15/WEAVE_V15_POSTSCALE_AND_POSTTRAIN_SUMMARY.md")
        row["n_cross"] = 600
        row.update({"dino_s": ds, "dino_c": dc, "lpips": lp, "target_top1": t1, "target_margin": mar, "G": g_val})
        push(row)

    # 7. V16 (9 arms)
    for ep in sorted((ROOT / "runs/v16/measured").glob("*/eval.json")):
        nm = ep.parent.name
        push(eval_json_to_row(f"V16.{nm}", "V16", ep, is_diagnostic=nm.endswith("_wrong")))

    # 8. V17 (excluding _canary_* sanity dirs)
    for ep in sorted((ROOT / "runs/v17/measured").glob("*/eval.json")):
        nm = ep.parent.name
        if nm.startswith("_canary"):
            continue
        is_diag = ("_wrong" in nm) or ("_swap_" in nm)
        push(eval_json_to_row(f"V17.{nm}", "V17", ep, is_diagnostic=is_diag))

    # 9. V18 Breakthrough arms
    for ep in sorted((ROOT / "runs/v18/measured").glob("*/eval.json")):
        nm = ep.parent.name
        push(eval_json_to_row(f"V18.{nm.removeprefix('V18_')}", "V18", ep, is_diagnostic=("_wrong" in nm)))

    # 10. V19 / PRISM v2 arms (Orthogonal Quotient-Fiber Duality + Anisotropic S10-VGG + 1-Step Chord)
    v19_speed = {
        "V19_S1_global_nosid": 17.37,
        "V19_S2_prism1_l075": 17.16,
        "V19_S2_prism1_v2": 17.04,
        "V19_S2_prism1_l130": 17.01,
        "V19_S2b_prism1_blind_same": 16.81,
        "V19_S2_prism1_wrong": 17.32,
        "V19_S3_oneshot_exemplar": 17.03,
        "V19_T1_aniso_r1": 55.5,
        "V19_T2_s10_vgg_nosid": 203.3,
        "V19_T4_sweet_vgg_nosid": 137.1,
        "V19_T5_sweet_hybrid_nosid": 245.0,
    }
    for ep in sorted((ROOT / "runs/v19/measured").glob("*/eval.json")):
        nm = ep.parent.name
        # Mark wrong-label controls and over-strong T2/T3 (DINO-C < 0.56) appropriately
        is_diag = ("_wrong" in nm) or (nm in ("V19_T1_aniso_r1", "V19_T2_s10_vgg_nosid", "V19_T3_hybrid_pro_nosid"))
        row = eval_json_to_row(f"V19.{nm.removeprefix('V19_')}", "V19-PRISM", ep, is_diagnostic=is_diag)
        if nm in v19_speed:
            row["ms_per_img"] = v19_speed[nm]
        push(row)

    # 11. V20 / PRISM-Flow & Submode Patch-SWD arms (remote 4090 measured summary or local eval.json)
    v20_done: set[str] = set()
    for ep in sorted((ROOT / "runs/v20/measured").glob("*/eval.json")):
        nm = ep.parent.name
        is_diag = ("_wrong" in nm)
        row = eval_json_to_row(f"V20.{nm.removeprefix('V20_')}", "V20-PRISM", ep, is_diagnostic=is_diag)
        push(row)
        v20_done.add(nm)

    v20_cache = read_json(ROOT / "docs/next1118/v20_measured_cache.json")
    if v20_cache and "arms" in v20_cache:
        for rec in v20_cache["arms"]:
            nm = rec["arm"]
            if nm in v20_done:
                continue
            is_diag = bool(rec.get("is_diagnostic", "_wrong" in nm))
            row = blank_row(
                f"V20.{nm.removeprefix('V20_')}", "V20-PRISM", "q2a_oracle_cross600", "S10",
                is_diag, "docs/next1118/v20_measured_cache.json",
            )
            row["n_cross"] = 600
            row.update({
                "dino_s": rec.get("dino_s"),
                "clip_s": rec.get("clip_s"),
                "style_affinity": rec.get("style_affinity"),
                "dino_c": rec.get("dino_c"),
                "lpips": rec.get("lpips"),
                "target_top1": rec.get("csd_top1"),
                "target_margin": rec.get("csd_margin"),
                "same_top1": rec.get("same150_top1"),
                "same_dino_s": rec.get("same150_dino_s"),
                "same_dino_c": rec.get("same150_dino_c"),
                "ms_per_img": rec.get("ms_per_img"),
            })
            push(row)

    # 12. V21 / PRISM 1-Shot Arbitrary-Reference & Gallery Cross-Attention arms
    v21_cache = read_json(ROOT / "docs/next1118/v21_measured_cache.json")
    if v21_cache and "arms" in v21_cache:
        for rec in v21_cache["arms"]:
            nm = rec["arm"]
            is_diag = bool(rec.get("is_diagnostic", "_wrong" in nm))
            row = blank_row(
                f"V21.{nm.removeprefix('V21_')}", "V21-1Shot", "q2a_oracle_cross600", "S10",
                is_diag, "docs/next1118/v21_measured_cache.json",
            )
            row["n_cross"] = 600
            row.update({
                "dino_s": rec.get("dino_s"),
                "clip_s": rec.get("clip_s"),
                "style_affinity": rec.get("style_affinity"),
                "dino_c": rec.get("dino_c"),
                "lpips": rec.get("lpips"),
                "target_top1": rec.get("csd_top1"),
                "target_margin": rec.get("csd_margin"),
                "same_top1": rec.get("same150_top1"),
                "ms_per_img": rec.get("ms_per_img"),
            })
            push(row)

    return rows


def mark_pareto(rows: list[dict]):
    for r in rows:
        r["pareto"] = False
    by_proto: dict[str, list[dict]] = {}
    for r in rows:
        if r["is_diagnostic"]:
            continue
        if None in (r["dino_s"], r["dino_c"], r["target_top1"], r["lpips"]):
            continue
        by_proto.setdefault(r["protocol"], []).append(r)

    for grp in by_proto.values():
        def _v(x):
            return (x["dino_s"], x["dino_c"], x["target_top1"], -x["lpips"])
        for r in grp:
            vr = _v(r)
            dom = any(
                all(b >= a for b, a in zip(_v(o), vr)) and any(b > a for b, a in zip(_v(o), vr))
                for o in grp if o is not r
            )
            r["pareto"] = not dom


def render_pareto_dashboard(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    valid = [
        r for r in rows
        if r["protocol"] == "q2a_oracle_cross600"
        and not r["is_diagnostic"]
        and None not in (r["dino_s"], r["dino_c"], r["target_top1"], r["lpips"])
    ]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=160)
    panels = [
        (axes[0], "lpips", "target_top1", "LPIPS ↓ (Perceptual Distortion vs Source)", "CSD Target Top-1 ↑ (5-Style Specificity)"),
        (axes[1], "dino_c", "dino_s", "DINO-C ↑ (Structural Content Preservation)", "DINO-S ↑ (Reference Style Similarity)"),
    ]
    highlight_names = {
        "StyleShot", "StyleID", "StyleGallery", "AesPA", "StyTr2", "Seedream 4.5",
        "Hist2Style", "Q2a + R1 Oracle (1 step) + S10", "P1_B4_distill50 (218K) + S10",
        "V9.LD4", "V10.E4", "V15.PT2_P2_s1.t04", "V16.B_teacher", "V17.P2_S4_eval",
    }
    for r in valid:
        if r["group"] in ("V18", "V19-PRISM", "V20-PRISM") and r["pareto"]:
            highlight_names.add(r["arm"])

    for ax, xk, yk, xl, yl in panels:
        ax.set_facecolor("#FCFDFE")
        for r in valid:
            is_ext = r["group"].startswith("Baseline")
            is_prism = r["group"] in ("V19-PRISM", "V20-PRISM")
            is_v18 = r["group"] == "V18"
            if is_prism and r["pareto"]:
                c, mk, sz, zo = "#C81D25", "*", 230, 7
            elif is_prism:
                c, mk, sz, zo = "#ED553B", "o", 56, 5
            elif is_v18:
                c, mk, sz, zo = "#E65100", "*", 165, 6
            elif r["pareto"] and not is_ext:
                c, mk, sz, zo = "#F59E0B", "*", 140, 5
            elif r["pareto"] and is_ext:
                c, mk, sz, zo = "#2B5B84", "D", 68, 4
            elif not is_ext:
                c, mk, sz, zo = "#7A4FA3", "o", 36, 3
            else:
                c, mk, sz, zo = "#9AA5B1", "X", 36, 2
            ax.scatter(r[xk], r[yk], c=c, marker=mk, s=sz, alpha=0.88,
                       edgecolors="#182A42", linewidths=0.45, zorder=zo)
            if r["arm"] in highlight_names:
                short_lbl = (
                    r["arm"]
                    .replace(" (1 step) + S10", "")
                    .replace(" (218K) + S10", "")
                )
                ax.annotate(short_lbl, (r[xk], r[yk]), fontsize=7, color="#182A42",
                            xytext=(4, 3), textcoords="offset points", zorder=7)
        ax.set_xlabel(xl, fontsize=10, color="#182A42")
        ax.set_ylabel(yl, fontsize=10, color="#182A42")
        ax.grid(True, color="#E8EEF4", linewidth=0.8)

    axes[0].set_title("A. Style Specificity (CSD Top-1) vs Distortion (LPIPS)", fontsize=11, fontweight="bold", color="#182A42")
    axes[1].set_title("B. Style Similarity (DINO-S) vs Content Fidelity (DINO-C)", fontsize=11, fontweight="bold", color="#182A42")
    fig.suptitle(
        "WEAVE Unified Cross600 Performance Landscape (S10 Referee Protocol) — "
        "Orange/Red ★ = WEAVE Pareto, Blue ◆ = External Pareto",
        fontsize=11.5, fontweight="bold", color="#182A42",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"pareto_frontier_cross600.{ext}")
    plt.close(fig)


def main():
    rows = collect_all()
    mark_pareto(rows)
    PERF.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with (PERF / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    json_payload = {
        "protocol": "eval_protocol/tasks_750.json (Cross600 primary, Same150 identity control, S10 decoder)",
        "n_rows": len(rows),
        "n_pareto": sum(1 for r in rows if r["pareto"]),
        "rows": rows,
    }
    (PERF / "metrics.json").write_text(json.dumps(json_payload, ensure_ascii=False, indent=1), encoding="utf-8")
    render_pareto_dashboard(rows)
    print(f"[Master Ledger] {len(rows)} total rows ({json_payload['n_pareto']} Pareto) -> docs/performance/metrics.{{csv,json}}")


if __name__ == "__main__":
    main()
