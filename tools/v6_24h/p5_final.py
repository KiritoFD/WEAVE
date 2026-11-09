"""V6 24h plan section 8 - P5 final independent evaluation.

Runs the two independent tiers on the finalist checkpoints and checks that the Val200
ranking survives them:

  Heldout400  400 cross tasks from 100 sources disjoint from Val200's 50 sources.
              Answers "does the Val200 ordering hold on unseen sources?"
  Same150     150 same-style tasks run THROUGH THE NETWORK (never code-level identity).
              Answers "is the operator self-consistent, not just stylising?"

Also reports the source-clustered paired CI between finalists, because the plan requires
arm-vs-arm claims to be judged on the paired interval rather than overlapping separate CIs.

Usage
    python3 -m tools.v6_24h.p5_final --ckpts A0=runs/v6/D1_c1_gentle/D1_c1_gentle.pt \
        N3=runs/v6/N3_dim32/N3_dim32.pt --skip-run     # reuse existing eval.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v6_24h.stats import paired_cluster_ci

OUT = ROOT / "runs/v6_24h"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TIERS = {"heldout400": OUT / "heldout400_tasks.json",
         "same150": OUT / "same150_tasks.json"}


def run_tier(name: str, ckpt: str, tier: str) -> Path:
    dst = OUT / f"eval_{tier}_{name}"
    cmd = [sys.executable, "-u", "-m", "tools.v6_24h.eval_tiers",
           "--ckpt", ckpt, "--tasks", str(TIERS[tier]), "--out", str(dst)]
    print(f"[p5] {name} on {tier} ...", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[p5] {name}/{tier} FAILED (rc={r.returncode})")
    return dst / "eval.json"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def val200(ckpt_name: str) -> dict | None:
    for cand in (ROOT / f"runs/v6/{ckpt_name}/report.json",
                 ROOT / f"runs/v6_24h/{ckpt_name}/report.json"):
        if cand.exists():
            return (json.loads(cand.read_text()).get("eval200") or None)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="NAME=path/to/ckpt.pt entries")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--out", default="runs/v6_24h/p5_final")
    args = ap.parse_args()

    cks = {}
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        cks[name] = path

    res = {name: {} for name in cks}
    for name, path in cks.items():
        for tier in TIERS:
            p = OUT / f"eval_{tier}_{name}" / "eval.json"
            if not args.skip_run:
                p = run_tier(name, path, tier)
            res[name][tier] = load(p)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "p5_final.json").write_text(
        json.dumps({k: {t: (v or None) for t, v in d.items()} for k, d in res.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# P5 final independent evaluation", ""]
    print(f"\n{'model':14s} {'tier':11s} {'dT':>9s} {'top1':>6s} {'DINO-C':>7s} "
          f"{'LPIPS':>7s} {'sty+':>5s} {'n':>5s}")
    for name in cks:
        v = val200(name)
        if v:
            print(f"{name:14s} {'val200':11s} {v['mean_dT']:+9.4f} "
                  f"{(v.get('per_target_top1') and 0) or 0:6.3f} {v['mean_dino_c']:7.4f} "
                  f"{v['mean_lpips']:7.4f} {str(v['styles_improved'])+'/5':>5s} "
                  f"{v['n_cross']:5d}")
        for tier in TIERS:
            r = res[name][tier]
            if not r:
                print(f"{name:14s} {tier:11s} {'MISSING':>9s}")
                continue
            print(f"{name:14s} {tier:11s} {r['mean_dT']:+9.4f} {r['mean_top1']:6.3f} "
                  f"{r['mean_dino_c']:7.4f} {r['mean_lpips']:7.4f} "
                  f"{str(r['styles_improved'])+'/5':>5s} {r['n']:5d}")

    # does the Val200 ordering survive Heldout400?
    names = [n for n in cks if res[n]["heldout400"] and val200(n)]
    if len(names) >= 2:
        order_val = sorted(names, key=lambda n: -val200(n)["mean_dT"])
        order_held = sorted(names, key=lambda n: -res[n]["heldout400"]["mean_dT"])
        print(f"\n  Val200 order   : {' > '.join(order_val)}")
        print(f"  Heldout400 order: {' > '.join(order_held)}")
        same = order_val == order_held
        print(f"  ordering preserved on source-disjoint sources: {same}")
        lines += [f"- Val200 order: {' > '.join(order_val)}",
                  f"- Heldout400 order: {' > '.join(order_held)}",
                  f"- ordering preserved: **{same}**", ""]

        # paired clustered comparison on Heldout400 between the top two
        a, b = order_held[0], order_held[1]
        ra = {r["task_id"]: r for r in res[a]["heldout400"]["rows"]}
        rb = {r["task_id"]: r for r in res[b]["heldout400"]["rows"]}
        common = sorted(set(ra) & set(rb))
        if common:
            pa = [ra[k]["dT"] for k in common]
            pb = [rb[k]["dT"] for k in common]
            cl = [ra[k]["src_stem"] for k in common]
            p = paired_cluster_ci(pa, pb, cl, n_boot=4000)
            print(f"\n  paired clustered diff ({a} - {b}) on Heldout400: "
                  f"{p['diff']:+.5f} CI [{p['ci'][0]:+.5f}, {p['ci'][1]:+.5f}] "
                  f"excludes_zero={p['excludes_zero']}")
            lines += [f"- paired clustered diff ({a} - {b}): {p['diff']:+.5f} "
                      f"CI [{p['ci'][0]:+.5f}, {p['ci'][1]:+.5f}], "
                      f"excludes zero: **{p['excludes_zero']}**", ""]

    # same-style self-consistency
    print(f"\n{'model':14s} {'Same150 dS':>11s} {'Same150 DINO-C':>15s}")
    lines += ["| model | Same150 ΔS | Same150 DINO-C |", "|---|---:|---:|"]
    for name in cks:
        r = res[name]["same150"]
        if r:
            print(f"{name:14s} {r['mean_dS']:+11.4f} {r['mean_dino_c']:15.4f}")
            lines.append(f"| {name} | {r['mean_dS']:+.4f} | {r['mean_dino_c']:.4f} |")
    print("\n  note: dS near 0 means the operator leaves a same-style image alone, i.e. it is")
    print("        self-consistent rather than applying style unconditionally.")

    (out / "p5_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[p5] wrote {out}/p5_final.md")


if __name__ == "__main__":
    main()
