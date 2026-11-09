"""Fold any completed background CPU score files into suite_state.json without
touching the running suite40 process. Safe to run repeatedly.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
SUITE = ROOT / "runs/blog_sbflow/suite40"


def main() -> None:
    state_file = SUITE / "suite_state.json"
    if not state_file.exists():
        print("no suite_state.json")
        return
    state = json.loads(state_file.read_text())
    changed = 0
    for aid, rec in state.items():
        if rec.get("eval") and "target_top1" in rec["eval"]:
            continue
        f = SUITE / f"score_{aid}_quick200" / "style_scores.json"
        if not f.exists():
            continue
        o = json.loads(f.read_text())["aggregate"]["overall"]
        rec["eval"] = {k: o[k] for k in
                       ("target_top1", "target_margin", "csd_dT", "csd_dS",
                        "csd_swd", "csd_mmd2_unbiased", "dino_c", "n")}
        rec["status"] = "done"
        changed += 1
        print(f"  {aid}: top1={o['target_top1']:.4f} margin={o['target_margin']:+.4f} "
              f"dT={o['csd_dT']:+.4f}")
    if changed:
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        from sbflow.suite40 import write_results
        write_results(state, SUITE)
        print(f"folded {changed} score(s)")
    else:
        print("nothing to fold")


if __name__ == "__main__":
    main()
