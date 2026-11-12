"""One-shot P5 index repair, run AFTER P0-P4 but BEFORE evaluate.report.

The running run_v7r_next.sh has a wrong Q2a_R1_Oracle path in its inline
index builder (eval_Q2a_R1 instead of eval_Q2a_oracle). We must NOT overwrite
a running bash script (byte-offset read hazard). This standalone script
rewrites runs/v7r_next/p5/eval_index.json with correct paths, then calls
evaluate.report. Idempotent.

Usage (3060 WSL): python3 -m tools.v7r_next.p5_fix_index
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402
from tools.v7r_next import evaluate as EV  # noqa: E402

HISTORY = {
    "B3": "runs/v6_r2/eval_B3_distill20/eval.json",
    "B4": "runs/v6_r2/eval_B4_distill50/eval.json",
    "Q2a_R1_Oracle": "runs/v6_r2/eval_Q2a_oracle/eval.json",
    "R3c": "runs/v7r/p3_corrector/R3c/eval.json",
    "R4c": "runs/v7r/p3_corrector/R4c/eval.json",
    "R3c_disp": "runs/v7r/p3_corrector/R3c_disp/eval.json",
    "R1_Full_repaired": "runs/v7r/p3_corrector/R1c/eval.json",
    "R1_TopK_repaired": "runs/v7r/p3_corrector/R2c/eval.json",
}


def build_index() -> dict:
    idx = {}
    for name, rel in HISTORY.items():
        p = C.ROOT / rel
        if p.exists():
            idx[name] = rel
        else:
            print(f"  [skip] {name}: missing {rel}", flush=True)
    # new CPT / student / C-ablations produced under v7r_next
    for p in sorted(C.RUNS.rglob("eval.json")):
        if "v7r_next" not in str(p):
            continue
        key = p.parent.name
        if key in idx:
            continue
        # exclude smoke / audit-internal evals from the formal table
        if any(t in str(p) for t in ("_smoke", "p0_audit")):
            continue
        idx[key] = str(p.relative_to(C.ROOT))
    return idx


def main() -> None:
    idx = build_index()
    out = C.RUNS / "p5/eval_index.json"
    C.write_json(out, idx)
    print("eval_index arms:", list(idx), flush=True)
    evals = {k: C.read_json(C.ROOT / v) for k, v in idx.items()}
    r = EV.build_report(evals)
    C.write_json(C.RUNS / "p5/final_report.json", r)
    EV.write_markdown(r)
    print("wrote final_report.json + V7R_NEXT_BREAKTHROUGH.md", flush=True)


if __name__ == "__main__":
    main()
