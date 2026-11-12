"""V7 P5: unified statistics, gates, runtime, and the final report.

Collects every V7 eval arm, checks protocol compliance against the B3 anchor,
computes paired clustered bootstrap vs B3, P4.3 gates, DINO-C bins, runtime
and gradient audits, then writes:

  runs/v7/final_matrix.json
  runs/v7/final_verdict.json
  runs/v7/final_report_inputs.json
  runs/v7/runtime.json
  docs/v7/V7_REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v6_24h.stats import paired_cluster_ci
from tools.v7.v7_common import (
    P0_DIR, P2_DIR, P3_EVAL, P4_DIR, ROOT, STYLES, tasks_sha256, write_json,
)
from tools.v7.p4_combine import dinoc_bins, load_eval

B3_EVAL = ROOT / "runs" / "v6_r2" / "eval_B3_distill20"
B4_EVAL = ROOT / "runs" / "v6_r2" / "eval_B4_distill50"

ARM_DEFS = [
    ("p0_q2a_direct", P0_DIR / "p0_q2a_direct", "P0"),
    ("p0_q2a_relative", P0_DIR / "p0_q2a_relative", "P0"),
    ("p0_b4_direct", P0_DIR / "p0_b4_direct", "P0"),
    ("p0_b4_relative", P0_DIR / "p0_b4_relative", "P0"),
    ("T0", P2_DIR / "T0", "P2"),
    ("T1", P2_DIR / "T1", "P2"),
    ("T2", P2_DIR / "T2", "P2"),
    ("T3", P2_DIR / "T3", "P2"),
    ("T4", P2_DIR / "T4", "P2"),
    ("T5", P2_DIR / "T5", "P2"),
    ("M1", P3_EVAL / "M1", "P3"),
    ("M2", P3_EVAL / "M2", "P3"),
    ("C0", P4_DIR / "C0", "P4"),
    ("C1", P4_DIR / "C1", "P4"),
    ("C2", P4_DIR / "C2", "P4"),
    ("C3", P4_DIR / "C3", "P4"),
    ("C4", P4_DIR / "C4", "P4"),
]

GATES = {
    "dino_s_delta": 0.020,
    "dino_c_min": 0.8125,
    "lpips_max": 0.4712,
    "top1_min_delta": -0.02,
}


def paired_vs_b3(eval_json: dict, b3: dict) -> dict:
    rb = {r["task_id"]: r for r in b3["rows"] if r["is_cross"]}
    ra = {r["task_id"]: r for r in eval_json["rows"] if r["is_cross"]}
    if set(rb) != set(ra):
        return {"protocol_mismatch": True,
                "missing": len(set(rb) - set(ra)), "extra": len(set(ra) - set(rb))}
    common = sorted(rb)
    out = {}
    for metric in ["target_top1", "dino_s", "dino_c", "lpips"]:
        a = [ra[k][metric] for k in common]
        b = [rb[k][metric] for k in common]
        clusters = [ra[k]["src_stem"] for k in common]
        p = paired_cluster_ci(a, b, clusters, n_boot=4000, seed=0)
        out[metric] = {"delta": p["diff"], "ci": p["ci"],
                       "excludes_zero": p["excludes_zero"]}
    return out


def gate_status(eval_json: dict, paired: dict) -> str:
    if paired.get("protocol_mismatch"):
        return "PROTOCOL_MISMATCH"
    c = eval_json["cross600"]
    ds = paired["dino_s"]
    if (ds["delta"] >= GATES["dino_s_delta"] and ds["ci"][0] > 0
            and c["dino_c"] >= GATES["dino_c_min"]
            and c["lpips"] <= GATES["lpips_max"]
            and paired["target_top1"]["delta"] >= GATES["top1_min_delta"]):
        return "PASS"
    if ds["ci"][0] > 0:
        return "PARTIAL"
    if (c["dino_c"] < GATES["dino_c_min"] - 0.01
            or c["lpips"] > GATES["lpips_max"] + 0.01
            or paired["dino_s"]["delta"] < -0.005):
        return "NO"
    return "PARTIAL"


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 P5 final report")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    b3 = load_eval(B3_EVAL)
    b4 = load_eval(B4_EVAL)
    if b3 is None:
        raise SystemExit("B3 anchor eval missing")

    matrix, inputs_log = [], []
    for name, path, phase in ARM_DEFS:
        e = load_eval(path)
        entry = {"arm_id": name, "phase": phase, "eval_path": str(path)}
        if e is None:
            entry.update({"status": "MISSING", "include_in_final": False,
                          "reason": "eval.json not found"})
            matrix.append(entry)
            inputs_log.append({"path": str(path), "included": False,
                               "reason": "missing"})
            continue
        paired = paired_vs_b3(e, b3)
        status = gate_status(e, paired)
        ok_tasks = not paired.get("protocol_mismatch")
        entry.update({
            "cross600": e["cross600"], "same150": e["same150"],
            "paired_vs_B3": paired,
            "dinoc_bins": dinoc_bins(e),
            "grad_balance": None,
            "status": status,
            "include_in_final": bool(ok_tasks),
            "reason": "" if ok_tasks else "task-id set mismatch vs B3",
        })
        gn = path / "grad_norm.json"
        if gn.exists():
            entry["grad_balance"] = json.loads(gn.read_text(encoding="utf-8")).get("audit")
        rt = path / "runtime.json"
        if rt.exists():
            entry["runtime"] = json.loads(rt.read_text(encoding="utf-8"))
        matrix.append(entry)
        inputs_log.append({"path": str(path), "included": bool(ok_tasks),
                           "status": status})

    # per-target / routes for final candidates
    for entry in matrix:
        p = Path(entry["eval_path"])
        e = load_eval(p)
        if e and entry["include_in_final"]:
            entry["per_target"] = e["cross600"]["per_target"]
            entry["directed_pairs"] = e["cross600"]["directed_pairs"]

    write_json({"arms": matrix, "gates": GATES,
                "tasks_sha256": tasks_sha256()},
               ROOT / "runs" / "v7" / "final_matrix.json")

    # runtime aggregate
    runtime = {"evals": {}, "train": {}, "notes": "3060 RTX3060 12GB, batch=8, TRT fp16 SD-VAE"}
    for entry in matrix:
        if entry.get("runtime"):
            runtime["evals"][entry["arm_id"]] = entry["runtime"]
    for name in ["M1", "M2"]:
        cfg = ROOT / "runs" / "v7" / "p3_models" / name / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            runtime["train"][name] = {
                "wall_sec": c.get("train_wall_sec"),
                "peak_vram_mib": c.get("peak_vram_mib"),
                "steps": c.get("steps"),
                "effective_batch": c.get("effective_batch"),
            }
    write_json(runtime, ROOT / "runs" / "v7" / "runtime.json")

    # verdicts + question answers
    p0v = _read(ROOT / "runs" / "v7" / "p0_verdict.json")
    p1v = _read(ROOT / "runs" / "v7" / "p1_energy" / "energy_validity.json")
    p2v = _read(ROOT / "runs" / "v7" / "p2_verdict.json")
    p3v = _read(ROOT / "runs" / "v7" / "p3_verdict.json")
    p4v = _read(ROOT / "runs" / "v7" / "p4_verdict.json")

    by_arm = {m["arm_id"]: m for m in matrix}
    q_a = answer_a(p1v, p2v, by_arm)
    q_b = answer_b(p0v, p3v, by_arm)
    q_c = answer_c(p2v, p3v, p4v)
    p4c = ((p4v or {}).get("c4_vs_c2") or {})
    stage_p4 = ("COMPLEMENTARY" if p4c.get("complementary")
                else "INCONCLUSIVE" if not p4c else "NOT_COMPLEMENTARY") if p4v else "SKIPPED"
    verdict = {
        "questions": {"A": q_a, "B": q_b, "C": q_c},
        "stage_verdicts": {
            "P0": (p0v or {}).get("verdict"),
            "P1": (p1v or {}).get("verdict"),
            "P2": (p2v or {}).get("verdict"),
            "P3": (p3v or {}).get("verdict"),
            "P4": stage_p4,
        },
        "best_overall": best_overall(matrix),
        "dev_set_statement": "All statistics are paired in-development-set results "
                             "on tasks_750 (Val200 ⊂ Test750 / Test750 repeated use); "
                             "no blind-test claim is made.",
    }
    write_json(verdict, ROOT / "runs" / "v7" / "final_verdict.json")
    write_json({"included": inputs_log, "b3_anchor": str(B3_EVAL),
                "excluded": [i for i in inputs_log if not i["included"]]},
               ROOT / "runs" / "v7" / "final_report_inputs.json")

    write_report(matrix, verdict, p0v, p1v, p2v, p3v, p4v)
    print("[p5] report written", flush=True)
    return 0


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def answer_a(p1v, p2v, by_arm) -> dict:
    if not p1v or p1v.get("verdict") != "GO_ENERGY":
        return {"answer": "NO — current real-style descriptions did not pass "
                          "validity; only T0/T1 controls were run",
                "evidence": ["p1_energy/energy_validity.json"]}
    if p2v and p2v.get("verdict") == "NEW_CORRECTOR":
        best = p2v.get("best_corrector")
        return {"answer": "YES — real-distribution target outperformed the R1 "
                          "classification boundary at matched DINO-C",
                "evidence": [f"p2_eval/{best} vs p2_eval/T0",
                             "p1_energy/energy_validity.json best_energy="
                             f"{p1v.get('best_energy')}"]}
    return {"answer": "NO — real-distribution targets did not beat T0 at "
                      "matched content level; negative result recorded",
            "evidence": ["runs/v7/p2_verdict.json"]}


def answer_b(p0v, p3v, by_arm) -> dict:
    ev = ["p0_verdict.json", "p3_verdict.json"]
    if p3v and p3v.get("verdict") == "NEW_PREDICTOR":
        return {"answer": "YES — explicit source→target residual modeling "
                          "(M2) beat B4 and M1 with lower source leakage",
                "evidence": ev}
    if p3v and p3v.get("verdict") == "STRUCTURE_ONLY":
        return {"answer": "PARTIAL — relative structure (M1) helped but the "
                          "real-distribution objective (M2) added nothing",
                "evidence": ev}
    return {"answer": "NO — relative residual parameterization / M2 did not "
                      "reduce the conditional shortcut or beat baselines; next "
                      "stage should target style-invariant content representation",
            "evidence": ev}


def answer_c(p2v, p3v, p4v) -> dict:
    comp = (p4v or {}).get("c4_vs_c2")
    if comp and comp.get("complementary"):
        return {"answer": "YES — a better initial transformation reduced the "
                          "required correction while preserving content and style",
                "evidence": ["p4_verdict.json c4_vs_c2"]}
    if comp:
        return {"answer": "NO — Predictor+Corrector did not show complementarity; "
                          "keep the effective single module only",
                "evidence": ["p4_verdict.json c4_vs_c2"]}
    return {"answer": "INCONCLUSIVE — C4 was not run (P2/P3 not both NEW)",
            "evidence": ["p4_skipped.json"]}


def best_overall(matrix) -> dict | None:
    cands = [m for m in matrix if m.get("include_in_final")
             and m["status"] in ("PASS", "PARTIAL")]
    if not cands:
        return None
    best = max(cands, key=lambda m: (m["status"] == "PASS",
                                     m["cross600"]["dino_s"]))
    return {"arm": best["arm_id"], "status": best["status"],
            "dino_s": best["cross600"]["dino_s"],
            "dino_c": best["cross600"]["dino_c"],
            "lpips": best["cross600"]["lpips"]}


def fmt_arm(m: dict) -> str:
    c, s = m.get("cross600"), m.get("same150")
    if not c:
        return f"| {m['arm_id']} | — missing — |"
    p = m.get("paired_vs_B3", {})
    ds = p.get("dino_s", {})
    return (f"| {m['arm_id']} | {c['top1']*100:.1f} | {c['dino_s']:.4f} "
            f"| {c['dino_c']:.4f} | {c['lpips']:.4f} | {c['dT']:+.4f} "
            f"| {c['minimalism_offtarget']*100:.1f} "
            f"| {s['same_top1']*100:.1f} | {s['same_lpips']:.4f} "
            f"| {ds.get('delta', 0):+.4f} [{ds.get('ci', [0, 0])[0]:+.4f},"
            f"{ds.get('ci', [0, 0])[1]:+.4f}] | {m['status']} |")


def write_report(matrix, verdict, p0v, p1v, p2v, p3v, p4v) -> None:
    lines = ["# WEAVE V7 Report (auto-generated by tools/v7/p5_report.py)", ""]
    lines += ["## 1. Stage Go/No-Go", ""]
    for k, v in verdict["stage_verdicts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## 2. All candidates — full Cross600 / Same150", "",
              "| arm | Top-1 | DINO-S | DINO-C | LPIPS | dT | MinOff% | Same-Top1 | Same-LPIPS | ΔDINO-S vs B3 [95% CI] | status |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [fmt_arm(m) for m in matrix]
    lines += ["", "## 3. Paired bootstrap vs B3",
              "Per-arm paired deltas for Top-1 / DINO-S / DINO-C / LPIPS are in "
              "`runs/v7/final_matrix.json` (150 source clusters, 4000 bootstrap).", ""]
    lines += ["## 4. DINO-C bins (等 DINO-C 档位)", ""]
    for m in matrix:
        if m.get("dinoc_bins") and m.get("include_in_final"):
            lines.append(f"### {m['arm_id']}")
            lines.append("| bin | n | DINO-S mean | LPIPS mean |")
            lines.append("|---|---|---|---|")
            for k, v in m["dinoc_bins"].items():
                lines.append(f"| {k} | {v['n']} | {v['dino_s_mean']:.4f} "
                             f"| {v['lpips_mean']:.4f} |")
            lines.append("")
    lines += ["## 5. Gradient audits", ""]
    for m in matrix:
        gb = m.get("grad_balance")
        if gb:
            lines.append(f"- {m['arm_id']}: ratio={gb.get('max_over_min', 0):.1f} "
                         f"imbalanced={gb.get('imbalanced')} "
                         f"policy={gb.get('scale_policy')}")
    lines += ["", "## 6. Inference overhead",
              "See `runs/v7/runtime.json` (wall seconds, per-task seconds, peak "
              "VRAM per arm; batch=8, TRT fp16 SD-VAE decoder).", ""]
    lines += ["## 7. Visual cases", "`runs/v7/visual_grid.png` (10 sources × 5 "
              "targets; mechanism diagnosis only).", ""]
    lines += ["## 8. Questions A / B / C", ""]
    for q, a in verdict["questions"].items():
        lines.append(f"- **{q}**: {a['answer']}  \n  evidence: {', '.join(a['evidence'])}")
    lines += ["", "## 9. Failed / excluded experiments"]
    for m in matrix:
        if m.get("status") in ("MISSING", "PROTOCOL_MISMATCH", "NO"):
            lines.append(f"- {m['arm_id']}: {m.get('status')} "
                         f"{m.get('reason', '')}")
    lines += ["", "## 10. Development-set limitation", "",
              verdict["dev_set_statement"], ""]
    (ROOT / "docs" / "v7" / "V7_REPORT.md").write_text("\n".join(lines),
                                                       encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
