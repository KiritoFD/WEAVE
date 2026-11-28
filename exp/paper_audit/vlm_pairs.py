"""Resumable, balanced two-position VLM comparisons; no model names in prompts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAIRS = [
    ("cleave_pe_s085", m) for m in
    ("cleave_e_l075", "cleave_one_t50", "stylegallery", "stytr2", "aespa", "seedream")
]


def run_pairs(tasks, server_url, n_pairwise=40):
    from .f2_vlm_eval import (
        F2_DIR, PAIRWISE_PROMPT, _chat_vlm, _find_cand_path, _img_b64_uri,
        _parse_json_reply,
    )
    from src.utils.runtime import REPO, read_json, save_json
    full_tasks = read_json(REPO / "eval_protocol/tasks_750.json")["tasks"]
    # Preserve the original 40 predetermined tasks. V2 always uses both orders.
    selected = tasks[:n_pairwise]
    p = F2_DIR / "vlm_pairwise_balanced.jsonl"
    if not p.exists():
        old_p = F2_DIR / "vlm_pairwise.jsonl"
        allowed = {int(t["idx"]) for t in selected}
        with p.open("w", encoding="utf-8") as initial:
            for line in old_p.read_text(encoding="utf-8").splitlines() if old_p.exists() else []:
                r = json.loads(line)
                raw = r.get("raw", {})
                if r["idx"] not in allowed or not all(str(raw.get(k, "")).upper() in ("X", "Y", "TIE")
                        for k in ("winner", "style_winner", "content_winner")):
                    continue
                r["src_stem"] = full_tasks[r["idx"]]["source_stem"]
                r["reused_original"] = True
                initial.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    done = {(r["idx"], r["m_a"], r["m_b"], r["swapped"]) for r in rows}
    attempts_p = F2_DIR / "vlm_pairwise_attempts.jsonl"
    with p.open("a", encoding="utf-8") as out, attempts_p.open("a", encoding="utf-8") as failures:
        for t in selected:
            idx = int(t["idx"])
            root = F2_DIR / "images"
            fixed = [_img_b64_uri(root / d / f"{idx:04d}.png") for d in ("source", "ref")]
            for ma, mb in PAIRS:
                pa, pb = _find_cand_path(ma, idx), _find_cand_path(mb, idx)
                if pa is None or pb is None:
                    raise FileNotFoundError(f"Missing pair image: {idx} {ma} {mb}")
                for swapped in (False, True):
                    if (idx, ma, mb, swapped) in done:
                        continue
                    xp, yp = (pb, pa) if swapped else (pa, pb)
                    xm, ym = (mb, ma) if swapped else (ma, mb)
                    for attempt in range(3):
                        reply = _chat_vlm(server_url, PAIRWISE_PROMPT,
                                          fixed + [_img_b64_uri(xp), _img_b64_uri(yp)],
                                          max_tokens=384)
                        parsed = _parse_json_reply(reply)
                        if parsed and all(str(parsed.get(k, "")).upper() in ("X", "Y", "TIE")
                                          for k in ("winner", "style_winner", "content_winner")):
                            break
                        failures.write(json.dumps({"idx": idx, "m_a": ma, "m_b": mb,
                                                   "swapped": swapped, "attempt": attempt,
                                                   "reply": reply}, ensure_ascii=False) + "\n")
                        failures.flush()
                    else:
                        raise ValueError(f"Invalid VLM response after 3 attempts: {idx} {ma} {mb} {swapped}")
                    raw_winner = str(parsed["winner"]).upper()
                    rec = {"idx": idx, "m_a": ma, "m_b": mb, "swapped": swapped,
                           "src_stem": full_tasks[idx]["source_stem"],
                           "winner": xm if raw_winner == "X" else ym if raw_winner == "Y" else "TIE",
                           "raw": parsed}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(rec)
                print(f"[F2 balanced] task={idx} vs={mb} valid={len(rows)}", flush=True)
    summary = summarize_pairs(rows, len(selected))
    save_json(F2_DIR / "f2_balanced_summary.json", summary)
    return summary


def summarize_pairs(rows, n_tasks):
    from src.eval.metrics import bootstrap_source_ci
    stats = {}
    for ma, mb in PAIRS:
        sub = [r for r in rows if r["m_a"] == ma and r["m_b"] == mb]
        groups = {}
        for r in sub:
            g = groups.setdefault(r["idx"], {})
            if r["swapped"] in g:
                raise ValueError("Duplicate position judgment")
            g[r["swapped"]] = r
        if len(groups) != n_tasks or any(len(g) != 2 for g in groups.values()):
            raise ValueError(f"Incomplete two-position coverage: {ma} vs {mb}")
        scores, sources, consistent = [], [], []
        for g in groups.values():
            rr = list(g.values())
            scores.append(float(np.mean([1 if r["winner"] == ma else 0 if r["winner"] == mb else .5 for r in rr])))
            sources.append(rr[0]["src_stem"])
            consistent.append(rr[0]["winner"] == rr[1]["winner"])
        ci = bootstrap_source_ci(scores, sources, seed=20261128)
        stats[f"{ma}_vs_{mb}"] = {
            "n_tasks": n_tasks, "n_judgments": len(sub),
            "position_consistency": float(np.mean(consistent)),
            "a_score_tie_half": ci,
            "win_a": sum(r["winner"] == ma for r in sub),
            "win_b": sum(r["winner"] == mb for r in sub),
            "tie": sum(r["winner"] == "TIE" for r in sub),
        }
    return {"protocol": "both positions, per-task average, source-cluster bootstrap; ties=0.5",
            "expected_judgments": n_tasks * len(PAIRS) * 2, "valid_judgments": len(rows),
            "comparisons": stats}


if __name__ == "__main__":
    import argparse
    from .f2_vlm_eval import F2_DIR
    from src.utils.runtime import read_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:8089")
    args = ap.parse_args()
    run_pairs(read_json(F2_DIR / "f2_manifest.json")["tasks"], args.server_url)
