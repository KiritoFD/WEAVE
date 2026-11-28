"""Seen-only checkpoint selection, matched E comparisons, then complete official splits."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from exp.v36_art_foundation import CKPT_DIR
from exp.v36_art_foundation.eval_ecore import load_reference_ecore, _generate_ecore_raw
from exp.v36_art_foundation.eval_v36 import load_v36_models, sample_v36_ode, reconstruct_reference_a3
from exp.v36_art_foundation.diag_v36 import _load_diag_assets
from exp.v37_editing import OUT, atomic_json
from exp.v37_editing.evaluation.common import P_CKPTS, split_pack, save_latents, measure_embeddings
from src.utils import load_s10_decoder
from src.eval.runner import run_measure_batch_subprocess


def select_ecores(label="matched"):
    result = {"initial": str(CKPT_DIR / "v36_ecore_step04500.pt")}
    details = {}
    if label == "matched":
        steps = []
        for route in ("E0", "E1"):
            steps.append({int(p.stem.rsplit("step", 1)[1]) for p in (OUT / "checkpoints").glob(f"{route}_step*.pt")})
        common = steps[0] & steps[1]
        assert common, "No common saved exposure checkpoint; do not pretend arms are matched"
        step = max(common)
        for route in ("E0", "E1"):
            result[route] = str(OUT / "checkpoints" / f"{route}_step{step:05d}.pt")
            details[route] = {"step": step, "extra_source_exposure": {"A": (step-4500)*6, "B": (step-4500)*4, "C": (step-4500)*4}}
        atomic_json(OUT / f"{label}_selection.json", {"checkpoints": result, "details": details,
                    "criterion": "last common saved step, fixed exposure control; no test-based checkpoint selection", "test_used": False})
        return result
    for route in ("E0", "E1"):
        state = json.loads((OUT / f"{route}_state.json").read_text())
        history = state["history"]
        # Declared criterion, all recorded Seen-Dev checks; never use test750.
        rows = [r for r in history if (OUT / "checkpoints" / f"{route}_step{r['step']:05d}.pt").exists()]
        if rows:
            best = max(rows, key=lambda r: r["dev_vgg_style_margin"] - .5 * r["dev_gray_content_l1"])
            result[route] = str(OUT / "checkpoints" / f"{route}_step{best['step']:05d}.pt")
            details[route] = best
        else:
            result[route] = str(OUT / "checkpoints" / f"{route}_final.pt")
            details[route] = {"step": state["step"], "reason": "no intermediate Seen-Dev check"}
    atomic_json(OUT / f"{label}_selection.json", {"checkpoints": result, "details": details,
                "criterion": "max Seen-Dev32 VGG style margin - .5 gray-content-L1; initial diagnostic only",
                "test_used": False})
    return result


@torch.no_grad()
def generate_split(route, ckpt, split, n, folder, strengths=(.6, .8, 1.), taus=(.35, .50, .65, .80)):
    pck = P_CKPTS["p2_50k"]  # Fixed final P, identical to official V36 P+A3 baseline.
    p, se, _ = load_v36_models(pck)
    pack = split_pack(split, n)
    count = len(pack["zc"])
    if route != "P_A3":
        ec, priors = load_reference_ecore(Path(ckpt))
    else:
        ec, priors = None, torch.load(CKPT_DIR.parent / "cache/v36_seen_train_priors.pt", weights_only=False)
    es = se(pack["sr"].cuda()).cpu()
    eps_ec = torch.randn(pack["zc"].shape, generator=torch.Generator().manual_seed(20261129))
    specs = [(f"E_l{int(l * 100)}", l, None) for l in strengths] + [(f"EP_t{int(t * 100)}", 1., t) for t in taus]
    for suffix, strength, tau in specs:
        name = f"{route}_{split}_{suffix}"
        path = folder / f"{name}.pt"
        if path.exists():
            continue
        outputs = []
        for s in range(0, count, 24):
            sl = slice(s, min(s + 24, count)); b = len(pack["zc"][sl])
            zc, zr, q, sr = [pack[k][sl].cuda().float() for k in ("zc", "zr", "q", "sr")]
            if ec is not None:
                draft = ec(zc, zr, q, sr, torch.full((b,), strength, device="cuda"), priors, noise=eps_ec[sl].cuda())
            else:
                draft = reconstruct_reference_a3(zc, zr, priors)
            out = draft if tau is None else sample_v36_ode(p, draft, pack["y"][sl].cuda(), q,
                        es[sl].cuda(), pack["eps"][sl].cuda(), tau0=tau, wj=3.8, wc=1.6)
            outputs.append(out.cpu())
        save_latents(path, torch.cat(outputs), {"arm": name, "route": route, "ckpt": str(ckpt),
                    "pcore": str(pck), "split": split, "n": count, "strength": strength, "tau": tau,
                    "nfe": (0 if ec is None else 1) + (0 if tau is None else 6)})
        print("GENERATED " + name, flush=True)
    del ec, p, se
    gc.collect(); torch.cuda.empty_cache()


def seen_generate():
    selected = select_ecores()
    folder = OUT / "matched_seen"
    for route, ck in selected.items():
        generate_split(route, ck, "seen_dev", 300, folder, (.6, 1.), (.35, .5, .65))
    generate_split("P_A3", "", "seen_dev", 300, folder, (), (.35, .5, .65))


def decide_extension():
    rows = json.loads((OUT / "matched_seen/summary.json").read_text())
    scores = {}
    for route in ("initial", "E0", "E1", "P_A3"):
        rr = sorted([r for r in rows if r["route"] == route and r["tau"] is not None], key=lambda r: r["dino_c"])
        cc, ss = [r["dino_c"] for r in rr], [r["dino_s"] for r in rr]
        scores[route] = {str(c): float(np.interp(c, cc, ss)) for c in (.60, .65, .70, .75)
                         if cc and cc[0] <= c <= cc[-1]}
    gains = {c: s - scores["P_A3"][c] for c, s in scores["E1"].items() if c in scores["P_A3"]}
    wins = sum(v >= .005 for v in gains.values())
    standalone = {}
    for route in ("initial", "E0", "E1"):
        rr = sorted([r for r in rows if r["route"] == route and r["tau"] is None], key=lambda r: r["dino_c"])
        cc, ss = [r["dino_c"] for r in rr], [r["dino_s"] for r in rr]
        standalone[route] = {str(c): float(np.interp(c, cc, ss)) for c in (.80, .85, .90)
                             if cc and cc[0] <= c <= cc[-1]}
    standalone_gains = {c: s - standalone["initial"][c] for c, s in standalone["E1"].items() if c in standalone["initial"]}
    decision = "A_extend_E1" if wins >= 2 and any(v > 0 for v in standalone_gains.values()) else "skip_P4_no_supported_branch"
    atomic_json(OUT / "p4_decision.json", {"decision": decision, "scores_interpolated": scores,
                "E1_gain_over_P_A3": gains, "n_workpoints_ge_0005": wins,
                "standalone_interpolated": standalone, "standalone_gains_over_initial": standalone_gains,
                "reason": "Existing Oracle works; extend E1 only if Seen-Dev dual improves .005 at two points against P+A3 and standalone also improves. Final acceptance remains .01.",
                "no_new_P_architecture": True, "test750_used_for_selection": False})
    return decision


@torch.no_grad()
def official_core():
    selected = json.loads((OUT / "matched_selection.json").read_text())["checkpoints"]
    selected.pop("initial")
    if (OUT / "extended_selection.json").exists():
        selected["E1plus"] = json.loads((OUT / "extended_selection.json").read_text())["checkpoints"]["E1"]
    tasks, z, q, y, refs, idx, _ = _load_diag_assets()
    final_rows = []
    for route, ck in selected.items():
        ec, priors = load_reference_ecore(Path(ck))
        p, se, _ = load_v36_models(P_CKPTS["p2_50k"])
        dec = load_s10_decoder("cuda")
        pending, metadata = [], {}
        strengths = (1.,) if route == "E1plus" else (.6, .8, 1.)
        specs = [(f"E_l{int(l * 100)}", l, None) for l in strengths] + [(f"EP_t{int(t * 100)}", 1., t) for t in (.35, .50, .65, .80)]
        for suffix, strength, tau in specs:
            tag = f"V37_{route}_{suffix}"
            raw, meas = _generate_ecore_raw(tag, ecore=ec, priors=priors, z_src_750=z, q32_750=q, y16_750=y,
                z_refs=refs["z_ref"], st_refs=refs["stat_1561"], idx_ref=idx, tasks=tasks, dec=dec,
                lam=strength, pcore=p if tau is not None else None, style_enc=se if tau is not None else None,
                tau0=tau, wj=3.8, wc=1.6)
            pending.append((raw, meas, tag))
            metadata[tag] = {"route": route, "ckpt": str(ck), "lambda": strength, "tau": tau,
                             "nfe": 1 + (0 if tau is None else 6), "official_one_shot": True}
        del ec, p, se, dec
        gc.collect(); torch.cuda.empty_cache()
        summaries = run_measure_batch_subprocess(pending, write_png=False, micro_batch=24)
        for raw, meas, tag in pending:
            row = {**summaries[tag], **metadata[tag], "measurement_dir": str(meas)}
            atomic_json(meas / "summary_v37.json", row)
            final_rows.append(row)
            # Completed measurements survive; remove only this arm's regenerable RGB tensors.
            for f in raw.glob("batch_*.pt"):
                f.unlink()
        atomic_json(OUT / "official_core750.json", final_rows)
    from exp.v37_editing.evaluation.diagnostics import core_q_controls
    core_q_controls()
    from exp.v37_editing.evaluation.visuals import fixed_examples
    fixed_examples()


def holdout_generate():
    selected = json.loads((OUT / "matched_selection.json").read_text())["checkpoints"]
    selected.pop("initial")
    if (OUT / "extended_selection.json").exists():
        selected["E1plus"] = json.loads((OUT / "extended_selection.json").read_text())["checkpoints"]["E1"]
    selected["P_A3"] = ""
    for route, ck in selected.items():
        for split in ("artist_ood", "movement_ood"):
            generate_split(route, ck, split, 0, OUT / "holdouts", (), (.35, .50, .65, .80))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("seen_generate", "seen_measure", "decide", "core", "holdout_generate", "holdout_measure", "select_extended"))
    a = ap.parse_args()
    if a.stage == "seen_generate": seen_generate()
    elif a.stage == "seen_measure": measure_embeddings(OUT / "matched_seen")
    elif a.stage == "decide": decide_extension()
    elif a.stage == "core": official_core()
    elif a.stage == "holdout_generate": holdout_generate()
    elif a.stage == "holdout_measure": measure_embeddings(OUT / "holdouts")
    else: select_ecores("extended")
