"""V7-R arm registry for r2_eval --gen v7r.

Arm kinds (all start from the frozen B4 predictor unless stated):
  predictor    : B4 (+ optional H corrector ckpt passed via --ckpt) output
  probe_step   : P1-A single-step RMS-matched edit along one objective gradient
                 (sign +1 descent / -1 ascent / random / wrong-target)
  energy_edit  : P1-C / P3 budgeted 4-step edit with one style objective
                 (E0..E4) plus content/anchor protection
  r1_edit      : R1-margin correctors (full / topk25 / short paths)
  combo        : best predictor (via --ckpt) + best corrector (via p3_gate.json)

Every optimizing arm records optim_trace.json + grad_norm.json + run_meta.json
(per-step norms, accepted/rejected alphas, displacement — P3-B requirement).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.v7.v7_common import (
    STYLES, load_c3unet, load_canon, load_critic, model_direct,
    rms_normalize, s1_loss, s2_loss, tasks_sha256, write_json,
)
from tools.v7r.v7r_common import (
    BudgetedEditor, P1_DIR, P3_DIR, ROOT, V7_P1, audit_summary,
    corrector_output, load_corrector,
)

_task_by_id = None

CANON_MODE = "c1_gentle"
PROBE_RMS = 0.08          # P1-A single-step RMS (same as one T0 step)
EDIT_STEPS = 4
EDIT_RMS = 0.08           # total budget cap identical to v6_w T0/T1
W_CONTENT = 0.525
W_ANCHOR = 0.5


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_energies():
    out = {}
    for name in ["S1_sd", "S2_sd"]:
        p = V7_P1 / f"{name}.pt"
        if p.exists():
            out[name] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def load_patch_banks():
    p = P1_DIR / "patch_banks.pt"
    return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None


def load_v6_style_targets(device: str):
    from tools.v6.train_eval_v6 import build_indomain_style_targets
    from tools.v7r.v7r_common import SD_POOL
    cache = ROOT / "runs" / "v6" / "cache" / "indomain_style_targets_v2_k64.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    train_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(SD_POOL / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        train_by_style[s] = d["latents"].float()[100:]
    return build_indomain_style_targets(train_by_style, ROOT.parent,
                                        device=device, top_k=64)


def make_style_objective(name: str, pack: dict, tid: torch.Tensor,
                         wrong_target: bool = False):
    """Bind target ids into an objective closure for the editor."""
    if wrong_target:
        tid = (tid + 2) % len(STYLES)
    if name == "R1":
        critic = pack["critic"]

        def f_r1(z):
            q = critic(z.float()).float()
            oh = F.one_hot(tid, 5).bool()
            m = q.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(
                q.masked_fill(oh, -1e9), dim=1)
            return F.softplus(-m).mean()
        return f_r1
    if name == "S1":
        stats = pack["energies"]["S1_sd"]
        return lambda z: s1_loss(z, tid, stats)
    if name in ("S2", "E2"):
        stats = pack["energies"]["S2_sd"]
        return lambda z: s2_loss(z, tid, stats, pack["feat_ext"])
    if name == "E0":
        targets = pack["v6_targets"]
        from tools.v6.train_eval_v6 import compute_style_distribution_loss
        return lambda z: compute_style_distribution_loss(z, tid, targets)
    if name == "E1":
        targets = pack["v6_targets"]
        return lambda z: s1_loss(z, tid, targets)
    if name in ("E3", "E4"):
        banks = pack["patch_banks"]
        from tools.v7r.p1_patchbank import bank_softnn_loss
        return lambda z: bank_softnn_loss(z, tid, banks,
                                          use_ll=(name == "E4"))
    if name in ("REPR_A", "REPR_B"):
        from tools.v7r.e3_repr import repr_energy

        variant = name.split("_")[1]
        return lambda z: repr_energy(variant, z, tid)
    raise ValueError(f"unknown objective {name}")


ARM_CONFIGS = {
    # ---- P1-A single-step probes ------------------------------------------------
    **{f"p1a_{o}_{v}": {"kind": "probe_step", "objective": o, "sign": s,
                        "random": r, "wrong": w, "steps": 1, "rms": PROBE_RMS}
       for o in ("S1", "S2", "R1")
       for (v, s, r, w) in (("fwd", 1, False, False), ("rev", -1, False, False),
                            ("rand", 1, True, False), ("wrong", 1, False, True))},
    # ---- P1-C objective comparison (budget-matched 4-step edits) -----------------
    **{e: {"kind": "energy_edit", "objective": e, "steps": EDIT_STEPS,
           "rms": EDIT_RMS, "w_content": W_CONTENT, "w_anchor": W_ANCHOR,
           "w_r1": 0.0, "w_struct": 0.0}
       for e in ("E0", "E1", "E2", "E3", "E4")},
    # ---- P3 correctors -----------------------------------------------------------
    "R0": {"kind": "predictor"},
    "NPred": {"kind": "predictor"},
    "R1c": {"kind": "r1_edit", "mode": "full", "steps": 4, "rms": EDIT_RMS},
    "R2c": {"kind": "r1_edit", "mode": "topk", "steps": 4, "rms": EDIT_RMS,
            "topk_frac": 0.25},
    "R3c": {"kind": "energy_edit", "objective": "BEST_E", "steps": EDIT_STEPS,
            "rms": EDIT_RMS, "w_content": W_CONTENT, "w_anchor": W_ANCHOR,
            "w_r1": 0.0, "w_struct": 0.0},
    "R4c": {"kind": "energy_edit", "objective": "BEST_E", "steps": EDIT_STEPS,
            "rms": EDIT_RMS, "w_content": W_CONTENT, "w_anchor": W_ANCHOR,
            "w_r1": 0.25, "w_struct": 0.0, "r1_as_component": True},
    "R5c": {"kind": "combo", "objective": "BEST_R", "steps": EDIT_STEPS,
            "rms": EDIT_RMS, "w_content": W_CONTENT, "w_anchor": W_ANCHOR,
            "w_r1": 0.25, "w_struct": 0.0},
    # ---- E1 conditioning-intervention arms (need --ckpt = trained corrector) ----
    **{a: {"kind": "predictor_cond", "src_mode": sm, "tgt_mode": tm,
           "fixed_src": 0}
       for a, sm, tm in [("E1_A", "correct", "correct"),
                         ("E1_B", "wrong", "correct"),
                         ("E1_C", "fixed", "correct"),
                         ("E1_D", "correct", "wrong"),
                         ("E1_R1src", "r1_pred", "correct")]},
    # ---- E3 phase-2: learned-representation energy edits (from B4) --------------
    "E3A_repr": {"kind": "energy_edit", "objective": "REPR_A", "steps": EDIT_STEPS,
                 "rms": EDIT_RMS, "w_content": W_CONTENT, "w_anchor": W_ANCHOR,
                 "w_r1": 0.0, "w_struct": 0.0},
    "E3B_repr": {"kind": "energy_edit", "objective": "REPR_B", "steps": EDIT_STEPS,
                 "rms": EDIT_RMS, "w_content": W_CONTENT, "w_anchor": W_ANCHOR,
                 "w_r1": 0.0, "w_struct": 0.0},
    # short R1 paths (only when no new objective passed the P1 gate)
    "R1c_s1": {"kind": "r1_edit", "mode": "full", "steps": 1, "rms": EDIT_RMS},
    "R1c_s2": {"kind": "r1_edit", "mode": "full", "steps": 2, "rms": EDIT_RMS},
    # P3-C content-constraint variants (only when R3/R4 show positive gain)
    "R3c_noC": {"kind": "energy_edit", "objective": "BEST_E", "steps": EDIT_STEPS,
                "rms": EDIT_RMS, "w_content": 0.0, "w_anchor": W_ANCHOR,
                "w_r1": 0.0, "w_struct": 0.0},
    "R3c_disp": {"kind": "energy_edit", "objective": "BEST_E", "steps": EDIT_STEPS,
                 "rms": EDIT_RMS, "w_content": 0.0, "w_anchor": 2.0,
                 "w_r1": 0.0, "w_struct": 0.0},
}


def resolve_best_e(pack: dict) -> str:
    v = _load_json(P1_DIR / "p1_gate.json") or {}
    best = v.get("best_energy")
    if best:
        return best
    raise SystemExit("[v7r] arm requires P1 gate result (p1_gate.json missing)")


def resolve_best_r(pack: dict) -> dict:
    v = _load_json(P3_DIR / "p3_gate.json") or {}
    arm = v.get("best_corrector")
    if not arm:
        raise SystemExit("[v7r] R5c requires P3 gate result (p3_gate.json)")
    return ARM_CONFIGS[arm]


def build_v7r_pack(args, device: str) -> dict:
    from tools.v6_next.w_distill import R1FeatureExtractor

    ckpt_path = (ROOT / args.ckpt) if getattr(args, "ckpt", "") else None
    corrector, corrector_ck = None, None
    if ckpt_path is not None and Path(ckpt_path).exists():
        head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(head, dict) and "corrector" in head:
            corrector, corrector_ck = load_corrector(ckpt_path, device)
    pack = {
        "kind": "v7r",
        "arm": args.v7r_arm,
        "ckpt_path": str(ckpt_path or ""),
        "b4": load_c3unet(ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
                          device),
        "canon": load_canon(device),
        "canon_mode": CANON_MODE,
        "critic": load_critic(device=device),
        "feat_ext": R1FeatureExtractor(load_critic(device=device)),
        "energies": load_energies(),
        "patch_banks": load_patch_banks(),
        "v6_targets": load_v6_style_targets(device),
        "corrector": corrector,
        "corrector_meta": (corrector_ck or {}).get("args", {}),
        "trace_task_ids": set(),
        "trace_state": {"steps": {}, "displacement": {}, "final_z": {}},
        "editor_audit": [],
        "models": None, "src_cache": None, "ref_dino": None,
    }
    pack["cfg"] = ARM_CONFIGS[args.v7r_arm]
    return pack


def _apply_edit(pack: dict, cfg: dict, start: torch.Tensor,
                z_in: torch.Tensor, tid: torch.Tensor,
                task_batch=None) -> torch.Tensor:
    """Energy/R1 edit loop from an arbitrary start latent; records audit."""
    canon = pack["canon"]
    if cfg["kind"] == "r1_edit":
        from tools.v6_next.w_gating import topk_spatial_mask
        z_cur = start.clone()
        for step in range(cfg["steps"]):
            with torch.enable_grad():
                zr = z_cur.detach().clone().requires_grad_(True)
                q = pack["critic"](zr.float()).float()
                oh = F.one_hot(tid, 5).bool()
                m = (q.gather(1, tid[:, None]).squeeze(1)
                     - torch.logsumexp(q.masked_fill(oh, -1e9), dim=1))
                g = torch.autograd.grad(m.sum(), zr)[0]
            pack["editor_audit"].append({
                "step": step,
                "raw_norms": {"r1": float(g.flatten(1).norm(dim=1).mean())},
                "post_scale_norms": {"r1": 1.0},
                "alpha_used": 1.0,
                "alpha_trials": [{"alpha": 1.0, "accepted": True}],
                "n_rejected": 0,
                "cum_displacement_rms": float(
                    (z_cur - start).flatten(1).pow(2).mean(dim=1).sqrt().mean()),
            })
            if cfg["mode"] == "topk":
                g = topk_spatial_mask(g, cfg.get("topk_frac", 0.25))
            g = rms_normalize(g)
            z_cur = (z_cur + cfg["rms"] * g).detach()
        _record_editor(pack, task_batch, start, z_cur, editor_audit_only=True)
        return z_cur

    if cfg["kind"] == "energy_edit":
        obj_name = cfg["objective"]
        if obj_name == "BEST_E":
            obj_name = resolve_best_e(pack)
        objectives = {"style": make_style_objective(obj_name, pack, tid)}
        weights = {"style": 1.0, "content": cfg["w_content"],
                   "anchor": cfg["w_anchor"]}
        if cfg.get("r1_as_component"):
            objectives["r1_margin"] = make_style_objective("R1", pack, tid)
            weights["r1_margin"] = cfg.get("w_r1", 0.0)
        editor = BudgetedEditor(
            start, z_in, tid, canon=canon, canon_mode=CANON_MODE,
            objectives=objectives, weights=weights,
            steps=cfg["steps"], per_step_rms=cfg["rms"])
        z_cur = start
        for _ in range(cfg["steps"]):
            z_cur = editor.step(z_cur)
        pack["editor_audit"] = editor.audit
        _record_editor(pack, task_batch, start, z_cur, editor)
        return z_cur

    raise ValueError(f"unknown edit kind: {cfg['kind']}")


def run_v7r_edit(pack: dict, z_batch: torch.Tensor, sid_batch: torch.Tensor,
                 tid_batch: torch.Tensor, task_batch=None) -> torch.Tensor:
    cfg = pack["cfg"]
    kind = cfg["kind"]
    b4, canon = pack["b4"], pack["canon"]
    with torch.inference_mode(False), torch.enable_grad():
        z_in = z_batch.clone()
        sid = sid_batch.clone()
        tid = tid_batch.clone()
        c = canon(z_in, mode=CANON_MODE, augment=False).detach()
        z0 = model_direct(b4, z_in, canon, CANON_MODE, tid).detach()

        if kind == "predictor":
            if pack["corrector"] is None:
                return z0
            cross = (sid != tid)
            return corrector_output(pack["corrector"], z0, c, sid, tid,
                                    cross).detach()

        if kind == "predictor_cond":
            """E1 intervention: swap source/target Style-IDs fed to the frozen
            pipeline. z0 is rebuilt from the EFFECTIVE target (the intervention
            covers the whole predictor), H is applied on the original cross-style
            task set; per-trace-task source/target sensitivity is recorded."""
            if pack["corrector"] is None:
                return z0
            sid_eff, tid_eff = sid.clone(), tid.clone()
            if cfg["src_mode"] == "wrong":
                sid_eff = (sid + 2) % len(STYLES)
            elif cfg["src_mode"] == "fixed":
                sid_eff = torch.full_like(sid, int(cfg.get("fixed_src", 0)))
            elif cfg["src_mode"] == "r1_pred":
                sid_eff = pack["critic"](c).argmax(dim=1)
            if cfg["tgt_mode"] == "wrong":
                tid_eff = (tid + 2) % len(STYLES)
            z0_eff = model_direct(b4, z_in, canon, CANON_MODE, tid_eff).detach()
            cross = (sid != tid)  # H applies exactly where the original task does
            out = corrector_output(pack["corrector"], z0_eff, c, sid_eff,
                                   tid_eff, cross)
            _record_cond_sensitivity(pack, task_batch, c, z0_eff, sid, tid)
            _record_displacement(pack, task_batch, out, z0_eff)
            return out.detach()

        if kind == "probe_step":
            obj = make_style_objective(cfg["objective"], pack, tid,
                                       wrong_target=cfg["wrong"])
            zr = z0.clone().requires_grad_(True)
            if cfg["random"]:
                g = torch.randn(z0.shape, device=z0.device,
                                dtype=torch.float32)
                g = rms_normalize(g)
            else:
                loss = obj(zr)
                g = torch.autograd.grad(loss, zr)[0]
                g = rms_normalize(g) * cfg["sign"]
            z_cur = (z0 + cfg["rms"] * g).detach()
            _record_probe(pack, task_batch, z_cur, z0, obj)
            return z_cur

        if kind == "r1_edit":
            return _apply_edit(pack, cfg, z0, z_in, tid, task_batch)

        if kind == "energy_edit":
            return _apply_edit(pack, cfg, z0, z_in, tid, task_batch)

        if kind == "combo":
            rcfg = resolve_best_r(pack)
            if pack["corrector"] is None:
                start = z0
            else:
                cross = (sid != tid)
                start = corrector_output(pack["corrector"], z0, c, sid, tid,
                                         cross).detach()
            if rcfg["kind"] == "r1_edit":
                return _apply_edit(pack, rcfg, start, z_in, tid, task_batch)
            obj_name = rcfg.get("objective", "BEST_E")
            if obj_name == "BEST_E":
                obj_name = resolve_best_e(pack)
            ecfg = dict(rcfg)
            ecfg["objective"] = obj_name
            return _apply_edit(pack, ecfg, start, z_in, tid, task_batch)

    raise ValueError(f"unknown v7r arm kind: {kind}")


def _record_probe(pack, task_batch, z_cur, z0, obj):
    with torch.no_grad():
        try:
            e = float(obj(z_cur))
        except Exception:  # noqa: BLE001
            e = None
    for i, t in enumerate(task_batch or []):
        k = t["task_id"]
        if k not in pack["trace_task_ids"]:
            continue
        pack["trace_state"]["steps"].setdefault(k, []).append({
            "energy_after": e,
            "cum_displacement_rms": float(
                (z_cur[i] - z0[i]).flatten().pow(2).mean().sqrt()),
        })
        pack["trace_state"]["displacement"][k] = float(
            (z_cur[i] - z0[i]).flatten().pow(2).mean().sqrt())


def _record_displacement(pack, task_batch, z_cur, start):
    for i, t in enumerate(task_batch or []):
        k = t["task_id"]
        if k not in pack["trace_task_ids"]:
            continue
        pack["trace_state"]["displacement"][k] = float(
            (z_cur[i] - start[i]).flatten().pow(2).mean().sqrt())
        pack["trace_state"]["final_z"][k] = z_cur[i].detach().cpu()


def _record_cond_sensitivity(pack, task_batch, c, z0_eff, sid, tid):
    """E1 mechanism metric on trace tasks: how much H's residual moves when the
    TARGET id varies vs when the SOURCE id varies (same content, same z0)."""
    H = pack["corrector"]
    if H is None:
        return
    sens = pack["trace_state"].setdefault("cond_sensitivity", {})
    with torch.inference_mode(False), torch.enable_grad():
        for i, t in enumerate(task_batch or []):
            k = t["task_id"]
            if k not in pack["trace_task_ids"] or k in sens:
                continue
            c1 = c[i:i + 1]
            z01 = z0_eff[i:i + 1]
            s0 = int(sid[i])
            t0 = int(tid[i])
            d_tgt, d_src = [], []
            for j in range(5):
                if j != t0:
                    d_tgt.append(float((H(c1, z01, torch.tensor([s0], device=c.device),
                                             torch.tensor([t0], device=c.device))
                                        - H(c1, z01, torch.tensor([s0], device=c.device),
                                            torch.tensor([j], device=c.device)))
                                       .flatten().pow(2).mean().sqrt()))
                if j != s0:
                    d_src.append(float((H(c1, z01, torch.tensor([s0], device=c.device),
                                             torch.tensor([t0], device=c.device))
                                        - H(c1, z01, torch.tensor([j], device=c.device),
                                            torch.tensor([t0], device=c.device)))
                                       .flatten().pow(2).mean().sqrt()))
            sens[k] = {
                "target_sensitivity": float(np.mean(d_tgt)) if d_tgt else 0.0,
                "source_sensitivity": float(np.mean(d_src)) if d_src else 0.0,
            }


def _record_editor(pack, task_batch, start, z_cur, editor=None,
                   editor_audit_only: bool = False):
    """Record per-trace-task displacement / final latents after an edit."""
    last = (editor.audit[-1] if editor is not None and editor.audit else {})
    n_steps = len(editor.audit) if editor is not None else len(
        [a for a in pack["editor_audit"]])
    for i, t in enumerate(task_batch or []):
        k = t["task_id"]
        if k not in pack["trace_task_ids"]:
            continue
        rec = pack["trace_state"]["steps"].setdefault(k, [])
        while len(rec) < max(n_steps, 1):
            rec.append({})
        if last:
            rec[-1].update({kk: last.get(kk) for kk in
                            ("alpha_used", "cum_displacement_rms", "losses")})
        pack["trace_state"]["displacement"][k] = float(
            (z_cur[i] - start[i]).flatten().pow(2).mean().sqrt())
        pack["trace_state"]["final_z"][k] = z_cur[i].detach().cpu()


def finish_v7r_trace(pack: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    tr = pack["trace_state"]
    final_cols = {}
    models, src_cache, ref_dino = (pack.get("models"), pack.get("src_cache"),
                                   pack.get("ref_dino"))
    if models is not None and tr["final_z"]:
        from tools.v5.route_c import decode_trt
        from tools.v6.train_eval_v6 import encode_dino
        from utils.run_evaluation import to_lpips_input
        import numpy as np
        from PIL import Image
        task_ids = sorted(tr["final_z"].keys())
        dev = next(models["vae"].parameters()).device
        ref_np = {s: r.detach().cpu().numpy() for s, r in ref_dino.items()}
        z = torch.stack([tr["final_z"][k] for k in task_ids], 0).to(dev)
        with torch.inference_mode():
            img = decode_trt(models["trt"], models["vae"], z, batch=8)
            u8 = img.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[j]) for j in range(arr.shape[0])]
            dg, _ = encode_dino(models["dino"], pils, batch=8, device=str(dev))
            dg = F.normalize(dg.float(), dim=-1)
            gen01 = u8.float().div(255.0).to(dev)
        for j, k in enumerate(task_ids):
            t = _task_by_id[k]
            raw = src_cache["raw_rgb"][t["source_stem"]].unsqueeze(0).to(dev)
            with torch.inference_mode():
                lp = float(models["lpips"](to_lpips_input(gen01[j:j + 1]),
                                           to_lpips_input(raw)).flatten()[0])
            final_cols[k] = {
                "dino_s": float(np.max(dg[j].cpu().numpy() @ ref_np[t["target_style"]].T)),
                "dino_c": float(dg[j].cpu().numpy()
                                @ src_cache["raw_dino"][t["source_stem"]]),
                "lpips_vs_source": lp,
            }
    for k, cols in final_cols.items():
        if tr["steps"].get(k):
            tr["steps"][k][-1].update(cols)

    summary = audit_summary(pack["editor_audit"]) if pack["editor_audit"] else {}
    write_json({"steps": tr["steps"], "displacement": tr["displacement"],
                "final": final_cols, "audit_summary": summary,
                "cond_sensitivity": tr.get("cond_sensitivity", {})},
               out_dir / "optim_trace.json")
    write_json({
        "audit": summary,
        "policy": "fixed step-0 component scales + backtracking line search; "
                  "imbalance judged on POST-SCALE norms (predefined)",
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_dir / "grad_norm.json")
    write_json({
        "arm": pack["arm"],
        "ckpt": pack["ckpt_path"],
        "corrector_meta": pack["corrector_meta"],
        "config": dict(pack["cfg"]),
        "tasks_sha256": tasks_sha256(),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_dir / "run_meta.json")


def set_eval_context(tasks) -> set:
    """Called by r2_eval once: trace task ids + task lookup for final metrics."""
    global _task_by_id
    from tools.v7.v7_common import pick_trace_tasks
    _task_by_id = {t["task_id"]: t for t in tasks}
    return set(pick_trace_tasks(tasks, 4))
