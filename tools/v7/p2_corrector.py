"""V7 P2 corrector: real-distribution-guided test-time latent optimization.

Integrated into tools/v6_next/r2_eval.py as additive generator mode ``v7``
(existing modes untouched). Arms (V7_IMPLEMENTATION_PLAN.md P2.1):

  T0  R1 margin, full gradient          (v6_w "full" baseline, replicated exactly)
  T1  R1 margin, fixed TopK25           (v6_w fixed topk, per-sample mask)
  T2  S1 real-style energy              + predictor anchor + content protection
  T3  S2 R1-feature energy              + predictor anchor + content protection
  T4  best(S1,S2) energy + weak R1      + predictor anchor + content protection
  T5  T4 + source structure constraint  (RMS/step budget identical to T0-T4)

All arms: start from the predictor's direct output z0, 4 steps, per-step RMS
0.08 (same budget as v6_w T0/T1). Component gradients are per-sample
RMS-normalized before combining (the predefined scale policy of
V7_EVAL_PROTOCOL.md section 6); raw norms are audited into grad_norm.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.v6_next.r2_teacher import spatial_self_similarity_matrix
from tools.v6_next.w_gating import topk_spatial_mask
from tools.v6_next.w_guided import r1_gradient, r1_margin_per_sample
from tools.v7.v7_common import (
    ANCHOR_WEIGHT, CONTENT_WEIGHT, P1_DIR, ROOT, STRUCT_WEIGHT, STYLES,
    WEAK_R1_WEIGHT, audit_grad_balance, load_c3unet, load_canon, load_critic,
    model_output, rms_normalize, s1_loss, s1_loss_per_sample, s2_loss,
    s2_loss_per_sample, tasks_sha256, write_json,
)

ARM_STYLES = {"T2": "S1", "T3": "S2", "T4": "best", "T5": "best"}
ARM_R1_WEIGHT = {"T0": 1.0, "T1": 1.0, "T2": 0.0, "T3": 0.0,
                 "T4": WEAK_R1_WEIGHT, "T5": WEAK_R1_WEIGHT}


class _FeatWrap:
    """Single R1FeatureExtractor per pack (hooks registered once)."""

    def __init__(self, critic):
        from tools.v6_next.w_distill import R1FeatureExtractor
        self.SCALES = R1FeatureExtractor.SCALES
        self._ext = R1FeatureExtractor(critic)

    def __call__(self, z):
        return self._ext(z)


def load_energies(energies_dir: Path) -> dict:
    out = {}
    for name in ["S1_sd", "S2_sd"]:
        p = energies_dir / f"{name}.pt"
        if p.exists():
            out[name] = torch.load(p, map_location="cpu", weights_only=False)
    v = energies_dir / "energy_validity.json"
    if v.exists():
        out["validity"] = json.loads(v.read_text(encoding="utf-8"))
    return out


def resolve_style_energy(arm: str, energies: dict) -> tuple:
    if arm not in ARM_STYLES:
        return "", {}
    want = ARM_STYLES[arm]
    if want == "S1" and "S1_sd" in energies:
        return "S1_sd", energies["S1_sd"]
    if want == "S2" and "S2_sd" in energies:
        return "S2_sd", energies["S2_sd"]
    best = (energies.get("validity") or {}).get("best_energy")
    if best in energies:
        return best, energies[best]
    if "S1_sd" in energies:
        return "S1_sd", energies["S1_sd"]
    raise SystemExit(f"[v7] arm {arm} requires P1 energies; none found")


def build_v7_pack(args, device: str) -> dict:
    ckpt_path = ROOT / args.ckpt
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    energies = load_energies(Path(getattr(args, "v7_energies", "")
                                  or (ROOT / "runs" / "v7" / "p1_energy")))
    energy_name, energy_stats = resolve_style_energy(args.v7_arm, energies)
    critic = load_critic(ROOT / args.gated_critic, device)
    return {
        "kind": "v7",
        "arm": args.v7_arm,
        "ckpt_path": str(ckpt_path),
        "model": load_c3unet(ckpt_path, device),
        "canon": load_canon(device),
        "canon_mode": args.canon_mode or ck.get("args", {}).get("canon_mode", "c1_gentle"),
        "critic": critic,
        "feat_ext": _FeatWrap(critic),
        "start_mode": getattr(args, "v7_start_mode", "") or "direct",
        "steps": args.gating_steps,
        "per_step_rms": args.gating_rms,
        "r1_weight": ARM_R1_WEIGHT.get(args.v7_arm, WEAK_R1_WEIGHT),
        "content_weight": CONTENT_WEIGHT,
        "anchor_weight": ANCHOR_WEIGHT,
        "struct_weight": STRUCT_WEIGHT if args.v7_arm == "T5" else 0.0,
        "energy_name": energy_name,
        "energy_stats": energy_stats,
        "trace_task_ids": set(),
        "trace_state": {"steps": {}, "displacement": {}, "final_z": {}},
        "grad_norms": [],
        "models": None, "src_cache": None, "ref_dino": None,
    }


def _style_loss(z, tid, energy_name, energy_stats, feat_ext):
    if energy_name.startswith("S1"):
        return s1_loss(z, tid, energy_stats)
    return s2_loss(z, tid, energy_stats, feat_ext)


def _energy_distance(z, tid, energy_name, energy_stats, feat_ext):
    with torch.no_grad():
        if energy_name.startswith("S1"):
            return s1_loss_per_sample(z, tid, energy_stats)
        return s2_loss_per_sample(z, tid, energy_stats, feat_ext)


def _record_step(gen_pack, task_batch, step, z_cur, z0, vals_fn):
    tr = gen_pack["trace_state"]
    for i, t in enumerate(task_batch or []):
        k = t["task_id"]
        if k not in gen_pack["trace_task_ids"]:
            continue
        rec = tr["steps"].setdefault(k, [])
        if len(rec) <= step:
            rec.append({})
        disp = float((z_cur[i].detach() - z0[i]).flatten().pow(2).mean().sqrt())
        rec[step].update(vals_fn(i))
        rec[step]["cum_displacement_rms"] = disp
        tr["displacement"][k] = disp
        tr["final_z"][k] = z_cur[i].detach().cpu()


def run_v7_edit(gen_pack: dict, z_batch: torch.Tensor, sid_batch: torch.Tensor,
                tid_batch: torch.Tensor, task_batch=None) -> torch.Tensor:
    arm = gen_pack["arm"]
    steps = max(1, gen_pack["steps"])
    per_rms = gen_pack["per_step_rms"]
    model, canon = gen_pack["model"], gen_pack["canon"]
    canon_mode = gen_pack["canon_mode"]

    with torch.inference_mode(False), torch.enable_grad():
        # clone: inputs were created under the outer inference_mode; autograd
        # cannot track inference tensors (same workaround as the oracle mode).
        z_in = z_batch.clone()
        sid = sid_batch.clone()
        tid = tid_batch.clone()
        z0 = model_output(model, z_in, canon, canon_mode, sid, tid,
                          gen_pack["start_mode"]).detach()
        if arm in ("T0", "T1"):
            z_cur = _r1_gated_loop(z0, tid, gen_pack, arm, steps, per_rms,
                                   task_batch)
        else:
            z_cur = _energy_loop(z0, z_in, tid, gen_pack, steps,
                                 per_rms, task_batch)
    return z_cur.detach()


def _r1_gated_loop(z0, tid, gen_pack, arm, steps, per_rms, task_batch):
    """Exact replication of apply_gated_edit modes full/topk (fixed mask) + trace."""
    critic = gen_pack["critic"]
    z_cur = z0.detach().clone()
    for step in range(steps):
        with torch.inference_mode(False), torch.enable_grad():
            zr = z_cur.detach().clone().requires_grad_(True)
            q = critic(zr.float()).float()
            oh = F.one_hot(tid, 5).bool()
            m = q.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(
                q.masked_fill(oh, -1e9), dim=1)
            g = torch.autograd.grad(m.sum(), zr)[0]
        gen_pack["grad_norms"].append({
            "step": step,
            "grad_style_norm": float(g.flatten(1).norm(dim=1).mean()),
            "grad_content_norm": 0.0, "grad_anchor_norm": 0.0,
            "grad_r1_norm": float(g.flatten(1).norm(dim=1).mean()),
        })
        if arm == "T1":
            g = topk_spatial_mask(g, 0.25)
        g = rms_normalize(g)
        with torch.no_grad():
            m_now = r1_margin_per_sample(z_cur, tid, critic)
        z_cur = (z_cur + per_rms * g).detach()
        _record_step(gen_pack, task_batch, step, z_cur, z0,
                     lambda i, mm=m_now: {"r1_margin": float(mm[i]),
                                          "style_loss": None, "content_loss": None,
                                          "energy_distance": None})
    return z_cur


def _energy_loop(z0, z_src, tid, gen_pack, steps, per_rms, task_batch):
    """T2-T5: multi-component descent with the predefined scale policy.

    Scale policy (V7_EVAL_PROTOCOL.md section 6, "predefined scale, applied once"):
    component gradients are divided by fixed scales measured ONCE at step 0 of
    the first batch (grad RMS per component), then combined with the arm weights
    and RMS-normalized per sample. Equalized per-step normalization was measured
    (smoke, 16 tasks) to dilute the style term 4:1 and INCREASE the style energy
    (E 7.2 -> 40) — the fixed-scale policy keeps the style term dominant while
    the anchor/content terms act as their raw norms grow with displacement.
    A backtracking line search (alpha grid 1 .. 1/32) on the combined objective
    prevents overshoot; the per-step displacement cap stays per_rms=0.08 and the
    4-step budget matches T0/T1.
    """
    canon, critic, feat_ext = (gen_pack["canon"], gen_pack["critic"],
                               gen_pack["feat_ext"])
    canon_mode = gen_pack["canon_mode"]
    energy_name, energy_stats = gen_pack["energy_name"], gen_pack["energy_stats"]
    w_c, w_a, w_r1, w_s = (gen_pack["content_weight"], gen_pack["anchor_weight"],
                           gen_pack["r1_weight"], gen_pack["struct_weight"])
    z_cur = z0.detach().clone()
    c_src = canon(z_src, mode=canon_mode, augment=False).detach()
    ssm_src = (spatial_self_similarity_matrix(z_src).detach()
               if w_s > 0 else None)
    scales = gen_pack.setdefault("scales", None)
    alpha_grid = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]

    def total_obj(zz, g_r1_needed=False):
        l_style = _style_loss(zz, tid, energy_name, energy_stats, feat_ext)
        c_z = canon(zz.float(), mode=canon_mode, augment=False)
        l_content = (c_z - c_src).square().mean(dim=(1, 2, 3)).mean()
        l_struct = ((spatial_self_similarity_matrix(zz) - ssm_src).square().mean()
                    if w_s > 0 else zz.new_tensor(0.0))
        l_anchor = (zz.float() - z0.float()).square().flatten(1).mean(dim=1).mean()
        m = r1_margin_per_sample(zz, tid, critic)
        l_r1 = F.softplus(-m).mean()
        return (l_style + w_c * l_content + w_s * l_struct + w_a * l_anchor
                + w_r1 * l_r1), l_style, l_content, l_struct, l_anchor, l_r1

    for step in range(steps):
        zr = z_cur.detach().clone().requires_grad_(True)
        l_style, l_content, l_struct, l_anchor, l_r1 = total_obj(zr)[1:]

        g_style = torch.autograd.grad(l_style, zr, retain_graph=True)[0]
        g_content = torch.autograd.grad(l_content, zr, retain_graph=True)[0]
        g_anchor = torch.autograd.grad(l_anchor, zr, retain_graph=True)[0]
        g_struct = (torch.autograd.grad(l_struct, zr, retain_graph=True)[0]
                    if w_s > 0 else torch.zeros_like(g_style))
        g_r1 = (torch.autograd.grad(l_r1, zr)[0] if w_r1 > 0
                else torch.zeros_like(g_style))

        step_norms = {
            "grad_style_norm": float(g_style.flatten(1).norm(dim=1).mean()),
            "grad_content_norm": float(g_content.flatten(1).norm(dim=1).mean()),
            "grad_anchor_norm": float(g_anchor.flatten(1).norm(dim=1).mean()),
            "grad_r1_norm": float(g_r1.flatten(1).norm(dim=1).mean()),
        }
        if w_s > 0:
            step_norms["grad_struct_norm"] = float(
                g_struct.flatten(1).norm(dim=1).mean())
        gen_pack["grad_norms"].append({"step": step, **step_norms})

        if scales is None:
            def grms(g):
                return float(g.flatten(1).pow(2).mean().sqrt()) + 1e-12
            scales = {"style": grms(g_style), "content": grms(g_content),
                      "anchor": grms(g_anchor), "r1": grms(g_r1),
                      "struct": grms(g_struct) if w_s > 0 else 1.0}
            gen_pack["scales"] = scales
        d = (-g_style / scales["style"]
             + w_c * (-g_content / scales["content"])
             + w_a * (-g_anchor / scales["anchor"])
             + w_s * (-g_struct / scales["struct"])
             + w_r1 * (-g_r1 / scales["r1"]))
        d = rms_normalize(d)

        # backtracking line search on the combined objective, every step
        f0 = float(total_obj(z_cur)[0])
        alpha_used = alpha_grid[-1]
        for a in alpha_grid:
            zt = z_cur + per_rms * a * d
            if float(total_obj(zt)[0]) < f0:
                alpha_used = a
                break
        z_cur = (zr + per_rms * alpha_used * d).detach()

        with torch.no_grad():
            e_dist = _energy_distance(z_cur, tid, energy_name, energy_stats, feat_ext)
            l_inv_now = (canon(z_cur.float(), mode=canon_mode, augment=False)
                         - c_src).square().mean(dim=(1, 2, 3))
            m_now = r1_margin_per_sample(z_cur, tid, critic)
            ls_val = float(l_style.detach())
            lss_val = float(l_struct.detach())

        def vals_fn(i, e=e_dist, li=l_inv_now, mm=m_now):
            return {"style_loss": ls_val, "struct_loss": lss_val,
                    "content_loss": float(li[i]), "r1_margin": float(mm[i]),
                    "energy_distance": float(e[i]),
                    "step_alpha": alpha_used}

        _record_step(gen_pack, task_batch, step, z_cur, z0, vals_fn)
    return z_cur


def finish_v7_trace(gen_pack: dict, out_dir: Path) -> None:
    """Write optim_trace.json (+ final DINO-S/DINO-C/LPIPS per trace task),
    grad_norm.json, run_meta.json."""
    out_dir = Path(out_dir)
    tr = gen_pack["trace_state"]
    final_cols = {}
    models, src_cache, ref_dino = (gen_pack.get("models"), gen_pack.get("src_cache"),
                                   gen_pack.get("ref_dino"))
    if models is not None and tr["final_z"]:
        from tools.v5.route_c import decode_trt
        from tools.v6.train_eval_v6 import encode_dino
        from utils.run_evaluation import to_lpips_input
        task_ids = sorted(tr["final_z"].keys())
        dev = next(models["vae"].parameters()).device
        ref_np = {s: r.detach().cpu().numpy() for s, r in ref_dino.items()}
        z = torch.stack([tr["final_z"][k] for k in task_ids], 0).to(dev)
        vae, trt, dino, lp = (models["vae"], models["trt"], models["dino"],
                              models["lpips"])
        dev = z.device
        with torch.inference_mode():
            img = decode_trt(trt, vae, z, batch=8)
            u8 = img.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[j]) for j in range(arr.shape[0])]
            dg, _ = encode_dino(dino, pils, batch=8, device=str(dev))
            dg = F.normalize(dg.float(), dim=-1)
            gen01 = u8.float().div(255.0).to(dev)
            lp_vals = []
            for j, k in enumerate(task_ids):
                t = _task_by_id.get(k) if _task_by_id else None
                raw = src_cache["raw_rgb"][t["source_stem"]].unsqueeze(0).to(dev)
                lp_vals.append(float(lp(to_lpips_input(gen01[j:j+1]),
                                        to_lpips_input(raw)).flatten()[0]))
            for j, k in enumerate(task_ids):
                t = _task_by_id[k]
                tgt = t["target_style"]
                final_cols[k] = {
                    "dino_s": float(np.max(dg[j].cpu().numpy() @ ref_np[tgt].T)),
                    "dino_c": float(dg[j].cpu().numpy()
                                    @ src_cache["raw_dino"][t["source_stem"]]),
                    "lpips_vs_source": lp_vals[j],
                }
    for k, cols in final_cols.items():
        if tr["steps"].get(k):
            tr["steps"][k][-1].update(cols)

    gn = gen_pack["grad_norms"]
    agg = {}
    if gn:
        for k in gn[0]:
            if k == "step":
                continue
            vals = [g[k] for g in gn]
            agg[k] = {"mean": float(sum(vals) / len(vals)), "min": float(min(vals)),
                      "max": float(max(vals)), "n": len(vals)}
    audit = audit_grad_balance({k: v["mean"] for k, v in agg.items()})
    if gen_pack.get("scales"):
        audit["scales"] = dict(gen_pack["scales"])
        audit["scale_policy"] = ("fixed step-0 component scales + line search "
                                 "(predefined, applied uniformly to T2-T5)")
    write_json({"steps": tr["steps"], "displacement": tr["displacement"],
                "final": final_cols}, out_dir / "optim_trace.json")
    write_json({
        "per_step": gn,
        "aggregate": agg,
        "audit": audit,
        "scales": gen_pack.get("scales"),
        "policy": "fixed component scales measured once at step 0 (style/content/"
                  "anchor/r1 raw grad RMS), then weighted sum + per-sample RMS "
                  "normalize; backtracking line search (alpha in "
                  "[1,.5,.25,.125,.0625,.03125]) on the combined objective every "
                  "step; per-step displacement cap per_rms",
    }, out_dir / "grad_norm.json")
    write_json({
        "arm": gen_pack["arm"],
        "start_mode": gen_pack["start_mode"],
        "ckpt": gen_pack.get("ckpt_path", ""),
        "steps": gen_pack["steps"],
        "per_step_rms": gen_pack["per_step_rms"],
        "weights": {
            "content": gen_pack["content_weight"],
            "anchor": gen_pack["anchor_weight"],
            "r1": gen_pack["r1_weight"],
            "struct": gen_pack["struct_weight"],
        },
        "energy": gen_pack["energy_name"],
        "canon_mode": gen_pack["canon_mode"],
        "tasks_sha256": tasks_sha256(),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_dir / "run_meta.json")


_task_by_id = None


def set_eval_context(tasks) -> set:
    """Called by r2_eval once: trace task ids + task lookup for final metrics."""
    global _task_by_id
    from tools.v7.v7_common import pick_trace_tasks
    _task_by_id = {t["task_id"]: t for t in tasks}
    return set(pick_trace_tasks(tasks, 4))
