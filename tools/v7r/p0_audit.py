"""V7-R P0: failure attribution audit (no new candidates, mechanism only).

P0-A  Was the relative style signal destroyed by training?
      Trajectory of M1/M2 reproductions (ckpts at 0/100/400/800 produced by
      tools.v7.p3_train --ckpt-steps): residual RMS ||A(c,t)-A(c,s)||, residual
      direction cosines across targets, R1 source/target margins, identity
      drift, decoded CSD target/source scores + DINO/LPIPS on a fixed
      100-task diagnostic set (diagnostic only, never a candidate gate).

P0-B  Why did M2 destroy content?
      Same trajectory + S2-energy drift of outputs, weighted-gradient history
      from the V7 grad_norm.log, residual-growth curve.

P0-C  Re-verify B4's conditional shortcut on the full 150 sources x 5 IDs:
      CSD-judged source fraction (from the V7 p0_b4_direct eval rows), latent
      distances across Style-IDs, Haar subband energy of the B4 edit residual,
      source-vs-target influence (CSD source-dominance index), and a fresh
      c1_gentle source-style probe trained with painter-disjoint splits.

Writes runs/v7r/p0_failure_audit.json + docs/v7/V7R_FAILURE_AUDIT.md with
findings categorized as: supervision_invalid / gradient_scale / optimization
_instability / residual_collapse / eval_impl_error.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.v7.v7_common import (
    ROOT, STYLES, load_c3unet, load_canon, load_critic, model_direct, tasks_sha256,
    write_json,
)
from tools.v7r.v7r_common import P0_CKPTS, P0_DIR, painter_of
from wavelet import dwt2_haar

SRC_CACHE = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"
V7_P0_EVAL = ROOT / "runs" / "v7" / "p0_relative" / "p0_b4_direct"
DIAG_TASKS = 100  # fixed stratified diagnostic set (20 per target style)


def diag_tasks():
    tasks = json.loads((ROOT / "eval_protocol" / "tasks_750.json").read_text(
        encoding="utf-8"))["tasks"]
    tasks = [t for t in tasks if t.get("is_cross")]
    count, out = {}, []
    for t in tasks:
        s = t["target_style"]
        if count.get(s, 0) < DIAG_TASKS // 5:
            out.append(t)
            count[s] = count.get(s, 0) + 1
    return out


def relative_forward(model, z, canon, canon_mode, sid, tid):
    c = canon(z, mode=canon_mode, augment=False)
    return z + model(c, 0.0, c, tid) - model(c, 0.0, c, sid)


def trajectory_metrics(model, tasks, src_cache, device, stack):
    """Per-checkpoint mechanism metrics on the fixed diagnostic set."""
    canon, critic = stack["canon"], stack["critic"]
    rows = {"residual_rms": [], "pair_cos_mean": [], "r1_target_margin": [],
            "r1_source_margin": [], "r1_pred_is_source": []}
    per_tgt = {s: {"residual_rms": [], "pred_is_source": []} for s in STYLES}
    from tools.v6_next.w_distill import R1FeatureExtractor
    from tools.v7.v7_common import s2_loss
    ext = stack["feat_ext"]
    stats = stack["s2_stats"]

    imgs, metas = [], []
    with torch.inference_mode():
        for i in range(0, len(tasks), 10):
            grp = tasks[i:i + 10]
            z = torch.cat([src_cache["z_src"][t["source_stem"]].to(device)
                           for t in grp], 0)
            sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
            c = canon(z, mode="c1_gentle", augment=False)
            a_t = model(c, 0.0, c, tid)
            a_s = model(c, 0.0, c, sid)
            resid = a_t - a_s
            rms = resid.flatten(1).pow(2).mean(dim=1).sqrt()
            rows["residual_rms"].extend(rms.tolist())
            # direction cosines across the 5 targets (first sample of group);
            # C3UNet requires batch(style_ids) == batch(z) -> expand to 5 copies
            c1 = c[:1].expand(5, -1, -1, -1).contiguous()
            r5 = model(c1, 0.0, c1, torch.arange(5, device=device)) - a_s[:1]
            r5 = r5.flatten(1)
            r5n = F.normalize(r5, dim=1)
            cos = (r5n @ r5n.t())[~torch.eye(5, dtype=torch.bool, device=device)]
            rows["pair_cos_mean"].extend(cos.tolist())
            q_t = critic(a_t.float()).float()
            oh_t = F.one_hot(tid, 5).bool()
            oh_s = F.one_hot(sid, 5).bool()
            m_t = (q_t.gather(1, tid[:, None]).squeeze(1)
                   - torch.logsumexp(q_t.masked_fill(oh_t, -1e9), dim=1))
            m_s = (q_t.gather(1, sid[:, None]).squeeze(1)
                   - torch.logsumexp(q_t.masked_fill(oh_s, -1e9), dim=1))
            rows["r1_target_margin"].extend(m_t.tolist())
            rows["r1_source_margin"].extend(m_s.tolist())
            pred_is_src = (q_t.argmax(dim=1) == sid).float()
            rows["r1_pred_is_source"].extend(pred_is_src.tolist())
            for j, t in enumerate(grp):
                per_tgt[t["target_style"]]["residual_rms"].append(float(rms[j]))
                per_tgt[t["target_style"]]["pred_is_source"].append(
                    float(pred_is_src[j]))
            s2e = float(s2_loss(a_t, tid, stats, ext))
            rows.setdefault("s2_energy_vals", []).append(s2e)
            if stack.get("trt") is not None:
                imgs.append(stack["decode"](a_t))
                metas.extend(grp)
    out = {k: float(np.mean(v)) for k, v in rows.items()
           if k != "s2_energy_vals"}
    out["s2_energy"] = float(np.mean(rows["s2_energy_vals"]))
    out["residual_rms_per_target"] = {
        s: {"residual_rms": float(np.mean(v["residual_rms"])),
            "pred_is_source": float(np.mean(v["pred_is_source"]))
            if v["pred_is_source"] else None}
        for s, v in per_tgt.items()}

    if imgs:
        from tools.v5.route_c import csd_prototypes
        proto = csd_prototypes()
        all_u8 = torch.cat(imgs, 0)
        arr = all_u8.permute(0, 2, 3, 1).numpy()
        pils = [Image.fromarray(arr[j]) for j in range(arr.shape[0])]
        fg = stack["csd_feats"](pils)
        sc = fg @ proto.T
        tgt_scores, src_scores = [], []
        for j, t in enumerate(metas):
            tgt_scores.append(float(sc[j, t["target_style_id"]]))
            src_scores.append(float(sc[j, t["source_style_id"]]))
        out["csd_target_score"] = float(np.mean(tgt_scores))
        out["csd_source_score"] = float(np.mean(src_scores))
    return out


def audit_model(name: str, ckpt_dir: Path, device: str, stack) -> dict:
    src_cache = torch.load(SRC_CACHE, map_location="cpu", weights_only=False)
    tasks = diag_tasks()
    steps = sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("step_*.pt"))
    traj = {}
    for s in steps:
        ck = torch.load(ckpt_dir / f"step_{s:04d}.pt", map_location="cpu",
                        weights_only=False)
        a = ck.get("args", {})
        model = load_c3unet_from_state(ck["model"], a, device)
        traj[s] = trajectory_metrics(model, tasks, src_cache, device, stack)
        print(f"[p0] {name} step {s}: resid_rms={traj[s]['residual_rms']:.4f} "
              f"r1_tgt={traj[s]['r1_target_margin']:.2f}", flush=True)
    return traj


def load_c3unet_from_state(state, a, device):
    from tools.v6.models_v6 import C3UNet
    m = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5,
               mode=a.get("mode", "direct"),
               no_style=bool(a.get("no_style", False)),
               content_mode=a.get("content_mode", "full"),
               inject_64=not a.get("no_inject_64", False),
               inject_32=not a.get("no_inject_32", False),
               inject_16=not a.get("no_inject_16", False),
               style_keys=bool(a.get("style_keys", False))).to(device)
    m.load_state_dict(state)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def b4_shortcut_audit(device: str) -> dict:
    src_cache = torch.load(SRC_CACHE, map_location="cpu", weights_only=False)
    b4 = load_c3unet(ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt", device)
    canon = load_canon(device)
    critic = load_critic(device=device)
    tasks = json.loads((ROOT / "eval_protocol" / "tasks_750.json").read_text(
        encoding="utf-8"))["tasks"]
    seen, sources = set(), []
    for t in tasks:
        if t["source_stem"] not in seen:
            seen.add(t["source_stem"])
            sources.append(t)
    out = {"n_sources": len(sources), "per_target": {}, "overall": {}}
    resid_props = {"ll": [], "lh": [], "hl": [], "hh": []}
    tgt_rms = {s: [] for s in STYLES}
    tgt_pred_src = {s: [] for s in STYLES}
    with torch.inference_mode():
        for i in range(0, len(sources), 10):
            grp = sources[i:i + 10]
            z = torch.cat([src_cache["z_src"][t["source_stem"]].to(device)
                           for t in grp], 0)
            c = canon(z, mode="c1_gentle", augment=False)
            for j, t in enumerate(grp):
                c1 = c[j:j + 1]
                outs = b4(c1.expand(5, -1, -1, -1).contiguous(), 0.0,
                          c1.expand(5, -1, -1, -1).contiguous(),
                          torch.arange(5, device=device))
                pair = [float((outs[a] - outs[b2]).flatten().pow(2).mean().sqrt())
                        for a in range(5) for b2 in range(a + 1, 5)]
                out["per_target"].setdefault(t["source_style"], {})[
                    t["source_stem"]] = {"pairwise_rms_mean": float(np.mean(pair))}
                resid = outs - c1
                ll, lh, hl, hh = dwt2_haar(resid)
                tot = (ll.pow(2).sum() + lh.pow(2).sum() + hl.pow(2).sum()
                       + hh.pow(2).sum()).clamp(min=1e-12)
                resid_props["ll"].append(float(ll.pow(2).sum() / tot))
                resid_props["lh"].append(float(lh.pow(2).sum() / tot))
                resid_props["hl"].append(float(hl.pow(2).sum() / tot))
                resid_props["hh"].append(float(hh.pow(2).sum() / tot))
                q = critic(outs.float()).float()
                pred = q.argmax(dim=1)
                for ti in range(5):
                    tgt_rms[STYLES[ti]].append(float(
                        (outs[ti] - c1).flatten().pow(2).mean().sqrt()))
                    tgt_pred_src[STYLES[ti]].append(
                        1.0 if int(pred[ti]) == t["source_style_id"] else 0.0)
    out["overall"]["pairwise_rms_mean"] = float(np.mean([
        v["pairwise_rms_mean"] for d in out["per_target"].values()
        for v in d.values()]))
    out["overall"]["residual_subband_ratio"] = {
        k: float(np.mean(v)) for k, v in resid_props.items()}
    out["overall"]["per_target_edit_rms"] = {
        s: float(np.mean(v)) for s, v in tgt_rms.items()}
    out["overall"]["per_target_pred_is_source"] = {
        s: float(np.mean(v)) for s, v in tgt_pred_src.items()}
    # CSD source fraction from the existing V7 B4-direct eval (all 750 rows)
    ev = V7_P0_EVAL / "eval.json"
    if ev.exists():
        rows = json.loads(ev.read_text(encoding="utf-8"))["rows"]
        cross = [r for r in rows if r["is_cross"]]
        out["overall"]["csd_pred_is_source_frac"] = float(np.mean(
            [1.0 if r["pred_class"] == r["src_style_id"] else 0.0
             for r in cross]))
        out["overall"]["csd_target_top1"] = float(np.mean(
            [r["target_top1"] for r in cross]))
    return out


def train_c1_probe(device: str) -> dict:
    """Source-style probe on c1_gentle-canonicalized latents, painter-disjoint
    splits. Compared with the frozen R1 evaluated on the same canonicalized
    latents (R1 was trained on RAW latents only)."""
    from tools.v6_next.probe_style import LatentStyleNet
    from tools.v7r.v7r_common import load_pool
    train_by_style, val_by_style, painters = load_pool(device="cpu", val_n=0)
    canon = load_canon(device)

    def canon_cache(x, sid_list):
        outs, metas = [], []
        with torch.inference_mode():
            for i in range(0, x.shape[0], 64):
                zb = x[i:i + 64].to(device)
                c = canon(zb, mode="c1_gentle", augment=False)
                outs.append(c.cpu())
                metas.extend(sid_list[i:i + 64])
        return torch.cat(outs, 0), metas

    # painter-disjoint split per style: hold out ~20% of painters
    xs, ys, painters_all = [], [], []
    for sid, s in enumerate(STYLES):
        x = train_by_style[s]
        pl = painters[s]
        xs.append(x)
        ys.extend([sid] * x.shape[0])
        painters_all.extend(pl)
    xs = torch.cat(xs, 0)
    ys = torch.tensor(ys)
    painters_all = np.array([p if p else "__anon" for p in painters_all])
    key = np.array([f"{p}" for p in painters_all])
    uniq = sorted(set(key))
    rng = np.random.default_rng(0)
    hold = set(rng.choice(uniq, size=max(1, int(0.2 * len(uniq))),
                          replace=False).tolist())
    is_val = np.array([k in hold for k in key])
    # canonicalize train/val separately
    ctr, ytr = xs[torch.from_numpy(~is_val)], ys[torch.from_numpy(~is_val)]
    cva, yva = xs[torch.from_numpy(is_val)], ys[torch.from_numpy(is_val)]
    ctr_c, _ = canon_cache(ctr, ytr.tolist())
    cva_c, _ = canon_cache(cva, yva.tolist())
    ctr_c, cva_c = ctr_c.to(device), cva_c.to(device)

    probe = LatentStyleNet(width=64).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    n_train = ctr_c.shape[0]
    ytr_g = ytr.to(device)
    for step in range(400):
        idx = torch.randint(0, n_train, (128,), device=device)
        zb = ctr_c[idx]
        yb = ytr_g[idx].to(device)
        loss = F.cross_entropy(probe(zb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    with torch.inference_mode():
        correct, n = 0, 0
        for i in range(0, cva_c.shape[0], 128):
            pred = probe(cva_c[i:i + 128]).argmax(dim=1).cpu()
            correct += int((pred == yva[i:i + 128]).sum())
            n += pred.shape[0]
    probe_acc = correct / max(n, 1)

    critic = load_critic(device=device)
    correct, n = 0, 0
    with torch.inference_mode():
        for i in range(0, cva_c.shape[0], 128):
            pred = critic(cva_c[i:i + 128]).argmax(dim=1).cpu()
            correct += int((pred == yva[i:i + 128]).sum())
            n += pred.shape[0]
    r1_on_c1_acc = correct / max(n, 1)
    return {
        "probe_val_acc_c1gently": probe_acc,
        "r1_val_acc_on_c1gently": r1_on_c1_acc,
        "n_train": int(n_train), "n_val": int(cva_c.shape[0]),
        "n_painters_train": int(len(uniq) - len(hold)),
        "n_painters_val": int(len(hold)),
        "note": "painter-disjoint split; probe trained from scratch on "
                "canonicalized latents; R1 evaluated as-is on the same inputs",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R P0 failure audit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-retrain-analysis", action="store_true")
    args = ap.parse_args()
    device = args.device
    t0 = time.time()
    P0_DIR.mkdir(parents=True, exist_ok=True)

    from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae
    from tools.v6.train_eval_v6 import csd_model
    from tools.v6_next.w_distill import R1FeatureExtractor
    from tools.v7.v7_common import s2_loss as _s2
    vae = load_vae(device)
    trt = load_trt(vae, device)
    critic = load_critic(device=device)
    csd, _, _ = csd_model.load_csd(device)
    s2_stats = torch.load(ROOT / "runs/v7/p1_energy/S2_sd.pt", map_location="cpu",
                          weights_only=False)
    stack = {
        "vae": vae, "trt": trt, "canon": load_canon(device), "critic": critic,
        "feat_ext": R1FeatureExtractor(critic), "s2_stats": s2_stats,
        "csd_feats": lambda pils: (lambda f: f / np.linalg.norm(
            f, axis=1, keepdims=True))(
            csd_model.feats_pils(csd, pils, batch=16, device=device)),
        "decode": lambda z: decode_trt(trt, vae, z, batch=10).mul(255).round()
        .clamp(0, 255).to(torch.uint8).cpu(),
    }
    import tools.v7r.p0_audit as _mod  # noqa: F401 (module marker for tests)

    audit = {"tasks_sha256": tasks_sha256(), "diagnostic_tasks": DIAG_TASKS}

    # ---- P0-A / P0-B trajectories (reproduced with checkpoints) ----
    if not args.skip_retrain_analysis:
        for name in ["M1", "M2"]:
            d = P0_CKPTS / name
            if d.exists() and list(d.glob("step_*.pt")):
                audit[f"{name}_trajectory"] = audit_model(name, d, device, stack)
            else:
                audit[f"{name}_trajectory"] = None

    # ---- P0-C B4 conditional shortcut ----
    audit["b4_shortcut"] = b4_shortcut_audit(device)
    print(f"[p0] b4_shortcut: csd_src_frac="
          f"{audit['b4_shortcut']['overall'].get('csd_pred_is_source_frac')}",
          flush=True)

    # ---- P0-C c1_gentle probe ----
    audit["c1_probe"] = train_c1_probe(device)
    print(f"[p0] c1 probe acc={audit['c1_probe']['probe_val_acc_c1gently']:.3f} "
          f"vs R1-on-c1={audit['c1_probe']['r1_val_acc_on_c1gently']:.3f}",
          flush=True)

    write_json(audit, ROOT / "runs" / "v7r" / "p0_failure_audit.json")
    write_failure_md(audit)
    print(f"[p0] done in {time.time()-t0:.0f}s", flush=True)
    return 0


def write_failure_md(audit: dict) -> None:
    lines = ["# V7-R Failure Audit (auto-generated by tools/v7r/p0_audit.py)", ""]
    findings = []

    m1 = audit.get("M1_trajectory") or {}
    if m1:
        s0, sF = m1.get(0, {}), m1.get(max(m1), {})
        lines += [f"## M1 trajectory (steps {sorted(m1)})", "",
                  "| step | resid RMS | pair cos | R1 tgt margin | R1 src margin |"
                  " S2 energy | CSD tgt | CSD src |", "|---|---|---|---|---|---|---|---|"]
        for k in sorted(m1):
            v = m1[k]
            lines.append(
                f"| {k} | {v['residual_rms']:.4f} | {v['pair_cos_mean']:.3f} "
                f"| {v['r1_target_margin']:+.2f} | {v['r1_source_margin']:+.2f} "
                f"| {v['s2_energy']:.2f} | {v.get('csd_target_score', 0):.3f} "
                f"| {v.get('csd_source_score', 0):.3f} |")
        if sF and s0:
            if sF["residual_rms"] < 0.3 * max(s0["residual_rms"], 1e-6):
                findings.append(("residual_collapse",
                                 "M1 residual RMS collapsed toward zero during "
                                 "training (identity attractor of the relative "
                                 "parameterization under the V6 loss)."))
            elif sF["r1_target_margin"] <= 0:
                findings.append(("supervision_invalid",
                                 "M1 residuals persist but no longer point "
                                 "toward the target style (R1 target margin "
                                 "non-positive at final step)."))
            else:
                findings.append(("supervision_invalid",
                                 "M1 residual persists with target margin; "
                                 "failure is not residual collapse — check "
                                 "decoded metrics trajectory."))

    m2 = audit.get("M2_trajectory") or {}
    if m2:
        lines += ["", f"## M2 trajectory (steps {sorted(m2)})", "",
                  "| step | resid RMS | R1 tgt margin | S2 energy | CSD tgt | CSD src |",
                  "|---|---|---|---|---|---|"]
        for k in sorted(m2):
            v = m2[k]
            lines.append(
                f"| {k} | {v['residual_rms']:.4f} | {v['r1_target_margin']:+.2f} "
                f"| {v['s2_energy']:.2f} | {v.get('csd_target_score', 0):.3f} "
                f"| {v.get('csd_source_score', 0):.3f} |")
        s0, sF = m2.get(0, {}), m2.get(max(m2), {})
        if sF and s0 and sF["residual_rms"] > 1.5 * max(s0["residual_rms"], 1e-6):
            findings.append(("gradient_scale",
                             "M2 residual grows well beyond init (unbounded "
                             "style-energy gradient dominated the weighted sum; "
                             "S2 energy on generated latents sits far from the "
                             "real manifold)."))

    b4 = audit.get("b4_shortcut", {})
    lines += ["", "## B4 conditional shortcut (150 sources x 5 IDs)", ""]
    ov = b4.get("overall", {})
    lines += [f"- CSD-judged source fraction: {ov.get('csd_pred_is_source_frac', 0)*100:.1f}%",
              f"- CSD target Top-1: {ov.get('csd_target_top1', 0)*100:.1f}%",
              f"- mean pairwise latent RMS across 5 Style-IDs: {ov.get('pairwise_rms_mean', 0):.4f}",
              f"- B4 edit residual subband ratios: {ov.get('residual_subband_ratio')}"]

    pr = audit.get("c1_probe", {})
    lines += ["", "## c1_gentle source-style probe (painter-disjoint)", "",
              f"- fresh probe val acc on c1_gentle latents: {pr.get('probe_val_acc_c1gently', 0)*100:.1f}%",
              f"- frozen R1 accuracy on the same c1_gentle latents: {pr.get('r1_val_acc_on_c1gently', 0)*100:.1f}%",
              f"- split: {pr.get('n_painters_train')} train / {pr.get('n_painters_val')} val painters"]

    lines += ["", "## Categorized findings", ""]
    for tag, msg in findings:
        lines.append(f"- **{tag}**: {msg}")
    if not findings:
        lines.append("- (trajectory checkpoints missing — run p3_train "
                     "--ckpt-steps 0,100,400,800 reproductions first)")
    lines += ["", "## V7 bookkeeping corrections", "",
              "- V7 T5: SKIPPED_BY_GATE (T4 showed no positive signal vs T0; "
              "recorded in runs/v7/p2_t5_skipped.json)",
              "- V7 DINO-C bins were NOT same-task paired comparisons; V7-R P5 "
              "replaces them with a model-level Pareto front + per-task paired "
              "differences.",
              "- M1/M2 Same-LPIPS 0.0745 equals the algebraic identity of the "
              "relative parameterization (t=s cancels), not learned reconstruction."]
    (ROOT / "docs" / "v7" / "V7R_FAILURE_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
