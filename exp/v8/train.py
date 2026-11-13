"""Unified trainer for the V8 training arms.

Arms (24h plan section 6):
  A3  control policy + bounded hierarchical control, generator frozen
  A4  A3 + small-rank local control at 16/32
  L2  per-position diagonal SPD on the latent gradient, K=2
  L3  per-position 4x4 SPD, K=4
  L4  SPD in the frozen A3 control coordinates, K=2
  C3  small-radius robustness fine-tune of R1 on REAL paintings

Key rule for all of them: gradients come from the FINAL generated result, and no
teacher endpoint is ever fitted. The plan explicitly rejects distilling a
stored optimiser trajectory (that is the R2/W failure mode).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from exp.v8.arms import GROUPS
from exp.v8.energy import RealSupport, SupportedEnergy, build_real_support

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/v8"
TRAIN_DIR = RUNS / "train"

# Microbatches measured on this 3060 by exp/v8/probe_step.py, which runs the
# EXACT training step (the style loss backprops through the frozen generator
# activations, unlike a detached benstep). Chosen to sit at the ~10 GB target:
#
#   A3/A4  96 -> 3.40 GB  192 -> 6.77 GB  288 -> 10.14 GB  (364 samples/s, best)
#   L2/L3  288 -> 5.0-7.1 GB (K-step R1 unroll)
#   C3     512 -> 9.31 GB
#
# 384 was rejected: it OOMs, because real training needs ~1.8x the detached-step
# memory (384 would be ~13.5 GB on a 12 GB card).
MICROBATCH = {"A3": 288, "A4": 288, "L2": 288, "L3": 288, "L4": 288, "C3": 512,
              "N2A": 288, "N2B": 288, "N2C": 288,
              "F0": 288, "F1": 288, "F2": 288, "F3": 288}

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def find_arm(arm_id: str) -> dict:
    for group, arms in GROUPS.items():
        for a in arms:
            if a["id"] == arm_id:
                return {**a, "group": group}
    raise KeyError(arm_id)


class RealPool:
    """Training-painting latents: source + a random other-style target.

    Evaluation sources (tasks_750) and the independent reference set are never
    sampled here, which is the same rule the teacher caches used.
    """

    def __init__(self, device="cuda"):
        from tools.v7r.v7r_common import SD_POOL
        self.device = device
        self.by_style = {}
        for sid, name in enumerate(STYLES):
            d = torch.load(SD_POOL / f"{sid:02d}_{name}.pt", map_location="cpu",
                           weights_only=False)
            self.by_style[name] = d["latents"].float()[100:]
        self.styles = STYLES

    def sample(self, batch: int, generator):
        idx = torch.randint(0, len(self.styles), (batch,), generator=generator)
        srcs, sids = [], []
        for s in idx.tolist():
            name = self.styles[s]
            pool = self.by_style[name]
            k = int(torch.randint(0, pool.shape[0], (1,), generator=generator))
            srcs.append(pool[k]); sids.append(s)
        z_s = torch.stack(srcs).to(self.device)
        sid = torch.tensor(sids, device=self.device)
        offset = torch.randint(1, 5, (batch,), generator=generator).to(self.device)
        tid = (sid + offset) % 5
        return z_s, sid, tid


def _save(arm_id: str, step: int, payload: dict):
    d = TRAIN_DIR / arm_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"step_{step:05d}.pt"
    tmp = p.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    return p


def train_arm(arm_id: str, *, device: str = "cuda", log_every: int = 100):
    arm = find_arm(arm_id)
    # Resume guard: a DONE training arm is never restarted by the queue.
    st = TRAIN_DIR / arm_id / "state.json"
    if st.exists():
        try:
            if json.loads(st.read_text()).get("state") == "DONE":
                print(f"[{arm_id}] already DONE, skipping", flush=True)
                return
        except Exception:
            pass
    kind = arm["kind"]
    updates = arm["updates"]
    ckpts = set(arm.get("checkpoints", [0, updates]))
    micro = int(os.environ.get(f"V8_MICRO_{arm_id}", MICROBATCH[arm_id]))
    eff = 240 if arm_id in ("A3", "A4", "C3") else 128
    accum = max(1, round(eff / micro))

    from src.infer import canonicalizer, load_c3unet, load_critic
    from exp.v8.controls import ControlSpec, Controls, wrap
    from exp.v8.guidance import (ControlSPDHead, SPDHead, UnsupervisedUnroll)

    starts = {"Q2a": "runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
              "B4": "runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
              "B3": "runs/v6/P1_B3_distill20/P1_B3_distill20.pt"}
    base_path = REPO / starts.get(arm.get("start", "Q2a"), starts["Q2a"])
    base, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = load_critic(device)
    pool = RealPool(device)
    gen = torch.Generator().manual_seed(arm.get("seed", 42))

    log = []

    # ------------------------------------------------ C3: robustness fine-tune
    if kind == "train_critic":
        for p in critic.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(critic.parameters(), lr=5e-5, weight_decay=1e-4)
        from exp.v8.energy import robustness_finetune_step
        for step in range(1, updates + 1):
            z_s, sid, tid = pool.sample(micro, gen)
            # z_s are REAL paintings whose true style is sid; labeling them
            # with the random tid trained the critic to call every painting
            # the wrong style (C3 Top-1 collapse). Use sid.
            info = robustness_finetune_step(critic, z_s, sid, opt)
            if step % log_every == 0 or step == 1:
                log.append({"step": step, **info})
                print(f"[{arm_id}] {step}/{updates} loss={info['loss']:.4f}", flush=True)
            if step in ckpts:
                _save(arm_id, step, {"model": critic.state_dict(), "kind": arm_id,
                                     "step": step, "config": arm})
        _write_trace(arm_id, log)
        return

    # ------------------------------------------------ A3/A4: control policy
    if kind == "train_control":
        spec = ControlSpec(local_rank=2 if "local" in arm["spec"] else 0)
        gen_wrapped = wrap(base, spec, device)

        def _feat(c):
            c = c.float()
            return torch.cat([c.mean(dim=(2, 3)), c.std(dim=(2, 3))], dim=1)

        policy = None  # built lazily once feat_dim is known
        opt = None
        support_path = RUNS / "real_support.pt"
        support = None
        if support_path.exists():
            st = torch.load(support_path, map_location="cpu", weights_only=False)
            support = RealSupport(st["prototypes"].to(device), st["sigma"].to(device))
        scorer = SupportedEnergy(critic, support, 0.1)
        for step in range(1, updates + 1):
            z_s, sid, tid = pool.sample(micro, gen)
            with torch.no_grad():
                c = canon(z_s, mode="c1_gentle", augment=False)
                z0 = base(c, 0.0, c, tid).detach()
                feat = _feat(c)
            if policy is None:
                feat_dim = feat.shape[1]
                policy = nn.Sequential(nn.Linear(feat_dim + 5, 256), nn.SiLU(),
                                       nn.Linear(256, spec.dim)).to(device)
                # zero final layer -> u0 = 0 reproduces the base exactly,
                # then the policy has to earn any deviation
                nn.init.zeros_(policy[-1].weight); nn.init.zeros_(policy[-1].bias)
                opt = torch.optim.AdamW(policy.parameters(), lr=2e-4,
                                        weight_decay=1e-4)
            oh = nn.functional.one_hot(tid, 5).float()
            u = policy(torch.cat([feat, oh], 1))
            out = gen_wrapped(c, 0.0, c, tid, controls=Controls.from_tensor(u, spec))
            margin = scorer.value(out, tid)
            # direct objective on the FINAL output; no teacher endpoint
            loss = (nn.functional.softplus(-margin).mean()
                    + 1.0 * (out - z0).square().flatten(1).mean()
                    + 1e-3 * u.square().mean())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == 1:
                log.append({"step": step, "loss": float(loss),
                            "margin": float(margin.mean()),
                            "displacement_rms": float((out - z0).square().flatten(1).mean(1).sqrt().mean())})
                print(f"[{arm_id}] {step}/{updates} loss={float(loss):.4f} "
                      f"margin={float(margin.mean()):.3f}", flush=True)
            if step in ckpts:
                _save(arm_id, step, {"policy": policy.state_dict(), "spec": asdict(spec),
                                     "feat_dim": int(feat_dim),
                                     "kind": arm_id, "step": step, "config": arm})
        _write_trace(arm_id, log)
        return

    # ------------------------------------------------ L2/L3/L4: SPD operators
    if kind == "train_spd":
        rank = arm.get("rank", 2)
        if arm["space"] == "control":
            from exp.v8.controls import ControlSpec
            cspec = ControlSpec(local_rank=2 if "local" in arm.get("spec", "") else 0)
            head = ControlSPDHead(dim=cspec.dim, rank=rank).to(device)
        else:
            head = SPDHead(rank=rank).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
        ed = UnsupervisedUnroll(critic, head, base, canon, space=arm["space"],
                                steps=arm.get("steps_k", 2), rho=0.25, device=device,
                                control_spec=cspec if arm["space"] == "control" else None)
        for step in range(1, updates + 1):
            z_s, sid, tid = pool.sample(micro, gen)
            out = ed(z_s, tid)
            margin = ed.margin(out, tid)
            loss = (nn.functional.softplus(-margin).mean()
                    + 0.5 * ed.content_penalty(out, ed.z0))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == 1:
                log.append({"step": step, "loss": float(loss),
                            "margin": float(margin.mean())})
                print(f"[{arm_id}] {step}/{updates} loss={float(loss):.4f} "
                      f"margin={float(margin.mean()):.3f}", flush=True)
            if step in ckpts:
                _save(arm_id, step, {"head": head.state_dict(), "rank": rank,
                                     "space": arm["space"], "kind": arm_id,
                                     "step": step, "config": arm})
        _write_trace(arm_id, log)
        return

    # ------------------------------------------- N2: constrained style max
    # L = -margin + lam * relu(cost - eps)^2,  cost = ||z'-z0||^2 / ||z0-z_s||^2
    # The hinge only prices edit damage BEYOND the per-sample budget eps, so
    # inside the budget the objective is pure margin maximization. N2B adds a
    # per-sample lambda weight (samples whose generation already sits far from
    # the source get a tighter effective budget); N2C adds the V6 real
    # style-distribution loss so margin cannot be bought by a classifier
    # shortcut alone.
    if kind == "train_n2":
        rank = arm.get("rank", 0)
        head = SPDHead(rank=rank).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
        ed = UnsupervisedUnroll(critic, head, base, canon, space="latent",
                                steps=arm.get("steps_k", 2), rho=0.25, device=device)
        lam = float(arm.get("lam", 30.0))
        eps = float(arm.get("eps", 0.05))
        adaptive = bool(arm.get("adaptive", False))
        wsd = float(arm.get("w_style_dist", 0.0))
        style_targets = None
        if wsd > 0:
            from exp.v8.style_dist import load_style_targets
            style_targets = load_style_targets(device)
        for step in range(1, updates + 1):
            z_s, sid, tid = pool.sample(micro, gen)
            out = ed(z_s, tid)
            z0 = ed.z0
            margin = ed.margin(out, tid)
            d_edit = (out - z0).square().flatten(1).mean(1)
            d_gen = (z0 - z_s).square().flatten(1).mean(1).clamp_min(1e-8)
            cost = d_edit / d_gen
            viol = (cost - eps).clamp_min(0.0)
            if adaptive:
                w = (d_gen / d_gen.median().clamp_min(1e-8)).clamp(0.5, 2.0)
                pen = (lam * w * viol.square()).mean()
            else:
                pen = (lam * viol.square()).mean()
            loss = -margin.mean() + pen
            if style_targets is not None:
                from exp.v8.style_dist import style_distribution_loss
                loss = loss + wsd * style_distribution_loss(out, tid, style_targets)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == 1:
                log.append({"step": step, "loss": float(loss),
                            "margin": float(margin.mean()),
                            "cost": float(cost.mean()),
                            "viol": float(viol.mean())})
                print(f"[{arm_id}] {step}/{updates} loss={float(loss):.4f} "
                      f"margin={float(margin.mean()):.3f} "
                      f"cost={float(cost.mean()):.4f} viol={float(viol.mean()):.4f}",
                      flush=True)
            if step in ckpts:
                _save(arm_id, step, {"head": head.state_dict(), "rank": rank,
                                     "space": "latent", "kind": arm_id,
                                     "step": step, "config": arm})
        _write_trace(arm_id, log)
        return

    # ------------------------------- F0-F3: trainable style actuation
    # ctrl = A_phi(u, feat(c), t); z = G_B4(c,t;ctrl). Inner loop optimises u
    # against the R1 margin exactly as at inference (steps_k = 0/1/2), then the
    # FINAL output is scored by margin + content-hinge + optional V6 style
    # distribution loss, and gradients train phi only (u is detached between
    # steps). u=0 is exactly B4 by construction, for any phi.
    if kind == "train_actuation":
        from exp.v8.actuation import StyleActuator
        from exp.v8.guidance import target_margin

        u_dim = int(arm.get("u_dim", 32))
        steps_k = int(arm.get("steps_k", 0))
        lam = float(arm.get("lam", 50.0))
        eps = float(arm.get("eps", 0.15))
        wsd = float(arm.get("w_style_dist", 0.0))
        ctrl_reg = float(arm.get("ctrl_reg", 1e-3))
        cspec = ControlSpec(local_rank=int(arm.get("local_rank", 0)))
        gen_w = wrap(base, cspec, device)

        style_targets = None
        if wsd > 0:
            from exp.v8.style_dist import load_style_targets
            style_targets = load_style_targets(device)

        actuator = None
        opt = None
        feat_dim = None
        for step in range(1, updates + 1):
            z_s, sid, tid = pool.sample(micro, gen)
            with torch.no_grad():
                c = canon(z_s, mode="c1_gentle", augment=False)
                z0 = base(c, 0.0, c, tid).detach()
                feat = torch.cat([c.mean(dim=(2, 3)), c.std(dim=(2, 3))], dim=1)
            if actuator is None:
                feat_dim = feat.shape[1]
                actuator = StyleActuator(u_dim, feat_dim, cspec.dim).to(device)
                opt = torch.optim.AdamW(actuator.parameters(), lr=2e-4,
                                        weight_decay=1e-4)
            u = torch.zeros(z_s.shape[0], u_dim, device=device,
                            requires_grad=True)
            out = None
            margin = None
            ctrl = None
            for _ in range(steps_k + 1):
                ctrl = actuator(u, feat, tid)
                controls = Controls.from_tensor(ctrl, cspec)
                out = gen_w(c, 0.0, c, tid, controls=controls)
                margin = target_margin(critic, out, tid)
                if _ == steps_k:
                    break
                g = torch.autograd.grad(margin.sum(), u)[0]
                gr = g / g.flatten(1).pow(2).mean(1).sqrt() \
                    .clamp_min(1e-12).view(-1, 1)
                u = (u + arm.get("eta", 0.05) * gr).detach() \
                    .requires_grad_(True)
            d_edit = (out - z0).square().flatten(1).mean(1)
            d_gen = (z0 - z_s).square().flatten(1).mean(1).clamp_min(1e-8)
            cost = d_edit / d_gen
            viol = (cost - eps).clamp_min(0.0)
            loss = (-margin.mean() + (lam * viol.square()).mean()
                    + ctrl_reg * ctrl.square().mean())
            if style_targets is not None:
                from exp.v8.style_dist import style_distribution_loss
                loss = loss + wsd * style_distribution_loss(out, tid,
                                                            style_targets)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actuator.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == 1:
                log.append({"step": step, "loss": float(loss),
                            "margin": float(margin.mean()),
                            "cost": float(cost.mean()),
                            "viol": float(viol.mean())})
                print(f"[{arm_id}] {step}/{updates} loss={float(loss):.4f} "
                      f"margin={float(margin.mean()):.3f} "
                      f"cost={float(cost.mean()):.4f} "
                      f"viol={float(viol.mean()):.4f}", flush=True)
            if step in ckpts:
                _save(arm_id, step, {"actuator": actuator.state_dict(),
                                     "spec": asdict(cspec), "u_dim": u_dim,
                                     "feat_dim": int(feat_dim),
                                     "kind": arm_id, "step": step,
                                     "config": arm})
        _write_trace(arm_id, log)
        return

    raise KeyError(f"arm {arm_id} has no trainer (kind={kind})")


def _write_trace(arm_id: str, log: list):
    d = TRAIN_DIR / arm_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (d / "state.json").write_text(json.dumps(
        {"arm": arm_id, "state": "DONE", "updates": log[-1]["step"] if log else 0},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    train_arm(sys.argv[1])
