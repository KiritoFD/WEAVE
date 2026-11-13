"""WEAVE V7-R shared helpers (docs/v7/V7R_PLAN.md).

V7-R = mechanism-repair extension of V7:
  Q1 why real-style energies separate styles but yield no useful generation gradient;
  Q2 why the relative transform degrades after training although B4-relative already
     responds (Top-1 31.5%);
  Q3 can we keep B4's content mapping and learn a small source->target correction.

Everything V7-R writes to runs/v7r/ and docs/v7/V7R_*.md. Frozen assets (B3/B4,
V6/W/V7 results, tasks_750) are read-only.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v7.v7_common import (
    B4_CKPT, R1_CRITIC, ROOT, STYLES, load_c3unet, load_canon, load_critic,
    write_json,
)

RUNS_V7R = ROOT / "runs" / "v7r"
P0_DIR = RUNS_V7R / "p0_audit"
P0_CKPTS = RUNS_V7R / "p0_ckpts"
P1_DIR = RUNS_V7R / "p1_energy"
P2_DIR = RUNS_V7R / "p2_predictor"
P3_DIR = RUNS_V7R / "p3_corrector"
P4_DIR = RUNS_V7R / "p4_joint"
STATE = RUNS_V7R / "state"
LOGS = RUNS_V7R / "logs"

V7_P1 = ROOT / "runs" / "v7" / "p1_energy"
V7_P2 = ROOT / "runs" / "v7" / "p2_eval"

SD_POOL = Path(os.environ.get(
    "WEAVE_SD_POOL", "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"))

# P2 corrector defaults (predefined; documented in V7R_PLAN.md)
CORRECTOR_BASE_DIM = 48
N_STEPS = 3000
N_CKPT_STEPS = (500, 1500, 3000)
N_EFFECTIVE_BATCH = 240      # microbatch 40 x accum 6
N_LR = 2e-4
N_WARMUP = 100
W_STYLE = 1.0
W_CONTENT = 0.525            # V6 canon-inv weight
W_ANCHOR = 1.0
W_R1 = 0.003
# Predefined imbalance policy: at the step-100 audit, if any WEIGHTED term's
# gradient norm exceeds 100x the smallest, rescale that weight once toward the
# median (capped at 100x), then re-audit and record. No further tuning.
IMBALANCE_RATIO = 100.0


# ---------------------------------------------------------------------------
# painter parsing (pool ids look like wikiart__S__S__painter_title)
# ---------------------------------------------------------------------------

def painter_of(image_id: str) -> str:
    rest = image_id.split("__", 2)[-1] if "__" in image_id else image_id
    rest = rest.split("__")[-1]
    tok = re.split(r"[_]", rest)[0]
    if len(tok) < 5 or tok.isdigit():
        return ""
    return tok


# ---------------------------------------------------------------------------
# H_theta: small source/target-conditioned residual corrector on top of B4
# ---------------------------------------------------------------------------

class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, dim)
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.n2 = nn.GroupNorm(8, dim)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        gb = self.film(cond)
        gamma, beta = gb[:, : x.shape[1]].unsqueeze(-1).unsqueeze(-1), \
            gb[:, x.shape[1]:].unsqueeze(-1).unsqueeze(-1)
        h = self.n1(x)
        h = F.silu(h + gamma + beta)
        h = self.c1(h)
        h = self.n2(h)
        h = F.silu(h)
        h = self.c2(h)
        return x + h


class ResidualCorrector(nn.Module):
    """z_out = z0 + H(c, z0, [source], target); output layer zero-initialized so
    step 0 is exactly B4. N1 vs N2 differ ONLY in `use_source`."""

    def __init__(self, base_dim: int = CORRECTOR_BASE_DIM, num_styles: int = 5,
                 use_source: bool = True, cond_dim: int = 32):
        super().__init__()
        self.use_source = use_source
        in_ch = 12  # [c, z0, z0 - c]
        self.style_emb = nn.Embedding(num_styles, cond_dim)
        self.stem = nn.Conv2d(in_ch, base_dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResBlock(base_dim, cond_dim) for _ in range(3)])
        self.out_norm = nn.GroupNorm(8, base_dim)
        self.out_conv = nn.Conv2d(base_dim, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, c: torch.Tensor, z0: torch.Tensor,
                source_id: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        e = self.style_emb(target_id)
        if self.use_source:
            e = e + self.style_emb(source_id)
        x = torch.cat([c, z0, z0 - c], dim=1)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, e)
        h = self.out_norm(h)
        return z0 + self.out_conv(F.silu(h))


def save_corrector(path: Path, model: ResidualCorrector, args: dict,
                   step: int, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "corrector": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "use_source": model.use_source,
        "base_dim": int(model.blocks[0].c1.in_channels),
        "args": args,
        "step": step,
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_corrector(path: Path, device: str) -> Tuple[ResidualCorrector, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ResidualCorrector(base_dim=int(ck.get("base_dim", CORRECTOR_BASE_DIM)),
                          use_source=bool(ck["use_source"]))
    m.load_state_dict(ck["corrector"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


def corrector_output(model: ResidualCorrector, z0: torch.Tensor,
                     c: torch.Tensor, sid: torch.Tensor, tid: torch.Tensor,
                     cross_mask: torch.Tensor) -> torch.Tensor:
    """Apply H only on cross-style tasks; same-style keeps B4 output z0."""
    res = model(c, z0, sid, tid)
    keep = (~cross_mask).view(-1, 1, 1, 1)
    return torch.where(keep, z0, res)


# ---------------------------------------------------------------------------
# pools / probes
# ---------------------------------------------------------------------------

def load_pool(device: str = "cpu", val_n: int = 100):
    train_by_style, val_by_style, painters = {}, {}, {}
    for sid, s in enumerate(STYLES):
        d = torch.load(SD_POOL / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        x = d["latents"].float()
        val_by_style[s] = x[:val_n]
        train_by_style[s] = x[val_n:]
        painters[s] = [painter_of(i) for i in d.get("ids", [val_n + k for k in range(x.shape[0] - val_n)])]
    return train_by_style, val_by_style, painters


def canon_inv_loss(z_pred: torch.Tensor, z_real: torch.Tensor,
                   canon) -> torch.Tensor:
    """V6 content-canonical invariance: c1 LL + 0.35*(LH,HL)."""
    from wavelet import dwt2_haar
    c_clean = canon(z_real, mode="c1", augment=False)
    c_cross = canon(z_pred.float(), mode="c1", augment=False)
    r_ll, r_lh, r_hl, _ = dwt2_haar(c_clean)
    g_ll, g_lh, g_hl, _ = dwt2_haar(c_cross)
    return F.mse_loss(g_ll, r_ll) + 0.35 * (F.mse_loss(g_lh, r_lh)
                                            + F.mse_loss(g_hl, r_hl))


def randomized_canon(z: torch.Tensor, canon,
                     ll_mean_range: Tuple[float, float] = (0.8, 1.2),
                     ll_std_range: Tuple[float, float] = (0.8, 1.2),
                     hf_amp_range: Tuple[float, float] = (0.6, 1.4)) -> torch.Tensor:
    """P2 alternative route: constrained style randomization of c1_gentle output.

    Same content, different low-freq colour statistics and HF amplitude, so the
    network cannot rely on source-style appearance and must use the target
    Style-ID. Ranges are fixed (recorded in the config), not tuned; randomness
    follows the global training seed.
    """
    from wavelet import dwt2_haar, idwt2_haar
    c = canon(z, mode="c1_gentle", augment=False)
    B = c.shape[0]
    dev = c.device

    def u(lo, hi):
        return torch.rand(B, device=dev) * (hi - lo) + lo

    ll, lh, hl, hh = dwt2_haar(c.float())
    m_ll = ll.mean(dim=(2, 3), keepdim=True)
    s_ll = ll.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
    lm = u(*ll_mean_range).view(B, 1, 1, 1)
    ls = u(*ll_std_range).view(B, 1, 1, 1)
    ll2 = (ll - m_ll) * ls + m_ll * lm
    a = u(*hf_amp_range).view(B, 1, 1, 1)
    c2 = idwt2_haar(ll2, lh * a, hl * a, hh * a)
    return c2.to(c.dtype)


# ---------------------------------------------------------------------------
# generic budgeted multi-component optimizer (P1-C / P3), with strict auditing
# ---------------------------------------------------------------------------

def rms_normalize(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    rms = g.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
    return g / (rms + eps)


class BudgetedEditor:
    """Multi-component latent editor with fixed step-0 scales, backtracking
    line search, and full per-step audit (raw + post-scale norms, accepted and
    rejected alphas, cumulative displacement)."""

    ALPHA_GRID = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)

    def __init__(self, z0, z_src, tid, *, canon, canon_mode, objectives: dict,
                 weights: dict, steps: int, per_step_rms: float,
                 struct_src: Optional[torch.Tensor] = None):
        """objectives: name -> callable(z) -> scalar loss (differentiable).
        weights: name -> float (component '-name' means ascent direction)."""
        self.z0, self.z_src, self.tid = z0, z_src, tid
        self.canon, self.canon_mode = canon, canon_mode
        self.objectives, self.weights = objectives, weights
        self.steps, self.per_step_rms = steps, per_step_rms
        self.struct_src = struct_src
        self.c_src = canon(z_src, mode=canon_mode, augment=False).detach()
        self.scales = None
        self.audit: List[dict] = []

    def _builtin_terms(self, zz):
        c_z = self.canon(zz.float(), mode=self.canon_mode, augment=False)
        l_content = (c_z - self.c_src).square().mean(dim=(1, 2, 3)).mean()
        l_anchor = (zz.float() - self.z0.float()).square().flatten(1).mean()
        out = {"content": l_content, "anchor": l_anchor}
        if "struct" in self.weights and self.struct_src is not None:
            from tools.v6_next.r2_teacher import spatial_self_similarity_matrix
            l_struct = (spatial_self_similarity_matrix(zz)
                        - self.struct_src).square().mean()
            out["struct"] = l_struct
        return out

    def step(self, z_cur):
        zr = z_cur.detach().clone().requires_grad_(True)
        losses = dict(self._builtin_terms(zr))
        for name, fn in self.objectives.items():
            losses[name] = fn(zr)
        grads, raw_norms, post_norms = {}, {}, {}
        for name, loss in losses.items():
            g = torch.autograd.grad(loss, zr, retain_graph=True)[0]
            grads[name] = g
            raw_norms[name] = float(g.flatten(1).norm(dim=1).mean())
        if self.scales is None:
            self.scales = {k: max(raw_norms[k], 1e-12) for k in grads}
        direction = zr.new_zeros(zr.shape)
        for name, g in grads.items():
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            # components are DESCENTS on their loss; a negative weight flips
            direction = direction + w * (-g / self.scales[name])
        for name in grads:
            w = self.weights.get(name, 0.0)
            post_norms[name] = w * raw_norms[name] / self.scales[name]
        d = rms_normalize(direction)

        f0 = self._total(losses)
        alphas = []
        alpha_used = self.ALPHA_GRID[-1]
        for a in self.ALPHA_GRID:
            zt = z_cur + self.per_step_rms * a * d
            accepted = self._total(self._eval_losses(zt)) < f0
            alphas.append({"alpha": a, "accepted": bool(accepted)})
            if accepted:
                alpha_used = a
                break
        z_new = (zr + self.per_step_rms * alpha_used * d).detach()
        disp = float((z_new - self.z0).flatten(1).pow(2).mean(dim=1)
                     .sqrt().mean())
        self.audit.append({
            "step": len(self.audit), "raw_norms": raw_norms,
            "post_scale_norms": post_norms, "scales": dict(self.scales),
            "alpha_used": alpha_used, "alpha_trials": alphas,
            "n_rejected": sum(1 for t in alphas if not t["accepted"]),
            "cum_displacement_rms": disp,
            "losses": {k: float(v.detach()) for k, v in losses.items()},
        })
        return z_new

    def _eval_losses(self, zz):
        out = dict(self._builtin_terms(zz))
        for name, fn in self.objectives.items():
            out[name] = fn(zz)
        return out

    def _total(self, losses):
        t = None
        for name, w in self.weights.items():
            if w == 0.0 or name not in losses:
                continue
            term = w * losses[name]
            t = term if t is None else t + term
        return t if t is not None else z0_like(losses)


def z0_like(losses):
    for v in losses.values():
        return v * 0.0
    raise ValueError("empty losses")


def audit_summary(audit: List[dict]) -> dict:
    if not audit:
        return {}
    keys = [k for k in audit[0]["post_scale_norms"]]
    out = {}
    for k in keys:
        vals = [a["post_scale_norms"][k] for a in audit]
        out[f"post_scale_{k}"] = {"mean": sum(vals) / len(vals),
                                  "min": min(vals), "max": max(vals)}
    ratio = (max(v["mean"] for v in out.values())
             / max(min(v["mean"] for v in out.values()), 1e-12))
    out["post_scale_max_over_min"] = ratio
    out["imbalanced_post_scale"] = bool(ratio > IMBALANCE_RATIO)
    out["alpha_hist"] = {str(a): sum(1 for x in audit
                                     if x["alpha_used"] == a)
                         for a in BudgetedEditor.ALPHA_GRID}
    out["final_displacement_rms"] = audit[-1]["cum_displacement_rms"]
    return out
