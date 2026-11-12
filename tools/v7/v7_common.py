"""WEAVE V7 shared helpers: loaders, relative parameterization, energies, audits.

V7 = Amortized Style Transport + Instance-Adaptive Refinement.
All V7-specific code lives in tools/v7/ (see docs/v7/V7_BOUNDARIES.md section 8).
Frozen V6 assets are loaded read-only; nothing in this module modifies them.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet
from tools.v6_next.w_guided import r1_gradient, r1_margin_per_sample, rms_normalize
from wavelet import dwt2_haar

ROOT = Path(__file__).resolve().parents[2]

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

CANON_STATS = ROOT / "runs" / "v6" / "cache" / "canonicalizer_stats.pt"
B4_CKPT = ROOT / "runs" / "v6" / "P1_B4_distill50" / "P1_B4_distill50.pt"
Q2A_CKPT = ROOT / "runs" / "v6" / "Q2a_embed_style" / "Q2a_embed_style.pt"
R1_CRITIC = ROOT / "runs" / "v6_next" / "probes" / "r1_r1.pt"
TASKS_750 = ROOT / "eval_protocol" / "tasks_750.json"

RUNS_V7 = ROOT / "runs" / "v7"
P0_DIR = RUNS_V7 / "p0_relative"
P1_DIR = RUNS_V7 / "p1_energy"
P2_DIR = RUNS_V7 / "p2_eval"
P3_MODELS = RUNS_V7 / "p3_models"
P3_EVAL = RUNS_V7 / "p3_eval"
P4_DIR = RUNS_V7 / "p4_eval"
STATE_DIR = RUNS_V7 / "state"

# P2/P3 corrector/predictor RMS budget must match v6_w T0/T1 (per_step_rms=0.08, 4 steps).
PER_STEP_RMS = 0.08
STEPS = 4
# Predefined weak-R1 / content / anchor weights (gradient norms are audited separately;
# each component gradient is per-sample RMS-normalized before combining, which is the
# predefined scale policy required by V7_EVAL_PROTOCOL.md section 6).
WEAK_R1_WEIGHT = 0.25
CONTENT_WEIGHT = 0.5
ANCHOR_WEIGHT = 0.5
STRUCT_WEIGHT = 0.5  # T5 only: spatial self-similarity constraint


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_c3unet(ckpt_path: Path, device: str) -> C3UNet:
    """Load a V6-family C3UNet checkpoint ({model, args}) read-only."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = C3UNet(
        base_dim=a.get("base_dim", 16),
        num_styles=5,
        mode=a.get("mode", "direct"),
        no_style=bool(a.get("no_style", False)),
        content_mode=a.get("content_mode", "full"),
        inject_64=not a.get("no_inject_64", False),
        inject_32=not a.get("no_inject_32", False),
        inject_16=not a.get("no_inject_16", False),
        style_keys=bool(a.get("style_keys", False)),
        style_keys_n=int(a.get("style_keys_n", 8) or 8),
        style_keys_rank=int(a.get("style_keys_rank", 32) or 32),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_critic(ckpt_path: Path = R1_CRITIC, device: str = "cuda") -> LatentStyleNet:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=ck.get("width", 64)).to(device).eval()
    critic.load_state_dict(ck["model"])
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic


def load_canon(device: str) -> ContentCanonicalizer:
    canon = ContentCanonicalizer(
        torch.load(CANON_STATS, map_location="cpu", weights_only=False)
    ).to(device)
    canon.eval()
    return canon


def canon_latent(z: torch.Tensor, canon: ContentCanonicalizer,
                 canon_mode: str = "c1_gentle") -> torch.Tensor:
    return canon(z, mode=canon_mode, augment=False)


def model_direct(model: C3UNet, z: torch.Tensor, canon: ContentCanonicalizer,
                 canon_mode: str, tid: torch.Tensor) -> torch.Tensor:
    """Original V6 output parameterization: z_direct = F(c(zs), target)."""
    c = canon_latent(z, canon, canon_mode)
    return model(c, 0.0, c, tid)


def model_relative(model: C3UNet, z: torch.Tensor, canon: ContentCanonicalizer,
                   canon_mode: str, sid: torch.Tensor,
                   tid: torch.Tensor) -> torch.Tensor:
    """V7 relative parameterization: z_rel = zs + F(c(zs), t) - F(c(zs), s).

    source == target  =>  exactly zs (structural identity).
    """
    c = canon_latent(z, canon, canon_mode)
    z_t = model(c, 0.0, c, tid)
    z_s = model(c, 0.0, c, sid)
    return z + z_t - z_s


def model_output(model: C3UNet, z: torch.Tensor, canon: ContentCanonicalizer,
                 canon_mode: str, sid: torch.Tensor, tid: torch.Tensor,
                 start_mode: str = "direct") -> torch.Tensor:
    if start_mode == "relative":
        return model_relative(model, z, canon, canon_mode, sid, tid)
    return model_direct(model, z, canon, canon_mode, tid)


def tasks_sha256() -> str:
    return sha256_of(TASKS_750)


def load_tasks() -> List[dict]:
    return json.loads(TASKS_750.read_text(encoding="utf-8"))["tasks"]


def pick_stratified_sources(tasks: List[dict], per_style: int) -> List[dict]:
    """Deterministic per-source-style stratified pick (first N stems in file order)."""
    seen = {}
    out = []
    for t in tasks:
        k = t["source_stem"]
        if k in seen:
            continue
        seen[k] = True
        if len([o for o in out if o["source_style"] == t["source_style"]]) < per_style:
            out.append(t)
        if len(out) >= per_style * len(STYLES):
            break
    return out


def pick_trace_tasks(tasks: List[dict], per_style: int = 4) -> List[str]:
    """Fixed trace task ids: first `per_style` cross tasks per TARGET style (file order)."""
    count: Dict[str, int] = {}
    out = []
    for t in tasks:
        is_cross = t.get("is_cross")
        if is_cross is None:
            is_cross = t.get("target_style_id", 0) != t.get("source_style_id", 1)
        if not is_cross:
            continue
        s = t["target_style"]
        if count.get(s, 0) < per_style:
            out.append(t["task_id"])
            count[s] = count.get(s, 0) + 1
        if len(out) >= per_style * len(STYLES):
            break
    return out


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v7] wrote {path}", flush=True)


def grad_norm_of(loss: torch.Tensor, params) -> float:
    """Total L2 norm of gradients of `loss` w.r.t. params (no .backward on main graph)."""
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.zeros((), device=next(iter(params)).device if len(params) else "cpu")
    for g in gs:
        if g is not None:
            total = total + g.detach().float().square().sum()
    return float(total.sqrt())


# ---------------------------------------------------------------------------
# P1 style energies (S1 = latent Haar/patch statistics, S2 = frozen R1 features)
# ---------------------------------------------------------------------------

def s1_loss(z: torch.Tensor, tgt_ids: torch.Tensor, targets: dict) -> torch.Tensor:
    """S1 energy: V6 multi-scale Haar moment + LL covariance + patch-SWD loss.

    Reuses tools.v6.train_eval_v6.compute_style_distribution_loss verbatim (the
    V6 distribution loss is kept as the S1 implementation and structural对照).
    """
    from tools.v6.train_eval_v6 import compute_style_distribution_loss
    return compute_style_distribution_loss(z, tgt_ids, targets)


def s2_stats_of(z: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Spatial-agnostic statistics of one R1 feature scale tensor (B,C,H,W).

    Returns per-sample spatial mean (B,C), spatial std (B,C) and channel
    correlation matrix (B,C,C). Position-independent by construction — this is
    deliberately different from W1's Oracle per-position feature fitting.
    """
    B, C = z.shape[0], z.shape[1]
    flat = z.float().flatten(2)                      # (B,C,HW)
    mu = flat.mean(dim=2)                            # (B,C)
    sd = flat.std(dim=2, unbiased=False).clamp(min=1e-4)
    f = F.normalize(flat, dim=2, eps=1e-6)
    corr = torch.bmm(f, f.transpose(1, 2))           # (B,C,C)
    return {"mu": mu, "sd": sd, "corr": corr, "n_ch": C}


def build_s2_style_stats(feat_ext, latents_by_style: Dict[str, torch.Tensor],
                         per_style: int = 1500, device: str = "cuda",
                         batch: int = 32) -> dict:
    """Build S2 statistics per style from real latents through the frozen R1 net."""
    out: Dict[str, dict] = {}
    for s in STYLES:
        z = latents_by_style[s][:per_style]
        mus, sds, corrs = [], [], []
        for i in range(0, z.shape[0], batch):
            zb = z[i:i + batch].to(device)
            with torch.no_grad():
                _, feats = feat_ext(zb)
            for scale, ft in feats.items():
                st = s2_stats_of(ft)
                mus.append((scale, st["mu"].cpu()))
                sds.append((scale, st["sd"].cpu()))
                corrs.append((scale, st["corr"].cpu()))
        out[s] = {}
        for scale in feat_ext.SCALES:
            mu = torch.cat([m for k, m in mus if k == scale], dim=0)
            sd = torch.cat([v for k, v in sds if k == scale], dim=0)
            co = torch.cat([c for k, c in corrs if k == scale], dim=0)
            out[s][scale] = {
                "n_ch": int(mu.shape[1]),
                "mu_mean": mu.mean(dim=0), "mu_std": mu.std(dim=0).clamp(min=1e-6),
                "sd_mean": sd.mean(dim=0), "sd_std": sd.std(dim=0).clamp(min=1e-6),
                "corr_mean": co.mean(dim=0),
            }
    return out


def s2_loss(z: torch.Tensor, tgt_ids: torch.Tensor, stats: dict,
            feat_ext) -> torch.Tensor:
    """S2 energy: match frozen-R1 per-scale spatial mean/std/correlation to style stats.

    Standardized residuals (z-scored against stored per-style std) so no scale
    dominates by magnitude. Fully differentiable w.r.t. z.
    """
    with torch.enable_grad():
        _, feats = feat_ext(z)
    total = z.new_tensor(0.0)
    B = z.shape[0]
    for scale, ft in feats.items():
        st = s2_stats_of(ft)
        for sid in torch.unique(tgt_ids):
            mask = tgt_ids == sid
            if not mask.any():
                continue
            ss = stats[STYLES[int(sid)]][scale]
            mu_t = ss["mu_mean"].to(z.device).unsqueeze(0)
            sd_t = ss["sd_mean"].to(z.device).unsqueeze(0)
            zmu = (st["mu"][mask] - mu_t) / ss["mu_std"].to(z.device).unsqueeze(0)
            zsd = (st["sd"][mask] - sd_t) / ss["sd_std"].to(z.device).unsqueeze(0)
            l_mu = zmu.square().mean()
            l_sd = zsd.square().mean()
            l_corr = (st["corr"][mask].mean(dim=0) - ss["corr_mean"].to(z.device)).square().mean()
            total = total + (l_mu + l_sd + 0.5 * l_corr) * (mask.sum().float() / B)
    return total


def s2_distance(z: torch.Tensor, tgt_ids: torch.Tensor, stats: dict, feat_ext) -> float:
    """Non-gradient S2 energy value (for audits/traces)."""
    with torch.no_grad():
        return float(s2_loss(z, tgt_ids, stats, feat_ext))


def s1_distance(z: torch.Tensor, tgt_ids: torch.Tensor, targets: dict) -> float:
    with torch.no_grad():
        return float(s1_loss(z, tgt_ids, targets))


def s1_loss_per_sample(z: torch.Tensor, tgt_ids: torch.Tensor,
                       targets: dict) -> torch.Tensor:
    """Per-sample S1 energy (audit/scoring form of the V6 distribution loss).

    Same terms as compute_style_distribution_loss but evaluated per latent:
    band moments + LL covariance + patch-SWD quantile L1 (LL and HF).
    """
    from tools.v6.train_eval_v6 import matrix_sqrt_invsqrt  # noqa: F401 (parity)
    dev = z.device
    ll, lh, hl, hh = dwt2_haar(z.float())
    ll2, _, _, _ = dwt2_haar(ll)
    R_ll = targets["R_ll"].to(dev)
    R_hf = targets["R_hf"].to(dev)
    B = z.shape[0]
    out = z.new_zeros(B)
    for sid, s in enumerate(STYLES):
        mask = (tgt_ids == sid)
        if not mask.any():
            continue
        st = targets["styles"][s]
        idx = mask.nonzero(as_tuple=True)[0]
        ll_s, lh_s, hl_s, hh_s, ll2_s = ll[idx], lh[idx], hl[idx], hh[idx], ll2[idx]

        def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a - b).square().mean(dim=tuple(range(1, a.dim())))

        l_mom = (
            _mse(ll_s.mean(dim=(2, 3)), st["ll_mu"].to(dev))
            + _mse(ll_s.std(dim=(2, 3)), st["ll_sd"].to(dev))
            + _mse(lh_s.std(dim=(2, 3)), st["lh_sd"].to(dev))
            + _mse(hl_s.std(dim=(2, 3)), st["hl_sd"].to(dev))
            + _mse(hh_s.std(dim=(2, 3)), st["hh_sd"].to(dev))
            + _mse(ll2_s.std(dim=(2, 3)), st["ll2_sd"].to(dev))
        )
        flat = ll_s.reshape(ll_s.shape[0], 4, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / max(flat.shape[2] - 1, 1)
        l_cov = (cov - st["ll_cov"].to(dev).unsqueeze(0)).square().mean(dim=(1, 2))

        pat_ll = F.unfold(ll_s, kernel_size=4, stride=4).permute(0, 2, 1)
        proj_ll, _ = torch.sort(pat_ll @ R_ll.t(), dim=1)
        q_idx = torch.linspace(0, proj_ll.shape[1] - 1, 128, device=dev).long()
        l_swd_ll = (proj_ll[:, q_idx] - st["qgrid_ll"].to(dev).unsqueeze(0)).abs().mean(dim=(1, 2))

        hf_cat = torch.cat([lh_s, hl_s, hh_s], dim=1)
        pat_hf = F.unfold(hf_cat, kernel_size=4, stride=4).permute(0, 2, 1)
        proj_hf, _ = torch.sort(pat_hf @ R_hf.t(), dim=1)
        l_swd_hf = (proj_hf[:, q_idx] - st["qgrid_hf"].to(dev).unsqueeze(0)).abs().mean(dim=(1, 2))

        sub = l_mom + 0.5 * l_cov + 0.5 * (l_swd_ll + l_swd_hf)
        out = out.index_put((idx,), sub.to(out.dtype))
    return out


def s2_loss_per_sample(z: torch.Tensor, tgt_ids: torch.Tensor, stats: dict,
                       feat_ext) -> torch.Tensor:
    """Per-sample S2 energy (audit/scoring form)."""
    with torch.enable_grad():
        _, feats = feat_ext(z)
    B = z.shape[0]
    out = z.new_zeros(B)
    for scale, ft in feats.items():
        st_all = s2_stats_of(ft)
        for sid in torch.unique(tgt_ids):
            mask = tgt_ids == sid
            if not mask.any():
                continue
            ss = stats[STYLES[int(sid)]][scale]
            idx = mask.nonzero(as_tuple=True)[0]
            mu_t = ss["mu_mean"].to(z.device).unsqueeze(0)
            sd_t = ss["sd_mean"].to(z.device).unsqueeze(0)
            zmu = (st_all["mu"][idx] - mu_t) / ss["mu_std"].to(z.device).unsqueeze(0)
            zsd = (st_all["sd"][idx] - sd_t) / ss["sd_std"].to(z.device).unsqueeze(0)
            l_ms = zmu.square().sum(dim=1) + zsd.square().sum(dim=1)
            corr_dev = (st_all["corr"][idx] - ss["corr_mean"].to(z.device).unsqueeze(0))
            l_corr = corr_dev.square().sum(dim=(1, 2)) / (ss["n_ch"] ** 2)
            out = out.index_put((idx,), (out[idx] + l_ms + 0.5 * l_corr))
    return out


# ---------------------------------------------------------------------------
# Gradient audit (V7_EVAL_PROTOCOL.md section 6)
# ---------------------------------------------------------------------------

def audit_grad_balance(norms: Dict[str, float]) -> dict:
    vals = {k: float(v) for k, v in norms.items() if float(v) > 0}
    if not vals:
        return {"imbalanced": False, "ratio": 0.0}
    mx, mn = max(vals.values()), min(vals.values())
    ratio = mx / max(mn, 1e-12)
    return {
        "norms": vals,
        "max_over_min": ratio,
        "imbalanced": bool(ratio > 100.0),
        "scale_policy": "per-component per-sample RMS normalization before combining",
    }
