"""V9 diagnostic suite — locate the real bottleneck without retraining.

All experiments reuse the existing checkpoints, the same DCAE and the same frozen
Test600 Impressionism<->Ukiyo_e subset (60 ordered tasks). No new data split.

Experiments
  A1  residual ablation      scale the DiT output by {0, .25, .5, 1, 1.5}
  B1  endpoint sensitivity   t_floor in {.002, .01, .02, .05}
  B2  ODE convergence        heun 8/16/32/64/128, distance to the 128-step result
  B3  drift decomposition    b+, b-, EDM skip and network terms along the trajectory
  C1  DCAE style fidelity    real image -> DCAE -> reconstruction, scored the same way
  D1  train vs test          identical checkpoint on training latents vs test sources
  D2  bidirectional cycle    A->B->A and B->A->B latent error + style scores
  D3  distribution match     per direction: Top-1, margin, CSD-MMD, CSD-SWD

Usage
  python -m sbflow.diagnose a1 --ckpt runs/blog_sbflow/ImpUki_OT0.5/model_step4000.pt --tag OT4k
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
A_STYLE, B_STYLE = "Impressionism", "Ukiyo_e"


# --------------------------------------------------------------------------- #
# shared infrastructure
# --------------------------------------------------------------------------- #
class Ctx:
    """Loaded checkpoint, DCAE, evaluator and frozen tasks."""

    def __init__(self, ckpt: str, repo_root: str = ".", which: str = "raw", device: str = "cuda"):
        from .cache import load_dcae
        from .infer import load_bundle

        self.repo = Path(repo_root).resolve()
        self.device = device
        self.ck, self.model, self.mom, self.mean, self.std = load_bundle(ckpt, device, which)
        self.ae = load_dcae(self.ck["ae"], device)
        self.sf = float(self.ck["ae_scaling_factor"])
        self.eps = float(self.ck["config"]["eps"])
        self.size = 512
        self.tasks = frozen_pair_tasks(self.repo, A_STYLE, B_STYLE)

        import sys
        sys.path.insert(0, str(self.repo))
        from tools.evaluation_v2.features import EvaluatorModels, build_reference_features_cache

        self.ev = EvaluatorModels(device)
        self.refs = build_reference_features_cache(self.ev)
        self.proto = np.asarray(self.ev.proto_csd, dtype=np.float32)
        self.gamma = float(self.refs["rbf_gamma"])
        self._src_cache: Dict[str, dict] = {}

    # ---- latents ---------------------------------------------------------- #
    def z_from_pil(self, pils: Sequence[Image.Image]) -> torch.Tensor:
        x = torch.stack([to_tensor(p) for p in pils]).to(self.device) * 2 - 1
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                z = self.ae.encode(x).latent.float() * self.sf
        return (z - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def decode(self, z: torch.Tensor) -> List[Image.Image]:
        zx = (z * self.std[None, :, None, None] + self.mean[None, :, None, None]) / self.sf
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                im = self.ae.decode(zx).sample.float().clamp(-1, 1)
        u8 = ((im.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        return [Image.fromarray(a) for a in u8]

    def dcae_noop(self, z: torch.Tensor) -> List[Image.Image]:
        return self.decode(z)

    def source_pils(self, tasks: Sequence[dict]) -> List[Image.Image]:
        return [Image.open(self.repo / t["source_rel_path"]).convert("RGB").resize((512, 512), Image.BICUBIC)
                for t in tasks]

    # ---- scoring ---------------------------------------------------------- #
    def score(self, pils: Sequence[Image.Image], tasks: Sequence[dict], src_pils: Sequence[Image.Image]) -> dict:
        """Top-1 / margin / dT / dS / LPIPS / MMD / SWD, grouped by direction."""
        feats = self.ev.extract_image_features(list(pils))
        sf = self.ev.extract_image_features(list(src_pils))
        rows = []
        for j, t in enumerate(tasks):
            tid = STYLES.index(t["target_style"])
            sid = STYLES.index(t["source_style"])
            v = feats["csd"][j].numpy() @ self.proto.T
            sv = sf["csd"][j].numpy() @ self.proto.T
            other = np.delete(v, tid)
            a = torch.from_numpy(np.asarray(pils[j], dtype=np.float32).copy()).permute(2, 0, 1).to(self.device) / 127.5 - 1
            b = torch.from_numpy(np.asarray(src_pils[j], dtype=np.float32).copy()).permute(2, 0, 1).to(self.device) / 127.5 - 1
            with torch.inference_mode():
                lp = float(self.ev.lpips(a[None], b[None]).mean().item())
            rows.append({
                "task_id": t["task_id"],
                "direction": f"{t['source_style']}->{t['target_style']}",
                "target": t["target_style"],
                "csd_dT": float(v[tid] - sv[tid]),
                "csd_dS": float(v[sid] - sv[sid]),
                "target_margin": float(v[tid] - other.max()),
                "target_top1": int(v.argmax() == tid),
                "lpips": lp,
                "csd": feats["csd"][j].numpy(),
                "hf": hf_energy(pils[j]),
            })
        return summarise(rows, self.refs, self.gamma)


def to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0


def hf_energy(im: Image.Image) -> float:
    """Mean absolute Laplacian: a cheap high-frequency / sharpness proxy."""
    a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    lap = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    return float(np.abs(lap).mean())


def mmd2(gen: np.ndarray, real: np.ndarray, gamma: float) -> float:
    x = torch.as_tensor(gen, dtype=torch.float32)
    y = torch.as_tensor(real, dtype=torch.float32)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan")

    def k(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).square())

    xx, yy = k(x, x), k(y, y)
    return float((xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
                 + (yy.sum() - yy.diagonal().sum()) / (m * (m - 1))
                 - 2 * k(x, y).mean())


def swd(gen: np.ndarray, real: np.ndarray, n_proj: int = 128, seed: int = 0) -> float:
    """Sliced Wasserstein distance between two CSD feature clouds.

    For each random direction the two 1-D empirical distributions are compared by
    integrating |F_g^{-1}(q) - F_r^{-1}(q)| over q in [0,1], which handles
    unequal sample counts without numpy's missing `axis` argument.
    """
    g = torch.as_tensor(gen, dtype=torch.float32)
    r = torch.as_tensor(real, dtype=torch.float32)
    if len(g) < 2 or len(r) < 2:
        return float("nan")
    d = g.shape[1]
    gen_ = torch.Generator().manual_seed(seed)
    proj = F.normalize(torch.randn(d, n_proj, generator=gen_), dim=0)
    pg = torch.sort(g @ proj, dim=0).values.numpy()   # (n, P) sorted per projection
    pr = torch.sort(r @ proj, dim=0).values.numpy()   # (m, P)
    q = np.linspace(0.0, 1.0, 256)
    xg = np.linspace(0.0, 1.0, pg.shape[0])
    xr = np.linspace(0.0, 1.0, pr.shape[0])
    total = 0.0
    for p in range(n_proj):
        total += float(np.abs(np.interp(q, xg, pg[:, p]) - np.interp(q, xr, pr[:, p])).mean())
    return total / n_proj


def summarise(rows: List[dict], refs: dict, gamma: float) -> dict:
    out = {"n": len(rows), "by_direction": {}, "overall": {}}
    keys = ("csd_dT", "csd_dS", "target_margin", "target_top1", "lpips", "hf")
    groups = {"overall": rows}
    for d in sorted({r["direction"] for r in rows}):
        groups[d] = [r for r in rows if r["direction"] == d]
    for gname, sub in groups.items():
        res = {k: float(np.mean([r[k] for r in sub])) for k in keys}
        tgt = sub[0]["target"]
        gen = np.stack([r["csd"] for r in sub])
        real = refs["styles"][tgt]["csd"].numpy()
        res["csd_mmd2"] = mmd2(gen, real, gamma)
        res["csd_swd"] = swd(gen, real)
        res["n"] = len(sub)
        if gname == "overall":
            out["overall"] = res
        else:
            out["by_direction"][gname] = res
    return out


def frozen_pair_tasks(repo: Path, a: str, b: str) -> List[dict]:
    raw = json.loads((repo / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))
    tasks = [t for t in raw["tasks"]
             if t["source_style"] in (a, b) and t["target_style"] in (a, b)
             and t["source_style"] != t["target_style"]]
    for t in tasks:
        if "source_rel_path" not in t:
            t["source_rel_path"] = f"data/test/{t['source_style']}/{t['source_filename']}"
    return tasks


# --------------------------------------------------------------------------- #
# instrumented probability-flow ODE
# --------------------------------------------------------------------------- #
def _vel(ctx: Ctx, z: torch.Tensor, t_phys: torch.Tensor, direction: str,
         t_floor: float, residual_scale: float = 1.0, probe: Optional[dict] = None,
         step_idx: int = 0) -> torch.Tensor:
    from .preconditioning import precondition_input, reconstruct_drift

    t = t_phys.clamp(t_floor, 1 - t_floor)
    if direction == "AtoB":
        mp, mm, dp, dm = ctx.mom, ctx.mom.reversed(), 1, 0
    else:
        mp, mm, dp, dm = ctx.mom.reversed(), ctx.mom, 0, 1
    xp, cp = precondition_input(z, t, ctx.eps, mp)
    xm, cm = precondition_input(z, 1 - t, ctx.eps, mm)
    raw_p = ctx.model(xp, t, torch.full_like(t, dp, dtype=torch.long)).float() * residual_scale
    raw_m = ctx.model(xm, 1 - t, torch.full_like(t, dm, dtype=torch.long)).float() * residual_scale
    bp = reconstruct_drift(z, raw_p, cp)
    bm = reconstruct_drift(z, raw_m, cm)
    if probe is not None:
        mxp, myp, cip, csp, cop = cp
        mxm, mym, cim, csm, com = cm
        probe.setdefault(step_idx, {
            "t": float(t.mean()),
            "bplus_rms": float(bp.pow(2).mean().sqrt()),
            "bminus_rms": float(bm.pow(2).mean().sqrt()),
            "pf_rms": float(((bp - bm) * 0.5).pow(2).mean().sqrt()),
            "skip_plus_rms": float((csp * (z - mxp)).pow(2).mean().sqrt()),
            "skip_minus_rms": float((csm * (z - mxm)).pow(2).mean().sqrt()),
            "net_plus_rms": float((cop * raw_p).pow(2).mean().sqrt()),
            "net_minus_rms": float((com * raw_m).pow(2).mean().sqrt()),
            "mu_y_plus_rms": float(myp.pow(2).mean().sqrt()),
            "mu_y_minus_rms": float(mym.pow(2).mean().sqrt()),
        })
    return (bp - bm) * 0.5


@torch.inference_mode()
def solve(ctx: Ctx, z: torch.Tensor, direction: str, steps: int = 4, method: str = "heun",
          t_floor: float = 0.002, residual_scale: float = 1.0,
          probe: Optional[dict] = None) -> torch.Tensor:
    h = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((len(z),), i * h, device=z.device)
        k0 = _vel(ctx, z, t0, direction, t_floor, residual_scale, probe, i * 2)
        if method == "euler":
            z = z + h * k0
        else:
            t1 = torch.full_like(t0, (i + 1) * h)
            k1 = _vel(ctx, z + h * k0, t1, direction, t_floor, residual_scale, probe, i * 2 + 1)
            z = z + h * 0.5 * (k0 + k1)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f"non-finite state at step {i}")
    return z


def direction_of(t: dict) -> str:
    return "AtoB" if t["source_style"] == A_STYLE else "BtoA"


def run_all_directions(ctx: Ctx, z: torch.Tensor, tasks: Sequence[dict], **kw) -> torch.Tensor:
    """Solve each task with its own physical direction."""
    out = torch.empty_like(z)
    for d in ("AtoB", "BtoA"):
        pos = [i for i, t in enumerate(tasks) if direction_of(t) == d]
        if pos:
            idx = torch.tensor(pos, device=z.device)
            out[idx] = solve(ctx, z[idx].clone(), d, **kw)
    return out


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def exp_a1(ctx: Ctx, out_dir: Path) -> dict:
    """Residual ablation: how much comes from the analytic EDM drift vs the network."""
    z = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    src = ctx.source_pils(ctx.tasks)
    res = {}
    for scale in (0.0, 0.25, 0.5, 1.0, 1.5):
        zg = run_all_directions(ctx, z, ctx.tasks, steps=4, method="heun", residual_scale=scale)
        sc = ctx.score(ctx.decode(zg), ctx.tasks, src)
        sc["latent_l2"] = float((zg - z).flatten(1).norm(dim=1).mean())
        res[f"scale_{scale}"] = sc
        print(f"  [A1 scale={scale}] top1={sc['overall']['target_top1']:.4f} "
              f"margin={sc['overall']['target_margin']:+.4f} dT={sc['overall']['csd_dT']:+.4f} "
              f"dS={sc['overall']['csd_dS']:+.4f} lpips={sc['overall']['lpips']:.4f} "
              f"latL2={sc['latent_l2']:.2f}", flush=True)
    return res


def exp_b1(ctx: Ctx, out_dir: Path) -> dict:
    """Endpoint sensitivity: t_floor sweep."""
    z = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    src = ctx.source_pils(ctx.tasks)
    res = {}
    for tf in (0.002, 0.01, 0.02, 0.05):
        zg = run_all_directions(ctx, z, ctx.tasks, steps=4, method="heun", t_floor=tf)
        sc = ctx.score(ctx.decode(zg), ctx.tasks, src)
        sc["latent_l2"] = float((zg - z).flatten(1).norm(dim=1).mean())
        sc["nonfinite"] = int((~torch.isfinite(zg)).sum())
        res[f"t_floor_{tf}"] = sc
        print(f"  [B1 t_floor={tf}] top1={sc['overall']['target_top1']:.4f} "
              f"margin={sc['overall']['target_margin']:+.4f} dT={sc['overall']['csd_dT']:+.4f} "
              f"lpips={sc['overall']['lpips']:.4f} latL2={sc['latent_l2']:.2f}", flush=True)
    return res


def exp_b2(ctx: Ctx, out_dir: Path) -> dict:
    """ODE convergence against the 128-step solution."""
    z = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    src = ctx.source_pils(ctx.tasks)
    res = {}
    ref = None
    for steps in (8, 16, 32, 64, 128):
        zg = run_all_directions(ctx, z, ctx.tasks, steps=steps, method="heun")
        if steps == 128:
            ref = zg
        sc = ctx.score(ctx.decode(zg), ctx.tasks, src)
        sc["latent_l2"] = float((zg - z).flatten(1).norm(dim=1).mean())
        res[f"heun_{steps}"] = sc
        print(f"  [B2 heun{steps}] nfe={2*steps} top1={sc['overall']['target_top1']:.4f} "
              f"margin={sc['overall']['target_margin']:+.4f} dT={sc['overall']['csd_dT']:+.4f} "
              f"lpips={sc['overall']['lpips']:.4f} latL2={sc['latent_l2']:.2f}", flush=True)
    if ref is not None:
        for steps in (8, 16, 32, 64):
            zg = run_all_directions(ctx, z, ctx.tasks, steps=steps, method="heun")
            res[f"heun_{steps}"]["dist_to_128"] = float((zg - ref).flatten(1).norm(dim=1).mean())
            res[f"heun_{steps}"]["rel_dist_to_128"] = res[f"heun_{steps}"]["dist_to_128"] / max(
                res["heun_128"]["latent_l2"], 1e-9)
            print(f"  [B2 heun{steps}] dist_to_128={res[f'heun_{steps}']['dist_to_128']:.3f}", flush=True)
    return res


def exp_b3(ctx: Ctx, out_dir: Path) -> dict:
    """Drift decomposition along the trajectory, bucketed by physical time."""
    z = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    res = {}
    for d in ("AtoB", "BtoA"):
        pos = [i for i, t in enumerate(ctx.tasks) if direction_of(t) == d]
        idx = torch.tensor(pos, device=z.device)
        probe: dict = {}
        solve(ctx, z[idx].clone(), d, steps=4, method="heun", probe=probe)
        # bucket by physical time
        buckets = {"0.00-0.25": [], "0.25-0.50": [], "0.50-0.75": [], "0.75-1.00": []}
        for _, rec in sorted(probe.items()):
            t = rec["t"]
            key = ("0.00-0.25" if t < .25 else "0.25-0.50" if t < .5 else "0.50-0.75" if t < .75 else "0.75-1.00")
            buckets[key].append(rec)
        agg = {}
        for k, recs in buckets.items():
            if not recs:
                continue
            agg[k] = {f: float(np.mean([r[f] for r in recs])) for f in recs[0] if f != "t"}
            agg[k]["n"] = len(recs)
        res[d] = agg
        print(f"  [B3 {d}]")
        for k, v in agg.items():
            print(f"     {k}: b+={v['bplus_rms']:.3f} b-={v['bminus_rms']:.3f} pf={v['pf_rms']:.3f} "
                  f"skip+={v['skip_plus_rms']:.3f} net+={v['net_plus_rms']:.3f} "
                  f"skip-={v['skip_minus_rms']:.3f} net-={v['net_minus_rms']:.3f}", flush=True)
    return res


def exp_c1(ctx: Ctx, out_dir: Path) -> dict:
    """DCAE style fidelity: real images of both domains, round-tripped, scored as targets."""
    # Build pseudo-tasks: each real target-domain image is scored against its own domain.
    res = {}
    for style in (A_STYLE, B_STYLE):
        src_tasks = [t for t in ctx.tasks if t["target_style"] == style]
        src = ctx.source_pils(src_tasks)
        real_pils = [Image.open(ctx.repo / t["source_rel_path"]).convert("RGB").resize((512, 512), Image.BICUBIC)
                     for t in src_tasks]
        z = ctx.z_from_pil(real_pils)
        rt = ctx.dcae_noop(z)
        sc = ctx.score(rt, src_tasks, src)
        sc["hf_original"] = float(np.mean([hf_energy(p) for p in real_pils]))
        sc["hf_roundtrip"] = float(np.mean([hf_energy(p) for p in rt]))
        res[f"target_{style}"] = sc
        print(f"  [C1 target={style}] top1={sc['overall']['target_top1']:.4f} "
              f"margin={sc['overall']['target_margin']:+.4f} mmd={sc['overall']['csd_mmd2']:.4f} "
              f"hf {sc['hf_original']:.5f}->{sc['hf_roundtrip']:.5f}", flush=True)
    return res


def exp_d1(ctx: Ctx, out_dir: Path) -> dict:
    """Train vs test: same checkpoint on cached training latents vs frozen test sources."""
    res = {}
    # test side
    zt = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    srct = ctx.source_pils(ctx.tasks)
    zgt = run_all_directions(ctx, zt, ctx.tasks, steps=4, method="heun")
    res["test"] = ctx.score(ctx.decode(zgt), ctx.tasks, srct)
    res["test"]["latent_l2"] = float((zgt - zt).flatten(1).norm(dim=1).mean())

    # train side: reuse cached training latents and build mirror tasks
    import sys
    sys.path.insert(0, str(ctx.repo))
    train_tasks, train_z, train_src = [], [], []
    for tgt_style, src_style in ((B_STYLE, A_STYLE), (A_STYLE, B_STYLE)):
        cache_path = ctx.repo / "runs/blog_sbflow/cache" / f"{src_style}.pt"
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        n = min(30, len(c["ids"]))
        for i in range(n):
            p = Path(c["ids"][i])
            train_tasks.append({
                "task_id": f"TRAIN/{src_style}/{p.stem}/{tgt_style}",
                "source_style": src_style, "target_style": tgt_style,
                "source_rel_path": str(p.relative_to(ctx.repo)),
            })
            train_src.append(Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC))
    train_z = ctx.z_from_pil(train_src)
    zgt2 = run_all_directions(ctx, train_z, train_tasks, steps=4, method="heun")
    res["train"] = ctx.score(ctx.decode(zgt2), train_tasks, train_src)
    res["train"]["latent_l2"] = float((zgt2 - train_z).flatten(1).norm(dim=1).mean())
    for k in ("test", "train"):
        o = res[k]["overall"]
        print(f"  [D1 {k}] n={o['n']} top1={o['target_top1']:.4f} margin={o['target_margin']:+.4f} "
              f"dT={o['csd_dT']:+.4f} mmd={o['csd_mmd2']:.4f} lpips={o['lpips']:.4f}", flush=True)
    return res


def exp_d2(ctx: Ctx, out_dir: Path) -> dict:
    """Bidirectional cycle consistency in latent space."""
    z = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    fwd = torch.empty_like(z)
    for d in ("AtoB", "BtoA"):
        pos = [i for i, t in enumerate(ctx.tasks) if direction_of(t) == d]
        idx = torch.tensor(pos, device=z.device)
        fwd[idx] = solve(ctx, z[idx].clone(), d, steps=4, method="heun")
    back = torch.empty_like(fwd)
    for d in ("AtoB", "BtoA"):
        pos = [i for i, t in enumerate(ctx.tasks) if direction_of(t) == d]
        idx = torch.tensor(pos, device=z.device)
        inv = "BtoA" if d == "AtoB" else "AtoB"
        back[idx] = solve(ctx, fwd[idx].clone(), inv, steps=4, method="heun")
    err = (back - z).flatten(1).norm(dim=1)
    res = {
        "cycle_l2_mean": float(err.mean()),
        "cycle_l2_median": float(err.median()),
        "cycle_rel": float(err.mean() / (fwd - z).flatten(1).norm(dim=1).mean()),
        "one_way_l2_mean": float((fwd - z).flatten(1).norm(dim=1).mean()),
        "n": int(len(z)),
    }
    print(f"  [D2] one_way_l2={res['one_way_l2_mean']:.3f} cycle_l2={res['cycle_l2_mean']:.3f} "
          f"rel={res['cycle_rel']:.3f}", flush=True)
    return res


def exp_d3(ctx: Ctx, out_dir: Path) -> dict:
    """Per-direction distribution match (Top-1, margin, MMD, SWD) plus the no-op reference."""
    z = ctx.z_from_pil(ctx.source_pils(ctx.tasks))
    src = ctx.source_pils(ctx.tasks)
    zg = run_all_directions(ctx, z, ctx.tasks, steps=4, method="heun")
    gen = ctx.score(ctx.decode(zg), ctx.tasks, src)
    noop = ctx.score(ctx.dcae_noop(z), ctx.tasks, src)
    for tag, sc in (("generated", gen), ("dcae_noop", noop)):
        for d, v in sc["by_direction"].items():
            print(f"  [D3 {tag} {d}] top1={v['target_top1']:.4f} margin={v['target_margin']:+.4f} "
                  f"dT={v['csd_dT']:+.4f} dS={v['csd_dS']:+.4f} mmd={v['csd_mmd2']:.4f} "
                  f"swd={v['csd_swd']:.4f} hf={v['hf']:.5f}", flush=True)
    return {"generated": gen, "dcae_noop": noop}


EXPS = {"a1": exp_a1, "b1": exp_b1, "b2": exp_b2, "b3": exp_b3,
        "c1": exp_c1, "d1": exp_d1, "d2": exp_d2, "d3": exp_d3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=sorted(EXPS) + ["all"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V9 diagnose {args.experiments} ckpt={args.ckpt} weights={args.weights} ===", flush=True)
    ctx = Ctx(args.ckpt, args.repo_root, args.weights, device)

    names = sorted(EXPS) if "all" in args.experiments else [e for e in args.experiments if e in EXPS]
    out_path = Path(args.out) if args.out else (
        Path(args.repo_root).resolve() / "runs/blog_sbflow/diagnostics" / f"{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"checkpoint": args.ckpt, "weights": args.weights, "tag": args.tag,
               "protocol": {"tasks": len(ctx.tasks), "eps": ctx.eps, "directions": [A_STYLE + "->" + B_STYLE,
                                                                                   B_STYLE + "->" + A_STYLE]},
               "results": {}}
    for name in names:
        print(f"--- {name.upper()} ---", flush=True)
        payload["results"][name] = EXPS[name](ctx, out_path.parent)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
