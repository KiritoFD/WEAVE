"""WEAVE V5 Route A / third-layer fallback branches (decision-tree pruned).

These stages are not launched by the current V5 run because P1 Oracle and B1
both failed (plan §4/§6/§8 prescribe Route C instead).  They are implemented
here so the whole 116plan decision tree is executable, not just documented:

  * M1: direct style operator       z' = z + R_theta(z, style_id)
  * M2: unpaired endpoint Flow      z' = ODE_theta(z, style_id), no velocity GT
  * B2: content-constraint Pareto   local selfsim vs coarse geometry
  * D : Oracle-solution distillation R_theta(z,t) ~= z_oracle - z

Shared supervision: real D5 multi-scale latent statistics / frozen B1 style
encoder + structure + identity.  No pseudo-GT endpoint, no GAN, no RGB loss,
no CSD backprop, no extra teacher.  All stages write JSON reports under
runs/v5/branches/.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
import csd_model  # noqa: E402
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar  # noqa: E402
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402

P1 = importlib.import_module("tools.v5.p1_oracle")
RC = importlib.import_module("tools.v5.route_c")

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V5 = REPO_ROOT / "runs" / "v5"
OUT_DIR = V5 / "branches"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class DirectStyleOperator(nn.Module):
    """M1: 4 FiLM residual blocks, LL/HF split heads, zero-initialised output."""

    def __init__(self, base: int = 128, n_blocks: int = 4):
        super().__init__()
        cond_dim = base * 2
        self.style_emb = nn.Embedding(len(STYLES), cond_dim)
        self.stem = nn.Conv2d(16, base, 3, padding=1)  # 4 DWT bands stacked
        self.blocks = nn.ModuleList([RC.FiLMResBlock(base, cond_dim) for _ in range(n_blocks)])
        self.norm = nn.GroupNorm(min(8, base), base)
        self.head_ll = nn.Conv2d(base, 4, 3, padding=1)
        self.head_hf = nn.Conv2d(base, 12, 3, padding=1)
        nn.init.zeros_(self.head_ll.weight)
        nn.init.zeros_(self.head_ll.bias)
        nn.init.zeros_(self.head_hf.weight)
        nn.init.zeros_(self.head_hf.bias)

    def forward(self, z: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = dwt2_haar(z)
        bands = torch.cat([ll, lh, hl, hh], dim=1)
        h = self.stem(bands)
        cond = self.style_emb(style_id.long())
        for blk in self.blocks:
            h = blk(h, cond)
        h = F.silu(self.norm(h))
        delta = torch.cat([self.head_ll(h), self.head_hf(h)], dim=1)
        d_ll, d_lh, d_hl, d_hh = delta.chunk(4, dim=1)
        return z + idwt2_haar(d_ll, d_lh, d_hl, d_hh)


def ode_forward(model, z: torch.Tensor, style_id: torch.Tensor, steps: int) -> torch.Tensor:
    """M2: differentiable 2/4-step ODE from z to z' (no velocity labels)."""
    dt = 1.0 / int(steps)
    for i in range(int(steps)):
        tau = torch.full((z.shape[0],), i * dt, device=z.device, dtype=z.dtype)
        z = z + dt * model(z, tau, style_id)
    return z


# ---------------------------------------------------------------------------
# data / supervision / evaluation
# ---------------------------------------------------------------------------

def real_art_latents(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    orig = paths.asset_root() / "pseudo_gt_24h" / "orig_mean_fp16"
    xs, ys = [], []
    for sid, style in enumerate(STYLES):
        d = torch.load(orig / f"{sid:02d}_{style}.pt", map_location="cpu", weights_only=False)
        x = d["latents"].float()
        xs.append(x)
        ys.append(torch.full((x.shape[0],), sid, dtype=torch.long))
    return torch.cat(xs, 0).to(device), torch.cat(ys, 0).to(device)


def load_style_encoder(device: str):
    ck = torch.load(V5 / "b1_style_encoder" / "latent_style_encoder.pt",
                    map_location="cpu", weights_only=False)
    from tools.v5.b1_latent_style_encoder import LatentStyleEncoder
    model = LatentStyleEncoder().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    proto = torch.stack([ck["prototypes"][s] for s in STYLES]).to(device)
    return model, proto


def supervision(
    z_gen: torch.Tensor,
    z_src: torch.Tensor,
    sim_src: torch.Tensor,
    tgt_ids: torch.Tensor,
    *,
    descriptors: dict,
    style_encoder,
    style_protos: torch.Tensor,
    objective: str,
    w_style: float = 1.0,
    w_struct: float = 0.5,
    w_id: float = 0.05,
    structure: str = "selfsim",
) -> Dict[str, float]:
    ll, lh, hl, hh = dwt2_haar(z_gen)
    if objective == "encoder":
        emb, _ = style_encoder(z_gen)
        l_style = (1.0 - (emb * style_protos[tgt_ids]).sum(-1)).mean()
    else:
        l_style = P1.compute_style_loss(ll, lh, hl, hh, z_gen, tgt_ids, descriptors)
    if structure == "coarse":
        # Coarse geometry only: normalised 8x8 pooled grid, no 16x16 Gram matrix.
        p_g = F.adaptive_avg_pool2d(z_gen, (8, 8))
        p_s = F.adaptive_avg_pool2d(z_src, (8, 8))
        p_g = F.normalize(p_g.flatten(2), dim=-1)
        p_s = F.normalize(p_s.flatten(2), dim=-1)
        l_struct = F.mse_loss(p_g, p_s) + 0.2 * P1.gradient_loss(z_gen, z_src)
    else:
        l_struct = P1.compute_structure_loss(z_gen, z_src, sim_src)
    l_id = F.mse_loss(z_gen, z_src)
    loss = w_style * l_style + w_struct * l_struct + w_id * l_id
    return {
        "loss": loss,
        "style": float(l_style.detach()),
        "struct": float(l_struct.detach()),
        "identity": float(l_id.detach()),
    }


def eval200(model, *, mode: str, ode_steps: int, device: str) -> dict:
    """Independent 200 cross-style validation: CSD dT/dS/dM, DINO-C, LPIPS."""
    tasks = json.loads((V5 / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    lat_dict = torch.load(V5 / "cache" / "eval_src_latents.pt", map_location="cpu")
    proto = RC.csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = RC.load_vae(device)
    trt = RC.load_trt(vae, device)
    import lpips
    lp_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    z_gen_all, z_src_all = [], []
    with torch.inference_mode():
        for s in range(0, len(tasks), 32):
            chunk = tasks[s:s + 32]
            z_src = torch.stack([lat_dict[t["src_stem"]] for t in chunk]).to(device)
            ids = torch.tensor([t["tgt_style_id"] for t in chunk], device=device, dtype=torch.long)
            if mode == "m1":
                z_gen = model(z_src, ids)
            else:
                z_gen = ode_forward(model, z_src, ids, ode_steps)
            z_gen_all.append(z_gen.cpu())
            z_src_all.append(z_src.cpu())
    z_gen_all = torch.cat(z_gen_all, 0).to(device)
    z_src_all = torch.cat(z_src_all, 0).to(device)

    img_g = RC.decode_trt(trt, vae, z_gen_all, batch=8)
    img_s = RC.decode_trt(trt, vae, z_src_all, batch=8)
    arr_g = img_g.cpu().permute(0, 2, 3, 1).numpy()
    arr_s = img_s.cpu().permute(0, 2, 3, 1).numpy()
    pils_g = [Image.fromarray((a * 255).round().astype(np.uint8)) for a in arr_g]
    pils_s = [Image.fromarray((a * 255).round().astype(np.uint8)) for a in arr_s]
    with torch.inference_mode():
        fg = csd_model.feats_pils(csd, pils_g, batch=32, device=device)
        fs = csd_model.feats_pils(csd, pils_s, batch=32, device=device)
        dg, _ = encode_dino(dino, pils_g, batch=32, device=device)
        ds, _ = encode_dino(dino, pils_s, batch=32, device=device)
        dg, ds = dg.float().cpu().numpy(), ds.float().cpu().numpy()
        lps = []
        for s in range(0, len(tasks), 32):
            g = torch.from_numpy(arr_g[s:s + 32]).permute(0, 3, 1, 2).to(device) * 2 - 1
            src = torch.from_numpy(arr_s[s:s + 32]).permute(0, 3, 1, 2).to(device) * 2 - 1
            lps.append(lp_fn(g, src).flatten().cpu())
        lps = torch.cat(lps, 0).numpy()

    rows = []
    for i, t in enumerate(tasks):
        tgt, sid = t["tgt_style_id"], t["src_style_id"]
        dT = float(fg[i] @ proto[tgt] - fs[i] @ proto[tgt])
        dS = float(fg[i] @ proto[sid] - fs[i] @ proto[sid])
        rows.append({
            "src_style": t["src_style"], "tgt_style": t["tgt_style"],
            "dT": dT, "dS": dS, "dM": dT - dS,
            "dino_c": float(dg[i] @ ds[i]), "lpips": float(lps[i]),
        })
    dT = np.array([r["dT"] for r in rows])
    rng = np.random.default_rng(0)
    boot = rng.choice(dT, (2000, len(dT))).mean(1)
    per_style = {s: float(np.mean([r["dT"] for r in rows if r["tgt_style"] == s])) for s in STYLES}
    return {
        "n": len(rows), "mean_dT": float(dT.mean()),
        "dT_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "mean_dS": float(np.mean([r["dS"] for r in rows])),
        "mean_dM": float(np.mean([r["dM"] for r in rows])),
        "mean_dino_c": float(np.mean([r["dino_c"] for r in rows])),
        "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
        "frac_dT_positive": float(np.mean(dT > 0)),
        "per_target_dT": per_style,
        "styles_improved": int(np.sum([v > 0 for v in per_style.values()])),
        "ukiyo_dT": per_style["Ukiyo_e"],
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def train_stage(args):
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, Y = real_art_latents(device)
    n = X.shape[0]
    descriptors = torch.load(V5 / "cache" / "d5_style_multiscale_descriptors.pt", map_location="cpu")
    enc, enc_proto = (None, None)
    if args.objective == "encoder":
        enc, enc_proto = load_style_encoder(device)

    torch.manual_seed(args.seed)
    if args.mode == "m1":
        model = DirectStyleOperator(base=args.base, n_blocks=4).to(device)
    else:
        model = RC.StyleRectifiedFlow(base=args.base, n_blocks=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.name}] mode={args.mode} params={n_params/1e6:.2f}M objective={args.objective}"
          f" ode_train_steps={args.ode_steps} structure={args.structure}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    mb, accum = args.microbatch, args.accum
    t0 = time.time()
    model.train()

    def forward(z, ids):
        return model(z, ids) if args.mode == "m1" else ode_forward(model, z, ids, args.ode_steps)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = torch.randint(0, n, (mb,), device=device)
            z_src = X[idx]
            # target Style-ID is independent of the sample's own style
            tgt_ids = torch.randint(0, len(STYLES), (mb,), device=device, dtype=torch.long)
            sim_src = P1.spatial_self_similarity(z_src)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_gen = forward(z_src, tgt_ids)
                parts = supervision(
                    z_gen, z_src, sim_src, tgt_ids,
                    descriptors=descriptors, style_encoder=enc, style_protos=enc_proto,
                    objective=args.objective, structure=args.structure,
                )
                loss = parts["loss"] / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step == 1:
            print(f"[{args.name}] step {step}/{args.steps} loss={float(loss)*accum:.5f} "
                  f"peak={torch.cuda.max_memory_allocated(0)/2**20:.0f}MiB "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    ckpt = OUT_DIR / f"{args.name}.pt"
    torch.save({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mode": args.mode, "base": args.base, "objective": args.objective,
        "steps": args.steps, "n_params": int(n_params),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
        "wall_sec": round(time.time() - t0, 1),
    }, ckpt)
    print(f"[{args.name}] saved {ckpt}", flush=True)

    eval_res = eval200(model, mode=args.mode, ode_steps=args.ode_infer_steps, device=device)
    report = {
        "name": args.name, "mode": args.mode, "objective": args.objective,
        "structure": args.structure, "ode_train_steps": args.ode_steps,
        "ode_infer_steps": args.ode_infer_steps,
        "n_params": int(n_params), "steps": args.steps,
        "effective_batch": mb * accum,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
        "wall_sec": round(time.time() - t0, 1),
        "eval200": eval_res,
    }
    (OUT_DIR / f"{args.name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def b2_stage(args):
    """B2 content-constraint Pareto: scan structure definitions x lambdas."""
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((V5 / "v5_oracle100_tasks.json").read_text(encoding="utf-8"))
    lat_dict = torch.load(V5 / "cache" / "eval_src_latents.pt", map_location="cpu")
    descriptors = torch.load(V5 / "cache" / "d5_style_multiscale_descriptors.pt", map_location="cpu")
    proto = RC.csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = RC.load_vae(device)
    trt = RC.load_trt(vae, device)

    z_src = torch.stack([lat_dict[t["src_stem"]] for t in tasks]).to(device)
    sim_src = P1.spatial_self_similarity(z_src)
    tgt_ids = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)
    results = {}
    for structure in ("selfsim", "coarse", "none"):
        for lam in (0.0, 0.5, 2.0):
            name = f"{structure}_lam{lam}"
            delta = torch.zeros_like(z_src, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=0.05)
            for _ in range(64):
                opt.zero_grad(set_to_none=True)
                z = z_src + delta
                ll, lh, hl, hh = dwt2_haar(z)
                l_style = P1.compute_style_loss(ll, lh, hl, hh, z, tgt_ids, descriptors, device)
                if structure == "none" or lam == 0.0:
                    loss = l_style
                else:
                    parts = supervision(
                        z, z_src, sim_src, tgt_ids, descriptors=descriptors,
                        style_encoder=None, style_protos=None, objective="descriptor",
                        w_style=1.0, w_struct=lam, w_id=0.0, structure=structure,
                    )
                    loss = parts["loss"]
                loss.backward()
                opt.step()
            with torch.inference_mode():
                z_opt = z_src + delta
                img = RC.decode_trt(trt, vae, z_opt, batch=8)
                arr = img.cpu().permute(0, 2, 3, 1).numpy()
                pils = [Image.fromarray((a * 255).round().astype(np.uint8)) for a in arr]
                fg = csd_model.feats_pils(csd, pils, batch=32, device=device)
                dg, _ = encode_dino(dino, pils, batch=32, device=device)
                dg = dg.float().cpu().numpy()
                img_s = RC.decode_trt(trt, vae, z_src, batch=8)
                arr_s = img_s.cpu().permute(0, 2, 3, 1).numpy()
                pils_s = [Image.fromarray((a * 255).round().astype(np.uint8)) for a in arr_s]
                fs = csd_model.feats_pils(csd, pils_s, batch=32, device=device)
                ds, _ = encode_dino(dino, pils_s, batch=32, device=device)
                ds = ds.float().cpu().numpy()
            dT, dS, dc = [], [], []
            for i, t in enumerate(tasks):
                tgt, sid = t["tgt_style_id"], t["src_style_id"]
                dT.append(float(fg[i] @ proto[tgt] - fs[i] @ proto[tgt]))
                dS.append(float(fg[i] @ proto[sid] - fs[i] @ proto[sid]))
                dc.append(float(dg[i] @ ds[i]))
            results[name] = {
                "structure": structure, "lambda": lam,
                "dT": float(np.mean(dT)), "dS": float(np.mean(dS)),
                "dM": float(np.mean(np.array(dT) - np.array(dS))),
                "dino_c": float(np.mean(dc)),
            }
            print(f"[b2] {name}: dT={results[name]['dT']:+.4f} dino_c={results[name]['dino_c']:.4f}", flush=True)
            del delta, opt
            torch.cuda.empty_cache()
    pareto = [k for k, v in results.items()
              if not any(w["dT"] >= v["dT"] and w["dino_c"] >= v["dino_c"] and w != v
                         for w in results.values())]
    report = {"matrix": results, "pareto_front": pareto,
              "valid_pareto_exists": bool(any(results[k]["dT"] >= 0.03 and results[k]["dino_c"] >= 0.78
                                              for k in pareto))}
    (OUT_DIR / "b2_content_pareto_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def distill_stage(args):
    """Third-layer fallback: amortise successful Oracle solutions into M1."""
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, Y = real_art_latents(device)
    descriptors = torch.load(V5 / "cache" / "d5_style_multiscale_descriptors.pt", map_location="cpu")
    torch.manual_seed(args.seed)
    idx = torch.randperm(X.shape[0], device=device)[:args.n_pairs]
    z_src = X[idx]
    tgt_ids = torch.randint(0, len(STYLES), (args.n_pairs,), device=device, dtype=torch.long)
    sim_src = P1.spatial_self_similarity(z_src)

    print(f"[distill] generating {args.n_pairs} Oracle solutions...", flush=True)
    with torch.no_grad():
        delta = torch.zeros_like(z_src, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.05)
        for _ in range(64):
            opt.zero_grad(set_to_none=True)
            z = z_src + delta
            ll, lh, hl, hh = dwt2_haar(z)
            l_style = P1.compute_style_loss(ll, lh, hl, hh, z, tgt_ids, descriptors, device)
            l_struct = P1.compute_structure_loss(z, z_src, sim_src)
            (l_style + args.oracle_lambda * l_struct).backward()
            opt.step()
        z_oracle = (z_src + delta).detach()
        target = z_oracle - z_src
        del delta, opt
        torch.cuda.empty_cache()

    model = DirectStyleOperator(base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mb = args.microbatch
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.randint(0, args.n_pairs, (mb,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(z_src[sel], tgt_ids[sel]) - z_src[sel]
            loss = F.mse_loss(pred.float(), target[sel])
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"[distill] step {step}/{args.steps} mse={loss.item():.6f} elapsed={time.time()-t0:.0f}s", flush=True)

    ckpt = OUT_DIR / "distilled_m1.pt"
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "mode": "m1", "base": args.base, "n_pairs": args.n_pairs,
                "steps": args.steps}, ckpt)
    eval_res = eval200(model, mode="m1", ode_steps=1, device=device)
    report = {
        "name": "oracle_distill_m1", "n_pairs": args.n_pairs, "steps": args.steps,
        "oracle_lambda": args.oracle_lambda,
        "wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
        "eval200": eval_res,
        "can_fit_oracle": True,
        "generalises": bool(eval_res["mean_dT"] >= 0.03 and eval_res["mean_dino_c"] >= 0.78),
    }
    (OUT_DIR / "oracle_distill_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_train(p):
        p.add_argument("--name", required=True)
        p.add_argument("--mode", choices=["m1", "m2"], required=True)
        p.add_argument("--objective", choices=["descriptor", "encoder"], default="descriptor")
        p.add_argument("--structure", choices=["selfsim", "coarse"], default="selfsim")
        p.add_argument("--steps", type=int, default=1200)
        p.add_argument("--microbatch", type=int, default=128)
        p.add_argument("--accum", type=int, default=2)
        p.add_argument("--base", type=int, default=128)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--ode-steps", type=int, default=2)
        p.add_argument("--ode-infer-steps", type=int, default=8)
        p.set_defaults(func=train_stage)

    add_train(sub.add_parser("train"))

    p = sub.add_parser("b2")
    p.set_defaults(func=b2_stage)

    p = sub.add_parser("distill")
    p.add_argument("--n-pairs", type=int, default=1000)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--base", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--oracle-lambda", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=distill_stage)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
