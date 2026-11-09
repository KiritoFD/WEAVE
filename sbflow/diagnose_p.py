"""Four diagnostics that separate representation, conditioning and trajectory failures.

P0  EDM analytic term vs DiT learned term
    Compares full Test200 transfer for: trained / EDM-only (residual=0) / shuffled
    condition (EDM pair statistics kept, network style ids replaced). The decisive
    number is dT_trained - dT_EDM_only: if it is near zero, the style response comes
    from the analytic statistical mapping and the network has not learned transport.

P1  SD-VAE round trip per style
    Real images -> VAE -> reconstruction, scored with the same CSD protocol, so VAE
    loss is separated from transport loss. Also reports the generator for comparison.

P2  Real vs generated distribution statistics
    Within-class distance, effective rank, Real-Real SWD as the finite-sample floor,
    Generated-Real SWD, generated diversity and nearest-neighbour distance.

P3  Teacher forcing vs free ODE
    Bridge drift error on real Brownian bridges per time bucket, then the free-ODE
    marginal deviation, to tell local-supervision failure from trajectory drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(REPO))

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
CACHE_FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
               "03_Rococo.pt", "04_Ukiyo_e.pt"]


# --------------------------------------------------------------------------- #
def load_tasks(repo: Path, subset: str = "quick200") -> List[dict]:
    raw = json.loads((repo / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))
    tasks = [t for t in raw["tasks"]
             if t.get("is_cross", t["source_style"] != t["target_style"])]
    if subset == "quick200":
        buckets: Dict[tuple, List[dict]] = {}
        for t in tasks:
            buckets.setdefault((t["source_style"], t["target_style"]), []).append(t)
        picked = []
        for k in sorted(buckets):
            picked.extend(sorted(buckets[k], key=lambda x: x["task_id"])[:10])
        tasks = picked
    for t in tasks:
        t.setdefault("source_rel_path", f"data/test/{t['source_style']}/{t['source_filename']}")
    return tasks


def to_u8(im: torch.Tensor) -> np.ndarray:
    a = (im.detach().float().clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
    return a.round().clip(0, 255).astype(np.uint8)


class Scorer:
    """Reuse the pure-CPU scorer so P0/P1 share the exact metric definitions."""

    def __init__(self, threads: int = 6):
        from sbflow.score_cpu import CpuFeatures, _reference_csd, aggregate
        self.feat = CpuFeatures(REPO, threads)
        self.refs = _reference_csd(REPO, self.feat,
                                  REPO / "eval_protocol/cache/references_v1_csd_cpu.pt")
        self.gamma = float(self.refs["rbf_gamma"])
        self.aggregate = aggregate
        self.proto = self.feat.proto

    def rows(self, gen: Sequence[np.ndarray], src: Sequence[np.ndarray],
             tasks: Sequence[dict], variant: str) -> List[dict]:
        out = []
        for i, t in enumerate(tasks):
            fg = self.feat.features([Image.fromarray(gen[i])])
            fs = self.feat.features([Image.fromarray(src[i])])
            vg = fg["csd"][0] @ self.proto.T
            vs = fs["csd"][0] @ self.proto.T
            tid = STYLES.index(t["target_style"])
            sid = STYLES.index(t["source_style"])
            other = np.delete(vg, tid)
            a = torch.from_numpy(gen[i].copy()).permute(2, 0, 1) / 127.5 - 1
            b = torch.from_numpy(src[i].copy()).permute(2, 0, 1) / 127.5 - 1
            out.append({
                "task_id": t["task_id"],
                "direction": f"{t['source_style']}->{t['target_style']}",
                "source": t["source_style"], "target": t["target_style"], "variant": variant,
                "csd_dT": float(vg[tid] - vs[tid]), "csd_dS": float(vg[sid] - vs[sid]),
                "target_margin": float(vg[tid] - other.max()),
                "target_top1": int(vg.argmax() == tid),
                "lpips_to_input": self.feat._lpips(a[None], b[None]),
                "dino_c": float(F.cosine_similarity(torch.as_tensor(fg["dino"]),
                                                    torch.as_tensor(fs["dino"]), dim=1)[0]),
                "csd_embedding": fg["csd"][0].tolist(),
            })
        return out

    def summarise(self, rows: List[dict]) -> dict:
        a = self.aggregate(rows, self.refs, self.gamma)
        return a


# --------------------------------------------------------------------------- #
# P0
# --------------------------------------------------------------------------- #
def p0_edm_vs_network(ckpt: str, scorer: Scorer, device: str, steps: int = 4):
    from sbflow.infer5 import load_model5, solve_transfer
    from sbflow.vae import decode_latents, encode_images, load_sdvae
    from sbflow.data5 import load_caches

    ck, model, stats = load_model5(ckpt, device, "raw")
    vae, trt = load_sdvae(device)
    sf = float(ck["ae_scaling_factor"])
    eps = float(ck["config"]["eps"])
    objective = ck.get("objective", "bidir")
    data = load_caches([Path(f"{CACHE_ROOT}/{f}") for f in CACHE_FILES], device, STYLES)
    gmean, gstd = stats.global_mean, stats.global_std

    tasks = load_tasks(REPO, "quick200")
    uniq = sorted({t["source_rel_path"] for t in tasks})
    zsrc = {}
    with torch.inference_mode():
        for s in range(0, len(uniq), 8):
            chunk = uniq[s:s + 8]
            x = torch.stack([torch.from_numpy(np.asarray(
                Image.open(REPO / p).convert("RGB").resize((512, 512)), dtype=np.uint8).copy()
            ).permute(2, 0, 1).float() / 255.0 for p in chunk]).to(device) * 2 - 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                z = encode_images(vae, x, sf)
            z = (z - gmean[None, :, None, None]) / gstd[None, :, None, None]
            for i, p in enumerate(chunk):
                zsrc[p] = z[i:i + 1]

    src_u8 = [np.asarray(Image.open(REPO / t["source_rel_path"]).convert("RGB")
                         .resize((512, 512)), dtype=np.uint8) for t in tasks]
    z = torch.cat([zsrc[t["source_rel_path"]] for t in tasks], 0)

    def run(tag: str, residual_scale: float, shuffle_cond: bool):
        zout = torch.empty_like(z)
        rng = np.random.RandomState(0)
        with torch.inference_mode():
            for src_name in STYLES:
                for tgt_name in STYLES:
                    if src_name == tgt_name:
                        continue
                    pos = [j for j, t in enumerate(tasks)
                           if t["source_style"] == src_name and t["target_style"] == tgt_name]
                    if not pos:
                        continue
                    idx = torch.tensor(pos, device=device)
                    si, ti = STYLES.index(src_name), STYLES.index(tgt_name)
                    ca = cb = None
                    if shuffle_cond:
                        # keep the EDM pair statistics, replace only the network style ids
                        while True:
                            x, y = rng.randint(0, 5), rng.randint(0, 5)
                            if x != y:
                                break
                        ca, cb = x, y
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        zout[idx] = solve_transfer(model, z[idx].clone(), si, ti, stats, eps,
                                                   steps, "heun", 0.002, objective,
                                                   cond_a=ca, cond_b=cb,
                                                   residual_scale=residual_scale)
            dec = decode_latents(vae, trt, (zout * gstd[None, :, None, None]
                                            + gmean[None, :, None, None]) / sf, batch=8)
        u8 = to_u8(dec)
        rows = scorer.rows([a for a in u8], src_u8, tasks, tag)
        agg = scorer.summarise(rows)
        o = agg["overall"]
        print(f"  [{tag:16s}] top1={o['target_top1']:.4f} margin={o['target_margin']:+.4f} "
              f"dT={o['csd_dT']:+.4f} dS={o['csd_dS']:+.4f} swd={o['csd_swd']:.4f} "
              f"dino_c={o['dino_c']:.4f}", flush=True)
        return {"overall": {k: o[k] for k in ("target_top1", "target_margin", "csd_dT",
                                             "csd_dS", "csd_swd", "dino_c", "n")},
                "by_target": {k: {kk: v[kk] for kk in ("target_top1", "csd_dT")}
                              for k, v in agg["by_target"].items()}}

    print("\n=== P0: EDM analytic term vs DiT learned term (Quick200) ===", flush=True)
    res = {"trained": run("trained", 1.0, False),
           "edm_only": run("edm_only", 0.0, False),
           "cond_shuffle": run("cond_shuffle", 1.0, True)}
    d_tr = res["trained"]["overall"]["csd_dT"]
    d_edm = res["edm_only"]["overall"]["csd_dT"]
    res["delta_dT_network_contribution"] = d_tr - d_edm
    res["trained_over_edm_ratio"] = (d_tr / d_edm) if abs(d_edm) > 1e-9 else float("nan")
    print(f"\n  dT_trained - dT_EDM_only = {d_tr - d_edm:+.4f} "
          f"(trained {d_tr:+.4f}, edm-only {d_edm:+.4f})")
    print(f"  dT_shuffle = {res['cond_shuffle']['overall']['csd_dT']:+.4f} "
          f"(vs trained {d_tr:+.4f})")
    return res


# --------------------------------------------------------------------------- #
# P1
# --------------------------------------------------------------------------- #
def p1_vae_roundtrip(scorer: Scorer, device: str, per_style: int = 100):
    """Real image vs its SD-VAE reconstruction, scored identically."""
    from sbflow.vae import decode_latents, encode_images, load_sdvae

    vae, trt = load_sdvae(device)
    sf = 0.18215
    print("\n=== P1: SD-VAE round trip per style ===", flush=True)
    out = {}
    for s in STYLES:
        files = sorted((REPO / "data/test" / s).glob("*.jpg"))[:per_style]
        if not files:
            files = sorted((REPO / "data/test" / s).glob("*.png"))[:per_style]
        orig = [np.asarray(Image.open(p).convert("RGB").resize((512, 512),
                                                               Image.Resampling.BICUBIC),
                           dtype=np.uint8) for p in files]
        x = torch.stack([torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.0
                         for a in orig]).to(device) * 2 - 1
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                z = encode_images(vae, x, sf)
            rec = decode_latents(vae, trt, z, batch=8)
        rec_u8 = to_u8(rec)
        f_o = scorer.feat.features([Image.fromarray(a) for a in orig])
        f_r = scorer.feat.features([Image.fromarray(a) for a in rec_u8])
        vo = f_o["csd"] @ scorer.proto.T
        vr = f_r["csd"] @ scorer.proto.T
        tid = STYLES.index(s)
        top1_o = float((vo.argmax(1) == tid).mean())
        top1_r = float((vr.argmax(1) == tid).mean())
        margin_o = float(np.mean(vo[:, tid] - np.delete(vo, tid, 1).max(1)))
        margin_r = float(np.mean(vr[:, tid] - np.delete(vr, tid, 1).max(1)))
        from sbflow.score_cpu import mmd2, swd
        real = scorer.refs["styles"][s]["csd"]
        out[s] = {"n": len(files), "top1_original": top1_o, "top1_roundtrip": top1_r,
                  "margin_original": margin_o, "margin_roundtrip": margin_r,
                  "mmd_original": mmd2(f_o["csd"], real, scorer.gamma),
                  "mmd_roundtrip": mmd2(f_r["csd"], real, scorer.gamma),
                  "swd_original": swd(f_o["csd"], real),
                  "swd_roundtrip": swd(f_r["csd"], real),
                  "dino_c": float(F.cosine_similarity(
                      torch.as_tensor(f_o["dino"]), torch.as_tensor(f_r["dino"]), dim=1).mean())}
        print(f"  {s:22s} top1 {top1_o:.3f} -> {top1_r:.3f} | margin {margin_o:+.4f} -> "
              f"{margin_r:+.4f} | MMD {out[s]['mmd_original']:.4f} -> "
              f"{out[s]['mmd_roundtrip']:.4f} | DINO-C {out[s]['dino_c']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# P2
# --------------------------------------------------------------------------- #
def p2_distribution(scorer: Scorer, ckpt: str, device: str):
    """Within-class spread and real-vs-generated distance in CSD space."""
    from sbflow.score_cpu import mmd2, swd
    from sbflow.data5 import load_caches

    print("\n=== P2: real vs generated distribution statistics (CSD space) ===", flush=True)
    gen_path = REPO / "runs/blog_sbflow/v10/score_04M_quick200/style_scores.json"
    out = {}
    gen_rows = json.loads(gen_path.read_text())["rows"] if gen_path.exists() else []
    for s in STYLES:
        real = scorer.refs["styles"][s]["csd"]
        # two independent halves of the reference pool give the finite-sample floor
        half = len(real) // 2
        floor = swd(real[:half], real[half:])
        gen = np.asarray([r["csd_embedding"] for r in gen_rows if r["target"] == s])
        if len(gen) < 4:
            print(f"  {s:22s} not enough generated samples")
            continue
        def spread(x):
            c = x.mean(0)
            return float(np.linalg.norm(x - c, axis=1).mean())
        def eff_rank(x):
            xc = x - x.mean(0)
            sv = np.linalg.svd(xc, compute_uv=False)
            p = sv ** 2 / max((sv ** 2).sum(), 1e-12)
            return float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        d_rr = swd(real, real)
        out[s] = {
            "n_real": len(real), "n_gen": len(gen),
            "spread_real": spread(real), "spread_gen": spread(gen),
            "eff_rank_real": eff_rank(real), "eff_rank_gen": eff_rank(gen),
            "swd_real_real_floor": floor,
            "swd_gen_real": swd(gen, real),
            "mmd_gen_real": mmd2(gen, real, scorer.gamma),
        }
        print(f"  {s:22s} spread {out[s]['spread_real']:.3f}->{out[s]['spread_gen']:.3f} | "
              f"effRank {out[s]['eff_rank_real']:.2f}->{out[s]['eff_rank_gen']:.2f} | "
              f"SWD floor {floor:.4f} gen-real {out[s]['swd_gen_real']:.4f} | "
              f"MMD {out[s]['mmd_gen_real']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# P3
# --------------------------------------------------------------------------- #
def p3_teacher_vs_free(ckpt: str, device: str, n_batches: int = 12, batch: int = 32):
    """Bridge drift error (teacher forcing) vs free-ODE marginal deviation."""
    from sbflow.bridge import bridge_sample
    from sbflow.infer5 import load_model5, probability_flow_velocity, solve_transfer
    from sbflow.preconditioning import (moment_coefficients, normalized_target,
                                        precondition_input)
    from sbflow.data5 import load_caches

    ck, model, stats = load_model5(ckpt, device, "raw")
    eps = float(ck["config"]["eps"])
    objective = ck.get("objective", "bidir")
    data = load_caches([Path(f"{CACHE_ROOT}/{f}") for f in CACHE_FILES], device, STYLES)
    pairs = data.pairs()
    g = torch.Generator(device=device).manual_seed(0)

    print("\n=== P3: teacher forcing vs free ODE ===", flush=True)
    buckets = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
    acc = {b: {"err": 0.0, "tgt": 0.0, "n": 0} for b in buckets}
    free = {}
    with torch.inference_mode():
        for (a, b) in pairs:
            ia = torch.randint(0, data.counts[a], (batch,), generator=g)
            ib = torch.randint(0, data.counts[b], (batch,), generator=g)
            z0, z1 = data.z[a][ia], data.z[b][ib]
            t = torch.rand(batch, device=device, generator=g) * 0.96 + 0.02
            xi = torch.randn(z0.shape, device=device, generator=g, dtype=z0.dtype)
            xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)
            pm = stats.pair(a, b)
            half = batch // 2
            cf = moment_coefficients(t[:half], eps, pm)
            cb = moment_coefficients(1 - t[half:], eps, pm.reversed())
            xf, cf = precondition_input(xt[:half], t[:half], eps, pm)
            xb, cb = precondition_input(xt[half:], 1 - t[half:], eps, pm.reversed())
            yf = normalized_target(xt[:half], bplus[:half], cf)
            yb = normalized_target(xt[half:], bminus[half:], cb)
            tau = torch.cat([t[:half], 1 - t[half:]], 0)
            d = torch.cat([torch.ones(half, device=device, dtype=torch.long),
                           torch.zeros(batch - half, device=device, dtype=torch.long)])
            sa = torch.full((batch,), a, device=device, dtype=torch.long)
            sb = torch.full((batch,), b, device=device, dtype=torch.long)
            pred = model(torch.cat([xf, xb], 0), tau, d, sa, sb).float()
            y = torch.cat([yf, yb], 0).float()
            err = (pred - y).square().mean(dim=(1, 2, 3))
            tt = tau
            for i, bk in enumerate(buckets):
                lo, hi = [0.0, 0.25, 0.5, 0.75][i], [0.25, 0.5, 0.75, 1.01][i]
                m = (tt >= lo) & (tt < hi)
                if m.any():
                    acc[bk]["err"] += float(err[m].sum())
                    acc[bk]["tgt"] += float(y[m].square().mean(dim=(1, 2, 3)).sum())
                    acc[bk]["n"] += int(m.sum())
            # free ODE from the real source, then compare marginal statistics
            zout = solve_transfer(model, z0.clone(), a, b, stats, eps, 4, "heun", 0.002,
                                  objective)
            free[(a, b)] = (float((zout - z0).flatten(1).norm(dim=1).mean()),
                            float((z1 - z0).flatten(1).norm(dim=1).mean()))

    for bk in buckets:
        v = acc[bk]
        if v["n"]:
            print(f"  {bk}: normalised MSE {v['err']/v['n']:.4f} "
                  f"(target var {v['tgt']/v['n']:.4f}, n={v['n']})", flush=True)
    shifts = np.mean([f[0] for f in free.values()])
    targets = np.mean([f[1] for f in free.values()])
    print(f"  free-ODE displacement {shifts:.3f} vs real endpoint distance {targets:.3f} "
          f"(ratio {shifts/max(targets,1e-9):.3f})")
    return {"teacher": {k: {"mse": v["err"]/max(v["n"], 1), "n": v["n"]}
                        for k, v in acc.items() if v["n"]},
            "free_ode_shift": shifts, "real_endpoint_distance": targets,
            "ratio": shifts / max(targets, 1e-9)}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/blog_sbflow/v10/exp_04M/model_step6000.pt")
    ap.add_argument("--tests", default="p0,p1,p2,p3")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="runs/blog_sbflow/v10/diagnostics_04M.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    want = [t.strip() for t in args.tests.split(",") if t.strip()]
    ckpt = str(REPO / args.ckpt)
    scorer = Scorer(args.threads)
    print(f"[diag] ckpt={args.ckpt} tests={want} device={device}", flush=True)

    out = {}
    if "p0" in want:
        out["p0"] = p0_edm_vs_network(ckpt, scorer, device)
    if "p1" in want:
        out["p1"] = p1_vae_roundtrip(scorer, device)
    if "p2" in want:
        out["p2"] = p2_distribution(scorer, ckpt, device)
    if "p3" in want:
        out["p3"] = p3_teacher_vs_free(ckpt, device)

    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
