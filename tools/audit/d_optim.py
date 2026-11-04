"""S12 / S13 / S14: false optima, direct latent optimisation, distribution coverage.

S12  do candidate losses prefer degenerate images (constant / blurred /
     whitened / shuffled / copied reference / identical outputs)?
S13  constrained optimisation of the 4x64x64 latent:
        min L_style(z', s)  s.t.  RMS(z' - z_c) <= eps
     no U-Net, no flow model; evaluated ONLY with independent offline
     features (DINO-S / DINO-C / CLIP-S / LPIPS on decoded images).
S14  does the optimised set cover the real style distribution, or collapse
     to a mean / few modes?

Usage: python3 tools/audit/d_optim.py [s12] [s13] [s14]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C
import latent_losses as L
from b_grad import offline_metrics, ref_pools, perturb


def rms_proj(z: torch.Tensor, z0: torch.Tensor, eps: float) -> torch.Tensor:
    r = L.content_rms(z, z0).clamp_min(1e-12)
    scale = torch.where(r > eps, eps / r, torch.ones_like(r)).view(-1, 1, 1, 1)
    return z0 + (z - z0) * scale


def run_s12(loss: L.LatentStyleLoss, z_ref: torch.Tensor, zc: torch.Tensor) -> dict:
    out: dict = {"n_content": int(zc.shape[0]), "per_loss": {}}
    real = {s: z_ref[s][:32] for s in range(5)}
    mu = zc.mean(dim=(0, 2, 3), keepdim=True)
    sd = zc.std(dim=(0, 2, 3), keepdim=True)
    for k in L.KINDS:
        rows = {}
        for s in range(5):
            # sampling floor: loss on 16 real target-style latents
            floor = float(loss(k, real[s][:16], s))
            cand = {
                "real_target_16": floor,
                "content_identity": float(loss(k, zc[:16], s)),
                "const_mean": float(loss(k, mu.expand(zc[:16].shape), s)),
                "blur_ll1": float(loss(k, perturb(zc[:16], "blur_ll1"), s)),
                "whiten": float(loss(k, ((zc[:16] - mu) / sd.clamp_min(1e-6)), s)),
                "patch_shuffle": float(loss(k, perturb(zc[:16], "patch_shuffle"), s)),
                "copy_ref_multi": float(loss(k, real[s][16:32], s)),
                "copy_single_ref": float(loss(k, real[s][:1].repeat(16, 1, 1, 1), s)),
            }
            rows[C.SHORT[s]] = {kk: vv for kk, vv in cand.items()}
        out["per_loss"][k] = rows
        best = {C.SHORT[s]: min(rows[C.SHORT[s]], key=rows[C.SHORT[s]].get) for s in range(5)}
        print(f"[S12] {k:8s} floor={rows['ER']['real_target_16']:.4f} "
              f"best_candidate={ {C.SHORT[s]: best[C.SHORT[s]] for s in range(5)} }")
    C.save_json("s12_degenerate.json", out)
    return out


def optimise(loss: L.LatentStyleLoss, kind: str, z0: torch.Tensor, s: int, *, eps: float,
             steps: int = 15, step_rms: float = 0.04) -> tuple[torch.Tensor, list[float], list[float]]:
    z = z0.clone()
    traces, rmss = [], []
    for _ in range(steps):
        zb = z.clone().requires_grad_(True)
        v = loss(kind, zb, s)
        (gr,) = torch.autograd.grad(v, zb)
        d = gr.flatten(1)
        d = d / d.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        z = rms_proj((z - step_rms * d.reshape_as(z)).detach(), z0, eps)
        traces.append(float(v.detach()))
        rmss.append(float(L.content_rms(z, z0).mean()))
    return z, traces, rmss


def run_s13(loss: L.LatentStyleLoss, z_q: torch.Tensor, pools: dict, *, n: int = 8,
            eps: float = 0.2, losses: tuple[str, ...] = L.KINDS, steps: int = 15,
            src_img: torch.Tensor | None = None,
            out_name: str = "s13_latent_opt.json") -> dict:
    """32 content latents (8/style); target = source + 1, +2, +3 styles."""
    z0_all = z_q[:, :n].reshape(-1, 4, 64, 64).detach()
    src = torch.arange(5).repeat_interleave(n).to(z0_all.device)
    if src_img is None:
        src_img = C.decode(z0_all)
    src_img = src_img.to(z0_all.device)
    out: dict = {"n_content": int(z0_all.shape[0]), "eps": eps, "steps": steps,
                 "per_loss": {}}
    for kind in losses:
        groups = {}
        for off in (1, 2, 3):
            tgt = (src + off) % 5
            z = z0_all.clone()
            tr = []
            for step in range(steps):
                zb = z.clone().requires_grad_(True)
                # per-sample target style -> loop over the five style groups
                v = zb.new_zeros(())
                for s in range(5):
                    m = tgt == s
                    if m.any():
                        v = v + loss(kind, zb[m], s) * int(m.sum()) / z0_all.shape[0]
                (gr,) = torch.autograd.grad(v, zb)
                d = gr.flatten(1)
                d = d / d.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
                z = rms_proj((z - 0.04 * d.reshape_as(z)).detach(), z0_all, eps)
                tr.append(float(v.detach()))
            # independent offline evaluation, per target style
            got = {}
            for s in range(5):
                m = tgt == s
                if not m.any():
                    continue
                img = C.decode(z[m])
                met = offline_metrics(img, src_img[m], pools, s)
                others = [met["dino_s_per_style"][C.SHORT[t]] for t in range(5) if t != s]
                base = offline_metrics(src_img[m], src_img[m], pools, s)
                got[C.SHORT[s]] = {
                    "n": int(m.sum()),
                    "loss_trace": [tr[0], tr[len(tr) // 2], tr[-1]],
                    "content_rms": float(L.content_rms(z[m], z0_all[m]).mean()),
                    "dino_s_target": met["dino_s_target"],
                    "dino_s_gain": met["dino_s_target"] - base["dino_s_target"],
                    "dino_s_selectivity": met["dino_s_target"] - float(sum(others) / len(others)),
                    "dino_c": met["dino_c"],
                    "clip_s": met["clip_s"],
                    "clip_s_gain": met["clip_s"] - base["clip_s"],
                    "lpips": met["lpips"],
                }
            groups[f"offset{off}"] = got
        out["per_loss"][kind] = groups
        for off in (1, 2, 3):
            g = groups[f"offset{off}"]
            print(f"[S13] {kind:8s} offset+{off}: dinoS_gain="
                  f"{sum(v['dino_s_gain'] for v in g.values())/len(g):+.4f} "
                  f"sel={sum(v['dino_s_selectivity'] for v in g.values())/len(g):+.4f} "
                  f"dinoC={sum(v['dino_c'] for v in g.values())/len(g):.4f} "
                  f"lpips={sum(v['lpips'] for v in g.values())/len(g):.3f}")
    C.save_json(out_name, out)
    return out


def run_s13_sweep(loss, z_q, pools, *, n: int = 8, src_img=None) -> dict:
    out: dict = {"n_content": int(n * 5), "loss": "swd", "sweep": {}}
    for eps in (0.1, 0.2, 0.4, 0.8):
        res = run_s13(loss, z_q, pools, n=n, eps=eps, losses=("swd",), steps=15,
                      src_img=src_img, out_name=f"s13_sweep_eps{eps}.json")
        out["sweep"][str(eps)] = res["per_loss"]["swd"]
    C.save_json("s13_sweep.json", out)
    return out


def run_s14(loss, z_q, pools, *, n: int = 8, eps: float = 0.2, n_real: int = 100) -> dict:
    z0_all = z_q[:, :n].reshape(-1, 4, 64, 64).detach()
    src = torch.arange(5).repeat_interleave(n).to(z0_all.device)
    z_ref = torch.load(C.AUDIT_DIR / "cache" / "z_ref.pt", map_location="cpu",
                       weights_only=False).to(z0_all.device)
    out: dict = {"eps": eps, "per_target_style": {}, "loss": "swd"}
    for off in (1, 2, 3):
        tgt = (src + off) % 5
        z = z0_all.clone()
        for _ in range(15):
            zb = z.clone().requires_grad_(True)
            v = zb.new_zeros(())
            for s in range(5):
                m = tgt == s
                if m.any():
                    v = v + loss("swd", zb[m], s) * int(m.sum()) / z0_all.shape[0]
            (gr,) = torch.autograd.grad(v, zb)
            d = gr.flatten(1)
            d = d / d.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
            z = rms_proj((z - 0.04 * d.reshape_as(z)).detach(), z0_all, eps)
        for s in range(5):
            m = tgt == s
            if not m.any():
                continue
            opt_feat = C.feat_global(z[m]).float()
            real_feat = C.feat_global(z_ref[s][:n_real]).float()
            ctrl_feat = C.feat_global(z_ref[s][n_real:n_real + 8]).float()
            gmu, gsd = C.standardize(real_feat)
            o = (opt_feat - gmu) / gsd
            r = (real_feat - gmu) / gsd

            def _stats(o, r):
                d_rr = torch.cdist(r, r)
                nn_rr = d_rr[~torch.eye(d_rr.shape[0], dtype=torch.bool)].min()
                d_or = torch.cdist(r, o)
                cov = float((d_or.min(1).values <= nn_rr * 1.5).float().mean())
                fo = F.normalize(o, dim=1)
                return {
                    "mmd_vs_real": float(C.mmd2(o * gsd + gmu, real_feat)),
                    "mean_pairwise_cos": float((fo @ fo.T)[~torch.eye(fo.shape[0], dtype=torch.bool)].mean()),
                    "var_ratio": float(o.var(0).mean() / r.var(0).mean()),
                    "coverage_frac": cov,
                }

            st = _stats(o, r)
            ctl = _stats((ctrl_feat - gmu) / gsd, r)
            out["per_target_style"].setdefault(C.SHORT[s], {})[f"offset{off}"] = {
                "mmd_opt_vs_real": st["mmd_vs_real"],
                "mean_pairwise_cos_opt": st["mean_pairwise_cos"],
                "var_ratio_opt_over_real": st["var_ratio"],
                "coverage_frac": st["coverage_frac"],
                "n_opt": int(m.sum()),
                "control_real8": {
                    "mmd_vs_real": ctl["mmd_vs_real"],
                    "mean_pairwise_cos": ctl["mean_pairwise_cos"],
                    "var_ratio": ctl["var_ratio"],
                    "coverage_frac": ctl["coverage_frac"],
                },
            }
    C.save_json("s14_coverage.json", out)
    return out


def main() -> None:
    torch.manual_seed(0)
    steps = [s.lower() for s in sys.argv[1:]] or ["s12", "s13", "s14"]
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])
    z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to("cuda")
    (C.AUDIT_DIR / "cache").mkdir(parents=True, exist_ok=True)
    torch.save(z_ref, C.AUDIT_DIR / "cache" / "z_ref.pt")
    tg = L.Targets.build(z_ref, device="cuda")
    loss = L.LatentStyleLoss(tg)
    pools = ref_pools("cuda")
    n = 8
    zc = z_q[:, :24].reshape(-1, 4, 64, 64)
    if "s12" in steps:
        run_s12(loss, z_ref.to("cuda"), zc)
    if "s13" in steps:
        src13 = C.decode(z_q[:, :n].reshape(-1, 4, 64, 64))
        run_s13(loss, z_q, pools, n=n, src_img=src13)
        run_s13_sweep(loss, z_q, pools, n=n, src_img=src13)
    if "s14" in steps:
        run_s14(loss, z_q, pools, n=n)


if __name__ == "__main__":
    main()
