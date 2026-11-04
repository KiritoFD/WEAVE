"""S5 / S6 / S7: what should the content constraint measure?

S5  LL-only / HF-only / per-band masking / multi-scale smoothing: offline
    DINO-C, DINO-S, LPIPS + latent structure change (no decoder backprop).
S6  ranking consistency of latent content metrics under mild texture change
    vs severe structural damage.
S7  Pareto curve: style gain vs content damage for small gradient steps.

Usage: python3 tools/audit/c_content.py [s5] [s6] [s7]
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


def run_s5(zc: torch.Tensor, pools: dict, *, target: int = 1) -> dict:
    from wavelet import dwt2_haar, idwt2_haar
    src = C.decode(zc)
    ll1, lh1, hl1, hh1 = dwt2_haar(zc)
    vars_: dict[str, torch.Tensor] = {
        "identity": zc.clone(),
        "ll1_only": idwt2_haar(ll1, torch.zeros_like(lh1), torch.zeros_like(hl1), torch.zeros_like(hh1)),
        "hf_only": idwt2_haar(torch.zeros_like(ll1), lh1, hl1, hh1),
        "no_LH": idwt2_haar(ll1, torch.zeros_like(lh1), hl1, hh1),
        "no_HL": idwt2_haar(ll1, lh1, torch.zeros_like(hl1), hh1),
        "no_HH": idwt2_haar(ll1, lh1, hl1, torch.zeros_like(hh1)),
        "blur_8px": F.avg_pool2d(zc, 2, 2),
        "blur_16px": F.avg_pool2d(zc, 4, 4),
    }
    out: dict = {"target_style": target, "n_images": int(zc.shape[0]), "variants": {}}
    for name, zv in vars_.items():
        # upscale the heavily pooled variants back to the latent grid (nearest)
        if zv.shape[-1] != zc.shape[-1]:
            zv = F.interpolate(zv, size=zc.shape[-2:], mode="nearest")
        met = offline_metrics(C.decode(zv), src, pools, target)
        # latent structural change
        out["variants"][name] = {
            "dino_c": met["dino_c"], "dino_s_target": met["dino_s_target"],
            "dino_s_per_style": met["dino_s_per_style"], "clip_s": met["clip_s"],
            "lpips": met["lpips"],
            "latent_rms": float(L.content_rms(zv, zc).mean()),
            "latent_structure": float(L.content_structure(zv, zc)),
            "subband_energy": _band_energy(zv),
        }
        print(f"[S5] {name:12s} dinoC={met['dino_c']:.4f} dinoS*={met['dino_s_target']:.4f} "
              f"lpips={met['lpips']:.4f} rms={out['variants'][name]['latent_rms']:.3f}")
    C.save_json("s5_content_parts.json", out)
    return out


def _band_energy(z: torch.Tensor) -> dict:
    from wavelet import dwt2_haar
    ll, lh, hl, hh = dwt2_haar(z)
    e = [float(b.pow(2).mean()) for b in (ll, lh, hl, hh)]
    tot = sum(e) or 1.0
    return {k: v / tot for k, v in zip(("LL", "LH", "HL", "HH"), e)}


def run_s6(zc: torch.Tensor) -> dict:
    """Do latent content metrics rank severe structural damage above mild texture change?"""
    severe = ["patch_shuffle", "blur_ll1", "shift8"]
    mild = ["chan_affine", "sharpen", "lowpass_x13"]
    metrics = {
        "latent_l1": lambda z: float((z - zc).abs().mean()),
        "latent_l2": lambda z: float((z - zc).pow(2).mean().sqrt()),
        "lowpass_rms": lambda z: float((_ll(z) - _ll(zc)).abs().mean()),
        "structure": lambda z: float(L.content_structure(z, zc)),
    }
    out: dict = {"metrics": {}, "n_images": int(zc.shape[0])}
    for name, fn in metrics.items():
        mild_v = {k: fn(perturb(zc, k)) for k in mild}
        sev_v = {k: fn(perturb(zc, k)) for k in severe}
        out["metrics"][name] = {
            "mild": mild_v, "severe": sev_v,
            "mild_mean": sum(mild_v.values()) / len(mild_v),
            "severe_mean": sum(sev_v.values()) / len(sev_v),
            "ranking_ok": bool(min(sev_v.values()) > max(mild_v.values())),
            "margin_ratio": float(sum(sev_v.values()) / max(sum(mild_v.values()), 1e-8)),
        }
        print(f"[S6] {name:12s} mild={out['metrics'][name]['mild_mean']:.4f} "
              f"severe={out['metrics'][name]['severe_mean']:.4f} "
              f"rank_ok={out['metrics'][name]['ranking_ok']}")
    C.save_json("s6_content_metrics.json", out)
    return out


def _ll(z: torch.Tensor) -> torch.Tensor:
    from wavelet import dwt2_haar_lowpass
    return dwt2_haar_lowpass(z, levels=2)


def run_s7(loss: L.LatentStyleLoss, z_q: torch.Tensor, pools: dict, *, n: int = 8,
           step_rms: float = 0.03) -> dict:
    z0 = z_q[:, :n].reshape(-1, 4, 64, 64).detach()
    src_style = torch.arange(5).repeat_interleave(n).to(z0.device)
    src = C.decode(z0).to(z0.device)
    out: dict = {"n_images": int(z0.shape[0]), "step_rms": step_rms, "points": []}
    snapshots = {0: z0.clone()}
    z = z0.clone()
    for step in range(1, 13):
        zb = z.clone().requires_grad_(True)
        v = zb.new_zeros(())
        for s in range(5):
            m = src_style == s
            if m.any():
                v = v + loss("swd", zb[m], s) * int(m.sum()) / z0.shape[0]
        (gr,) = torch.autograd.grad(v, zb)
        d = gr.flatten(1)
        d = d / d.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        z = (z - step_rms * d.reshape_as(z)).detach()
        if step in (2, 4, 6, 8, 10, 12):
            snapshots[step] = z.clone()
    for step, zs in snapshots.items():
        img = C.decode(zs).to(z0.device)
        # style gain measured against each image's own source style
        gains, sel, dcs, lp = [], [], [], []
        for s in range(5):
            m = src_style == s
            met = offline_metrics(img[m], src[m], pools, s)
            base = offline_metrics(src[m], src[m], pools, s)
            gains.append(met["dino_s_target"] - base["dino_s_target"])
            others = [met["dino_s_per_style"][C.SHORT[t]] for t in range(5) if t != s]
            sel.append(met["dino_s_target"] - sum(others) / 4)
            dcs.append(met["dino_c"])
            lp.append(met["lpips"])
        out["points"].append({
            "steps": step,
            "content_rms": float(L.content_rms(zs, z0).mean()),
            "dino_s_gain": float(torch.tensor(gains).mean()),
            "dino_s_selectivity": float(torch.tensor(sel).mean()),
            "dino_c": float(torch.tensor(dcs).mean()),
            "lpips": float(torch.tensor(lp).mean()),
        })
        print(f"[S7] steps={step:2d} rms={out['points'][-1]['content_rms']:.3f} "
              f"dS={out['points'][-1]['dino_s_gain']:+.4f} sel={out['points'][-1]['dino_s_selectivity']:+.4f} "
              f"dinoC={out['points'][-1]['dino_c']:.4f} lpips={out['points'][-1]['lpips']:.3f}")
    C.save_json("s7_pareto.json", out)
    return out


def main() -> None:
    torch.manual_seed(0)
    steps = [s.lower() for s in sys.argv[1:]] or ["s5", "s6", "s7"]
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])
    z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to("cuda")
    tg = L.Targets.build(z_ref, device="cuda")
    loss = L.LatentStyleLoss(tg)
    pools = ref_pools("cuda")
    n = 8
    zc = z_q[:, :n].reshape(-1, 4, 64, 64)
    if "s5" in steps:
        run_s5(zc, pools)
    if "s6" in steps:
        run_s6(zc)
    if "s7" in steps:
        run_s7(loss, z_q, pools, n=n)


if __name__ == "__main__":
    main()
