"""S4 / S8 / S9 / S10 / S11: does the latent style gradient point anywhere useful?

S4  controlled perturbations: loss response + offline image metrics.
S8  direction of -grad L_s for five target styles: cosines, effective rank,
    subband energy.
S9  small-step gradient descent on the latent (no model training), then
    offline decode + independent metrics.
S10 cos(grad L_s, grad L_c) overall / per subband.
S11 gradient stability under target/projection randomness and cancellation.

Usage: python3 tools/audit/b_grad.py [s4] [s8] [s9] [s10] [s11]
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


# ---------------------------------------------------------------------------
# offline reference pools from the official test split (30 images/style)
# ---------------------------------------------------------------------------
def ref_pools(device: str = "cuda") -> dict:
    cache = C.AUDIT_DIR / "cache" / "ref_pools.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test = C.ROOT / "data/test"
    C.load_metric_models(device)
    dino_cls, clip_vec, clip_proto = {}, {}, {}
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for s, style in enumerate(C.STYLES):
        paths = sorted((test / style).glob("*.jpg"))
        px = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
        img01 = (px * std + mean).clamp(0, 1)
        dino_cls[s] = C.dino_cls(img01, device=device)
        cv = C.clip_img(img01, device=device)
        clip_vec[s] = cv
        clip_proto[s] = F.normalize(cv.mean(0, keepdim=True), dim=-1)[0]
    payload = {"dino_cls": dino_cls, "clip_proto": clip_proto, "clip_vec": clip_vec,
               "n_ref": {s: len(dino_cls[s]) for s in range(5)}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def offline_metrics(gen: torch.Tensor, src: torch.Tensor, pools: dict, target: int) -> dict:
    """gen/src [N,3,512,512] in [0,1]. DINO-S / DINO-C / CLIP-S / LPIPS."""
    gen = C.quantize01(gen)
    src = C.quantize01(src)
    gd = C.dino_cls(gen)
    gc = C.clip_img(gen)
    cd = C.dino_cls(src)
    cc = C.clip_img(src)
    dino_s = float((gd @ pools["dino_cls"][target].T).max(1).values.mean())
    per_style = {C.SHORT[s]: float((gd @ pools["dino_cls"][s].T).max(1).values.mean()) for s in range(5)}
    clip_s = float((gc @ pools["clip_proto"][target]).mean())
    return {
        "dino_s_target": dino_s,
        "dino_s_per_style": per_style,
        "dino_c": float((gd * cd).sum(1).mean()),
        "clip_s": clip_s,
        "clip_c": float((gc * cc).sum(1).mean()),
        "lpips": float(C.lpips_dist(gen, src).mean()),
    }


# ---------------------------------------------------------------------------
# perturbations (latent space)
# ---------------------------------------------------------------------------
def perturb(z: torch.Tensor, kind: str) -> torch.Tensor:
    from wavelet import dwt2_haar_lowpass
    if kind == "identity":
        return z.clone()
    if kind == "chan_affine":
        mu = z.mean(dim=(2, 3), keepdim=True)
        sd = z.std(dim=(2, 3), keepdim=True)
        return (z - mu) * 1.25 + mu + 0.5 * sd
    if kind == "lowpass_x13":
        lp = dwt2_haar_lowpass(z, levels=2)
        return z - lp + 1.3 * lp
    if kind == "hf_x15":
        lp = dwt2_haar_lowpass(z, levels=1)
        return z + 1.5 * (z - lp)
    if kind == "blur_ll1":
        return dwt2_haar_lowpass(z, levels=1)
    if kind == "sharpen":
        lp = dwt2_haar_lowpass(z, levels=1)
        return z + 0.8 * (z - lp)
    if kind == "patch_shuffle":
        b, c, h, w = z.shape
        p = 8
        patches = z.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(b, -1, c, p, p)
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(patches.shape[1], generator=g)
        patches = patches[:, perm]
        patches = patches.reshape(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(b, c, h, w)
    if kind == "shift8":
        return torch.roll(z, shifts=(8, 8), dims=(2, 3))
    raise ValueError(kind)


PERTURB_KINDS = ["identity", "chan_affine", "lowpass_x13", "hf_x15", "blur_ll1",
                 "sharpen", "patch_shuffle", "shift8"]


# ---------------------------------------------------------------------------
def run_s4(loss, zc, pools) -> dict:
    out: dict = {"n_images": int(zc.shape[0]), "target_style": 1, "perturbations": {}}
    src_img = C.decode(zc)
    base = offline_metrics(src_img, src_img, pools, 1)
    out["baseline_metrics"] = base
    for kind in PERTURB_KINDS:
        zp = perturb(zc, kind)
        row = {"delta_loss": {}}
        for k in L.KINDS:
            with torch.no_grad():
                v, v0 = float(loss(k, zp, 1)), float(loss(k, zc, 1))
            row["delta_loss"][k] = v - v0
        row["metrics"] = base if kind == "identity" else offline_metrics(C.decode(zp), src_img, pools, 1)
        out["perturbations"][kind] = row
        print(f"[S4] {kind:14s} dL={ {k: round(v,4) for k,v in row['delta_loss'].items()} } "
              f"dinoS={row['metrics']['dino_s_target']:.4f} dinoC={row['metrics']['dino_c']:.4f} "
              f"clipS={row['metrics']['clip_s']:.4f} lpips={row['metrics']['lpips']:.4f}")
    C.save_json("s4_perturb.json", out)
    return out


def run_s8(loss, zc) -> dict:
    z8 = zc[:40].detach()
    out: dict = {"n_images": int(z8.shape[0]), "loss": "swd"}
    grads = {}
    for s in range(5):
        g = []
        for i in range(0, z8.shape[0], 8):
            zb = z8[i:i + 8].clone().requires_grad_(True)
            v = loss("swd", zb, s)
            (gr,) = torch.autograd.grad(v, zb)
            g.append((-gr).detach())
        grads[s] = torch.cat(g)
    cos_m = torch.zeros(5, 5)
    for a in range(5):
        for b in range(5):
            x, y = grads[a].flatten(1), grads[b].flatten(1)
            cos_m[a, b] = float((F.normalize(x, dim=1) * F.normalize(y, dim=1)).sum(1).mean())
    emb = torch.stack([grads[s].flatten(1) for s in range(5)], 0)
    sv = torch.linalg.svdvals(emb.permute(1, 0, 2))
    p = sv.pow(2) / sv.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    eff = float(torch.exp(-(p.clamp_min(1e-12).log() * p).sum(-1)).mean())
    from wavelet import dwt2_haar
    band = {}
    for s in range(5):
        ll, lh, hl, hh = dwt2_haar(grads[s])
        e = [float(b.pow(2).mean()) for b in (ll, lh, hl, hh)]
        tot = sum(e) or 1.0
        band[C.SHORT[s]] = {"LL": e[0] / tot, "LH": e[1] / tot, "HL": e[2] / tot, "HH": e[3] / tot}
    gnorm = {C.SHORT[s]: float(grads[s].pow(2).flatten(1).sum(1).sqrt().mean()) for s in range(5)}
    off = cos_m[~torch.eye(5, dtype=torch.bool)]
    out.update({"grad_norm": gnorm, "cos_matrix_neg_grad": cos_m.tolist(),
                "mean_offdiag_cos": float(off.mean()), "max_pair_cos": float(off.max()),
                "effective_rank": eff, "subband_energy_frac": band})
    print(f"[S8] offdiag cos={float(off.mean()):.3f} max={float(off.max()):.3f} effrank={eff:.2f} "
          f"grad_norm={ {k: round(v,3) for k,v in gnorm.items()} }")
    C.save_json("s8_directions.json", out)
    return out


def run_s9(loss, z_q, pools, *, n9: int = 8, steps_n: int = 12, step_rms: float = 0.05) -> dict:
    z0 = z_q[:, :n9].reshape(-1, 4, 64, 64).detach()
    src9 = C.decode(z0)
    out: dict = {"n_images_per_style": n9, "steps": steps_n, "step_rms": step_rms, "targets": {}}
    for s in range(5):
        z = z0.clone()
        hist, rms = [], []
        for _ in range(steps_n):
            zb = z.clone().requires_grad_(True)
            v = loss("swd", zb, s)
            (gr,) = torch.autograd.grad(v, zb)
            d = gr.flatten(1)
            d = d / d.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)   # RMS-normalised
            z = (z - step_rms * d.reshape_as(z)).detach()
            hist.append(float(v.detach()))
            rms.append(float(L.content_rms(z, z0).mean()))
        img = C.decode(z)
        met = offline_metrics(img, src9, pools, s)
        others = [met["dino_s_per_style"][C.SHORT[t]] for t in range(5) if t != s]
        out["targets"][C.SHORT[s]] = {
            "loss_start": hist[0], "loss_end": hist[-1],
            "content_rms_change": float(L.content_rms(z, z0).mean()),
            "rms_trace": rms,
            "metrics": met,
            "dino_s_selectivity": met["dino_s_target"] - float(sum(others) / len(others)),
        }
        print(f"[S9] {C.SHORT[s]}: L {hist[0]:.4f}->{hist[-1]:.4f} rms={rms[-1]:.3f} "
              f"dinoS={met['dino_s_target']:.4f} other={sum(others)/4:.4f} "
              f"dinoC={met['dino_c']:.4f} lpips={met['lpips']:.4f}")
    C.save_json("s9_ascent.json", out)
    return out


def run_s10(loss, zc) -> dict:
    from wavelet import dwt2_haar
    z10 = zc[:30].detach()
    out: dict = {"n_images": int(z10.shape[0]), "per_style": {}}
    for s in range(5):
        rows, band_rows, gn = [], [], []
        gnoise = torch.Generator().manual_seed(3)
        for i in range(0, z10.shape[0], 10):
            # content losses have zero gradient at the exact start point, so
            # evaluate the cosine at a small random offset (0.05 RMS).
            base = z10[i:i + 10]
            zb = (base + 0.05 * torch.randn(base.shape, generator=gnoise).to(base.device)
                  ).clone().requires_grad_(True)
            ls = loss("swd", zb, s)
            lc = L.content_structure(zb, z10[i:i + 10])
            gs = torch.autograd.grad(ls, zb, retain_graph=True)[0]
            gc = torch.autograd.grad(lc, zb)[0]
            gn.append([float(gs.flatten(1).norm(dim=1).mean()), float(gc.flatten(1).norm(dim=1).mean())])
            cos = float((F.normalize(gs.flatten(1), dim=1) * F.normalize(gc.flatten(1), dim=1)).sum(1).mean())
            rows.append(cos)
            bl = [max(float(b.pow(2).mean()), 1e-20) for b in dwt2_haar(gs)]
            cl = [max(float(b.pow(2).mean()), 1e-20) for b in dwt2_haar(gc)]
            band_rows.append({"gs": [x / sum(bl) for x in bl], "gc": [x / sum(cl) for x in cl]})
        out["per_style"][C.SHORT[s]] = {
            "cos_s_c_mean": float(torch.tensor(rows).mean()),
            "cos_per_chunk": rows,
            "grad_norm_style_vs_content": gn,
            "grad_s_band_frac": [float(torch.tensor([r["gs"][k] for r in band_rows]).mean()) for k in range(4)],
            "grad_c_band_frac": [float(torch.tensor([r["gc"][k] for r in band_rows]).mean()) for k in range(4)],
            "bands": ["LL", "LH", "HL", "HH"],
        }
        print(f"[S10] {C.SHORT[s]} cos(gS,gC)={out['per_style'][C.SHORT[s]]['cos_s_c_mean']:.4f} "
              f"|gS|={gn[0][0]:.2e} |gC|={gn[0][1]:.2e}")
    C.save_json("s10_conflict.json", out)
    return out


def run_s11(loss, zc) -> dict:
    z11 = zc[:16].detach()
    out: dict = {"style": 1, "variants": {}, "loss": "swd"}
    variants = {
        "target_seed0": 0,
        "target_seed1": 11,
        "target_seed2": 22,
        "seed_101": 101,
        "seed_202": 202,
    }
    gvec = {}
    for name, seed in variants.items():
        g = []
        for i in range(0, z11.shape[0], 8):
            zb = z11[i:i + 8].clone().requires_grad_(True)
            v = loss("swd", zb, 1, seed=seed)
            (gr,) = torch.autograd.grad(v, zb)
            g.append(gr.detach().flatten(1))
        gvec[name] = torch.cat(g)
    ref = gvec["target_seed0"]
    for name in gvec:
        cos = float((F.normalize(gvec[name], dim=1) * F.normalize(ref, dim=1)).sum(1).mean())
        out["variants"][name] = {
            "grad_norm_mean": float(gvec[name].norm(dim=1).mean()),
            "cos_to_base": cos,
            "norm_ratio_to_base": float((gvec[name].norm(dim=1) / ref.norm(dim=1)).mean()),
        }
    mixed = []
    for s in range(5):
        zb = z11[:8].clone().requires_grad_(True)
        v = loss("swd", zb, s)
        (gr,) = torch.autograd.grad(v, zb)
        mixed.append(gr.detach().flatten(1))
    mixed = torch.stack(mixed)
    mean_g = mixed.mean(0)
    out["mixed_style"] = {
        "norm_of_mean": float(mean_g.norm(dim=1).mean()),
        "mean_of_norms": float(mixed.norm(dim=2).mean()),
        "cancellation_ratio": float(mean_g.norm(dim=1).mean() / mixed.norm(dim=2).mean()),
        "within_style_variance": float((mixed - mixed.mean(0, keepdim=True)).pow(2).mean()),
    }
    print(f"[S11] variants: { {k: round(v['cos_to_base'],3) for k,v in out['variants'].items()} } "
          f"cancel={out['mixed_style']['cancellation_ratio']:.3f}")
    C.save_json("s11_stability.json", out)
    return out


def main() -> None:
    torch.manual_seed(0)
    steps = [s.lower() for s in sys.argv[1:]] or ["s4", "s8", "s9", "s10", "s11"]
    dev = "cuda"
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])
    z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to(dev)
    tg = L.Targets.build(z_ref, device=dev)
    loss = L.LatentStyleLoss(tg)
    pools = ref_pools(dev)
    n_eval = 24
    zc = z_q[:, :n_eval].reshape(-1, 4, 64, 64).to(dev)

    if "s4" in steps:
        run_s4(loss, zc, pools)
    if "s8" in steps:
        run_s8(loss, zc)
    if "s9" in steps:
        run_s9(loss, z_q, pools)
    if "s10" in steps:
        run_s10(loss, zc)
    if "s11" in steps:
        run_s11(loss, zc)


if __name__ == "__main__":
    main()
