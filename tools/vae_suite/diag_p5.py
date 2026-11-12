"""WEAVE-VAE V2 Diagnosis: P5 Decoder Intervention & Latent Oracle.

P5.1 Fixed covariance transfer baseline with alpha=0 verification:
  Delta_T_edit = S_t(D(z_alpha)) - S_t(D(z_0))  (must be exactly 0 at alpha=0)

P5.2 Four-way intervention control:
  - No-op (plain VAE roundtrip)
  - Same-style statistics transfer
  - Target-style statistics transfer
  - Random-style transfer
  - Random latent direction with matched norm

P5.3 Latent oracle: directly optimize z along -grad d_SWD(z, Q_target)
  Measure: SWD distance drop, CSD target gain, DINO-C, LPIPS, perturbation norm.

P5.4 cos(g_SWD, g_CSD) alignment test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
import csd_model
import lpips

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CALIB = Path("/root/Workspace/WEAVE_VAE/swd_calibration.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
V2_CKPT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/V2")
B0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"


def build_desc():
    c = json.loads(CALIB.read_text())
    d = LatentSWDDescriptor()
    d.set_calibration(
        torch.tensor(c["chan_mean"]), torch.tensor(c["chan_std"]),
        c["scale_swd1"], c["scale_swd2"], c["scale_moment"], c["median_inter_dist"],
    )
    return d, c


def load_model(kind, ckpt=None, device="cuda"):
    if kind == "B0":
        vae = AutoencoderKL.from_pretrained(B0_DIR)
    else:
        vae = AutoencoderKL.from_pretrained(A0_DIR)
        if ckpt:
            ck = torch.load(V2_CKPT_DIR / ckpt, map_location="cpu", weights_only=False)
            vae.load_state_dict(ck.get("ema", ck["model"]))
    return vae.to(device).eval()


def mat_sqrt(A, eps=1e-4):
    A = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    L, Q = torch.linalg.eigh(A)
    return Q @ torch.diag(torch.sqrt(torch.clamp(L, min=1e-6))) @ Q.t()


def mat_inv_sqrt(A, eps=1e-4):
    A = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    L, Q = torch.linalg.eigh(A)
    return Q @ torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-6))) @ Q.t()


def cov_transfer(z, mu_t, sigma_t):
    C, H, W = z.shape
    mu_s = z.mean(dim=(1, 2), keepdim=True)
    zc = (z - mu_s).reshape(C, -1)
    sigma_s = zc @ zc.t() / float(H * W - 1)
    M = mat_sqrt(sigma_t) @ mat_inv_sqrt(sigma_s)
    return (M @ zc).view(C, H, W) + mu_t.view(C, 1, 1)


def style_stats(vae, dataset, style_to_idx, styles_list, device, n=20):
    out = {}
    with torch.no_grad():
        for st in styles_list:
            picks = style_to_idx.get(st, [])[:n]
            if len(picks) < 4:
                continue
            x = torch.stack([dataset[p]["image"] for p in picks]).to(device)
            z = vae.encode(x).latent_dist.mean.float()
            mu = z.mean(dim=(0, 2, 3))
            zc = (z - mu.view(1, 4, 1, 1)).permute(1, 0, 2, 3).reshape(4, -1)
            sig = zc @ zc.t() / float(zc.shape[1] - 1)
            out[st] = (mu, sig)
    return out


def csd_feat(csd, img, norm_mean, norm_std):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    xin = (x224 - norm_mean) / norm_std
    _, _, f = csd(xin.half())
    return f


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    desc, _ = build_desc()
    desc = desc.to(device)
    dataset = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    csd, _, _ = csd_model.load_csd(device)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    # style index
    style_to_idx = {}
    for i, idx in enumerate(dataset.indices):
        rec = dataset.records_by_idx[idx]
        style_to_idx.setdefault(rec["style_id"], []).append(i)

    valid_styles = [s for s, v in style_to_idx.items() if len(v) >= 20][:40]
    rng = np.random.default_rng(42)
    src_picks = rng.choice(len(dataset), size=20, replace=False)
    sources = [dataset[i] for i in src_picks]

    results = {}
    alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]

    for name, kind, ckpt in [
        ("B0", "B0", None),
        ("A0", "A0", None),
        ("V2_1800", "V2", "checkpoint_step_001800.pt"),
    ]:
        print(f"\n========== P5 Evaluating {name} ==========", flush=True)
        vae = load_model(kind, ckpt, device)
        stats = style_stats(vae, dataset, style_to_idx, valid_styles, device)
        avail = [s for s in valid_styles if s in stats]
        if not avail:
            print("  [skip] no valid target styles")
            continue

        # ---------------- P5.1 / P5.2 ----------------
        alpha_res = {a: {"dt_edit": [], "dt_rgb": [], "lpips": [], "dino_c": []} for a in alphas}
        ctrl_res = {"noop": [], "same_style": [], "target_style": [], "random_style": [], "random_dir": []}

        with torch.no_grad():
            for src in sources:
                x = src["image"].unsqueeze(0).to(device)
                src_style = src["style_id"].item()
                z_s = vae.encode(x).latent_dist.mean.squeeze(0).float()
                rec0 = vae.decode(z_s.unsqueeze(0)).sample
                f_src = csd_feat(csd, x, norm_mean, norm_std)
                f_rec0 = csd_feat(csd, rec0, norm_mean, norm_std)

                tgt_cands = [s for s in avail if s != src_style]
                if not tgt_cands:
                    continue
                tgt = int(rng.choice(tgt_cands))
                mu_t, sig_t = stats[tgt]

                tgt_idx = style_to_idx[tgt][0]
                tgt_img = dataset[tgt_idx]["image"].unsqueeze(0).to(device)
                f_tgt = csd_feat(csd, tgt_img, norm_mean, norm_std)

                base_rgb = torch.sum(f_src * f_tgt, dim=-1).item()
                base_edit = torch.sum(f_rec0 * f_tgt, dim=-1).item()

                T = cov_transfer(z_s, mu_t, sig_t)

                # alpha sweep
                for a in alphas:
                    if a == 0.0:
                        rec = rec0
                        f_rec = f_rec0
                    else:
                        za = (1 - a) * z_s + a * T
                        rec = vae.decode(za.unsqueeze(0)).sample
                        f_rec = csd_feat(csd, rec, norm_mean, norm_std)
                    sim = torch.sum(f_rec * f_tgt, dim=-1).item()
                    alpha_res[a]["dt_edit"].append(sim - base_edit)
                    alpha_res[a]["dt_rgb"].append(sim - base_rgb)
                    alpha_res[a]["lpips"].append(lpips_fn(rec.float(), x.float()).mean().item())
                    alpha_res[a]["dino_c"].append(torch.sum(f_rec * f_src, dim=-1).item())

                # controls at alpha=0.3
                a = 0.3
                # same-style transfer
                mu_s2, sig_s2 = stats[src_style] if src_style in stats else (mu_t, sig_t)
                Ts = cov_transfer(z_s, mu_s2, sig_s2)
                zs_ = (1 - a) * z_s + a * Ts
                rec_s = vae.decode(zs_.unsqueeze(0)).sample
                f_s = csd_feat(csd, rec_s, norm_mean, norm_std)
                ctrl_res["same_style"].append(torch.sum(f_s * f_tgt, dim=-1).item() - base_edit)

                # target-style transfer
                zt_ = (1 - a) * z_s + a * T
                rec_t = vae.decode(zt_.unsqueeze(0)).sample
                f_t = csd_feat(csd, rec_t, norm_mean, norm_std)
                ctrl_res["target_style"].append(torch.sum(f_t * f_tgt, dim=-1).item() - base_edit)

                # random-style transfer
                rnd = int(rng.choice(avail))
                mu_r, sig_r = stats[rnd]
                Tr = cov_transfer(z_s, mu_r, sig_r)
                zr_ = (1 - a) * z_s + a * Tr
                rec_r = vae.decode(zr_.unsqueeze(0)).sample
                f_r = csd_feat(csd, rec_r, norm_mean, norm_std)
                ctrl_res["random_style"].append(torch.sum(f_r * f_tgt, dim=-1).item() - base_edit)

                # random direction matched norm
                delta = T - z_s
                rdir = torch.randn_like(delta)
                rdir = rdir / rdir.norm() * delta.norm()
                zrd = z_s + a * rdir
                rec_rd = vae.decode(zrd.unsqueeze(0)).sample
                f_rd = csd_feat(csd, rec_rd, norm_mean, norm_std)
                ctrl_res["random_dir"].append(torch.sum(f_rd * f_tgt, dim=-1).item() - base_edit)

                ctrl_res["noop"].append(0.0)

        p5_1 = {}
        for a in alphas:
            r = alpha_res[a]
            p5_1[str(a)] = {
                "delta_T_edit_mean": round(float(np.mean(r["dt_edit"])), 4),
                "delta_T_rgb_mean": round(float(np.mean(r["dt_rgb"])), 4),
                "pct_positive_edit": round(float(np.mean(np.array(r["dt_edit"]) > 0) * 100), 1),
                "lpips": round(float(np.mean(r["lpips"])), 4),
                "dino_c": round(float(np.mean(r["dino_c"])), 4),
            }
        print(f"  alpha=0 dt_edit = {p5_1['0.0']['delta_T_edit_mean']} (must be 0)", flush=True)
        for a in alphas[1:]:
            print(f"  alpha={a}: dt_edit={p5_1[str(a)]['delta_T_edit_mean']:+.4f} "
                  f"({p5_1[str(a)]['pct_positive_edit']}% pos) LPIPS={p5_1[str(a)]['lpips']:.4f}", flush=True)

        p5_2 = {k: round(float(np.mean(v)), 4) for k, v in ctrl_res.items()}
        print(f"  controls@0.3: {p5_2}", flush=True)

        # ---------------- P5.3 Latent Oracle ----------------
        print("  [P5.3] Latent oracle along -grad SWD ...", flush=True)
        oracle = []
        for src in sources[:8]:
            x = src["image"].unsqueeze(0).to(device)
            src_style = src["style_id"].item()
            with torch.no_grad():
                z0 = vae.encode(x).latent_dist.mean.squeeze(0).float()
                f_src = csd_feat(csd, x, norm_mean, norm_std)
                rec0 = vae.decode(z0.unsqueeze(0)).sample
                f_rec0 = csd_feat(csd, rec0, norm_mean, norm_std)

            tgt_cands = [s for s in avail if s != src_style]
            if not tgt_cands:
                continue
            tgt = int(rng.choice(tgt_cands))
            mu_t, sig_t = stats[tgt]
            tgt_img = dataset[style_to_idx[tgt][0]]["image"].unsqueeze(0).to(device)
            f_tgt = csd_feat(csd, tgt_img, norm_mean, norm_std)
            base_edit = torch.sum(f_rec0 * f_tgt, dim=-1).item()

            # build target reference descriptors
            ref_picks = style_to_idx[tgt][:8]
            with torch.no_grad():
                ref_x = torch.stack([dataset[p]["image"] for p in ref_picks]).to(device)
                ref_z = vae.encode(ref_x).latent_dist.mean.float()
                rq1, rq2, rmom = desc(ref_z)

            z = z0.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.02)
            d0_val = None
            for step in range(30):
                opt.zero_grad()
                q1, q2, mom = desc(z.unsqueeze(0))
                dmat = desc.compute_pairwise_distance(q1, q2, mom, rq1, rq2, rmom)
                loss = dmat.mean()
                if d0_val is None:
                    d0_val = float(loss.item())
                loss.backward()
                opt.step()
            d1_val = float(loss.item())
            pert_norm = float((z.detach() - z0).norm().item())

            with torch.no_grad():
                rec = vae.decode(z.detach().unsqueeze(0)).sample
                f_rec = csd_feat(csd, rec, norm_mean, norm_std)
            oracle.append({
                "swd_before": round(d0_val, 4),
                "swd_after": round(d1_val, 4),
                "swd_drop_pct": round((d0_val - d1_val) / max(d0_val, 1e-8) * 100, 2),
                "csd_gain": round(torch.sum(f_rec * f_tgt, dim=-1).item() - base_edit, 4),
                "dino_c": round(torch.sum(f_rec * f_src, dim=-1).item(), 4),
                "lpips": round(lpips_fn(rec.float(), x.float()).mean().item(), 4),
                "pert_norm": round(pert_norm, 4),
                "pert_rel": round(pert_norm / max(float(z0.norm().item()), 1e-8), 4),
            })

        p5_3 = {
            "n_sources": len(oracle),
            "mean_swd_drop_pct": round(float(np.mean([o["swd_drop_pct"] for o in oracle])), 2) if oracle else None,
            "mean_csd_gain": round(float(np.mean([o["csd_gain"] for o in oracle])), 4) if oracle else None,
            "pct_positive_csd": round(float(np.mean([o["csd_gain"] > 0 for o in oracle]) * 100), 1) if oracle else None,
            "mean_dino_c": round(float(np.mean([o["dino_c"] for o in oracle])), 4) if oracle else None,
            "mean_lpips": round(float(np.mean([o["lpips"] for o in oracle])), 4) if oracle else None,
            "mean_pert_rel": round(float(np.mean([o["pert_rel"] for o in oracle])), 4) if oracle else None,
            "per_source": oracle,
        }
        print(f"    SWD drop: {p5_3['mean_swd_drop_pct']}% | CSD gain: {p5_3['mean_csd_gain']} "
              f"({p5_3['pct_positive_csd']}% pos) | LPIPS={p5_3['mean_lpips']}", flush=True)

        # ---------------- P5.4 cos(g_SWD, g_CSD) ----------------
        print("  [P5.4] cos(g_SWD, g_CSD) alignment ...", flush=True)
        cos_list = []
        for src in sources[:6]:
            x = src["image"].unsqueeze(0).to(device)
            src_style = src["style_id"].item()
            tgt_cands = [s for s in avail if s != src_style]
            if not tgt_cands:
                continue
            tgt = int(rng.choice(tgt_cands))
            with torch.no_grad():
                z0 = vae.encode(x).latent_dist.mean.squeeze(0).float()
                ref_x = torch.stack([dataset[p]["image"] for p in style_to_idx[tgt][:8]]).to(device)
                ref_z = vae.encode(ref_x).latent_dist.mean.float()
                rq1, rq2, rmom = desc(ref_z)

            # g_SWD
            z1 = z0.clone().requires_grad_(True)
            q1, q2, mom = desc(z1.unsqueeze(0))
            dmat = desc.compute_pairwise_distance(q1, q2, mom, rq1, rq2, rmom).mean()
            g_swd = torch.autograd.grad(dmat, z1)[0].flatten()

            # g_CSD via decoder
            z2 = z0.clone().requires_grad_(True)
            rec = vae.decode(z2.unsqueeze(0)).sample
            f_rec = csd_feat(csd, rec, norm_mean, norm_std)
            # target direction: maximize similarity to target reference
            tgt_img = dataset[style_to_idx[tgt][0]]["image"].unsqueeze(0).to(device)
            f_tgt = csd_feat(csd, tgt_img, norm_mean, norm_std)
            sim = torch.sum(f_rec * f_tgt, dim=-1).mean()
            g_csd = torch.autograd.grad(sim, z2)[0].flatten()

            c = float(F.cosine_similarity(g_swd.unsqueeze(0), g_csd.unsqueeze(0)).item())
            cos_list.append(c)

        p5_4 = {
            "cos_gswd_gcsd_mean": round(float(np.mean(cos_list)), 4) if cos_list else None,
            "cos_values": [round(c, 4) for c in cos_list],
            "interpretation": ("aligned" if cos_list and np.mean(cos_list) > 0.2
                               else "weak/negative - SWD direction does not raise CSD"),
        }
        print(f"    cos(g_SWD, g_CSD) = {p5_4['cos_gswd_gcsd_mean']} -> {p5_4['interpretation']}", flush=True)

        results[name] = {
            "p5_1_alpha_sweep": p5_1,
            "p5_2_controls": p5_2,
            "p5_3_latent_oracle": p5_3,
            "p5_4_gradient_alignment": p5_4,
        }
        del vae
        torch.cuda.empty_cache()

    (OUT_DIR / "11_decoder_interventions.json").write_text(
        json.dumps({k: {"p5_1": v["p5_1_alpha_sweep"], "p5_2": v["p5_2_controls"]}
                    for k, v in results.items()}, indent=2), encoding="utf-8")
    (OUT_DIR / "12_latent_oracle.json").write_text(
        json.dumps({k: {"p5_3": v["p5_3_latent_oracle"], "p5_4": v["p5_4_gradient_alignment"]}
                    for k, v in results.items()}, indent=2), encoding="utf-8")
    print(f"\n[Saved] P5 outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
