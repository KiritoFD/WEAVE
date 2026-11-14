"""V14 Experiment 1: style-information scale scan (fixed).
Run: python -m exp.v14.scale_scan
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from .features import (REPO, STYLES, load_paintings, load_image,
                       s0_latent_stats, s1_dwt_tokens, s2_dwt_large,
                       s3_rgb64, s4_rgb128, s5_rgb224, s6_dino_tokens,
                       s8_multiscale)

RUNS = REPO / "runs/v14"
N_TRAIN = 200
N_HOLD = 80
N_BOOT = 1000


def _paint_means(feats, n_paint):
    """(n_paint*64, D) -> (n_paint, D) per-painting mean."""
    return feats.reshape(n_paint, 64, -1).mean(dim=1)


def _paint_labels(y_per_painting):
    return y_per_painting


def extract(rep, item, latents, device):
    si, img_path, lat_idx = item
    z = latents[si][lat_idx:lat_idx + 1].float().to(device)
    rgb = load_image(img_path, 512).to(device)
    from .features import (s0_latent_stats, s1_dwt_tokens, s2_dwt_large,
                           s3_rgb64, s4_rgb128, s5_rgb224, s6_dino_tokens,
                           s8_multiscale, to_grayscale)
    if rep == "S0": return s0_latent_stats(z)
    if rep == "S1": return s1_dwt_tokens(z)
    if rep == "S2": return s2_dwt_large(rgb)
    if rep == "S3": return s3_rgb64(rgb, device)
    if rep == "S4": return s4_rgb128(rgb, device)
    if rep == "S5": return s5_rgb224(rgb, device)
    if rep == "S6": return s6_dino_tokens(rgb, device)
    if rep == "S8": return s8_multiscale(rgb, z, device)
    raise ValueError(rep)


def gray_extract(rep, item, latents, device):
    si, img_path, lat_idx = item
    from .features import load_image as li, to_grayscale
    rgb = li(img_path, 512).to(device)
    g = to_grayscale(rgb)
    from .features import s3_rgb64, s4_rgb128, s5_rgb224, s6_dino_tokens
    if rep == "S3": return s3_rgb64(g, device)
    if rep == "S4": return s4_rgb128(g, device)
    if rep == "S5": return s5_rgb224(g, device)
    if rep == "S6": return s6_dino_tokens(g, device)
    raise ValueError(rep)


def ncm(train_f, train_y, test_f):
    mu = train_f.mean(0)
    sd = train_f.std(0).clamp_min(1e-4)
    tr = (train_f - mu) / sd
    te = (test_f - mu) / sd
    means = torch.stack([tr[train_y == c].mean(0) for c in range(5)])
    return torch.cdist(te, means).argmin(1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()

    pick = load_paintings(n_train=N_TRAIN, n_hold=N_HOLD)
    report = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "counts": {s: {"train": len(pick[s]["train"]),
                             "holdout": len(pick[s]["holdout"])}
                         for s in STYLES}}

    reps = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S8"]
    datasets = {}
    for split in ("train", "holdout"):
        items = pick[split]
        print(f"[{split}] {len(items)} paintings", flush=True)
        feats = {r: [] for r in reps}
        labels = []
        for n_done, (si, img_path, lat_idx) in enumerate(items):
            for r in reps:
                feats[r].append(extract(r, (si, img_path, lat_idx),
                                        pick["latents"], device).half().cpu())
            labels.append(si)
            if (n_done + 1) % 50 == 0:
                print(f"  {n_done+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        for r in reps:
            datasets[(split, r)] = torch.cat(feats[r]).float()
        datasets[(split, "labels")] = torch.tensor(labels)

    # ---- per-representation evaluation ----
    report["results"] = {}
    for r in reps:
        tr = datasets[("train", r)]
        ho = datasets[("holdout", r)]
        tr_y = datasets[("train", "labels")]
        ho_y = datasets[("holdout", "labels")]
        n_tr_p = tr.shape[0] // 64
        n_ho_p = ho.shape[0] // 64

        # painting-level means
        tr_pm = tr.reshape(n_tr_p, 64, -1).mean(1)
        ho_pm = ho.reshape(n_ho_p, 64, -1).mean(1)

        # NCM accuracy
        pred = ncm(tr_pm, tr_y, ho_pm)
        acc = (pred == ho_y).float().mean().item()

        # correct-vs-wrong margin
        mu, sd = tr_pm.mean(0), tr_pm.std(0).clamp_min(1e-4)
        e = Fn.normalize((ho_pm - mu) / sd, dim=1)
        mn = torch.nn.functional.normalize(
            torch.stack([tr_pm[tr_y == c].mean(0) for c in range(5)]), dim=1)
        sims = e @ mn.T
        correct = torch.gather(sims, 1, ho_y[:, None].to(sims.device)).squeeze(1).cpu()
        wrong = sims.cpu().scatter(1, ho_y[:, None].to(sims.device),
                                   float("-inf")).max(1).values.cpu()
        margin = (correct - wrong).mean().item()

        # painting-clustered bootstrap for margin
        g = torch.Generator().manual_seed(0)
        boots = [margin]  # will do properly below
        diffs = correct - wrong
        boots = [diffs[torch.randint(0, len(diffs), (len(diffs),),
                                     generator=g)].mean().item()
                 for _ in range(N_BOOT)]
        ci_lo = float(np.quantile(boots, 0.025))
        ci_hi = float(np.quantile(boots, 0.975))

        e_dict = {"acc_painting": round(acc, 4), "margin": round(margin, 4),
                  "ci95": [round(ci_lo, 4), round(ci_hi, 4)]}

        # grayscale retention (RGB spaces)
        if r in ("S3", "S4", "S5", "S6"):
            ho_g_key = ("holdout_gray", r) if ("holdout_gray", r) in datasets \
                else None
        report["results"][r] = e_dict

    # ---- gray retention pass (RGB spaces) ----
    gray_feats = {}
    for r in ("S3", "S4", "S5", "S6"):
        gf = []
        for split in ("train", "holdout"):
            items = pick[split]
            for n_done, (si, img_path, lat_idx) in enumerate(items):
                from .features import load_image as li, to_grayscale
                rgb = li(img_path, 512).to(device)
                g = to_grayscale(rgb)
                from .features import s3_rgb64, s4_rgb128, s5_rgb224, s6_dino_tokens
                if r == "S3": gf.append(s3_rgb64(g, device).half().cpu())
                elif r == "S4": gf.append(s4_rgb128(g, device).half().cpu())
                elif r == "S5": gf.append(s5_rgb224(g, device).half().cpu())
                elif r == "S6": gf.append(s6_dino_tokens(g, device).half().cpu())
        gray_feats[r] = torch.cat(gf).float()

    report["gray_retention"] = {}
    for r in ("S3", "S4", "S5", "S6"):
        tr = datasets[("train", r)]
        ho = datasets[("holdout", r)]
        tr_y = datasets[("train", "labels")]
        ho_y = datasets[("holdout", "labels")]
        acc_orig = report["results"][r]["acc_painting"]
        acc_gray = ncm(tr, tr_y, gray_feats[r], ho_y).item()
        report["gray_retention"][r] = round(acc_gray / max(acc_orig, 1e-6), 3)

    print("[scan] gray_retention", report["gray_retention"], flush=True)

    # ---- verdict ----
    r = report["results"]
    report["verdict"] = {
        "H1_scale": bool(r["S5"]["acc_painting"] > r["S1"]["acc_painting"] + 0.10),
        "H2_dwt_vs_rgb": bool(r["S5"]["acc_painting"] > r["S2"]["acc_painting"] + 0.10),
        "dino_ceiling": r["S6"]["acc_painting"],
        "S0_baseline": r["S0"]["acc_painting"],
    }
    print("[scan] verdict", json.dumps(report["verdict"], indent=1), flush=True)

    (RUNS / "scale_scan.json").write_text(json.dumps(report, indent=1))
    print("saved ->", RUNS / "scale_scan.json")


if __name__ == "__main__":
    main()
