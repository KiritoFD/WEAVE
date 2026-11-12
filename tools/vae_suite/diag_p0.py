"""WEAVE-VAE V2 Diagnosis: P0 Code & Data Audit.

Outputs:
1. Per-style sample counts, artist counts, valid cross-artist positive rate.
2. Missing classes / single-artist classes.
3. Reference Bank composition & refresh integrity.
4. SWD metric correctness: self-distance, symmetry, non-negativity, finite-difference.
5. Projection seed stability: AUROC std across 5 seeds; 32/64/128 direction comparison.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
from sklearn.metrics import roc_auc_score

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CALIB = Path("/root/Workspace/WEAVE_VAE/swd_calibration.json")


def load_manifest():
    return json.loads(MANIFEST.read_text())


def p0_1_dataset_bank_audit():
    print("\n===== P0.1 Dataset & Reference Bank Audit =====", flush=True)
    m = load_manifest()
    val_idx = set(m["val_indices"])
    records = m["records"]
    id_to_style = {v: k for k, v in m["style_to_id"].items()}

    train_records = [r for r in records if r["raw_index"] not in val_idx]
    val_records = [r for r in records if r["raw_index"] in val_idx]

    # Per-style stats in training set
    style_samples = defaultdict(list)
    for r in train_records:
        style_samples[r["style_id"]].append(r)

    per_style = {}
    total_anchors = 0
    valid_cross_artist = 0

    for sid in range(m["num_styles"]):
        recs = style_samples.get(sid, [])
        artists = [r["artist_id"] for r in recs if r.get("artist_id", -1) >= 0]
        artist_counter = Counter(artists)
        n_samples = len(recs)
        n_artists = len(artist_counter)

        # An anchor has a valid cross-artist positive if its style has >=2 artists
        # and at least one other sample from a different artist
        anchors_here = 0
        valid_here = 0
        for r in recs:
            a = r.get("artist_id", -1)
            if a < 0:
                continue
            anchors_here += 1
            # exists another sample with different artist
            if any(x.get("artist_id", -1) >= 0 and x.get("artist_id", -1) != a for x in recs):
                valid_here += 1

        total_anchors += anchors_here
        valid_cross_artist += valid_here

        per_style[id_to_style.get(sid, f"style_{sid}")] = {
            "style_id": sid,
            "n_train_samples": n_samples,
            "n_artists": n_artists,
            "bank_refs_planned": min(12, n_samples),
            "anchors_with_valid_cross_artist_pos": valid_here,
            "valid_positive_rate": round(valid_here / max(anchors_here, 1), 4),
            "is_single_artist": n_artists <= 1,
            "artist_distribution": dict(artist_counter.most_common(10)),
        }

    # Single-artist / scarce classes
    single_artist_styles = [s for s, v in per_style.items() if v["is_single_artist"]]
    scarce_styles = {s: v["n_train_samples"] for s, v in per_style.items() if v["n_train_samples"] < 12}

    # Bank composition
    bank_total = sum(v["bank_refs_planned"] for v in per_style.values())

    audit = {
        "num_styles": m["num_styles"],
        "train_images": len(train_records),
        "val_images": len(val_records),
        "bank_total_refs": bank_total,
        "bank_total_expected_136x12": 136 * 12,
        "bank_deficit": 136 * 12 - bank_total,
        "overall_valid_cross_artist_positive_rate": round(valid_cross_artist / max(total_anchors, 1), 4),
        "total_anchors": total_anchors,
        "valid_cross_artist_anchors": valid_cross_artist,
        "single_artist_styles": single_artist_styles,
        "single_artist_style_count": len(single_artist_styles),
        "scarce_styles_under_12": scarce_styles,
        "scarce_style_count": len(scarce_styles),
        "per_style": per_style,
    }

    print(f"  Styles: {m['num_styles']} | Train: {len(train_records)} | Val: {len(val_records)}")
    print(f"  Bank refs: {bank_total} (expected 1632, deficit {1632 - bank_total})")
    print(f"  Valid cross-artist positive rate: {audit['overall_valid_cross_artist_positive_rate']*100:.2f}% "
          f"({valid_cross_artist}/{total_anchors})")
    print(f"  Single-artist styles ({len(single_artist_styles)}): {single_artist_styles}")
    print(f"  Scarce styles (<12 imgs, {len(scarce_styles)}): {scarce_styles}")

    (OUT_DIR / "01_code_bank_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def p0_2_swd_correctness():
    print("\n===== P0.2 SWD Metric Correctness =====", flush=True)
    calib = json.loads(CALIB.read_text())
    desc = LatentSWDDescriptor()
    desc.set_calibration(
        torch.tensor(calib["chan_mean"]), torch.tensor(calib["chan_std"]),
        calib["scale_swd1"], calib["scale_swd2"], calib["scale_moment"],
        calib["median_inter_dist"],
    )

    torch.manual_seed(0)
    z1 = torch.randn(1, 4, 64, 64)
    z2 = torch.randn(1, 4, 64, 64)

    q1a, q2a, ma = desc(z1)
    q1b, q2b, mb = desc(z2)

    d_self = desc.compute_pairwise_distance(q1a, q2a, ma, q1a, q2a, ma).item()
    d_12 = desc.compute_pairwise_distance(q1a, q2a, ma, q1b, q2b, mb).item()
    d_21 = desc.compute_pairwise_distance(q1b, q2b, mb, q1a, q2a, ma).item()

    # Differentiable check
    z_req = torch.randn(2, 4, 64, 64, requires_grad=True)
    q1r, q2r, mr = desc(z_req)
    dmat = desc.compute_pairwise_distance(q1r[0:1], q2r[0:1], mr[0:1], q1r[1:2], q2r[1:2], mr[1:2])
    dmat.backward()
    grad_ok = z_req.grad is not None
    grad_nonzero = float((z_req.grad != 0).float().mean().item()) if grad_ok else 0.0
    grad_norm = float(z_req.grad.norm().item()) if grad_ok else 0.0

    # Finite difference check on one latent element
    eps = 1e-4
    z_a = torch.randn(1, 4, 64, 64)
    z_b = torch.randn(1, 4, 64, 64)
    z_a_req = z_a.clone().requires_grad_(True)
    qa1, qa2, mma = desc(z_a_req)
    qb1, qb2, mmb = desc(z_b)
    d0 = desc.compute_pairwise_distance(qa1, qa2, mma, qb1, qb2, mmb)
    d0.backward()
    autograd_g = z_a_req.grad[0, 0, 0, 0].item()

    z_a_p = z_a.clone()
    z_a_p[0, 0, 0, 0] += eps
    with torch.no_grad():
        pa1, pa2, pma = desc(z_a_p)
        dp = desc.compute_pairwise_distance(pa1, pa2, pma, qb1, qb2, mmb).item()
    fd_g = (dp - d0.item()) / eps

    # Patch / projection / quantile counts
    patch_count = desc.num_samples
    proj_count = desc.num_projections
    quant_count = desc.num_quantiles
    q_shape = tuple(q1a.shape)

    res = {
        "self_distance": round(d_self, 8),
        "self_distance_near_zero": bool(d_self < 1e-6),
        "symmetry_abs_diff": round(abs(d_12 - d_21), 10),
        "is_symmetric": bool(abs(d_12 - d_21) < 1e-8),
        "is_nonnegative": bool(d_12 >= 0 and d_21 >= 0),
        "d_12": round(d_12, 6),
        "grad_exists": grad_ok,
        "grad_nonzero_pct": round(grad_nonzero * 100, 2),
        "grad_norm": round(grad_norm, 8),
        "autograd_grad": round(autograd_g, 8),
        "finite_diff_grad": round(fd_g, 8),
        "fd_vs_autograd_rel_error": round(abs(fd_g - autograd_g) / max(abs(autograd_g), 1e-8), 6),
        "patch_samples": patch_count,
        "projections": proj_count,
        "quantiles": quant_count,
        "descriptor_shape": q_shape,
        "proj_directions_unit_norm": bool(
            torch.allclose(desc.proj_directions.norm(dim=0), torch.ones(proj_count), atol=1e-5)
        ),
    }
    print(f"  d(z,z) = {res['self_distance']:.2e} (near zero: {res['self_distance_near_zero']})")
    print(f"  Symmetry |d12-d21| = {res['symmetry_abs_diff']:.2e} (symmetric: {res['is_symmetric']})")
    print(f"  Non-negative: {res['is_nonnegative']}")
    print(f"  Grad nonzero: {res['grad_nonzero_pct']}% | norm {res['grad_norm']:.2e}")
    print(f"  FD vs Autograd rel error: {res['fd_vs_autograd_rel_error']:.4e}")
    print(f"  Patches={patch_count} Proj={proj_count} Quantiles={quant_count} Shape={q_shape}")
    print(f"  Projections unit-norm: {res['proj_directions_unit_norm']}")
    return res


def p0_2b_projection_stability(vae, dataset, base_calib):
    print("\n===== P0.2b Projection Seed Stability & Direction Count =====", flush=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    z_all, y_all = [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= 16:
                break
            post = vae.encode(b["image"]).latent_dist
            z_all.append(post.mean.float())
            y_all.append(b["style_id"])
    z = torch.cat(z_all)
    y = torch.cat(y_all).numpy()

    rng = np.random.default_rng(42)
    N = len(y)
    same, diff = [], []
    for i in range(N):
        sc = np.where(y == y[i])[0]
        sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
        dc = np.where(y != y[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
    n = min(len(same), len(diff))
    ia = np.concatenate([np.array(same[:n])[:, 0], np.array(diff[:n])[:, 0]])
    ib = np.concatenate([np.array(same[:n])[:, 1], np.array(diff[:n])[:, 1]])
    lbs = np.concatenate([np.ones(n), np.zeros(n)])

    results = {}
    for seed in [42, 7, 123, 2024, 999]:
        for nproj in [32, 64, 128]:
            if seed != 42 and nproj != 32:
                continue
            desc = LatentSWDDescriptor(num_projections=nproj, seed=seed)
            desc.set_calibration(
                torch.tensor(base_calib["chan_mean"]), torch.tensor(base_calib["chan_std"]),
                base_calib["scale_swd1"], base_calib["scale_swd2"], base_calib["scale_moment"],
                base_calib["median_inter_dist"],
            )
            with torch.no_grad():
                q1, q2, mm = desc(z)
                ds = []
                for c in range(0, len(ia), 64):
                    ca, cb = ia[c:c+64], ib[c:c+64]
                    d = desc.compute_pairwise_distance(q1[ca], q2[ca], mm[ca], q1[cb], q2[cb], mm[cb])
                    ds.append(torch.diag(d))
                dall = torch.cat(ds).numpy()
            auc = float(roc_auc_score(lbs, -dall) * 100.0)
            key = f"seed{seed}_proj{nproj}"
            results[key] = round(auc, 2)
            print(f"  {key}: AUROC = {auc:.2f}%")

    seed42_aucs = [v for k, v in results.items() if "proj32" in k]
    std = float(np.std(seed42_aucs))
    print(f"  Seed std (32 proj): {std:.3f} pp")
    results["seed_std_32proj"] = round(std, 3)
    return results


def main():
    manifest_audit = p0_1_dataset_bank_audit()
    swd_audit = p0_2_swd_correctness()

    calib = json.loads(CALIB.read_text())
    dataset = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    vae = AutoencoderKL.from_pretrained(
        "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
    ).eval()
    proj_audit = p0_2b_projection_stability(vae, dataset, calib)

    combined = {
        "dataset_bank": manifest_audit,
        "swd_correctness": swd_audit,
        "projection_stability": proj_audit,
    }
    (OUT_DIR / "01_code_bank_audit.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUT_DIR / '01_code_bank_audit.json'}")


if __name__ == "__main__":
    main()
