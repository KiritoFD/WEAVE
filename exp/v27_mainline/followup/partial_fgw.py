"""Partial fused Gromov-Wasserstein geometry control on known RGB correspondences."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache, stiefel_polar_quotient
from exp.v27_mainline.followup.frozen import POOL, RESULTS, DEVICE
from exp.v27_mainline.part_bank import PART_CACHE, PartStyleBank
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.utils.runtime import save_json
from src.utils.structure import extract_whitened_dino_layout, rgb_to_gray


def partial_assignment(cost, keep):
    n = cost.shape[0]
    augmented = np.zeros((2 * n - keep, 2 * n - keep))
    augmented[:n, :n] = cost
    augmented[n:, n:] = np.inf
    src, ref = linear_sum_assignment(augmented)
    valid = (src < n) & (ref < n)
    p = np.zeros_like(cost)
    p[src[valid], ref[valid]] = 1
    return p


def partial_fgw(cost, dist, keep, weight=.6, iterations=10):
    p = partial_assignment(cost, keep).astype(np.float64)
    d2 = dist ** 2

    def product(plan):
        src = d2 @ plan.sum(axis=1)
        ref = d2 @ plan.sum(axis=0)
        return src[:, None] + ref[None] - 2 * dist @ plan @ dist.T

    for _ in range(iterations):
        grad = (1 - weight) * cost + 2 * weight * product(p) / keep
        q = partial_assignment(grad, keep)
        step = q - p
        linear = np.sum(grad * step)
        quadratic = weight * np.sum(product(step) * step) / keep
        if linear >= -1e-8:
            break
        rho = np.clip(-linear / (2 * quadratic), 0, 1) if quadratic > 0 else 1
        p += rho * step
    return p


@torch.no_grad()
def main():
    pack = build_or_load_v21_cache(str(POOL), device=DEVICE)
    tasks, latents, _ = tasks_and_sources()
    unique = {}
    for i, row in enumerate(tasks):
        unique.setdefault(row["source_stem"], i)
    ids = list(unique.values())[::6][:24]
    saved = torch.load(PART_CACHE, map_location="cpu", weights_only=False)
    router = PartStyleBank(saved["centers"], saved["bank"]).cuda()
    dec = load_s10_decoder(DEVICE)
    dino = load_dino_model(device=DEVICE)
    dist_grid = np.mgrid[:16, :16].reshape(2, -1).T / 15.
    dist = np.linalg.norm(dist_grid[:, None] - dist_grid[None], axis=-1) / np.sqrt(2.)
    oracle = np.arange(256).reshape(16, 16)[:, ::-1].ravel()
    gen = np.random.default_rng(20261123)
    keep = 205
    results = []
    c21 = pack["c21"]
    for idx in torch.tensor(ids).split(4):
        rgb = decode_s10(dec, latents[idx].cuda(), output_cpu=False)
        ref = rgb.flip(-1).contiguous()
        d_ref = extract_whitened_dino_layout(dino, ref, device=DEVICE)
        q_ref = stiefel_polar_quotient(d_ref, c21["W_stiefel"]).cuda()
        pi_s = router.assign(c21["test750"]["q32"][idx].cuda())
        pi_r = router.assign(q_ref)
        score = (pi_s @ pi_r.transpose(1, 2)).cpu().numpy()
        gray = rgb_to_gray(ref)
        high = gray - F.avg_pool2d(gray, 5, stride=1, padding=2, count_include_pad=False)
        textures = F.avg_pool2d(high.square(), 32).flatten(1).sqrt().cpu().numpy()
        for affinity, texture in zip(score, textures):
            cost = 1 - affinity / max(float(affinity.max()), 1e-8)
            local = partial_assignment(cost, keep)
            gw = partial_fgw(cost, dist, keep)
            assignment = {"oracle_full": oracle, "q32_soft_top1_full": affinity.argmax(-1),
                          "shuffled_full": gen.permutation(256)}
            masks = {key: np.ones(256, dtype=bool) for key in assignment}
            for key, matrix in (("q32_partial_linear", local), ("q32_partial_fgw", gw)):
                assignment[key] = matrix.argmax(1)
                masks[key] = matrix.sum(1) >= .5
            row = {}
            for key, r in assignment.items():
                m = masks[key]
                row[key] = {
                    "coverage": float(m.mean()),
                    "oracle_patch_top1_kept": float((r[m] == oracle[m]).mean()),
                    "oracle_patch_top1_total": float(((r == oracle) & m).mean()),
                    "texture_mae_kept": float(np.abs(texture[r[m]] - texture[oracle[m]]).mean()),
                }
            results.append(row)
    summary = {arm: {metric: float(np.mean([r[arm][metric] for r in results])) for metric in values}
               for arm, values in results[0].items()}
    save_json(RESULTS / "partial_fgw.json", {
        "study": "24 distinct test sources, horizontal flip oracle",
        "partial_mass": keep / 256, "geometry_weight": .6,
        "sources": [tasks[i]["source_stem"] for i in ids], "arms": summary,
    })
    print(summary, flush=True)


if __name__ == "__main__":
    main()
