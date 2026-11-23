"""Frozen Stage 1 controls on the canonical test750, without a Same150 bypass."""

import argparse
from itertools import product

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import apply_latent_chroma_shift
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache, stiefel_polar_quotient
from exp.v27_mainline import OUT, ROOT, RawWriterV27M, run_measure_v27m
from exp.v27_mainline.dir1_reflow import ALPHA_BRIDGE, SLFUNetGamma, prepare_q32_cond_inputs
from exp.v27_mainline.part_bank import PART_CACHE, PartStyleBank
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast, sanitize_cuda_env, save_json
from src.utils.structure import extract_whitened_dino_layout, rgb_to_gray

POOL = ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"
RESULTS = OUT / "followup"
DEVICE = "cuda"


def conditions(z, eps, y, d, q, tid, priors, bank, same):
    y_in, q_in, u, layout = prepare_q32_cond_inputs(y, d, q, tid, bank)
    anchor = apply_latent_chroma_shift(z, tid, priors, alpha=1.0)
    anchor = torch.where(same[:, None, None, None], z, anchor)
    return eps, anchor, q_in, (y_in, q_in, tid, u), layout


def rollout(model, cond, nodes, mode="global", gamma=1.0, alpha=ALPHA_BRIDGE):
    eps, anchor, q, args, layout = cond
    if mode != "global":
        local = F.avg_pool2d(q, 3, stride=1, padding=1, count_include_pad=False)
        salience = (q - local).square().mean(1, keepdim=True).sqrt()
        center = salience.mean((2, 3), keepdim=True)
        scale = salience.std((2, 3), keepdim=True).clamp_min(1e-4)
        delta = 0.18 * F.interpolate(((salience - center) / scale).tanh(), size=eps.shape[-2:],
                                     mode="bilinear", align_corners=False)
        alpha = alpha + delta
    z = eps.lerp(anchor, alpha)
    g = torch.full_like(layout, gamma)
    for a, b in zip(nodes, nodes[1:]):
        dt = b - a
        if mode == "clock":
            # Keep four scalar time embeddings. Warp local integration increments.
            dt = dt * (1 + (delta / 0.18) * 0.65 * (a + b - 1))
        with bf16_autocast(DEVICE):
            v = model(z, *args, torch.full_like(layout, a), layout_scale=layout, gamma=g)
        z = z + dt * v.float()
    return z


@torch.no_grad()
def fit_grid(model, bundle, c21, priors, bank):
    tr = bundle["v18_cache"]["train"]
    gen = torch.Generator().manual_seed(20261123)
    idx = torch.randperm(len(tr["z"]), generator=gen)[:96]
    z = tr["z"][idx].cuda().float()
    eps = torch.randn(z.shape, generator=gen).cuda()
    sid = tr["sid"][idx].cuda().long()
    tid = (sid + torch.randint(1, 5, sid.shape, generator=gen).cuda()) % 5
    y = coarse_y16_lcn(tr["y_64"][idx].cuda().float())
    d = tr["dino_16"][idx].cuda().float()
    q = c21["train"]["q32"][idx].cuda().float()
    batches = [conditions(z[s:s+12], eps[s:s+12], y[s:s+12], d[s:s+12], q[s:s+12],
                          tid[s:s+12], priors, bank, (sid == tid)[s:s+12])
               for s in range(0, len(idx), 12)]
    fine = [rollout(model, c, [i/16 for i in range(17)], gamma=g)
            for g, c in product((0.0, 0.5, 1.0), batches)]
    scores = []
    for mid in product((0.125, 0.25, 0.375), (0.375, 0.5, 0.625), (0.625, 0.75, 0.875)):
        nodes = (0.0, *mid, 1.0)
        if min(b-a for a, b in zip(nodes, nodes[1:])) < 0.125:
            continue
        errors = [float((rollout(model, c, nodes, gamma=g) - z_ref).square().mean())
                  for (g, c), z_ref in zip(product((0.0, 0.5, 1.0), batches), fine)]
        scores.append({"nodes": nodes, "mse": sum(errors) / len(errors)})
    return sorted(scores, key=lambda row: row["mse"])


@torch.no_grad()
def routing_probe(z, q, c21, dec):
    """Known horizontal-flip correspondence. Texture measured independently of Q32."""
    saved = torch.load(PART_CACHE, map_location="cpu", weights_only=False)
    router = PartStyleBank(saved["centers"], saved["bank"]).cuda()
    dino = load_dino_model(device=DEVICE)
    rgb = decode_s10(dec, z[:24].cuda(), output_cpu=False)
    rgb_ref = rgb.flip(-1).contiguous()
    d_ref = extract_whitened_dino_layout(dino, rgb_ref, device=DEVICE)
    q_ref = stiefel_polar_quotient(d_ref, c21["W_stiefel"]).cuda()
    del dino
    pi_s = router.assign(q[:24].cuda())
    pi_r = router.assign(q_ref)
    plan = pi_s @ pi_r.transpose(1, 2)
    plan = plan / plan.sum(-1, keepdim=True).clamp_min(1e-8)
    oracle_idx = torch.arange(256, device=DEVICE).reshape(16, 16).flip(-1).flatten()
    gray = rgb_to_gray(rgb_ref)
    high = gray - F.avg_pool2d(gray, 5, stride=1, padding=2, count_include_pad=False)
    texture = F.avg_pool2d(high.square(), 32).flatten(2).transpose(1, 2).sqrt()
    target = texture[:, oracle_idx]
    routed = plan @ texture
    # Match amplitude and marginal texture distribution before comparing spatial placement.
    ordered = routed.argsort(dim=1)
    matched = torch.empty_like(routed).scatter_(1, ordered, target.sort(dim=1).values)
    idx = torch.randperm(256, generator=torch.Generator().manual_seed(EVAL_SEED)).cuda()
    return {
        "study": "24 fixed test tasks, RGB horizontal flip; independently re-encoded Q32",
        "oracle_mae": 0.0,
        "q32_raw_texture_mae": float((routed - target).abs().mean()),
        "q32_equal_marginal_texture_mae": float((matched - target).abs().mean()),
        "shuffled_equal_marginal_texture_mae": float((matched[:, idx] - target).abs().mean()),
        "q32_oracle_patch_top1": float((plan.argmax(-1) == oracle_idx).float().mean()),
        "q32_mean_probability_at_oracle_patch": float(plan.gather(2, oracle_idx[None, :, None].expand(len(plan), -1, -1)).mean()),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--pure-noise-control", action="store_true",
                    help="Evaluate the frozen weights with the source latent removed from the start.")
    args = ap.parse_args()
    sanitize_cuda_env()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pack = build_or_load_v21_cache(args.pool, device=DEVICE)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=DEVICE)
    priors = {k: v.cuda() if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    model = SLFUNetGamma().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(OUT / "d1_reflow_v2.pt", map_location="cpu", weights_only=False)["ema"])
    tasks, z, _ = tasks_and_sources()
    y, d = get_test750_y16_and_d16(bundle)
    q = c21["test750"]["q32"]
    eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    same = torch.tensor([t["target_style_id"] == t["source_style_id"] for t in tasks])
    dec = load_s10_decoder(DEVICE)
    report = {"stage1_reference_channel": "absent", "arms": {}}
    uniform = (0, .25, .5, .75, 1)
    if args.pure_noise_control:
        output = RESULTS / "pure_noise_control.json"
        specs = [("pure_noise", uniform, "global")]
        alpha = 0.0
    else:
        output = RESULTS / "step_a.json"
        report["routing"] = routing_probe(z, q, c21, dec)
        save_json(output, report)
        print("ROUTING", report["routing"], flush=True)
        report["grid_train_scores"] = fit_grid(model, bundle, c21, priors, bank)
        tuned = report["grid_train_scores"][0]["nodes"]
        save_json(output, report)
        print("GRID", tuned, flush=True)
        specs = [("global", uniform, "global"), ("grid", tuned, "global"),
                 ("start", uniform, "start"), ("clock", uniform, "clock")]
        alpha = ALPHA_BRIDGE
    report["alpha_bridge"] = alpha
    for label, grid, mode in specs:
        arm = f"followup_a_{label}"
        wr = RawWriterV27M(arm, dec=dec, device=DEVICE)
        same_mae = []
        same_rgb_mae = []
        for s in range(0, len(tasks), 12):
            sl = slice(s, s+12)
            batch = [x[sl].cuda().float() for x in (z, eps, y, d, q)]
            out = rollout(model, conditions(*batch, tid[sl].cuda(), priors, bank, same[sl].cuda()),
                          grid, mode, alpha=alpha)
            wr.write_batch(out, tasks[sl])
            mask = same[sl].cuda()
            if mask.any():
                same_mae.extend((out[mask] - batch[0][mask]).abs().flatten(1).mean(1).tolist())
                ref, gen = decode_s10(dec, torch.cat((batch[0][mask], out[mask]))).chunk(2)
                same_rgb_mae.extend((ref - gen).abs().flatten(1).mean(1).tolist())
        wr.finalize(arm=arm, steps=4, nodes=grid, same_bypass=False, alpha_bridge=alpha)
        report["arms"][label] = run_measure_v27m(arm, device=DEVICE)
        report["arms"][label]["same150_latent_mae"] = sum(same_mae) / len(same_mae)
        report["arms"][label]["same150_s10_rgb_mae"] = sum(same_rgb_mae) / len(same_rgb_mae)
        save_json(output, report)


if __name__ == "__main__":
    main()
