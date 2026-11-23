"""Same-style grayscale donor swap with fixed source, seed, class and gamma."""

import argparse

import torch
import torch.nn.functional as F

from exp.v18 import STYLES, tasks_and_sources
from exp.v19.fiber_duality import (
    StyleFiberEncoder,
    extract_color_fiber_27d,
    extract_dino_rgb_moments_768d,
)
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27_mainline import OUT
from exp.v27_mainline.dir1_reflow import SLFUNetGamma
from exp.v27_mainline.followup.frozen import POOL, RESULTS, DEVICE, conditions, rollout
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import sanitize_cuda_env, save_json
from src.utils.structure import dwt2_haar, rgb_to_gray


def band_signature(gray):
    low, *fine = dwt2_haar(gray)
    _, *mid = dwt2_haar(low)
    return torch.stack([x.square().mean((1, 2, 3)).clamp_min(1e-10).log()
                        for x in fine + mid], dim=-1)


def normalized_gray(rgb):
    gray = rgb_to_gray(rgb)
    return ((gray - gray.mean((2, 3), keepdim=True))
            / gray.std((2, 3), keepdim=True).clamp_min(.01) * .15 + .5).clamp(0, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    args = ap.parse_args()
    sanitize_cuda_env()
    pack = build_or_load_v21_cache(args.pool, device=DEVICE)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=DEVICE)
    priors = {k: v.cuda() if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    dec = load_s10_decoder(DEVICE)
    dino = load_dino_model(device=DEVICE)
    fnd = bundle["fnd"]
    fiber = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).cuda().eval()
    fiber.load_state_dict(fnd["fiber_state"])
    model = SLFUNetGamma().cuda().eval()
    model.load_state_dict(torch.load(OUT / "d1_reflow_v2.pt", map_location="cpu", weights_only=False)["ema"])
    tasks, z, _ = tasks_and_sources()
    y, d = get_test750_y16_and_d16(bundle)
    q = c21["test750"]["q32"]
    eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    hold = bundle["v18_cache"]["holdout"]
    out = {"intervention": "inject grayscale donor fiber in Stage 1 u_vivid instead of class prototype",
           "arms": {}}
    for tid in (0, 4):
        pool = (hold["sid"] == tid).nonzero().flatten()[:32]
        sig = []
        for ids in pool.split(8):
            rgb = decode_s10(dec, hold["z"][ids].cuda(), output_cpu=False)
            sig.append(band_signature(normalized_gray(rgb)))
        sig = torch.cat(sig)
        distance = torch.cdist(sig, sig)
        a, b = divmod(int(distance.argmax()), len(pool))
        ab = pool[[a, b]]
        rgb = decode_s10(dec, hold["z"][ab].cuda(), output_cpu=False)
        gray = normalized_gray(rgb).repeat(1, 3, 1, 1)
        u = fiber.encode(torch.cat((extract_color_fiber_27d(gray),
                        extract_dino_rgb_moments_768d(dino, gray, device=DEVICE)), -1))
        ids = [i for i, row in enumerate(tasks) if row["target_style_id"] == tid
               and row["source_style_id"] != tid][:16]
        delta_rgb, delta_texture = [], []
        ref_delta = sig[b] - sig[a]
        for idx in torch.tensor(ids).split(4):
            data = [t[idx].cuda().float() for t in (z, eps, y, d, q)]
            c = conditions(*data, torch.full((len(idx),), tid, device=DEVICE),
                           priors, bank, torch.zeros(len(idx), device=DEVICE, dtype=torch.bool))
            ep, anchor, q_in, (y_in, q_cond, labels, _), layout = c
            outputs = []
            for donor in u:
                c_donor = ep, anchor, q_in, (y_in, q_cond, labels, donor.expand(len(idx), -1)), layout
                gen = rollout(model, c_donor, (0, .25, .5, .75, 1))
                outputs.append(decode_s10(dec, gen, output_cpu=False))
            x, v = outputs
            delta_rgb.extend((x - v).abs().flatten(1).mean(1).tolist())
            delta_texture.append(band_signature(normalized_gray(v)) - band_signature(normalized_gray(x)))
        d_texture = torch.cat(delta_texture)
        out["arms"][STYLES[tid]] = {
            "donor_ids": [hold["ids"][i] for i in ab.tolist()],
            "source_tasks": [tasks[i]["task_id"] for i in ids],
            "donor_fiber_cosine": float(F.cosine_similarity(u[:1], u[1:])),
            "output_rgb_mae": sum(delta_rgb) / len(delta_rgb),
            "donor_gray_band_l2": float(ref_delta.norm()),
            "output_gray_band_l2": float(d_texture.norm(dim=-1).mean()),
            "output_to_donor_band_direction_cosine": float(F.cosine_similarity(d_texture, ref_delta[None]).mean()),
        }
    save_json(RESULTS / "reference_ab.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
