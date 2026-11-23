"""Matched ER/Uki real-endpoint pilot with one shared destylized source pool."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import to_tensor

from exp.v18 import STYLES, canonical_pid, tasks_and_sources
from exp.v19.fiber_duality import extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v19.fiber_duality import apply_latent_chroma_shift
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache, stiefel_polar_quotient
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27_mainline import OUT, ROOT, Q32_FLAGSHIP_CKPT, RawWriterV27M, run_measure_v27m
from exp.v27_mainline.dir1_reflow import (
    ALPHA_BRIDGE, SLFUNetGamma, _build_train_anchors, _integrate_q32_seg, prepare_q32_cond_inputs,
)
from exp.v27_mainline.rollout_refine import latent_y32_lcn_loss
from exp.v27_mainline.followup.frozen import POOL, RESULTS, DEVICE, conditions, rollout
from src.utils import load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast, sanitize_cuda_env, save_json, seed_all
from src.utils.structure import dwt2_haar, extract_whitened_dino_layout, rgb_to_gray, rgb_to_structure_64

SEED = 20261124
GRID = (0, .25, .5, .75, 1)


def sharp_source(gray, tid):
    """Bilateral luma smoothing with an alternate low-chroma palette and no resampling."""
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    num, den = torch.zeros_like(gray), torch.zeros_like(gray)
    h, w = gray.shape[-2:]
    for dy in range(3):
        for dx in range(3):
            neighbor = padded[..., dy:dy+h, dx:dx+w]
            spatial = 1.0 / (1 + (dy - 1)**2 + (dx - 1)**2)
            weight = spatial * torch.exp(-0.5 * ((neighbor - gray) / .045).square())
            num += weight * neighbor
            den += weight
    y = num / den
    # Imp-like cool tint for ER, near-neutral warm tint for Uki.
    palette = gray.new_tensor([[-.035, .003, .087], [.026, -.006, -.017]])
    gamut = (4 * y * (1 - y)).clamp(0, 1)
    return (y + gamut * palette[(tid == 4).long(), :, None, None]).clamp(0, 1)


@torch.no_grad()
def build_pairs(args, bundle, c21, bank):
    version = "sharp_v2" if args.sharp_bridge else "v1"
    cache = RESULTS / f"endpoint_pairs_{version}_n{args.per_style}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    tr = bundle["v18_cache"]["train"]
    hold = bundle["v18_cache"]["holdout"]
    gen = torch.Generator().manual_seed(SEED)
    style_ids = (0, 4)
    files = {
        (sid, canonical_pid(path.name)): path
        for sid in style_ids
        for path in sorted((args.images / STYLES[sid]).iterdir())
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    }
    selected = {}
    for split, data, n in (("train", tr, args.per_style), ("holdout", hold, 16)):
        idx = []
        for tid in style_ids:
            style_rows = torch.tensor([
                i for i, (name, sid) in enumerate(zip(data["ids"], data["sid"].tolist()))
                if sid == tid and (sid, canonical_pid(name)) in files
            ])
            if len(style_rows) < n:
                raise ValueError(f"{split}/{STYLES[tid]} has only {len(style_rows)} matched RGB assets for {n} pairs")
            print(f"MATCHED {split}/{STYLES[tid]}: {len(style_rows)}, selecting {n}", flush=True)
            n_candidates = min(len(style_rows), n + 48) if args.sharp_bridge else n
            idx.append(style_rows[torch.randperm(len(style_rows), generator=gen)[:n_candidates]])
        selected[split] = (data, torch.cat(idx))
    vae = AutoencoderKL.from_pretrained(args.encoder, torch_dtype=torch.float16, local_files_only=True).cuda().eval()
    dino = load_dino_model(device=DEVICE)
    prepared = {}
    for split, (data, idx) in selected.items():
        ids = [data["ids"][i] for i in idx.tolist()]
        tid = data["sid"][idx].long()
        chunks = {k: [] for k in ("src", "target", "y", "d", "q", "c", "fiber", "gray_layout_cos",
                                   "gray_hf_ratio")}
        for start in range(0, len(idx), 8):
            sl = slice(start, start+8)
            images = []
            for name, sid in zip(ids[sl], tid[sl].tolist()):
                with Image.open(files[sid, canonical_pid(name)]) as im:
                    images.append(to_tensor(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)))
            rgb = torch.stack(images).cuda()
            gray = rgb_to_gray(rgb)
            if args.sharp_bridge:
                source = sharp_source(gray, tid[sl].cuda())
            else:
                smooth = F.interpolate(F.interpolate(gray, size=(96, 96), mode="bilinear", antialias=True),
                                       size=(512, 512), mode="bicubic", align_corners=False).clamp(0, 1)
                source = smooth.expand(-1, 3, -1, -1).contiguous()
            src = vae.encode((source * 2 - 1).half()).latent_dist.mean * vae.config.scaling_factor
            target = vae.encode((rgb * 2 - 1).half()).latent_dist.mean * vae.config.scaling_factor
            y_source = rgb_to_gray(source)
            y_map = F.avg_pool2d(y_source, 8)
            d = extract_whitened_dino_layout(dino, source, device=DEVICE)
            q = stiefel_polar_quotient(d, c21["W_stiefel"])
            d_raw = extract_whitened_dino_layout(dino, rgb, device=DEVICE)
            hf = lambda x: (x - F.avg_pool2d(x, 5, 1, 2, count_include_pad=False)).square().mean((1, 2, 3))
            vals = (src, target, y_map, d, q, rgb_to_structure_64(source),
                    torch.cat((extract_color_fiber_27d(source), extract_dino_rgb_moments_768d(dino, source, device=DEVICE)), dim=-1),
                    F.cosine_similarity(d.flatten(1), d_raw.flatten(1)),
                    hf(y_source) / hf(gray).clamp_min(1e-8))
            for key, val in zip(chunks, vals):
                chunks[key].append(val.cpu().half())
        data_rows = {key: torch.cat(vals) for key, vals in chunks.items()}
        if args.sharp_bridge:
            eligible = (data_rows["gray_layout_cos"].float() >= .88) & (data_rows["gray_hf_ratio"].float() >= .65)
            picks = []
            n = args.per_style if split == "train" else 16
            for style in style_ids:
                keep = ((tid == style) & eligible).nonzero().flatten()
                print(f"SHARP GATE {split}/{STYLES[style]} passed={len(keep)} required={n}", flush=True)
                if len(keep) < n:
                    raise ValueError(f"Insufficient sharp pairs for {split}/{STYLES[style]}")
                picks.append(keep[:n])
            chosen = torch.cat(picks)
            ids = [ids[i] for i in chosen.tolist()]
            tid = tid[chosen]
            data_rows = {key: vals[chosen] for key, vals in data_rows.items()}
        prepared[split] = dict(ids=ids, tid=tid, **data_rows)
        print(f"PAIRS {split}={len(ids)}", flush=True)
    del vae, dino
    torch.cuda.empty_cache()
    train = prepared["train"]
    source_bundle = {"fnd": {**bundle["fnd"], "tr_fib": train["fiber"]}}
    anc = _build_train_anchors("p1", source_bundle, train["src"], train["d"], train["c"],
                               torch.arange(len(train["ids"])), train["tid"], DEVICE)
    teacher = make_bottleneck_net(32).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"])
    for split, rows in prepared.items():
        rows["eps"] = torch.randn(rows["src"].shape, generator=gen).half()
        if split != "train":
            continue
        rows["teacher"], rows["hi_c"] = [], []
        hi_s = dict(D5_CFG, tau_min=.30, tau_hi=.68, ws_fig=5.5, wc_fig=1.65,
                    wc_abs=.70, wsp_fig_late=3.6, wsp_abs=5.2)
        hi_c = dict(D5_CFG, tau_min=.48, tau_hi=.78, ws_fig=5.2)
        for start in range(0, len(rows["tid"]), 4):
            sl = slice(start, start+4)
            args_ = (teacher, anc[sl].cuda().float(), rows["eps"][sl].cuda().float(),
                     coarse_y16_lcn(rows["y"][sl].cuda().float()), rows["d"][sl].cuda().float(),
                     rows["q"][sl].cuda().float(), rows["tid"][sl].cuda(), bank)
            for key, full, cfg in (("teacher", {0, 1}, hi_s), ("hi_c", set(), hi_c)):
                end = _integrate_q32_seg(*args_, steps=6, full_steps=full, cfg_kw=cfg)
                rows[key].append(end.cpu().half())
        for key in ("teacher", "hi_c"):
            rows[key] = torch.cat(rows[key])
    for data in prepared.values():
        assert len({canonical_pid(n) for n in data["ids"]}) == len(data["ids"])
    assert not ({canonical_pid(n) for n in train["ids"]} &
                {canonical_pid(n) for n in prepared["holdout"]["ids"]})
    torch.save(prepared, cache)
    return prepared


def add_replay(rows, bundle, c21):
    cache = torch.load(OUT / "cache/dir1_teacher_19800_p1.pt", map_location="cpu", weights_only=False)
    gen = torch.Generator().manual_seed(SEED + 2)
    same = cache["is_same"].bool()
    tid = cache["tid"].long()
    extra_rows = []
    for style in (1, 2, 3):
        idx = ((tid == style) & ~same).nonzero().flatten()
        extra_rows.append(idx[torch.randperm(len(idx), generator=gen)[:512]])
    same_idx = same.nonzero().flatten()
    extra_rows.append(same_idx[torch.randperm(len(same_idx), generator=gen)[:512]])
    idx = torch.cat(extra_rows)
    si = cache["src_idx"][idx].long()
    train = bundle["v18_cache"]["train"]
    replay = {
        "tid": tid[idx],
        "src": train["z"][si],
        "d": train["dino_16"][si],
        "y": train["y_64"][si],
        "q": c21["train"]["q32"][si],
        "eps": cache["eps"][idx],
        "hi_c": cache["z_hiC"][idx],
        "teacher": cache["z_hiS"][idx],
        "target": cache["z_hiS"][idx],
    }
    out = {k: torch.cat((rows[k], v)) for k, v in replay.items()}
    out["is_same"] = torch.cat((torch.zeros(len(rows["tid"]), dtype=torch.bool), same[idx]))
    out["n_focus"] = len(rows["tid"])
    out["same_begin"] = out["n_focus"] + sum(len(r) for r in extra_rows[:-1])
    print(f"REPLAY focus={out['n_focus']} other={out['same_begin']-out['n_focus']} same={len(extra_rows[-1])}", flush=True)
    return out


def fit(arm, args, rows, bank, bundle):
    run = f"{arm}_sharp_bridge" if args.sharp_bridge else arm
    path = RESULTS / f"endpoint_{run}_n{args.per_style}_s{args.updates}.pt"
    model = SLFUNetGamma().cuda()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        return model.eval()
    seed_all(SEED)
    model.load_state_dict(torch.load(OUT / "d1_reflow_v2.pt", map_location="cpu", weights_only=False)["ema"])
    ema = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    n = len(rows["tid"])
    p = torch.arange(n)
    if arm == "shuffled":
        for tid in (0, 4):
            ids = (rows["tid"][:rows.get("n_focus", n)] == tid).nonzero().flatten()
            random_order = ids[torch.randperm(len(ids))]
            p[random_order] = random_order.roll(1)
    # Separate sampling from derangement so all arms see the same draws.
    sample = torch.Generator().manual_seed(SEED + 1)
    target = rows["teacher"] if arm == "teacher" else rows["target"][p]
    luma = bundle["v18_cache"]["luma_proj"]
    w = luma["w"].cuda().float().reshape(1, 4, 1, 1)
    b = luma["b"].cuda().float()
    if args.sharp_bridge:
        priors = {k: v.cuda() if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
        n_focus, same_begin = rows["n_focus"], rows["same_begin"]
    for step in range(1, args.updates+1):
        opt.zero_grad(set_to_none=True)
        if args.sharp_bridge:
            # 18 out of every 120 samples are same-style, with 8 replay samples per update.
            n_same = 4 if (step - 1) % 5 < 3 else 3
            order = torch.cat((
                torch.randint(n_focus, (16-n_same,), generator=sample),
                torch.randint(n_focus, same_begin, (8,), generator=sample),
                torch.randint(same_begin, n, (n_same,), generator=sample),
            ))
            order = order[torch.randperm(len(order), generator=sample)]
        else:
            order = torch.randint(n, (24,), generator=sample)
        for idx in order.split(8):
            d, q, y, eps, hi_c = (rows[k][idx].cuda().float() for k in ("d", "q", "y", "eps", "hi_c"))
            tid = rows["tid"][idx].cuda()
            start = eps
            if args.sharp_bridge:
                src = rows["src"][idx].cuda().float()
                sm = rows["is_same"][idx].cuda()[:, None, None, None]
                anchor = torch.where(sm, src, apply_latent_chroma_shift(src, tid, priors, alpha=1.0))
                start = eps.lerp(anchor, ALPHA_BRIDGE)
            y_in, q_in, u, layout = prepare_q32_cond_inputs(coarse_y16_lcn(y), d, q, tid, bank)
            selector = torch.rand(8, device=DEVICE)
            gamma = torch.where(selector < .25, 0, torch.where(selector < .5, 1, torch.rand_like(selector)))
            end = hi_c.lerp(target[idx].cuda().float(), gamma[:, None, None, None])
            tau = torch.sigmoid(torch.randn_like(gamma) * .85)
            tau4 = tau[:, None, None, None]
            z_tau = start.lerp(end, tau4)
            with bf16_autocast(DEVICE):
                v = model(z_tau, y_in, q_in, tid, u, tau, layout_scale=layout, gamma=gamma).float()
            diff = v - (end - start)
            reg = (diff.square().mean((1, 2, 3)).add(.01).sqrt() - .1).mean()
            pred = z_tau + (1 - tau4) * v
            high = sum(F.l1_loss(a, b_) for a, b_ in zip(dwt2_haar(pred)[1:], dwt2_haar(end)[1:])) / 3
            struct = latent_y32_lcn_loss(pred, end, w, b)
            loss = reg + .35 * high + .25 * struct + .001 * v.square().mean()
            (loss / 3).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema, [p.detach() for p in model.parameters()], .01)
        if step == 1 or step % 50 == 0 or step == args.updates:
            print(f"TRAIN {arm} {step}/{args.updates} loss={loss.item():.4f}", flush=True)
    with torch.no_grad():
        for p, v in zip(model.parameters(), ema):
            p.copy_(v)
    torch.save(model.state_dict(), path)
    return model.eval()


@torch.no_grad()
def evaluate(model, arm, bundle, c21, bank, held, alpha=0.0):
    priors = {k: v.cuda() if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    mse = []
    for st in range(0, len(held["tid"]), 8):
        sl = slice(st, st+8)
        src, eps, y, d, q = [held[k][sl].cuda().float() for k in ("src", "eps", "y", "d", "q")]
        tid = held["tid"][sl].cuda()
        c = conditions(src, eps, coarse_y16_lcn(y), d, q, tid, priors, bank, torch.zeros_like(tid).bool())
        pred = rollout(model, c, GRID, alpha=alpha)
        mse.extend((pred - held["target"][sl].cuda().float()).square().flatten(1).mean(1).tolist())
    tasks, z, _ = tasks_and_sources()
    eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    y, d = get_test750_y16_and_d16(bundle)
    q = c21["test750"]["q32"]
    tids = torch.tensor([t["target_style_id"] for t in tasks])
    sames = torch.tensor([t["source_style_id"] == t["target_style_id"] for t in tasks])
    wr = RawWriterV27M(arm, dec=load_s10_decoder(DEVICE), device=DEVICE)
    for st in range(0, len(tasks), 12):
        sl = slice(st, st+12)
        batch = [x[sl].cuda().float() for x in (z, eps, y, d, q)]
        c = conditions(*batch, tids[sl].cuda(), priors, bank, sames[sl].cuda())
        wr.write_batch(rollout(model, c, GRID, alpha=alpha), tasks[sl])
    wr.finalize(arm=arm, steps=4, alpha_bridge=alpha, same_bypass=False)
    out = run_measure_v27m(arm, device=DEVICE)
    out["paired_destylized_holdout_mse"] = sum(mse) / len(mse)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--images", type=Path, default=ROOT.parent / "train_512clean")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--updates", type=int, default=350)
    ap.add_argument("--per-style", type=int, default=128)
    ap.add_argument("--sharp-bridge", action="store_true")
    args = ap.parse_args()
    sanitize_cuda_env()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pack = build_or_load_v21_cache(args.pool, device=DEVICE)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=DEVICE)
    pairs = build_pairs(args, bundle, c21, bank)
    alpha = ALPHA_BRIDGE if args.sharp_bridge else 0.0
    training_rows = add_replay(pairs["train"], bundle, c21) if args.sharp_bridge else pairs["train"]
    report = {"seed": SEED, "updates_per_arm": args.updates, "alpha_bridge": alpha,
              "inference_nfe": 4, "decoder": "S10", "arms": {},
              "train_ids": pairs["train"]["ids"], "holdout_ids": pairs["holdout"]["ids"]}
    for split, rows in pairs.items():
        report[split] = {k: float(rows[k].float().mean()) for k in ("gray_layout_cos", "gray_hf_ratio")}
        report[split]["gray_layout_cos_min"] = float(rows["gray_layout_cos"].min())
        report[split]["gray_hf_ratio_min"] = float(rows["gray_hf_ratio"].min())
    suffix = f"sharp_bridge_n{args.per_style}_s{args.updates}" if args.sharp_bridge else f"n{args.per_style}_s{args.updates}"
    output = RESULTS / f"step_b_{suffix}.json"
    save_json(output, report)
    for kind in ("teacher", "aligned", "shuffled"):
        print(f"START {kind}", flush=True)
        model = fit(kind, args, training_rows, bank, bundle)
        report["arms"][kind] = evaluate(model, f"followup_b_{kind}_{suffix}",
                                        bundle, c21, bank, pairs["holdout"], alpha=alpha)
        del model
        save_json(output, report)


if __name__ == "__main__":
    main()
