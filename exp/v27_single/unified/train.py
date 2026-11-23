"""Fit and evaluate the single-stage PRISM-1-Unified student."""

import argparse
import copy
import gc
import math
import time

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import UnnormalizedVGGFeatureExtractor
from exp.v27_mainline import OUT, RawWriterV27M, run_measure_v27m
from exp.v27_single.bare_q32 import load_cache
from exp.v27_single.flow_pnp import hard_hits
from exp.v27_single.train_prism1plus import haar_signature
from exp.v27_single.unified.cache import PAIR_PATH, RUN, atomic_save
from exp.v27_single.unified.model import UnifiedEditor
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, save_json, seed_all
from src.utils.structure import dwt2_haar

SEED = 20261127
MODE_PATTERN = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
FINAL_LEVELS = (0., .25, .45, .70, .85, 1., 1.15)


class PixelEnergy:
    def __init__(self):
        self.decoder = load_s10_decoder().to(torch.bfloat16).requires_grad_(False)
        self.vgg = UnnormalizedVGGFeatureExtractor().cuda().eval().requires_grad_(False)
        self.layers = []
        for layer in (self.vgg.s1, self.vgg.s2, self.vgg.s3):
            layer.register_forward_hook(self.capture_layer)
        self.scale = None

    def capture_layer(self, _module, _input, output):
        self.layers.append(output)

    def encode(self, z, grad):
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=True), bf16_autocast("cuda"):
            rgb = decode_s10(self.decoder, z, dtype=torch.bfloat16, batch=len(z),
                             enable_grad=grad, output_cpu=False)
            self.layers.clear()
            features = self.vgg(rgb, multi_scale=False).float()
            tokens = tuple(F.adaptive_avg_pool2d(x, 16).float().flatten(2).transpose(1, 2)
                           for x in self.layers)
            self.layers.clear()
        return features, tokens

    def loss(self, pred, goal, proj):
        current, tokens = self.encode(pred, grad=True)
        with torch.no_grad():
            target, ref_tokens = self.encode(goal, grad=False)
        moments = F.smooth_l1_loss((current-target)/self.scale, torch.zeros_like(current), beta=.20)
        swd = sum(((a @ p).sort(dim=1).values - (b @ p).sort(dim=1).values).abs().mean()
                  for a, b, p in zip(tokens, ref_tokens, proj)) / len(proj)
        return .15*moments + .10*swd


@torch.no_grad()
def conditioning_stats(tr, foundation, pixel):
    seed = torch.Generator().manual_seed(SEED+100)
    styles = []
    for sid in range(5):
        ids = (tr["sid"] == sid).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=seed)[:64]]
        rgba = []
        for idx in ids.split(8):
            rgb = decode_s10(pixel.decoder, tr["z"][idx].cuda().float(),
                             dtype=torch.bfloat16, batch=len(idx), output_cpu=False)
            rgba.append(haar_signature(rgb))
        rgb_stats = foundation["tr_fib"][ids].cuda().float()
        styles.append(torch.cat((rgb_stats[:, :27], torch.cat(rgba), rgb_stats[:, 27:411]), 1))
    stacked = torch.stack(styles)
    flat = stacked.flatten(0, 1)
    return ((stacked - flat.mean(0)) / flat.std(0, unbiased=False).clamp_min(.01)).mean(1)


def batch_indices(step, same_ids, cross_ids, gen):
    prev_same, next_same = ((step-1)*64)//5, (step*64)//5
    n_same = next_same-prev_same
    n_cross = 64-n_same
    si = same_ids[torch.randint(len(same_ids), (n_same,), generator=gen)]
    ci = cross_ids[torch.randint(len(cross_ids), (n_cross,), generator=gen)]
    prev_cross = (step-1)*64 - prev_same
    modes = MODE_PATTERN[torch.arange(prev_cross, prev_cross+n_cross) % len(MODE_PATTERN)]
    modes = modes[torch.randperm(n_cross, generator=gen)]
    r = torch.rand(n_cross, generator=gen)
    cross_l = torch.where(modes == 0, .45, torch.where(modes == 1, 1.,
                           torch.where(modes == 2, .05 + .40*r, .45 + .55*r)))
    strengths = torch.cat((cross_l, torch.rand(n_same, generator=gen)))
    order = torch.randperm(64, generator=gen)
    if step % 2 == 0:
        chosen = order[order < n_cross][:8]
        order = torch.cat((chosen, order[~torch.isin(order, chosen)]))
    return torch.cat((ci, si))[order], strengths[order]


def target_latents(source, hi_c, hi_s, strength):
    alpha = strength[:, None, None, None]
    low = source.lerp(hi_c, (alpha/.45).clamp(0, 1))
    high = hi_c.lerp(hi_s, ((alpha-.45)/.55).clamp(0, 1))
    return torch.where(alpha <= .45, low, high)


def latent_loss(pred, target, priors):
    direct = F.smooth_l1_loss(pred, target, beta=.06)
    fp, ft = dwt2_haar(pred), dwt2_haar(target)
    bands = sum(w * F.l1_loss(a, b) for w, a, b in zip((1., 1., .5), fp[1:], ft[1:]))
    luma = F.l1_loss(torch.einsum("ij,bjhw->bihw", priors["P_Y"], pred-target), torch.zeros_like(pred))
    return direct + .35*bands + .25*luma


@torch.no_grad()
def evaluate(model, raw_stats, priors, ts, q, *, tag, levels, wrong=False):
    tasks, src, _ = tasks_and_sources()
    sid = torch.tensor([t["source_style_id"] for t in tasks])
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    labels = pick_wrong_cross_tid(sid, tid) if wrong else tid
    dec = load_s10_decoder()
    out = {}
    name_tags = ("wrong",) if wrong else tuple(f"l{int(round(l*100)):03d}" for l in levels)
    for name, strength in zip(name_tags, levels):
        arm = f"prism1_unified_{tag}_{name}"
        identity = 0.
        meta = OUT / "raw" / arm / "meta.json"
        if not (OUT / "measured" / arm / "summary_v27m.json").exists():
            writer = RawWriterV27M(arm, dec=dec)
            for st in range(0, len(tasks), 12):
                sl = slice(st, st+12)
                zs = src[sl].cuda().float()
                tgt = labels[sl].cuda()
                z = model(zs, ts["c_64"][sl].cuda().float(), q[sl].cuda().float(), tgt,
                          torch.full_like(tgt, strength, dtype=torch.float32), raw_stats, priors)
                if strength == 0:
                    identity = max(identity, float((z-zs).abs().max()))
                    if identity:
                        raise RuntimeError(f"Lambda-zero latent identity failed: {identity}")
                writer.write_batch(z, tasks[sl])
            writer.finalize(arm=arm, strength=strength, nfe=1, same_bypass=False,
                            lambda_zero_max_abs=identity if strength == 0 else None)
        row = run_measure_v27m(arm)
        row.update(strength=strength, hits=hard_hits(arm), wrong=wrong,
                   lambda_zero_max_abs=read_json(meta).get("lambda_zero_max_abs"))
        out[name] = row
    return out


def final_eval(model, raw_stats, priors, ts, q, steps, *, checkpoint=None, tag="final"):
    levels = evaluate(model, raw_stats, priors, ts, q, tag=tag, levels=FINAL_LEVELS)
    levels.update(evaluate(model, raw_stats, priors, ts, q,
                           tag=tag, levels=(1.,), wrong=True))
    high, strong = levels["l045"], levels["l100"]
    report = {
        "checkpoint": checkpoint or "prism1_unified.pt", "student_updates": steps, "nfe": 1,
        "teacher_gate": read_json(RUN / "pairs_meta.json")["teacher_gate_passed"],
        "same_bypass": False, "arms": levels, "acceptance": {
            "exact_latent_identity": levels["l000"]["lambda_zero_max_abs"] == 0,
            "high_content": high["dino_c"] >= .68 and high["dino_s"] >= .49
                            and high["csd_top1"] >= .50 and high["same150_top1"] >= .90,
            "strong_style": strong["dino_c"] >= .58 and strong["dino_s"] >= .54
                            and strong["csd_top1"] >= .54
                            and all(v[0] >= 25 for v in strong["hits"].values()),
            "wrong_target": levels["wrong"]["csd_top1"] <= .025,
            "rtx3060_latency": None,
        },
    }
    result_path = RUN / ("results.json" if tag == "final" else f"results_{tag}.json")
    save_json(result_path, report)
    print(f"EVALUATION COMPLETE {result_path}", flush=True)


def run(args):
    seed_all(SEED)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 8
    pairs = torch.load(PAIR_PATH, map_location="cpu", weights_only=False)
    tr = load_cache("runs/v18/cache/ccf_pool_cache.pt")["train"]
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    c21 = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")
    foundation = load_cache("runs/v19/cache/v19_foundation.pt")
    priors = {k: foundation["priors"][k].cuda().float() for k in ("P_Y", "P_C", "mu_c", "std_c")}
    pixel = PixelEnergy()
    raw_stats = conditioning_stats(tr, foundation, pixel)
    del foundation
    # Freeze standardization on an independent deterministic 48-pair slice of training targets.
    ids = (~pairs["same"]).nonzero().flatten()[::125][:48]
    with torch.no_grad():
        xx = torch.cat([pixel.encode(pairs["hiS"][idx].cuda().float(), False)[0] for idx in ids.split(4)])
        pixel.scale = xx.std(0, unbiased=False).clamp_min(.03)
    model = UnifiedEditor(raw_stats.shape[-1]).cuda().to(memory_format=torch.channels_last)
    num_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(.9, .99), weight_decay=1e-4)
    gen = torch.Generator().manual_seed(SEED+1)
    resume = RUN / "resume.pt"
    first = 1
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["updates"] != args.updates:
            raise ValueError("Requested update budget differs from the saved schedule")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        gen.set_state(state["sampling"])
        first = state["step"]+1
    same_ids = pairs["same"].nonzero().flatten()
    cross_ids = (~pairs["same"]).nonzero().flatten()
    print(f"STUDENT START {first}/{args.updates}, {num_params:,} params, micro={args.micro}, batch=64", flush=True)
    t0 = time.monotonic()
    for step in range(first, args.updates+1):
        frac = max(0, step-200) / max(1, args.updates-200)
        lr = 8e-4 * step/200 if step <= 200 else 4e-5 + .5*(8e-4-4e-5)*(1+math.cos(math.pi*frac))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        order, strength = batch_indices(step, same_ids, cross_ids, gen)
        proj = None
        if step % 2 == 0:
            pg = torch.Generator(device="cuda").manual_seed(SEED+step)
            proj = tuple(torch.linalg.qr(torch.randn(c,64,device="cuda",generator=pg),mode="reduced").Q
                         for c in (64,128,256))
        total = torch.zeros(2, device="cuda")
        def predict(sl):
            batch = order[sl]
            src = pairs["src"][batch]
            zs = tr["z"][src].cuda().float()
            levels = strength[sl].cuda()
            target = target_latents(zs, pairs["hiC"][batch].cuda().float(),
                                    pairs["hiS"][batch].cuda().float(), levels)
            identity = pairs["same"][batch].cuda()[:, None, None, None]
            target = torch.where(identity, zs, target)
            pred = model(zs, tr["c_64"][src].cuda().float(), c21["train"]["q32"][src].cuda().float(),
                         pairs["tid"][batch].cuda(), levels, raw_stats, priors)
            return pred, target

        if proj is not None:
            for st in range(0, 8, args.pixel_micro):
                pred, target = predict(slice(st, st+args.pixel_micro))
                latent = latent_loss(pred, target, priors) * (args.pixel_micro/64)
                rgb = pixel.loss(pred, target, proj) * (args.pixel_micro/8)
                (latent + rgb).backward()
                total[0] += latent.detach()
                total[1] += rgb.detach()
        for st in range(8 if proj is not None else 0, 64, args.micro):
            stop = min(64, st+args.micro)
            pred, target = predict(slice(st, stop))
            latent = latent_loss(pred, target, priors) * ((stop-st)/64)
            latent.backward()
            total[0] += latent.detach()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], .004)
        if step == first or step % 25 == 0:
            print(f"STUDENT {step}/{args.updates} latent,pixel={total.tolist()} grad={float(grad):.3f} "
                  f"elapsed_s={time.monotonic()-t0:.1f}", flush=True)
        if step % 100 == 0 or step == args.updates:
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                         "sampling": gen.get_state(), "updates": args.updates, "step": step}, resume)
        if step % 1000 == 0 or step == args.updates:
            name = "prism1_unified.pt" if step == args.updates else f"ema_s{step:04d}.pt"
            atomic_save({"ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors,
                         "step": step, "params": num_params}, RUN / name)
            torch.cuda.empty_cache()
            if step < args.updates:
                report = evaluate(ema, raw_stats, priors, ts, c21["test750"]["q32"],
                                  tag=f"s{step:04d}", levels=(.45, .80, 1.))
                save_json(RUN / f"preview_s{step:04d}.json", report)
    del pixel, model, opt
    torch.cuda.empty_cache()
    final_eval(ema, raw_stats, priors, ts, c21["test750"]["q32"], args.updates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--micro", type=int, default=16, choices=(2, 4, 8, 16))
    ap.add_argument("--pixel-micro", type=int, default=4, choices=(2, 4))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, help="Named EMA snapshot in the experiment directory.")
    args = ap.parse_args()
    if args.eval_only:
        checkpoint = args.checkpoint or "prism1_unified.pt"
        saved = torch.load(RUN / checkpoint, map_location="cuda", weights_only=False)
        model = UnifiedEditor(saved["raw_stats"].shape[-1]).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(saved["ema"])
        ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
        q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
        tag = f"s{saved['step']:04d}" if args.checkpoint else "final"
        final_eval(model.eval(), saved["raw_stats"], saved["priors"], ts, q, saved["step"],
                   checkpoint=checkpoint, tag=tag)
        return
    while True:
        try:
            run(args)
            return
        except torch.cuda.OutOfMemoryError:
            if args.pixel_micro == 4:
                args.pixel_micro = 2
            elif args.micro == 2:
                raise
            else:
                args.micro //= 2
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Retry from the last recovery point with micro={args.micro}, "
              f"pixel_micro={args.pixel_micro}, effective batch=64", flush=True)


if __name__ == "__main__":
    main()
