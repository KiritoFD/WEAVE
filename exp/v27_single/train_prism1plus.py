"""Train PRISM-1+ from random initialization with frozen S10 RGB supervision."""

import argparse
import copy
import time

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import _kmeans_unit, load_convnext_tiny_offline
from exp.v27_mainline import OUT, RawWriterV27M, run_measure_v27m
from exp.v27_mainline.rollout_refine import FrozenVGG16Rollout
from exp.v27_single.bare_q32 import load_cache
from exp.v27_single.flow_pnp import axis_component, hard_hits
from exp.v27_single.prism1plus import Prism1Plus
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast, save_json, seed_all
from src.utils.structure import dwt2_haar, extract_color_fiber_27d, local_contrast_norm, rgb_to_gray

OUT_DIR = OUT / "prism1plus_v1"
SEED = 20261125


def luminance(z, w, b):
    return (z * w).sum(1, keepdim=True) + b


def scene(d):
    return F.normalize(F.avg_pool2d(d.float(), 4).flatten(1), dim=-1)


def haar_signature(rgb):
    ll, *hi = dwt2_haar(rgb_to_gray(rgb))
    ll2, *mid = dwt2_haar(ll)
    floor = ll2.var(dim=(2, 3), unbiased=False).clamp_min(1e-5)
    return torch.cat([(band.square().mean((2, 3)) / floor + 1e-5).log() for band in hi + mid], dim=1)


def tokens(x):
    size = min(x.shape[-1], 12)
    return F.adaptive_avg_pool2d(x.float(), size).flatten(2).transpose(1, 2)


class FeatureLoss:
    def __init__(self):
        self.vgg = FrozenVGG16Rollout().cuda().eval()
        self.cnx = load_convnext_tiny_offline("cuda")[:6]
        self.channels = (128, 256, 512, 96, 192, 384)

    def gray_features(self, rgb):
        gray = rgb_to_gray(rgb)
        gray = F.interpolate(gray, (128, 128), mode="area").repeat(1, 3, 1, 1)
        with bf16_autocast("cuda"):
            x = (gray - self.vgg.im_mean) / self.vgg.im_std
            feat = self.vgg.s4_2(self.vgg.s3(self.vgg.s2(self.vgg.s1(x))))
        vectors = F.normalize(F.adaptive_avg_pool2d(feat.float(), 8).flatten(2), dim=1)
        return vectors.transpose(1, 2) @ vectors

    def style_features(self, rgb):
        image = F.interpolate(rgb, (128, 128), mode="area")
        image = (image - self.vgg.im_mean) / self.vgg.im_std
        with bf16_autocast("cuda"):
            a = self.vgg.s2(self.vgg.s1(image))
            b = self.vgg.s3(a)
            c = self.vgg.s4_3(self.vgg.s4_2(b))
            p = self.cnx[:2](image)
            q = self.cnx[2:4](p)
            r = self.cnx[4:6](q)
        return (*(tokens(x) for x in (a, b, c, p, q, r)),
                extract_color_fiber_27d(F.avg_pool2d(rgb, 8)), haar_signature(rgb))


def slice_loss(output, ref, projection):
    a = (output @ projection).sort(dim=1).values
    b = (ref.float() @ projection).sort(dim=1).values
    return (a - b).abs().mean((1, 2))


@torch.no_grad()
def build_bank(tr, dec, features):
    path = OUT_DIR / "reference_bank.pt"
    if path.exists():
        return torch.load(path, map_location="cuda", weights_only=False)
    raw_dino = load_cache("runs/v19/cache/v19_foundation.pt")["tr_fib"][:, 27:].float()
    sc = scene(tr["dino_16"].cuda())
    centers, ref_ids = [], []
    for sid in range(5):
        ids = (tr["sid"] == sid).nonzero().flatten().cuda()
        sim = sc[ids] @ sc[ids].T
        density = sim.topk(31, dim=1).values[:, 1:].mean(1)
        core = ids[density.topk(192).indices]
        c, assignment = _kmeans_unit(sc[core], k=8, seed=SEED+sid)
        centers.append(c)
        for center in c:
            ref_ids.append(core[(sc[core] @ center).topk(4).indices])
    refs = torch.stack(ref_ids).flatten().cpu()
    blocks = []
    for ids in refs.split(4):
        rgb = decode_s10(dec, tr["z"][ids].cuda().float(), dtype=torch.bfloat16, output_cpu=False)
        blocks.append(features.style_features(rgb))
    feats = tuple(torch.cat(xs).half() for xs in zip(*blocks))
    condition = torch.cat((feats[-2].float(), feats[-1].float(), raw_dino[refs].cuda()), dim=1)
    mean = condition.mean(0)
    std = condition.std(0, unbiased=False).clamp_min(.01)
    condition = ((condition - mean)/std).reshape(5, 8, 4, -1).mean(2)
    bank = {"center": torch.stack(centers), "stats": condition, "feats": feats,
            "ref_ids": [tr["ids"][i] for i in refs.tolist()]}
    torch.save(bank, path)
    print(f"TRAIN-ONLY BANK {len(refs)} reference slots and {condition.shape[-1]} raw statistics", flush=True)
    return bank


def pick_targets(d, labels, bank):
    score = torch.einsum("bd,bkd->bk", scene(d), bank["center"][labels])
    group = score.argmax(dim=1)
    return group + 8*labels, bank["stats"][labels, group]


def make_batch(tr, q32, pools, step, batch, gen):
    # floor differences give exactly 15% identity over every block of five updates at batch 12.
    n_same = (step * batch * 15)//100 - ((step-1) * batch * 15)//100
    sid = (torch.randperm(batch, generator=gen) + step*batch) % 5
    ids = torch.stack([pools[int(i)][torch.randint(len(pools[int(i)]), (), generator=gen)] for i in sid])
    tid = (sid + torch.randint(1, 5, (batch,), generator=gen)) % 5
    strength = .08 + .92*torch.rand(batch, generator=gen)
    strength[torch.rand(batch, generator=gen) < .15] = 1.
    tid[:n_same], strength[:n_same] = sid[:n_same], 0.
    order = torch.randperm(batch, generator=gen)
    ids, tid, strength = ids[order], tid[order], strength[order]
    return (tr["z"][ids], tr["y_64"][ids], tr["dino_16"][ids], q32[ids], tid, strength)


def train(args, model, cache, q32, bank, dec, features, w, b):
    path = OUT_DIR / "prism1plus.pt"
    resume = OUT_DIR / "resume.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["ema"])
        return model.eval()
    tr = cache["train"]
    pools = [(tr["sid"] == style).nonzero().flatten() for style in range(5)]
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    gen = torch.Generator().manual_seed(SEED + 1)
    axis = F.normalize(w, dim=1)
    first = 1
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["budget"] != (args.updates, args.batch, args.micro):
            raise ValueError("Resume configuration differs from the original run")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["opt"])
        gen.set_state(ck["sampling"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        first = ck["step"] + 1
    t0 = time.monotonic()
    print(f"TRAIN START {first}/{args.updates}, params={sum(p.numel() for p in model.parameters()):,}, "
          f"batch={args.batch}, micro={args.micro}", flush=True)
    for step in range(first, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        projection_gen = torch.Generator(device="cuda").manual_seed(SEED+step//5)
        ortho = tuple(torch.linalg.qr(torch.randn(c, 24, device="cuda", generator=projection_gen),
                                         mode="reduced").Q for c in features.channels)
        batch = make_batch(tr, q32, pools, step, args.batch, gen)
        stats = torch.zeros(3, device="cuda")
        accum = args.batch // args.micro
        for parts in zip(*(items.split(args.micro) for items in batch)):
            src, y64, d, q, tid, strength = [item.cuda() for item in parts]
            src, y64, d, q = (x.float() for x in (src, y64, d, q))
            y16 = coarse_y16_lcn(y64)
            group, cond = pick_targets(d, tid, bank)
            ref = group*4 + torch.randint(4, (len(src),), device="cuda")
            luma = luminance(src, w, b)
            with torch.no_grad():
                src_rgb = decode_s10(dec, src, dtype=torch.bfloat16, output_cpu=False)
                gray_src = features.gray_features(src_rgb)
            with bf16_autocast("cuda"):
                z = model(luma, y16, q, tid, cond, strength, torch.randn_like(src)).float()
            rgb = decode_s10(dec, z, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
            output = features.style_features(rgb)
            targets = bank["feats"]
            swd = sum(slice_loss(x, ref_feats[ref], proj)
                      for x, ref_feats, proj in zip(output[:-2], targets[:-2], ortho)) / len(ortho)
            palette = (output[-2] - targets[-2][ref]).abs().mean(1)
            haar = (output[-1] - targets[-1][ref]).abs().mean(1)
            style = (strength * (swd + 4*palette + .12*haar)).mean()
            same = ((strength == 0) * (rgb - src_rgb).abs().mean((1, 2, 3))).mean()
            pred_y = luminance(z, w, b)
            coarse = (coarse_y16_lcn(pred_y) - y16).abs().mean((1, 2, 3))
            fine = ((local_contrast_norm(pred_y) - local_contrast_norm(luma)).abs()
                    .mean((1, 2, 3)) + (pred_y-luma).abs().mean((1, 2, 3)))
            perceptual = (features.gray_features(rgb) - gray_src).abs().mean((1, 2))
            content = (.45 * coarse + (1-strength) * (1.5*fine + .8*perceptual)).mean()
            content_grad = torch.autograd.grad(content, z, retain_graph=True)[0]
            orthogonal_content = axis_component(content_grad, axis) * axis
            torch.autograd.backward(((style + 4*same)/accum, z), (None, orthogonal_content/accum))
            stats += torch.stack((style.detach(), content.detach(), same.detach())) / accum
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_([p for p in ema.parameters()], [p.detach() for p in model.parameters()], .012)
        if step == first or step % 25 == 0 or step == args.updates:
            print(f"TRAIN {step}/{args.updates} style,content,identity={stats.tolist()} "
                  f"grad={float(norm):.3f} elapsed_s={time.monotonic()-t0:.1f}", flush=True)
        if step % 100 == 0 and step < args.updates:
            tmp = resume.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": optimizer.state_dict(),
                        "sampling": gen.get_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "budget": (args.updates, args.batch, args.micro), "step": step}, tmp)
            tmp.replace(resume)
    tmp = path.with_suffix(".tmp")
    torch.save({"ema": ema.state_dict(), "axis_w": w, "axis_b": b,
                "scene_centers": bank["center"], "raw_style_stats": bank["stats"],
                "updates": args.updates, "seed": SEED}, tmp)
    tmp.replace(path)
    print(f"CKPT {path}", flush=True)
    return ema


@torch.no_grad()
def evaluate(model, cache, c21, bank, w, b):
    tasks, src, _ = tasks_and_sources()
    y, d = get_test750_y16_and_d16(cache)
    q = c21["test750"]["q32"]
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    noise = torch.randn(src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder()
    report = {"checkpoint": "prism1plus.pt", "nfe": 1, "same_bypass": False, "arms": {}}
    for strength in (0., .2, .4, .6, .8, 1.):
        arm = f"prism1plus_v1_l{round(strength*10):02}"
        if not (OUT / "measured" / arm / "summary_v27m.json").exists():
            writer = RawWriterV27M(arm, dec=dec)
            for start in range(0, len(tasks), 16):
                sl = slice(start, start+16)
                zs, yb, db, qb, eb = (x[sl].cuda().float() for x in (src, y, d, q, noise))
                ti = tid[sl].cuda()
                _, cond = pick_targets(db, ti, bank)
                with bf16_autocast("cuda"):
                    gen = model(luminance(zs, w, b), yb, qb, ti, cond,
                                torch.full_like(ti, strength, dtype=torch.float32), eb).float()
                writer.write_batch(gen, tasks[sl])
            writer.finalize(arm=arm, strength=strength, nfe=1, same_bypass=False)
        row = run_measure_v27m(arm)
        row.update(hits=hard_hits(arm), strength=strength)
        report["arms"][str(strength)] = row
        save_json(OUT_DIR / "results.json", report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--micro", type=int, default=3)
    ap.add_argument("--eval-only", action="store_true", help="Evaluate the saved model and its embedded conditioning data.")
    args = ap.parse_args()
    if args.batch % args.micro:
        ap.error("--batch must be divisible by --micro")
    seed_all(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_cache = load_cache("runs/v18/cache/test750_struct_cache.pt")
    c21 = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")
    if args.eval_only:
        ck = torch.load(OUT_DIR / "prism1plus.pt", map_location="cpu", weights_only=False)
        bank = {"center": ck["scene_centers"].cuda(), "stats": ck["raw_style_stats"].cuda()}
        model = Prism1Plus(bank["stats"].shape[-1]).cuda().eval()
        model.load_state_dict(ck["ema"])
        cache = {"test_struct": test_cache}
        if "y_64" not in test_cache:
            cache["v18_cache"] = {"luma_proj": {"w": ck["axis_w"].flatten(), "b": ck["axis_b"]}}
        evaluate(model, cache, c21, bank, ck["axis_w"].cuda(), ck["axis_b"].cuda())
        return
    tr_cache = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    w = tr_cache["luma_proj"]["w"].cuda().float().reshape(1, 4, 1, 1)
    b = tr_cache["luma_proj"]["b"].cuda().float()
    dec = load_s10_decoder().to(torch.bfloat16).requires_grad_(False)
    features = FeatureLoss()
    bank = build_bank(tr_cache["train"], dec, features)
    model = Prism1Plus(bank["stats"].shape[-1]).cuda()
    model = train(args, model, tr_cache, c21["train"]["q32"], bank, dec, features, w, b)
    del dec, features
    torch.cuda.empty_cache()
    evaluate(model, {"v18_cache": tr_cache, "test_struct": test_cache}, c21, bank, w, b)
    print("COMPLETE PRISM-1+", flush=True)


if __name__ == "__main__":
    main()
