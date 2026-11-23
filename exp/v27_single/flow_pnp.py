"""Pure-noise ICME Q32 Flow-PnP with orthogonal endpoint updates."""

import argparse
import hashlib
import time

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27_mainline import OUT, Q32_FLAGSHIP_CKPT, RawWriterV27M, run_measure_v27m
from exp.v27_mainline.dir1_reflow import prepare_q32_cond_inputs
from exp.v27_mainline.rollout_refine import FrozenVGG16Rollout
from exp.v27_single.bare_q32 import load_cache
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast, read_json, save_json
from src.utils.structure import extract_color_fiber_27d, local_contrast_norm

LEVELS = (.1, .3, .5, .7, .9, 1.)
STEPS = 8
STYLE_STEPS = (3, 4)
OUT_DIR = OUT / "flow_pnp"


class StyleEnergy:
    def __init__(self, decoder, train):
        self.dec = decoder
        self.vgg = FrozenVGG16Rollout().cuda().eval()
        generator = torch.Generator().manual_seed(EVAL_SEED + 51)
        self.proj = [torch.linalg.qr(torch.randn(c, 32, generator=generator), mode="reduced").Q.cuda()
                     for c in (64, 128, 256, 512)]
        self.ids = []
        banks = []
        for style in range(5):
            idx = (train["sid"] == style).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=generator)[:24]]
            self.ids.append([train["ids"][i] for i in idx.tolist()])
            rows = []
            for batch in idx.split(4):
                with torch.no_grad():
                    rows.append(self.describe(train["z"][batch].cuda().float(), grad=False))
            banks.append([torch.cat(feats).mean(0) for feats in zip(*rows)])
        self.targets = [torch.stack(feats) for feats in zip(*banks)]

    def describe(self, z, *, grad):
        rgb = decode_s10(self.dec, z, batch=len(z), dtype=torch.bfloat16,
                         enable_grad=grad, output_cpu=False)
        with bf16_autocast("cuda"):
            levels = self.vgg.forward_style(rgb)
        stats = []
        for feature, proj in zip(levels, self.proj):
            # Deterministic 16 x 16 spatial subsample, followed by sliced quantiles.
            stride = feature.shape[-1] // 16
            tokens = feature[:, :, stride//2::stride, stride//2::stride].flatten(2).transpose(1, 2)
            stats.append((tokens.float() @ proj).sort(dim=1).values)
        palette = extract_color_fiber_27d(F.avg_pool2d(rgb, 8))
        return (*stats, palette)

    def loss(self, z, tid):
        stats = self.describe(z, grad=True)
        swd = sum(F.l1_loss(x, target[tid]) for x, target in zip(stats[:-1], self.targets[:-1])) / 4
        palette = F.l1_loss(stats[-1], self.targets[-1][tid])
        return swd + 4.0 * palette


def axis_component(z, axis):
    return (z * axis).sum(1, keepdim=True)


def structure_signature(y):
    mid = F.avg_pool2d(y, 2)
    low = F.avg_pool2d(y, 4)
    return low, local_contrast_norm(mid), local_contrast_norm(low)


def unit_rms(g):
    return g / g.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)


def proximal(end, desired, axis, tid, style, eta_c, eta_g):
    if eta_c == 0 and eta_g == 0:
        return end
    with torch.enable_grad():
        clean = end.detach().requires_grad_(True)
        update = torch.zeros_like(clean)
        if eta_c:
            low, lcn_mid, lcn_low = structure_signature(axis_component(clean, axis))
            loss = (F.mse_loss(low, desired[0]) + F.mse_loss(lcn_mid, desired[1])
                    + F.mse_loss(lcn_low, desired[2]))
            grad = torch.autograd.grad(loss, clean)[0]
            update += eta_c * unit_rms(axis_component(grad, axis) * axis)
        if eta_g:
            g = torch.autograd.grad(style.loss(clean, tid), clean)[0]
            update += eta_g * unit_rms(g - axis_component(g, axis) * axis)
    return end - update.detach()


@torch.no_grad()
def rollout(model, eps, src, y, d, q, tid, bank, axis, s, eta_c_max, eta_g_max, style):
    # Keep the frozen checkpoint's native cached conditioning vectors and layout rule.
    y_in, q_in, u, layout = prepare_q32_cond_inputs(y, d, q, tid, bank, lscale_abs=.34)
    scene_strength = ((layout - .34) / .66).clamp(0, 1)[:, None, None, None]
    eta_c = eta_c_max * (1 - s)**2
    desired = structure_signature(axis_component(src, axis)) if eta_c else None
    flags = torch.arange(2, device=eps.device).bool().repeat_interleave(len(eps))
    paired = (y_in.repeat(2, 1, 1, 1), q_in.repeat(2, 1, 1, 1), tid.repeat(2), u.repeat(2, 1))
    layout = layout.repeat(2)
    z = eps
    for step in range(STEPS):
        t = step / STEPS
        t_next = (step + 1) / STEPS
        tv = eps.new_full((2 * len(eps),), t)
        with bf16_autocast("cuda"):
            v_sc, v_0c = model(z.repeat(2, 1, 1, 1), *paired, tv,
                               drop_style=flags, layout_scale=layout).float().chunk(2)
        late = min(1., max(0., (t - .36) / .32))
        max_cfg = 1.4 + scene_strength * (6.5 - 1.1 * late - 1.4)
        v = v_0c + (1 + s * (max_cfg - 1)) * (v_sc - v_0c)
        clean = z + (1 - t) * v
        inferred_noise = z - t * v
        eta_g = eta_g_max * s if style is not None and step in STYLE_STEPS else 0.
        clean = proximal(clean, desired, axis, tid, style, eta_c, eta_g)
        z = (1 - t_next) * inferred_noise + t_next * clean
    return z


def hard_hits(arm):
    rows = read_json(OUT / "measured" / arm / "eval.json")["rows"]
    counts = {}
    for sid, key in ((0, "ER"), (4, "Uki")):
        group = [row for row in rows if row["is_cross"] and row["tgt_style_id"] == sid]
        counts[key] = [sum(row["target_top1"] for row in group), len(group)]
    return counts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--style-micro", type=int, default=4)
    ap.add_argument("--eta-c-max", type=float, default=.45)
    ap.add_argument("--eta-g-max", type=float, default=.20)
    args = ap.parse_args()
    tasks, sources, _ = tasks_and_sources()
    structure = load_cache("runs/v18/cache/test750_struct_cache.pt")
    ccf = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    y, d = get_test750_y16_and_d16({"test_struct": structure, "v18_cache": ccf})
    q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
    bank = load_cache("runs/v20/cache/core_knn_bank.pt")
    bank = {k: v.cuda() if k in ("sub_scene", "sub_u", "sub_dsd") else v for k, v in bank.items()}
    axis = F.normalize(ccf["luma_proj"]["w"].float().cuda().reshape(1, 4, 1, 1), dim=1)
    labels = torch.tensor([row["target_style_id"] for row in tasks])
    noise = torch.randn(sources.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    model = make_bottleneck_net(32).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"])
    dec = load_s10_decoder("cuda").requires_grad_(False)
    result = {"checkpoint": Q32_FLAGSHIP_CKPT.name,
              "sha256": hashlib.sha256(Q32_FLAGSHIP_CKPT.read_bytes()).hexdigest(),
              "seed": EVAL_SEED, "tau0": 0., "steps": STEPS, "flow_branch_evals": 2 * STEPS,
              "eta_c_max": args.eta_c_max, "eta_g_max": args.eta_g_max,
              "style_steps_zero_based": STYLE_STEPS, "same_bypass": False, "arms": {}}
    output = OUT_DIR / "results.json"
    print(f"FLOW-PNP {len(tasks)} tasks eta_c={args.eta_c_max} eta_g={args.eta_g_max}", flush=True)
    for mode in ("structure", "joint"):
        style = None
        batch_size = args.batch
        if mode == "joint":
            print("BUILD train-only VGG/Oklab prototypes", flush=True)
            style = StyleEnergy(load_s10_decoder("cuda").to(dtype=torch.bfloat16).requires_grad_(False), ccf["train"])
            result["reference_ids"] = style.ids
            batch_size = args.style_micro
        for s in LEVELS:
            arm = f"flow_pnp_{mode}_s{round(10*s):02d}_c{args.eta_c_max:g}_g{args.eta_g_max:g}"
            print(f"START {mode} s={s:.1f}", flush=True)
            start_time = time.monotonic()
            if not (OUT / "measured" / arm / "summary_v27m.json").exists():
                writer = RawWriterV27M(arm, dec=dec, device="cuda")
                for start in range(0, len(tasks), batch_size):
                    sl = slice(start, start + batch_size)
                    z = rollout(model, *(x[sl].cuda().float() for x in (noise, sources, y, d, q)),
                                labels[sl].cuda(), bank, axis, s, args.eta_c_max, args.eta_g_max, style)
                    if not torch.isfinite(z).all():
                        raise FloatingPointError(f"nonfinite output in {arm} at {start}")
                    writer.write_batch(z, tasks[sl])
                    if start == 0 or (start // batch_size) % 50 == 49:
                        print(f"  generate {min(start+batch_size, len(tasks))}/{len(tasks)}", flush=True)
                writer.finalize(arm=arm, tau0=0., steps=STEPS, mode=mode, s=s, same_bypass=False)
            infer_s = time.monotonic() - start_time
            row = run_measure_v27m(arm, device="cuda")
            row.update(s=s, mode=mode, hits=hard_hits(arm), arm=arm, infer_s=round(infer_s, 2),
                       elapsed_s=round(time.monotonic()-start_time, 2))
            result["arms"][f"{mode}_{s:.1f}"] = row
            save_json(output, result)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
