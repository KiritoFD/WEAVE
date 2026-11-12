"""Joint style-friendly VAE training: experiments A -> B -> C.

A  joint reconstruction baseline   : L1 + lam_p*LPIPS512 + beta*KL
B  style organisation              : + style supervision on the 8-dim s(z)
C  editability                     : + statistics-swap cycle consistency

Everything is encoded from RGB on the fly: the old 78,661 latent cache is invalid
now that the encoder trains. Throughput was measured first (see
runs/vae_style_vae/joint_infra_bench.json): partial@bs6 = 12.2 img/s,
full@bs4 = 9.1 img/s, and gradient checkpointing does NOT help.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_style_vae.data import (MANIFEST, BalancedStyleBatchSampler,
                                      StyleImageDS, collate, load_split)
from tools.vae_style_vae.losses import (StyleVAELoss, crop_consistency, edit_loss,
                                        kl_divergence, supcon)
from tools.vae_style_vae.model import (StyleHead, StyleVAE, style_content,
                                       unfreeze_encoder)

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_vae")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
S10_CKPT = "/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"

STAGES = [
    # name, encoder mode, updates, enc_lr, dec_lr, bs, use_style, use_edit
    dict(name="A_joint", enc="partial", updates=400, enc_lr=1e-5, dec_lr=2e-5,
         bs=6, style=False, edit=False),
    # balanced 2 styles x 2 images: SupCon and the swap both come from the batch
    dict(name="B_style", enc="full", updates=1500, enc_lr=1e-5, dec_lr=1e-5,
         bs=4, style=True, edit=False, balanced=(2, 2)),
    dict(name="C_edit", enc="full", updates=1500, enc_lr=5e-6, dec_lr=5e-6,
         bs=4, style=True, edit=True, balanced=(2, 2)),
    # Goal-required short control: train C's objective DIRECTLY from A (skip B),
    # identical loss/LR/bs, short budget. Answers "did stage B contribute?".
    dict(name="A2C_short", enc="full", updates=300, enc_lr=5e-6, dec_lr=5e-6,
         bs=4, style=True, edit=True, balanced=(2, 2)),
]


def build_models(dev):
    from tools.vae_suite.decoder_prune import build_decoder
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev)
    ck = torch.load(S10_CKPT, map_location="cpu", weights_only=False)
    dec = build_decoder(ck["meta"]["widths_forward"])
    dec.load_state_dict(ck["state_dict"])
    dec = dec.to(dev)
    model = StyleVAE(vae, dec).to(dev)
    for p in model.post_quant_conv.parameters():
        p.requires_grad = False
    return model, vae


@torch.no_grad()
def latent_monitor(model, ds_hold, dev, n=64, styles=None):
    """Latent distribution + per-style statistics + s(z) spread."""
    model.eval()
    zs, ss, ys = [], [], []
    idx = list(range(min(n, len(ds_hold))))
    for i in range(0, len(idx), 8):
        b = [ds_hold[j] for j in idx[i:i + 8]]
        x = torch.stack([t["x"] for t in b]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            post = model.encode(x)
        z = post.mean.float()
        s, _, _, _ = style_content(z)
        zs.append(z); ss.append(s)
        ys += [t["style_id"] for t in b]
    Z = torch.cat(zs); S = torch.cat(ss)
    y = torch.tensor(ys, device=dev)
    flat = Z.flatten(1)
    out = {
        "z_mean": round(float(Z.mean()), 5),
        "z_std": round(float(Z.std()), 5),
        "z_q01": round(float(flat.quantile(0.01)), 4),
        "z_q99": round(float(flat.quantile(0.99)), 4),
        "ch_mean": [round(float(v), 4) for v in Z.mean(dim=(0, 2, 3))],
        "ch_std": [round(float(v), 4) for v in Z.std(dim=(0, 2, 3))],
    }
    # intra/inter style variance of the style coordinate
    uniq = torch.unique(y)
    if len(uniq) > 1:
        means = torch.stack([S[y == u].mean(0) for u in uniq])
        intra = float(torch.stack(
            [S[y == u].std(0, unbiased=False).mean() for u in uniq]).mean())
        inter = float(means.std(0).mean())
        out["s_intra_std"] = round(intra, 5)
        out["s_inter_std"] = round(inter, 5)
        out["s_ratio"] = round(inter / (intra + 1e-9), 3)
    model.train()
    return out


@torch.no_grad()
def recon_eval(model, ds_hold, dev, lpips_fn, csd=None, n=128, bs=4):
    model.eval()
    l1s, lps, csds = [], [], []
    nm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    ns = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    for i in range(0, min(n, len(ds_hold)), bs):
        b = [ds_hold[j] for j in range(i, min(i + bs, min(n, len(ds_hold))))]
        x = torch.stack([t["x"] for t in b]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            post = model.encode(x)
            y = model.decode(post.mean)
        y = y.float()
        l1s.append(float((y - x).abs().mean()))
        lps.append(float(lpips_fn(y, x).mean()))
        if csd is not None:
            f = []
            for t in (y, x):
                t01 = (t.clamp(-1, 1) + 1) / 2
                t224 = F.interpolate(t01, (224, 224), mode="bicubic", align_corners=False)
                _, _, s = csd((t224 - nm) / ns)
                f.append(F.normalize(s.float(), dim=-1))
            csds.append(float((f[0] * f[1]).sum(-1).mean()))
        torch.cuda.empty_cache()
    model.train()
    r = {"l1": round(float(np.mean(l1s)), 5), "lpips": round(float(np.mean(lps)), 5)}
    if csds:
        r["csd"] = round(float(np.mean(csds)), 5)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="A_joint,B_style,C_edit")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--lam-p", type=float, default=0.5)
    ap.add_argument("--beta-kl", type=float, default=1e-4,
                    help="weight on the ELEMENTWISE-mean KL, so it is comparable "
                         "to the elementwise-mean L1. The diffusers per-sample SUM "
                         "convention is ~16384x larger and would dominate L1.")
    ap.add_argument("--kl-reduction", default="elem", choices=["elem", "sum"])
    ap.add_argument("--lam-s", type=float, default=0.1)
    ap.add_argument("--lam-s-ramp", type=float, default=3.0,
                    help="final multiplier on the calibrated lam_s; the plan asks to raise it gradually after the initial 10-20% gradient share")
    ap.add_argument("--lam-e", type=float, default=0.1)
    ap.add_argument("--lam-c", type=float, default=0.5)
    ap.add_argument("--lpips-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-updates", type=int, default=0,
                    help="override each stage's update budget (smoke tests)")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    man, train_recs, val_recs = load_split(MANIFEST)
    ds_tr = StyleImageDS(train_recs, man)
    ds_val = StyleImageDS(val_recs, man)
    print(f"[Data] train={len(ds_tr)} val={len(ds_val)} "
          f"(real-time RGB encode; latent cache NOT used)", flush=True)

    n_styles = max(r["style_id"] for r in train_recs) + 1

    model, vae = build_models(dev)
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    _csd_cache = {"m": None}

    def get_csd():
        """Load CSD only for evaluation: it holds ~1.2 GB while resident and we
        are already close to the 24 GB limit with the encoder unfrozen."""
        if _csd_cache["m"] is None:
            try:
                import csd_model
                m, _, _ = csd_model.load_csd(dev)
                m = m.float().eval()
                for q in m.parameters():
                    q.requires_grad = False
                _csd_cache["m"] = m
            except Exception as e:
                print(f"[CSD] unavailable: {e}", flush=True)
                _csd_cache["m"] = False
        return _csd_cache["m"] or None

    head = StyleHead(n_styles).to(dev)
    crit = StyleVAELoss(lpips_fn=lp, lam_p=args.lam_p, beta=args.beta_kl,
                        kl_reduction=args.kl_reduction)

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.decoder.load_state_dict(ck["decoder"])
        if "style_head" in ck:
            head.load_state_dict(ck["style_head"])
        print(f"[Init] resumed from {args.init_from}", flush=True)


    results = {}
    for st in STAGES:
        if st["name"] not in args.stages.split(","):
            continue
        print("\n" + "=" * 92)
        print(f"STAGE {st['name']}  enc={st['enc']}  updates={st['updates']}  "
              f"bs={st['bs']}  style={st['style']}  edit={st['edit']}")
        print("=" * 92, flush=True)
        if args.max_updates:
            st = {**st, "updates": args.max_updates}

        named = unfreeze_encoder(model, st["enc"])
        # diffusers' Encoder has no built-in checkpointing, so we wrap it manually
        # in StyleVAE.encode. Needed once the encoder trains at 512px.
        enc_ckpt = (st["enc"] == "full")
        print(f"[Unfreeze] {named}  manual_enc_ckpt={enc_ckpt}", flush=True)
        enc_ps = [p for p in model.encoder.parameters() if p.requires_grad]
        dec_ps = [p for p in model.decoder.parameters() if p.requires_grad]
        groups = [{"params": dec_ps, "lr": st["dec_lr"]}]
        if enc_ps:
            groups.append({"params": enc_ps, "lr": st["enc_lr"]})
        if st["style"]:
            groups.append({"params": head.parameters(), "lr": 1e-3})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) *
            (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / st["updates"])))))

        bs = st["bs"]
        accum = max(1, round(64 / bs))
        base = model.post_quant_conv
        bs_param = base.bias.detach().clone() if base.bias is not None else None

        if st.get("balanced"):
            ns_, ps_ = st["balanced"]
            bsampler = BalancedStyleBatchSampler(ds_tr, ns_, ps_, seed=args.seed)
            dl = DataLoader(ds_tr, batch_sampler=bsampler, num_workers=0,
                            collate_fn=collate)
        else:
            dl = DataLoader(ds_tr, batch_size=st["bs"], shuffle=True,
                            num_workers=args.workers, collate_fn=collate,
                            drop_last=True,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers > 0 else None)
        t0 = time.time()
        it = iter(dl)
        n_img = 0
        lam_s = args.lam_s          # auto-calibrated once, see below
        calib_done = False

        def _ramp(u):
            """1x until upd 200, then linear up to --lam-s-ramp by upd 1200."""
            if u < 200:
                return 1.0
            f = min(1.0, (u - 200) / 1000.0)
            return 1.0 + (args.lam_s_ramp - 1.0) * f
        best = None
        for upd in range(1, st["updates"] + 1):
            ramp = _ramp(upd) if st["style"] else 1.0
            opt.zero_grad(set_to_none=True)
            acc = {}
            style_grad_norm = 0.0
            for _ in range(accum):
                try:
                    b = next(it)
                except StopIteration:
                    it = iter(dl); b = next(it)
                x = b["x"].to(dev, non_blocking=True)
                sid = b["style_id"].to(dev)

                # ---- C: statistics swap + cycle consistency ----
                # Runs as its OWN forward+backward BEFORE the main graph so the
                # two activation sets are serialised, not summed (they OOMed when
                # both were alive at once).
                if st["edit"] and upd % 4 == 0:
                    half = len(x) // 2
                    xa, xb = x[:half], x[half:2 * half]
                    l_e, ep = edit_loss(model, xa, xb, lam_c=args.lam_c,
                                        use_ckpt=enc_ckpt)
                    (args.lam_e * l_e / accum).backward()
                    acc.update({k: acc.get(k, 0.0) + v / accum
                                for k, v in ep.items()})
                    del l_e
                    torch.cuda.empty_cache()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    post = model.encode(x, use_ckpt=enc_ckpt)
                    z = post.sample()
                    y = model.decode(z)
                tot, parts = crit(x, y, post, lpips_size=args.lpips_size)

                # ---- B: style supervision on the 8-dim coordinate only ----
                if st["style"]:
                    svec, cvec, _, _ = style_content(z.float())
                    l_ce = F.cross_entropy(head(svec), sid)
                    # SupCon uses the style groups already present in the batch
                    l_sc = supcon(svec, sid)
                    l_cc = torch.zeros((), device=dev)
                    if st.get("balanced") and upd % 4 == 0:
                        xc = F.interpolate(x, (448, 448), mode="bilinear",
                                           align_corners=False)
                        ox = int(torch.randint(0, 64, (1,)).item())
                        oy = int(torch.randint(0, 64, (1,)).item())
                        crop = xc[:, :, oy:oy + 384, ox:ox + 384]
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            sc_, _, _, _ = style_content(
                                model.encode(crop, use_ckpt=enc_ckpt).mean.float())
                        l_cc = crop_consistency(svec, sc_)
                    l_sty = l_ce + l_sc + 0.5 * l_cc
                    tot = tot + lam_s * ramp * l_sty
                    parts.update({"ce": float(l_ce.detach()),
                                  "supcon": float(l_sc.detach()),
                                  "crop_c": float(l_cc.detach())})
                    # ---- one-shot calibration of lam_s -------------------------
                    # The plan asks for the style gradient at the encoder output to
                    # be ~10-20% of the reconstruction gradient. Rather than guess
                    # a weight, measure the ratio once and rescale to hit 15%.
                    if upd == 20 and not calib_done:
                        gs = torch.autograd.grad(lam_s * l_sty, z,
                                                 retain_graph=True,
                                                 allow_unused=True)[0]
                        gr = torch.autograd.grad(tot, z, retain_graph=True,
                                                 allow_unused=True)[0]
                        if gs is not None and gr is not None:
                            ratio = float(gs.norm() / (gr.norm() + 1e-12))
                            style_grad_norm = ratio
                            if ratio > 0:
                                lam_s = float(min(max(lam_s * 0.15 / ratio,
                                                      1e-4), 10.0))
                            calib_done = True
                            print(f"  [{st['name']}] LAM_S CALIB: ratio={ratio:.4f} "
                                  f"-> lam_s={lam_s:.5f}", flush=True)

                (tot / accum).backward()
                n_img += len(x)
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + v / accum

            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step(); sched.step()

            if upd % 20 == 0 or upd == 1:
                el = time.time() - t0
                msg = " ".join(f"{k}={v:.4f}" for k, v in acc.items())
                if style_grad_norm:
                    msg += f" |gs/grec|={style_grad_norm:.3f}"
                if st["style"]:
                    msg += f" |lam_s={lam_s*ramp:.5f}(x{ramp:.2f})"
                print(f"  [{st['name']}] {upd:5d}/{st['updates']} | {msg} | "
                      f"{el/upd:.2f}s/upd {n_img/el:.2f} img/s | "
                      f"{(st['updates']-upd)*el/upd/60:.1f}min left", flush=True)

            if upd % 250 == 0 or upd == st["updates"]:
                mon = latent_monitor(model, ds_val, dev)
                rv = recon_eval(model, ds_val, dev, lp, get_csd())
                print(f"  [{st['name']}] MONITOR {json.dumps(mon)}", flush=True)
                print(f"  [{st['name']}] RECON   {json.dumps(rv)}", flush=True)
                rec = {"stage": st["name"], "upd": upd, **mon, **rv}
                (OUT / f"{st['name']}_metrics.jsonl").open("a").write(
                    json.dumps(rec) + "\n")
                if best is None or rv["lpips"] < best["lpips"]:
                    best = rv
                    torch.save({"encoder": {k: v.detach().cpu() for k, v
                                            in model.encoder.state_dict().items()},
                                "decoder": {k: v.detach().cpu() for k, v
                                            in model.decoder.state_dict().items()},
                                "style_head": {k: v.detach().cpu() for k, v
                                               in head.state_dict().items()},
                                "meta": {"stage": st["name"], "upd": upd,
                                         "enc_mode": st["enc"], **rv}},
                               OUT / f"{st['name']}_best.pt")
                    print(f"  [{st['name']}] saved best (lpips={rv['lpips']})",
                          flush=True)

        el = time.time() - t0
        torch.save({"encoder": {k: v.detach().cpu() for k, v
                                in model.encoder.state_dict().items()},
                    "decoder": {k: v.detach().cpu() for k, v
                                in model.decoder.state_dict().items()},
                    "style_head": {k: v.detach().cpu() for k, v
                                   in head.state_dict().items()},
                    "meta": {"stage": st["name"], "upd": st["updates"],
                             "enc_mode": st["enc"], "best": best}},
                   OUT / f"{st['name']}_last.pt")
        results[st["name"]] = {"best": best, "wall_min": round(el / 60, 1),
                               "img_per_sec": round(n_img / el, 2)}
        print(f"  [{st['name']}] DONE {json.dumps(results[st['name']])}", flush=True)

    (OUT / "train_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[Saved] {OUT/'train_summary.json'}")


if __name__ == "__main__":
    main()
