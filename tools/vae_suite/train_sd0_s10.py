"""SD0-S10 V2 trainer.

    python -m tools.vae_suite.train_sd0_s10 --stage p1_warm --init W
    python -m tools.vae_suite.train_sd0_s10 --stage c0     --init W
    python -m tools.vae_suite.train_sd0_s10 --stage c1     --init-from runs/sd0_s10/c0_best.pt

Stages share ONE loop. Losses:
  C0 : Charbonnier + λ_p LPIPS@512 + λ_h HF(2-level Haar, per-coefficient)
       + λ_e energy (default OFF) + λ_o off-manifold teacher (C1 only)
  C1 : adds response distillation (finite-difference Secant/Jacobian matching)
       on 25-40% of samples, with 3 perturbation kinds and 2 magnitude bands.

The official SD0 Encoder/quant_conv are FROZEN (hash checked at start). The
teacher decoder is eval() + requires_grad_(False) + all outputs detached.
StudentDecoder applies post_quant_conv EXACTLY ONCE.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "vae_suite"))

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
from tools.vae_suite.sd0_losses import (ReconLoss, WaveletHF,
                                        estimate_band_scales, make_perturbation,
                                        off_manifold_loss, prune_with_indices,
                                        response_loss)

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0 = str(ROOT / "runs_vae" / "A0" / "final_diffusers_vae")
WARM_CK = ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
IDX_CK = ROOT / "runs/vae_decoder_prune/s10_p1_l1/decoder.pt"
OUT = ROOT / "runs/sd0_s10"
FULL = OUT / "full"
WIDTHS = [64, 128, 192, 256]
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")

STAGES = {
    # name        updates  lr       hf    response  off   init
    "p1_warm": dict(updates=800, lr=3e-5, hf=True, resp=False, off=False),
    "p1_cold": dict(updates=800, lr=5e-5, hf=True, resp=False, off=False),
    "c0":      dict(updates=3000, lr=3e-5, hf=True, resp=False, off=False),
    "c1":      dict(updates=2500, lr=1.5e-5, hf=True, resp=True, off=True),
}


# --------------------------------------------------------------------------- data
class LatentImageDS(Dataset):
    """Cached SD0 latent row i + its exact 512px RGB image (raw_index[i])."""

    def __init__(self, bin_path, n, dim, raw_index, mmap, name):
        self.lat = np.memmap(bin_path, dtype=np.float16, mode="r", shape=(n, dim))
        self.img = mmap
        self.raw = np.asarray(raw_index, np.int64)
        self.n = n
        assert len(self.raw) == n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = np.ascontiguousarray(self.img[int(self.raw[i])])
        x = torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        z = torch.from_numpy(self.lat[i].astype(np.float32)).view(4, 64, 64)
        return {"x": x, "z": z}


def collate(batch):
    return {"x": torch.stack([b["x"] for b in batch]),
            "z": torch.stack([b["z"] for b in batch])}


def load_data():
    sp = json.loads((FULL / "split.json").read_text())
    m = json.loads(MANIFEST.read_text())
    mmap = np.memmap(m["bin_path"], dtype=np.uint8, mode="r",
                     shape=tuple(m["shape"]))
    ds_tr = LatentImageDS(sp["train_bin"], sp["n_train"], sp["dim"],
                          np.load(FULL / "raw_index_train.npy"), mmap, "train")
    ds_ho = LatentImageDS(sp["holdout_bin"], sp["n_holdout"], sp["dim"],
                          np.load(FULL / "raw_index_holdout.npy"), mmap, "holdout")
    ds_te = LatentImageDS(sp["test256_bin"], sp["n_test256"], sp["dim"],
                          np.load(FULL / "raw_index_test256.npy"), mmap, "test256")
    return sp, ds_tr, ds_ho, ds_te


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(student, ds, dev, lpips_fn, csd=None, n=256, bs=4):
    import lpips as _  # noqa
    student.eval()
    whf = WaveletHF({(l, b): 1.0 for l in range(2) for b in range(3)}).to(dev)
    acc = defaultdict(list)
    idx = np.random.default_rng(31).choice(min(n, len(ds)), min(n, len(ds)),
                                           replace=False)
    for i in range(0, len(idx), bs):
        b = [ds[int(j)] for j in idx[i:i + bs]]
        x = torch.stack([t["x"] for t in b]).to(dev)
        z = torch.stack([t["z"] for t in b]).to(dev)
        y = student(z).float()
        acc["l1"].append(float((y - x).abs().mean()))
        p = ((y.clamp(-1, 1) + 1) / 2 - (x + 1) / 2).pow(2).mean().item()
        acc["psnr"].append(10 * np.log10(1 / max(p, 1e-10)))
        acc["lpips"].append(float(lpips_fn(y.clamp(-1, 1), x).mean()))
        _, _, pb = whf(y, x)
        acc["hf_l1"].append(float(np.mean(list(pb.values()))))
        # HF correlation on finest band
        from tools.vae_suite.sd0_losses import haar_dwt
        wy, wx = haar_dwt(y, 1)[0], haar_dwt(x, 1)[0]
        cs = []
        for bi in range(3):
            a, c = wy[bi].flatten(), wx[bi].flatten()
            a, c = a - a.mean(), c - c.mean()
            cs.append(float((a * c).sum() / (a.norm() * c.norm() + 1e-8)))
        acc["hf_corr_L0"].append(float(np.mean(cs)))
        acc["hf_energy_ratio"].append(
            float(np.mean([float(wy[bi].std() / (wx[bi].std() + 1e-8))
                           for bi in range(3)])))
        # SSIM + colour (same definitions as eval_sd0_s10.py) so gates see a
        # consistent schema across scripts.
        a, b = (y.clamp(-1, 1) + 1) / 2, (x + 1) / 2
        C1, C2 = 1e-4, 9e-4
        ma, mb = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
        maa, mbb, mab = ma * ma, mb * mb, ma * mb
        sa = F.avg_pool2d(a * a, 11, 1, 5) - maa
        sb = F.avg_pool2d(b * b, 11, 1, 5) - mbb
        sab = F.avg_pool2d(a * b, 11, 1, 5) - mab
        acc["ssim"].append(float((((2 * mab + C1) * (2 * sab + C2)) /
                                  ((maa + mbb + C1) * (sa + sb + C2))).mean()))
        acc["colour"].append(
            float((y.mean((2, 3)) - x.mean((2, 3))).abs().mean()))
        if csd is not None:
            nm = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=dev).view(1, 3, 1, 1)
            ns = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                              device=dev).view(1, 3, 1, 1)
            f = []
            for t in (y, x):
                t01 = (t.clamp(-1, 1) + 1) / 2
                t224 = F.interpolate(t01, (224, 224), mode="bicubic",
                                     align_corners=False)
                _, _, sf = csd((t224 - nm) / ns)
                f.append(F.normalize(sf.float(), dim=-1))
            acc["csd"].append(float((f[0] * f[1]).sum(-1).mean()))
        del x, z, y
        torch.cuda.empty_cache()
    student.train()
    return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}


# --------------------------------------------------------------------------- grad calib
def calibrate_weights(student, teacher, batch, dev, loss_bundle, stage):
    """One-shot: λ_h ≈ 10-20% of base recon grad on shared decoder params."""
    # calibration uses only 2 images when a teacher is involved (C1): the
    # paired teacher+student forwards at 512px OOM at bs=4.
    n_cal = 2 if stage.get("resp") else min(4, len(batch["x"]))
    x = batch["x"][:n_cal].to(dev)
    z = batch["z"][:n_cal].to(dev).clone().requires_grad_(True)
    y = student(z)
    l_rec, _ = loss_bundle["recon"](x, y)
    g_rec, = torch.autograd.grad(l_rec, z, retain_graph=True)
    l_hf, _, _ = loss_bundle["hf"](y, x)
    g_hf, = torch.autograd.grad(l_hf, z, retain_graph=True)
    ratio = float(g_hf.norm() / (g_rec.norm() + 1e-12))
    lam_h = min(max(0.15 / max(ratio, 1e-8), 1e-4), 5.0)
    out = {"hf": lam_h, "hf_grad_ratio": ratio}
    if stage["resp"] and teacher is not None:
        delta = make_perturbation(z.detach(), "corr_gauss", 0.05,
                                  generator=torch.Generator(device=dev).manual_seed(0))
        l_r, _ = response_loss(student, teacher, z, delta)
        g_r, = torch.autograd.grad(l_r, z, retain_graph=True)
        rr = float(g_r.norm() / (g_rec.norm() + 1e-12))
        out["resp"] = min(max(0.15 / max(rr, 1e-8), 1e-4), 5.0)
        out["resp_grad_ratio"] = rr
        # off-manifold depends on z+delta, not z; calibrating its grad wrt z is
        # meaningless. Use a fixed small weight (plan: 5-10% level).
        out["off"] = 0.075
    del y, l_rec, g_rec, l_hf, g_hf, z, x
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--init", default="W", choices=["W", "C", "W-PQC"])
    ap.add_argument("--init-from", default="")
    ap.add_argument("--updates", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--band-scales", default="")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    st = dict(STAGES[args.stage]); st["name"] = args.stage
    if args.updates:
        st["updates"] = args.updates

    sp, ds_tr, ds_ho, ds_te = load_data()
    print(f"[Data] train={len(ds_tr)} holdout={len(ds_ho)} test256={len(ds_te)} "
          f"encoder={sp['encoder_hash']['config.json'][:12]}", flush=True)

    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    for p in official.parameters():
        p.requires_grad_(False)
    # Teacher only for C1 (response distillation). P1/C0 do NOT need it, and
    # keeping a 49.49M decoder resident alongside LPIPS@512 + bs8 OOMs.
    # CRITICAL: the teacher must apply post_quant_conv EXACTLY ONCE, just like
    # the student -- AutoencoderKL.decode(z) does pqc(z) then decoder. Passing
    # raw z straight to official.decoder would make the teacher and student see
    # different inputs and corrupt the finite-difference response target.
    pqc_sd0 = official.post_quant_conv
    teacher = (StudentDecoder(pqc_sd0, official.decoder).to(dev).eval()
               if st["resp"] else None)
    # Encoder/quant_conv hash check
    h = hashlib.sha256()
    for k, v in sorted(official.state_dict().items()):
        if k.startswith("encoder") or k.startswith("quant_conv"):
            h.update(k.encode()); h.update(v.cpu().numpy().tobytes())
    print(f"[Encoder] hash {h.hexdigest()[:16]} (frozen)", flush=True)

    idx = torch.load(IDX_CK, map_location="cpu",
                     weights_only=False)["meta"]["channel_index"]
    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        dec = build_decoder(WIDTHS)
        dec.load_state_dict(ck["state_dict"])
        pqc_w = ck.get("post_quant_conv")
        print(f"[Init] resumed {args.init_from} ({ck.get('meta',{}).get('stage','?')})",
              flush=True)
    elif args.init in ("W", "W-PQC"):
        wck = torch.load(WARM_CK, map_location="cpu", weights_only=False)
        dec = build_decoder(WIDTHS)
        dec.load_state_dict(wck["state_dict"])
        pqc_w = wck.get("post_quant_conv")
        if args.init == "W-PQC":
            pqc_w = {k: v.clone() for k, v in pqc_sd0.state_dict().items()}
        print(f"[Init] W={args.init} from s10_p5_lowlr", flush=True)
    else:  # Cold-SD0: Taylor-pruned official decoder, built by build_cold_sd0.py
        cck = torch.load(OUT / "cold_init.pt", map_location="cpu",
                         weights_only=False)
        dec = build_decoder(WIDTHS)
        dec.load_state_dict(cck["state_dict"])
        pqc_w = cck["post_quant_conv"]
        print("[Init] C=Cold-SD0 (Taylor-pruned official decoder, unrecovered)",
              flush=True)

    pqc = copy.deepcopy(pqc_sd0).to(dev)
    if pqc_w is not None:
        pqc.load_state_dict(pqc_w)
    # pqc is TRAINABLE as part of the Decoder (plan section 1.1); encoder is not.
    for p in pqc.parameters():
        p.requires_grad = True
    student = StudentDecoder(pqc, dec.to(dev)).to(dev).train()

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    csd = None
    try:
        sys.path.insert(0, str(ROOT / "tools" / "12h"))
        import csd_model
        csd, _, _ = csd_model.load_csd(dev)
        csd = csd.float().eval()
        for p in csd.parameters():
            p.requires_grad = False
    except Exception as e:
        print(f"[CSD] unavailable ({e}); CSD Keep skipped in-val", flush=True)

    # band scales FIXED from training GT (never from model output)
    bs_path = OUT / "band_scales.json" if not args.band_scales else Path(args.band_scales)
    if bs_path.exists():
        band_scales = {tuple(map(int, k.strip("()").split(","))): v
                       for k, v in json.loads(bs_path.read_text()).items()}
    else:
        dl0 = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=2,
                         collate_fn=collate)
        def _pairs():
            for i, b in enumerate(dl0):
                if i >= 64:
                    break
                yield (b["x"], b["z"])
        band_scales = estimate_band_scales(_pairs(), dev)
        bs_path.write_text(json.dumps({str(k): v for k, v in band_scales.items()}))
        del dl0
    print(f"[HF] fixed band scales: "
          f"{ {k: round(v,5) for k,v in band_scales.items()} }", flush=True)

    loss_bundle = {
        "recon": ReconLoss(lpips_fn=lp, lam_p=0.5, lpips_size=512),
        "hf": WaveletHF(band_scales).to(dev),
    }
    # ---- weight calibration (once, then fixed) ----
    dl_probe = DataLoader(ds_tr, batch_size=4, shuffle=True, collate_fn=collate)
    probe = next(iter(dl_probe))
    lambdas = calibrate_weights(student, teacher, probe, dev, loss_bundle, st)
    lambdas["p"] = 0.5
    lambdas["energy"] = 0.0     # OFF unless a measured amplitude deficit appears
    print(f"[CALIB] {json.dumps(lambdas)}", flush=True)

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=st["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / st["updates"])))))
    raw_decoder = student.decoder            # uncompiled ref for state_dict
    if args.compile:
        try:
            student.decoder.compile(dynamic=False)
            print("[Compile] on (state_dict read via raw module)", flush=True)
        except Exception as e:
            print(f"[Compile] skipped ({e})", flush=True)

    accum = max(1, 64 // args.bs)
    dl = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)
    dl_resp = DataLoader(ds_tr, batch_size=2, shuffle=True, num_workers=0,
                         collate_fn=collate)
    it_resp = iter(dl_resp)
    it = iter(dl)
    t0 = time.time()
    best = None
    n_img = 0
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    import random as _rnd
    pyrng = _rnd.Random(args.seed)
    # Accumulate TRAINING-side metrics across updates. The validation record
    # written every 250 updates must carry `resp` / `resp_cos` / `resp_amp_ratio`
    # so the C1 response Gate can read them; those live in the per-update `acc`,
    # not in evaluate()'s recon-only output.
    stage_sum = defaultdict(float)
    stage_cnt = defaultdict(int)
    log_path = OUT / f"train_{args.stage}_{args.init}.log"
    metrics_path = OUT / f"{args.stage}_{args.init}_metrics.jsonl"

    print(f"\n=== STAGE {args.stage} init={args.init} updates={st['updates']} "
          f"bs={args.bs} accum={accum} lr={st['lr']} ===", flush=True)

    for upd in range(1, st["updates"] + 1):
        opt.zero_grad(set_to_none=True)
        acc = {}
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            x, z = b["x"].to(dev, non_blocking=True), b["z"].to(dev, non_blocking=True)
            y = student(z)
            l_rec, parts = loss_bundle["recon"](x, y)
            l_hf, l_eng, _ = loss_bundle["hf"](y, x)
            total = l_rec + lambdas["hf"] * l_hf + lambdas["energy"] * l_eng
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v / accum
            acc["hf"] = acc.get("hf", 0.0) + float(l_hf.detach()) / accum
            acc["hf_w"] = acc.get("hf_w", 0.0) + float(lambdas["hf"] * l_hf.detach()) / accum

            (total / accum).backward()
            n_img += len(x)
        # ---- C1 response branch: isolated graph, AFTER the reconstruction
        # microbatches are done. Keeping it inside the accum loop stacked 16
        # student graphs plus the teacher's two forwards and OOMed at 21.9 GB.
        # Here one small (2-image) paired pass runs with a clean cache.
        if st["resp"] and (upd % 4 == 0):   # ~25% of updates
            torch.cuda.empty_cache()
            try:
                b = next(it_resp)
            except StopIteration:
                it_resp = iter(dl_resp); b = next(it_resp)
            zr = b["z"][:2].to(dev)
            kinds = ["corr_gauss"] * 5 + ["interp"] * 3 + ["posterior"] * 2
            kind = kinds[pyrng.randrange(len(kinds))]
            frac = [0.02, 0.05, 0.10, 0.20][pyrng.randrange(4)]
            partner = torch.roll(zr, 1, dims=0) if kind == "interp" else None
            delta = make_perturbation(zr, kind, frac, partner, generator=gen)
            l_r, rp = response_loss(student, teacher, zr, delta)
            l_extra = lambdas["resp"] * l_r
            if st["off"]:
                l_o, _ = off_manifold_loss(student, teacher, zr + delta)
                l_extra = l_extra + lambdas["off"] * l_o
            l_extra.backward()
            for k, v in rp.items():
                acc[k] = acc.get(k, 0.0) + v
            acc["resp"] = acc.get("resp", 0.0) + float(l_r.detach())
            if st["off"]:
                acc["off"] = acc.get("off", 0.0) + float(l_o.detach())
            del zr, delta, l_r, l_extra
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        for k, v in acc.items():
            if isinstance(v, (int, float)):
                stage_sum[k] += float(v)
                stage_cnt[k] += 1

        if upd % 50 == 0 or upd == 1:
            el = time.time() - t0
            msg = " ".join(f"{k}={v:.4f}" for k, v in acc.items())
            print(f"  [{args.stage}/{args.init}] {upd:5d}/{st['updates']} | {msg} | "
                  f"{el/upd:.2f}s/upd {n_img/el:.1f} img/s | "
                  f"{(st['updates']-upd)*el/upd/60:.1f}min left", flush=True)
            log_path.open("a").write(f"{upd} " + msg + "\n")

        if upd % 250 == 0 or upd == st["updates"]:
            val = evaluate(student, ds_ho, dev, lp, csd, n=256)
            stage_mean = {f"train_{k}": round(stage_sum[k] / max(stage_cnt[k], 1), 5)
                          for k in stage_sum}
            # Gate-facing aliases: C1 gate reads `resp` / `resp_cos` directly
            # from this jsonl; expose the accumulated training means there too.
            gate_fields = {}
            for k in ("resp", "resp_cos", "resp_amp_ratio", "off"):
                if stage_cnt.get(k, 0) > 0:
                    gate_fields[k] = round(stage_sum[k] / max(stage_cnt[k], 1), 5)
            rec = {"stage": args.stage, "init": args.init, "upd": upd,
                   "lambdas": lambdas, **val, **stage_mean, **gate_fields}
            print(f"  [{args.stage}/{args.init}] VALID {json.dumps(val)}", flush=True)
            metrics_path.open("a").write(json.dumps(rec) + "\n")
            if best is None or val["lpips"] < best["lpips"]:
                best = val
                torch.save({
                    "state_dict": {k: v.detach().cpu() for k, v in
                                   raw_decoder.state_dict().items()},
                    "post_quant_conv": {k: v.detach().cpu() for k, v in
                                        student.post_quant_conv.state_dict().items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "meta": {"stage": args.stage, "init": args.init, "upd": upd,
                             "lambdas": lambdas, "best": best,
                             "widths_forward": WIDTHS,
                             "needs_post_quant_conv": True,
                             "encoder": SD0,
                             "encoder_hash": sp["encoder_hash"],
                             "latent_convention": sp["latent_convention"]},
                }, OUT / f"{args.stage}_{args.init}_best.pt")
                print(f"  [{args.stage}/{args.init}] saved best lpips={val['lpips']}",
                      flush=True)

    el = time.time() - t0
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in
                       raw_decoder.state_dict().items()},
        "post_quant_conv": {k: v.detach().cpu() for k, v in
                            student.post_quant_conv.state_dict().items()},
        "opt": opt.state_dict(), "sched": sched.state_dict(),
        "meta": {"stage": args.stage, "init": args.init, "upd": st["updates"],
                 "best": best, "lambdas": lambdas, "widths_forward": WIDTHS,
                 "needs_post_quant_conv": True, "encoder": SD0,
                 "encoder_hash": sp["encoder_hash"]},
    }, OUT / f"{args.stage}_{args.init}_last.pt")
    (OUT / f"train_{args.stage}_{args.init}_summary.json").write_text(json.dumps({
        "stage": args.stage, "init": args.init, "updates": st["updates"],
        "best": best, "wall_min": round(el / 60, 1),
        "img_per_sec": round(n_img / el, 2), "lambdas": lambdas}, indent=2))
    print(f"  [{args.stage}/{args.init}] DONE {json.dumps(best)} "
          f"wall={el/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
