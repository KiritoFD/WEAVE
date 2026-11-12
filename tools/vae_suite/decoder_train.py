"""Multi-stage Decoder pruning + recovery training.

Pipeline (fast compression first, expensive refinement last):
    A0 49.5M --prune--> P1 28M --L1--> P2 18M --L1--> P3 10.7M --L1--> P4 10.7M --L1+LPIPS+HF--> export

P1-P3 use ONLY L1 against real RGB (cheap, fast capacity recovery).
P4 adds LPIPS + multi-scale Haar high-freq for final quality.
KL is NOT used: the Encoder is frozen, so d KL / d theta_decoder == 0.

Every stage: compute Taylor importance on the CURRENT weights -> structural prune
-> inherit channels -> L1 recovery -> keep best-on-holdout.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/root/Workspace/.inductor_cache")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.decoder_prune import (A0_DIR, MANIFEST, A0_SRC, make_plan,
                                           prune_decoder, taylor_importance, get_module,
                                           StudentDecoder, build_decoder)

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
STAGES = [
    # one-shot prune A0 -> 10.66M, then pure-L1 capacity recovery on the 13k cache
    {"name": "s10_p1_l1", "widths": [64, 128, 192, 256], "updates": 1200,
     "lr": 1e-4, "loss": "l1"},
    # refinement (needs the FULL WikiArt latent cache; run separately)
    # perceptual refinement: L1 + LPIPS only. No HF (experiment A showed the
    # high-frequency bands are NOT what S10 loses), no CSD (kept as a held-out
    # judge), no KL (encoder frozen).
    {"name": "s10_p3_lpips512", "widths": [64, 128, 192, 256], "updates": 600,
     "lr": 2e-5, "loss": "l1+lpips"},
    # P2: full WikiArt proper train pool, same minimal loss, lr 3e-5 cosine
    {"name": "s10_p4_full", "widths": [64, 128, 192, 256], "updates": 8000,
     "lr": 3e-5, "loss": "l1+lpips"},
    # P3 low-LR continuation: P2 plateaued while the cosine LR was still ~2.9e-5
    # (the schedule was stretched over 8000 steps), so try a short low-LR pass to
    # see whether CSD can break through 0.87 without any style loss.
    {"name": "s10_p5_lowlr", "widths": [64, 128, 192, 256], "updates": 1200,
     "lr": 1e-5, "loss": "l1+lpips"},
]


# ---------------------------------------------------------------- data
class LatentImageDS(torch.utils.data.Dataset):
    """Cached latent (row i) + the real image it came from (raw_index[i])."""

    def __init__(self, latent_bin, n, dim, raw_index, manifest_path, idx=None):
        self.lat = np.memmap(latent_bin, dtype=np.float16, mode="r", shape=(n, dim))
        man = json.loads(Path(manifest_path).read_text())
        self.img = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                             shape=tuple(man["shape"]))
        self.raw = np.asarray(raw_index, dtype=np.int64)
        assert len(self.raw) == n, f"raw_index {len(self.raw)} != latent rows {n}"
        self.idx = list(range(n)) if idx is None else idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        ri = int(self.raw[j])
        z = torch.from_numpy(self.lat[j].astype(np.float32)).view(4, 64, 64)
        a = np.ascontiguousarray(self.img[ri])
        x = torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return {"z": z, "x": x}


# ---------------------------------------------------------------- losses
def haar_bands(x, levels=3):
    outs = []
    for _ in range(levels):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        outs.append(((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5))
        x = (a + b + c + d) * 0.5
    return outs


def hf_loss(y, x):
    tot = 0.0
    for sy, sx in zip(haar_bands(y), haar_bands(x)):
        for a, b in zip(sy, sx):
            tot = tot + (a - b).abs().mean() / (b.abs().mean().detach() + 1e-6)
    return tot / 9.0


# ---------------------------------------------------------------- helpers
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)


@torch.no_grad()
def holdout_eval(dec, ds, idx, device, lpips_fn=None, csd=None, bs=4, n=200):
    """L1 always; LPIPS and CSD when the modules are supplied.

    CSD is a HELD-OUT judge here: it is never trained on, only observed.
    """
    dec.eval()
    sel = idx[:n]
    l1_sum, cnt = 0.0, 0
    lps, csds = [], []
    nm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    ns = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    for i in range(0, len(sel), bs):
        b = [ds[j] for j in sel[i:i + bs]]
        z = torch.stack([t["z"] for t in b]).to(device)
        x = torch.stack([t["x"] for t in b]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = dec(z)
        y = y.float()
        l1_sum += float((y - x).abs().mean()) * len(b)
        cnt += len(b)
        if lpips_fn is not None:
            lps.append(float(lpips_fn(y, x).mean()))
        if csd is not None:
            x01 = (y.clamp(-1, 1) + 1) / 2
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            _, _, sy = csd((x224 - nm) / ns)
            x01t = (x.clamp(-1, 1) + 1) / 2
            x224t = F.interpolate(x01t, (224, 224), mode="bicubic", align_corners=False)
            _, _, sx = csd((x224t - nm) / ns)
            csds.append(float((F.normalize(sy.float(), dim=-1) *
                               F.normalize(sx.float(), dim=-1)).sum(-1).mean()))
        torch.cuda.empty_cache()
    dec.train()
    out = {"l1": l1_sum / max(cnt, 1)}
    if lps:
        out["lpips"] = float(np.mean(lps))
    if csds:
        out["csd"] = float(np.mean(csds))
    return out


def holdout_l1(dec, ds, idx, device, bs=4, n=200):
    return holdout_eval(dec, ds, idx, device, bs=bs, n=n)["l1"]


def probe_batch(dec, ds, idx, device, cands=(24, 16, 12, 8, 6, 4, 2)):
    """Find the largest microbatch that fits with headroom (decoder-only, bf16).

    Re-probed per stage: the measured limit differs a lot (P1 bs=4, P3 bs=6).
    Only measures peak memory; NEVER steps the optimizer.
    """
    for bs in cands:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            z = torch.stack([ds[j]["z"] for j in idx[:bs]]).to(device)
            x = torch.stack([ds[j]["x"] for j in idx[:bs]]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = (dec(z).float() - x).abs().mean()
            loss.backward()
            peak = torch.cuda.max_memory_allocated(0) / 1024**3
            dec.zero_grad(set_to_none=True)
            del z, x, loss
            torch.cuda.empty_cache()
            if peak < 20.0:
                return bs, round(peak, 2)
        except torch.cuda.OutOfMemoryError:
            dec.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
    return 2, None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="s10_p1_l1,s10_p2_lpips")
    ap.add_argument("--init-from", default="",
                    help="load this stage checkpoint's decoder weights before training")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--bs", type=int, default=0, help="0 = auto probe")
    ap.add_argument("--accum", type=int, default=0, help="0 = 64//bs")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-dir", default="",
                    help="dir holding split.json + raw_index_*.npy "
                         "(default: the run dir)")
    ap.add_argument("--lpips-size", type=int, default=512,
                    help="resolution for the LPIPS term; 256 does NOT transfer to "
                         "512 (measured: -4.6%% on 512px LPIPS in 500 updates)")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    SD = Path(args.split_dir) if args.split_dir else OUT
    split = json.loads((SD / "split.json").read_text())
    n_tr, n_ho, dim = split["n_train"], split["n_holdout"], split["dim"]
    train_raw = np.load(SD / "raw_index_train.npy") if (SD / "raw_index_train.npy").exists()         else np.load(SD / "raw_index.npy")
    hold_raw = np.load(SD / "raw_index_holdout.npy") if (SD / "raw_index_holdout.npy").exists()         else np.load(SD / "holdout_raw_index.npy")
    print(f"[SplitDir] {SD} train={n_tr} holdout={n_ho}", flush=True)
    print(f"[Data] train={n_tr} holdout={n_ho} dim={dim} "
          f"(styles={split.get('n_styles','?')}, src={split.get('source','?')})",
          flush=True)

    ds = LatentImageDS(split["train_bin"], n_tr, dim, train_raw, MANIFEST)
    ds_hold = LatentImageDS(split["holdout_bin"], n_ho, dim, hold_raw, MANIFEST)
    hold_local = list(range(n_ho))
    print(f"[Data] ready", flush=True)

    vae_a0 = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(device).eval()
    dec_a0 = vae_a0.decoder
    pqc_a0 = vae_a0.post_quant_conv          # 4->4, never pruned (latent contract)
    a0_params = sum(p.numel() for p in dec_a0.parameters())
    print(f"[A0] decoder {a0_params/1e6:.3f} M (+ post_quant_conv)", flush=True)

    # reference wrapper so holdout L1 uses the SAME input contract as training.
    # post_quant_conv is deep-copied + frozen so the A0 reference never drifts when
    # the student's own post_quant_conv is trained.
    pqc_ref = copy.deepcopy(pqc_a0).eval()
    for p in pqc_ref.parameters():
        p.requires_grad = False
    ref_a0 = StudentDecoder(pqc_ref, dec_a0).eval()
    L_a0 = holdout_l1(ref_a0.to(device), ds_hold, hold_local, device)
    print(f"[A0 ref] holdout L1 = {L_a0:.5f} (with post_quant_conv)", flush=True)

    # calibration pool for Taylor importance
    rng = np.random.default_rng(0)
    calib_pos = rng.choice(n_tr, args.calib, replace=False)
    zs, xs = [], []
    for i in calib_pos:
        t = ds[int(i)]
        zs.append(t["z"]); xs.append(t["x"])
    Zc = torch.stack(zs); Xc = torch.stack(xs)
    print(f"[Calib] {tuple(Zc.shape)}", flush=True)

    # current decoder state, starts from A0
    cur_dec = copy.deepcopy(dec_a0)
    cur_meta = {"widths_forward": [128, 256, 512, 512], "params": a0_params}
    if args.init_from and Path(args.init_from).exists():
        _ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        _w = _ck["meta"]["widths_forward"]
        cur_dec = build_decoder(_w)
        cur_dec.load_state_dict(_ck["state_dict"])
        cur_meta = {"widths_forward": _w, "params": _ck["meta"]["params"]}
        print(f"[Init] loaded {args.init_from} widths={_w} "
              f"params={cur_meta['params']/1e6:.3f}M "
              f"holdout_l1={_ck['meta'].get('holdout_l1')}", flush=True)
    results = {}

    for st in STAGES:
        if st["name"] not in args.stages.split(","):
            continue
        print(f"\n{'='*78}\nSTAGE {st['name']}  widths={st['widths']}  "
              f"updates={st['updates']}  lr={st['lr']}  loss={st['loss']}\n{'='*78}",
              flush=True)

        # ---- 1. prune from current weights (skip if widths already match) ----
        if cur_meta["widths_forward"] != st["widths"]:
            print("[Prune] computing Taylor importance on current weights...", flush=True)
            imp = taylor_importance(cur_dec, make_plan(cur_dec), Zc, Xc, device,
                                    pre=pqc_a0)
            cur_dec = cur_dec.cpu()
            torch.cuda.empty_cache()
            cur_dec, cur_meta = prune_decoder(cur_dec, st["widths"], importance=imp)
            print(f"[Prune] -> {cur_meta['params']/1e6:.3f} M "
                  f"({cur_meta['params']/a0_params*100:.1f}% of A0)", flush=True)

        dec = StudentDecoder(pqc_a0, cur_dec.to(device)).to(device).train()
        params = sum(p.numel() for p in dec.decoder.parameters())

        # ---- post-prune baseline (A0 ref computed once above) ----
        L_prune = holdout_l1(dec, ds_hold, hold_local, device)
        gap = max(L_prune - L_a0, 1e-9)
        print(f"[Ref] holdout L1: A0={L_a0:.5f}  post-prune={L_prune:.5f}  "
              f"gap={gap:.5f} ({gap/max(L_a0,1e-9)*100:.1f}% above A0)", flush=True)

        # ---- 2. batch probe ----
        if args.bs > 0:
            bs = args.bs
        else:
            bs, peak = probe_batch(dec, ds, list(range(24)), device)
            print(f"[Batch] auto bs={bs} peak={peak} GB", flush=True)
        accum = args.accum if args.accum > 0 else max(1, 64 // bs)
        print(f"[Batch] microbatch={bs} accum={accum} effective={bs*accum}", flush=True)

        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4,
                                         drop_last=True, persistent_workers=False)

        # EMA must be created BEFORE in-place compile (deepcopy of a compiled module
        # is unreliable).
        ema = EMA(dec, 0.999)

        # ---- 3. compile (after pruning, per stage) ----
        if args.compile:
            t0 = time.time()
            try:
                dec.compile(dynamic=False)
                print(f"[Compile] {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[Compile] failed: {e}", flush=True)

        # ---- 4. losses ----
        lpips_fn, csd_judge = None, None
        if st["loss"] != "l1":
            import lpips as L
            lpips_fn = L.LPIPS(net="alex", verbose=False).to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad = False
            # CSD is a HELD-OUT JUDGE ONLY -- never in the loss.
            try:
                import csd_model
                csd_judge, _, _ = csd_model.load_csd(device)
                csd_judge = csd_judge.float().eval()
                for p in csd_judge.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[CSD judge] unavailable: {e}", flush=True)

        def compute_loss(y, x):
            y = y.float()
            l1 = (y - x).abs().mean()
            if lpips_fn is None:
                return l1, {"l1": float(l1.detach())}
            if args.lpips_size and args.lpips_size != y.shape[-1]:
                ys = F.interpolate(y, args.lpips_size, mode="bilinear", align_corners=False)
                xs = F.interpolate(x, args.lpips_size, mode="bilinear", align_corners=False)
            else:
                ys, xs = y, x
            l_p = lpips_fn(ys, xs).mean()
            # HF term only when the stage explicitly asks for it: experiment A
            # showed S10 does NOT lose high-frequency detail, so it is off by
            # default (loss string "l1+lpips").
            if "hf" in st["loss"]:
                l_h = hf_loss(y, x)
                tot = l1 + 0.5 * l_p + 0.25 * l_h
            else:
                l_h = torch.zeros((), device=y.device)
                tot = l1 + 0.5 * l_p
            return tot, {"l1": float(l1.detach()), "lpips": float(l_p.detach()),
                         "hf": float(l_h.detach())}

        opt = torch.optim.AdamW(dec.parameters(), lr=st["lr"], betas=(0.9, 0.99),
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) *
            (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / st["updates"])))))

        best = float("inf")
        best_state = None
        it = iter(dl)
        t0 = time.time()
        hist = []            # [(upd, holdout_l1)]
        plateau = []         # [(lpips, csd)] for the perceptual plateau rule
        n_img = 0
        stop_reason = "max_updates"
        for upd in range(1, st["updates"] + 1):
            opt.zero_grad(set_to_none=True)
            acc = {}
            for _ in range(accum):
                try:
                    zb = next(it)
                except StopIteration:
                    it = iter(dl); zb = next(it)
                z = zb["z"].to(device, non_blocking=True)
                x = zb["x"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = dec(z)
                loss, parts = compute_loss(y, x)
                (loss / accum).backward()
                n_img += z.shape[0]
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + v / accum
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(dec)

            if upd % 20 == 0 or upd == 1:
                el = time.time() - t0
                ips = n_img / el
                print(f"  [{st['name']}] {upd:5d}/{st['updates']} | "
                      + " ".join(f"{k}={v:.4f}" for k, v in acc.items())
                      + f" | {el/upd:.2f}s/upd {ips:.2f} img/s "
                      + f"| {(st['updates']-upd)*el/upd/60:.1f}min left", flush=True)

            if upd % 100 == 0 or upd == st["updates"]:
                he = holdout_eval(dec, ds_hold, hold_local, device,
                                  lpips_fn=lpips_fn, csd=csd_judge)
                hl = he["l1"]
                hist.append((upd, hl))
                # selection metric: LPIPS for perceptual stages, L1 for recovery
                if "lpips" in he:
                    sel = he["lpips"]
                    # the plan allows L1 <= 0.075 absolute; a pure ratio guard
                    # rejects checkpoints whose L1 rises slightly while LPIPS
                    # halves, which is exactly what we want to keep.
                    l1_ok = hl <= max(L_prune * 1.15, 0.075)
                    better = l1_ok and (best_state is None or sel < best)
                else:
                    sel = hl
                    better = hl < best
                if better:
                    best = sel
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in dec.state_dict().items()}
                recovered = (L_prune - hl) / gap
                extra = "".join(f" {k}={v:.4f}" for k, v in he.items() if k != "l1")
                print(f"  [{st['name']}] holdout L1={hl:.5f}{extra} "
                      f"(best_sel={best:.5f}) recovered={recovered*100:.1f}% "
                      f"[A0={L_a0:.5f}]", flush=True)
                # crash insurance: durable checkpoint every 250 updates
                if upd % 250 == 0 and best_state is not None:
                    _d = OUT / st["name"]; _d.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": dec.decoder.state_dict(),
                                "post_quant_conv": dec.post_quant_conv.state_dict(),
                                "meta": {**cur_meta, "widths_forward": st["widths"],
                                         "params": params, "stage": st["name"],
                                         "upd": upd, "partial": True,
                                         "needs_post_quant_conv": True}},
                               _d / "decoder_partial.pt")
                    print(f"  [{st['name']}] saved partial ckpt at upd {upd}",
                          flush=True)
                (OUT / f"{st['name']}_metrics.json").write_text(json.dumps(
                    {"stage": st["name"], "upd": upd, "holdout_l1": hl,
                     "best": best, "L_a0": L_a0, "L_prune": L_prune,
                     "recovered": round(recovered, 4), "params": params},
                    indent=2), encoding="utf-8")

                # ---- dynamic early stop: 2 consecutive <1% gains AND most of the
                # prune gap recovered ----
                if len(hist) >= 3 and st["loss"] == "l1":
                    prev2, prev1 = hist[-3][1], hist[-2][1]
                    g1 = (prev2 - prev1) / max(prev2, 1e-9)
                    g2 = (prev1 - hl) / max(prev1, 1e-9)
                    if g1 < 0.01 and g2 < 0.01 and recovered >= 0.70:
                        stop_reason = (f"converged (gains {g1*100:.2f}%/{g2*100:.2f}%, "
                                       f"recovered {recovered*100:.0f}%)")
                        print(f"  [{st['name']}] EARLY STOP at {upd}: {stop_reason}",
                              flush=True)
                        break

                # perceptual plateau rule from the 12h plan:
                # 3 consecutive checks with |dLPIPS| < 0.002 and dCSD < 0.005
                if "lpips" in he:
                    plateau.append((he["lpips"], he.get("csd")))
                    if len(plateau) >= 4:
                        w = plateau[-4:]
                        dlp = abs(w[-1][0] - w[0][0])   # NB: not `dl` (that is the DataLoader)
                        cs = [x[1] for x in w if x[1] is not None]
                        dc = abs(cs[-1] - cs[0]) if len(cs) >= 2 else 1.0
                        if dlp < 0.002 and dc < 0.005:
                            stop_reason = (f"perceptual plateau "
                                           f"(dLPIPS={dlp:.4f}, dCSD={dc:.4f})")
                            print(f"  [{st['name']}] EARLY STOP at {upd}: "
                                  f"{stop_reason}", flush=True)
                            break

        # ---- 5. save stage best ----
        if best_state is None:
            print(f"  [{st['name']}] WARNING no checkpoint passed the guard; "
                  f"saving the final weights instead", flush=True)
        else:
            dec.load_state_dict(best_state)
        dec = dec.cpu()
        d = OUT / st["name"]
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": dec.decoder.state_dict(),
                    "post_quant_conv": dec.post_quant_conv.state_dict(),
                    "meta": {**cur_meta, "widths_forward": st["widths"],
                             "params": params, "holdout_l1": round(float(L_prune), 5),
                             "best_sel": (None if best == float("inf") else round(float(best), 5)),
                             "stage": st["name"],
                             "needs_post_quant_conv": True}}, d / "decoder.pt")
        print(f"[Saved] {d}/decoder.pt  params={params/1e6:.3f} M  "
              f"holdout L1={best:.5f}", flush=True)
        results[st["name"]] = {"params": params, "holdout_l1": round(float(L_prune), 5),
                              "best_sel": (None if best == float("inf") else round(float(best), 5)),
                              "widths": st["widths"], "updates_run": upd,
                              "stop_reason": stop_reason,
                              "L_a0": round(L_a0, 5), "L_prune": round(L_prune, 5),
                              "recovered": round((L_prune - best) / gap, 4),
                              "img_per_sec": round(n_img / max(time.time() - t0, 1e-9), 2)}
        cur_dec = dec.decoder.cpu()
        cur_meta = {**cur_meta, "widths_forward": st["widths"], "params": params}
        del lpips_fn
        torch.cuda.empty_cache()
        (OUT / "train_summary.json").write_text(json.dumps(results, indent=2),
                                                encoding="utf-8")

    print("\n" + "=" * 70)
    print("PIPELINE DONE")
    for k, v in results.items():
        print(f"  {k:<10} {v['params']/1e6:>7.3f} M  holdout L1={v['holdout_l1']:.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
