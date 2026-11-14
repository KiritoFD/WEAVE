"""V13 fixed training: per-band sigma counters (#3), REAL optimizer-update
counter (#4), OneCycleLR over optimizer updates (#5), full resume (#7).

The old model keeps its original (÷√2) sigmas; new training uses the corrected
estimates (×√2 relative to the V11b values).

Run: python -m exp.v13.train_fixed --updates 4000 --out runs/v13/flow_v2_w56
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp.v11b.bands import dwt2, sobel_mag
from exp.v11b.dataset_real import (RamWaveletDataset, RealWaveletDataset,
                                   collate_batch, make_batch)
from exp.v11b.model import WaveletFlow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "v13"


def estimate_sigmas_fixed(n_images: int = 400, device="cuda") -> dict:
    """Per-style PER-BAND counters (fixes the V11b double-count: the old
    estimator added m.sum() to a shared 'n' once per band -> sigma /= sqrt(2))."""
    ds = RealWaveletDataset("train", crop=128)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    sums = {s: {"H1": 0., "H2": 0., "n_H1": 0, "n_H2": 0} for s in range(5)}
    items = [ds[i] for i in idx]
    for c0 in range(0, len(items), 16):
        batch = items[c0:c0 + 16]
        crops = torch.stack([b[0] for b in batch])
        sids = torch.tensor([b[1] for b in batch])
        b = make_batch(crops, sids, device)
        for sid in range(5):
            m = sids == sid
            if not m.any():
                continue
            for band in ("H1", "H2"):
                v = b[band][m].float()
                key = f"n_{band}"
                sums[sid][band] += v.pow(2).mean().item() * m.sum().item()
                sums[sid][key] += m.sum().item()
    out = {}
    for s in range(5):
        out[str(s)] = {
            "H1": math.sqrt(sums[s]["H1"] / max(sums[s]["n_H1"], 1)),
            "H2": math.sqrt(sums[s]["H2"] / max(sums[s]["n_H2"], 1)),
        }
    return out


def fm_step_full(model, batch, sigmas, device, p_drop=0.15):
    total = 0.0
    parts = {}
    for which, band, ll_key, edge_key in (
            ("H1", "H1", "LL1", "edge1"), ("H2", "H2", "LL2", "edge2")):
        h = batch[band].float()
        B = h.shape[0]
        sid = batch["style_id"]
        sig = torch.tensor(
            [sigmas[str(int(s))][band] for s in sid],
            device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        eps = torch.randn_like(h) * sig
        tau = torch.rand(B, device=device)
        ta = tau.view(-1, 1, 1, 1)
        h_tau = (1 - ta) * eps + ta * h
        v_star = h - eps
        drop = torch.rand(B, device=device) < p_drop
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(device) == "cuda"):
            v_cond = model.forward_band(
                which, h_tau, batch[ll_key], batch[edge_key], sid, tau,
                unconditional=False).float()
        if drop.any():
            idx = drop.nonzero(as_tuple=True)[0]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=str(device) == "cuda"):
                v_unc = model.forward_band(
                    which, h_tau[idx], batch[ll_key][idx],
                    batch[edge_key][idx], sid[idx], tau[idx],
                    unconditional=True).float()
            v_cond[idx] = v_unc
        loss = F.mse_loss(v_cond, v_star)
        parts[band] = loss.item()
        total = total + loss
    return total / 2, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=4000,
                    help="REAL optimizer updates (plan §7)")
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)   # eff batch 32
    ap.add_argument("--width", type=int, default=56)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=OUT / "flow_v2_w56")
    ap.add_argument("--sigma", type=Path, default=None)
    ap.add_argument("--g1-every", type=int, default=500,
                    help="every N optimizer updates")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.sigma and Path(args.sigma).is_file():
        sigmas = json.loads(Path(args.sigma).read_text())
        if "sigmas" in sigmas:
            sigmas = sigmas["sigmas"]
    else:
        print("[v13] estimating FIXED per-style band sigmas (400 imgs)...",
              flush=True)
        sigmas = estimate_sigmas_fixed(device=device)
        (args.out / "sigmas_fixed.json").write_text(json.dumps(sigmas, indent=1))
        print("sigmas:", json.dumps(sigmas)[:300], flush=True)

    model = WaveletFlow(width=args.width).to(device)
    print(f"[v13] params={model.n_params()/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # OneCycle over REAL optimizer updates (#5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.updates, pct_start=0.05)

    opt_step = 0
    micro_in_accum = 0
    if args.resume and Path(args.resume).is_file():
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        opt_step = st["opt_step"]
        if st.get("rng"):
            torch.set_rng_state(st["rng"].cpu())
        print(f"[v13] resumed at opt_step {opt_step}", flush=True)

    ds = RamWaveletDataset("train", crop=args.crop)
    print(f"[v13] train paintings={len(ds)} (RAM-resident)", flush=True)
    loader = DataLoader(ds, batch_size=args.micro, shuffle=True,
                        num_workers=0, collate_fn=collate_batch,
                        drop_last=True, pin_memory=True,
                        persistent_workers=False)

    curve = args.out / "train_curves.jsonl"
    state_path = args.out / "state.json"
    t0 = time.time()
    done = opt_step >= args.updates
    ckpt_marks = {0.25: "25", 0.5: "50", 0.75: "75", 1.0: "100"}
    while not done:
        for crops, sids in loader:
            crops, sids = crops.to(device), sids.to(device)
            bb = make_batch(crops, sids, device)
            loss, parts = fm_step_full(model, bb, sigmas, device,
                                       p_drop=args.p_drop)
            (loss / args.accum).backward()
            micro_in_accum += 1
            if micro_in_accum % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                rec = {"opt_step": opt_step, "loss": loss.item(),
                       "H1": parts.get("H1"), "H2": parts.get("H2"),
                       "sec": round(time.time() - t0, 1)}
                with curve.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                if opt_step % 100 == 0:
                    print(f"[v13] opt_step {opt_step}/{args.updates} "
                          f"loss={loss.item():.4f} "
                          f"{(time.time()-t0)/max(opt_step,1):.2f}s/opt",
                          flush=True)
                if opt_step % args.g1_every == 0:
                    from exp.v11b.train_flow import g1_eval
                    g1 = g1_eval(model, sigmas, device)
                    with (args.out / "g1.jsonl").open("a") as fp:
                        fp.write(json.dumps({"opt_step": opt_step, **g1}) + "\n")
                    print(f"[g1@{opt_step}] {json.dumps(g1)}", flush=True)
                frac = opt_step / args.updates
                for mark, tag in ckpt_marks.items():
                    if frac >= mark and not (args.out / f"ckpt_{tag}.pt").exists():
                        torch.save({"model": model.state_dict(),
                                    "opt": opt.state_dict(),
                                    "sched": sched.state_dict(),
                                    "opt_step": opt_step,
                                    "rng": torch.get_rng_state(),
                                    "args": vars(args),
                                    "sigmas": sigmas,
                                    "sigma_fix": "per-band counters (V13)"},
                                   args.out / f"ckpt_{tag}.pt")
                        print(f"[ckpt] {tag}% saved", flush=True)
                state_path.write_text(json.dumps(
                    {"opt_step": opt_step, "updates": args.updates,
                     "elapsed_s": time.time() - t0}))
            if opt_step >= args.updates:
                done = True
                break
    print(f"[v13] done opt_step={opt_step} elapsed={time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
