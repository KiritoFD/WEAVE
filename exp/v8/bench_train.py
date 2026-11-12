"""Find the microbatch that fills ~10 GB, for each NEW training arm.

The 24h plan fixes an EFFECTIVE batch (A: 240) and lets microbatch + gradient
accumulation vary with VRAM. The earlier probe stopped at batch 16 (448 MB) and
therefore answered nothing. This sweeps upward until VRAM approaches the 10 GB
target, then reports the accumulation factor that keeps the registered
effective batch.

Targets (from the plan):
  A3/A4 control : effective 240
  L2/L3/L4 SPD  : effective 128 (K-step R1 unroll is the expensive one)
  C3 critic     : effective 240

Grouped by memory profile: arms with the same footprint pick ONE microbatch, so
the matrix stays comparable instead of each arm inventing its own batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
Q2A = REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
TARGET_MB = 10 * 1024


def _load(device):
    from src.infer import canonicalizer, load_c3unet, load_critic
    base, _ = load_c3unet(Q2A, device)
    canon = canonicalizer(device)
    critic = load_critic(device)
    return base, canon, critic


def _margin(critic, z, tid):
    import torch.nn.functional as F
    logits = critic(z.float()).float()
    mask = F.one_hot(tid.long(), 5).bool()
    others = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(others, 1)


def _warm(device):
    """One throwaway step so CUDA/cuDNN compilation is not charged to the timer."""
    try:
        _step_control(2, device)
    except Exception:
        pass


def _timed(fn, batch, device, repeats=3, **kw):
    torch.cuda.empty_cache()
    fn(batch, device, **kw)                # compile / warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    import time
    t0 = time.time()
    for _ in range(repeats):
        fn(batch, device, **kw)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / repeats
    return dt, torch.cuda.max_memory_allocated() / 2**20


def _step_control(batch, device, local=False):
    from exp.v8.controls import ControlSpec, Controls, wrap
    base, canon, critic = _load(device)
    spec = ControlSpec(local_rank=2 if local else 0)
    gen = wrap(base, spec, device)
    head = nn.Sequential(nn.Linear(spec.dim, 256), nn.SiLU(),
                         nn.Linear(256, spec.dim)).to(device)
    nn.init.normal_(head[-1].weight, std=0.01); nn.init.zeros_(head[-1].bias)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4)
    z = torch.randn(batch, 4, 64, 64, device=device)
    tid = torch.randint(0, 5, (batch,), device=device)
    with torch.no_grad():
        c = canon(z, mode="c1_gentle", augment=False)
    u = head(torch.randn(batch, spec.dim, device=device))
    out = gen(c, 0.0, c, tid, controls=Controls.from_tensor(u, spec))
    loss = nn.functional.softplus(-_margin(critic, out, tid)).mean()
    opt.zero_grad(); loss.backward(); opt.step()


def _step_spd(batch, device, steps_k=2, rank=2):
    from exp.v8.guidance import LatentSPDEditor, SPDHead, UnrollConfig
    base, canon, critic = _load(device)
    head = SPDHead(rank=rank).to(device)
    ed = LatentSPDEditor(critic, head, UnrollConfig(steps=steps_k, step_rms=0.08,
                                                   rho=0.25)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4)
    z = torch.randn(batch, 4, 64, 64, device=device)
    tid = torch.randint(0, 5, (batch,), device=device)
    with torch.no_grad():
        c = canon(z, mode="c1_gentle", augment=False)
        z0 = base(c, 0.0, c, tid).detach()
    res = ed(z0, c, tid)
    loss = nn.functional.softplus(-_margin(critic, res["final"], tid)).mean()
    opt.zero_grad(); loss.backward(); opt.step()


def _step_critic(batch, device):
    from exp.v8.energy import robustness_finetune_step
    _, _, critic = _load(device)
    for p in critic.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(critic.parameters(), lr=5e-5)
    z = torch.randn(batch, 4, 64, 64, device=device)
    tid = torch.randint(0, 5, (batch,), device=device)
    robustness_finetune_step(critic, z, tid, opt)


ARMS = {
    # Round 2: the first sweep peaked at 3.5-5.1 GB, far below the 10 GB target,
    # so probe the extrapolated configs only. One probe each; the group's
    # microbatch is then fixed and every arm in that group inherits it.
    "A3/A4": (_step_control, {"effective": 240}, [384]),
    "L2/L3": (_step_spd, {"steps_k": 2, "rank": 2, "effective": 128}, [288]),
    "L3K4": (_step_spd, {"steps_k": 4, "rank": 2, "effective": 128}, [288]),
    "C3": (_step_critic, {"effective": 240}, [512]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-mb", type=int, default=TARGET_MB)
    a = ap.parse_args()
    results, picks = [], {}
    for name, (fn, kw, batches) in ARMS.items():
        kw = dict(kw)
        eff = kw.pop("effective")
        chosen = None
        for b in batches:                      # at most three probes per group
            try:
                dt, vram = _timed(fn, b, a.device, **kw)
            except torch.OutOfMemoryError:
                print(f"  {name}: OOM at {b}", flush=True)
                torch.cuda.empty_cache()
                continue
            accum = max(1, round(eff / b))
            rec = {"arm": name, "microbatch": b, "sec_per_update": round(dt, 4),
                   "peak_vram_mb": round(vram, 1), "target_effective": eff,
                   "accum_steps": accum, "achieved_effective": accum * b}
            results.append(rec)
            print("  " + json.dumps(rec), flush=True)
            # prefer the largest microbatch that stays under the VRAM target
            if vram <= a.target_mb and (chosen is None or b > chosen["microbatch"]):
                chosen = rec
        if chosen is None:
            cand = [r for r in results if r["arm"] == name]
            chosen = min(cand, key=lambda r: r["peak_vram_mb"]) if cand else None
        picks[name] = chosen
        if chosen:
            print(f"=> {name}: microbatch={chosen['microbatch']} "
                  f"accum={chosen['accum_steps']} achieved_eff={chosen['achieved_effective']} "
                  f"(target {eff}) vram={chosen['peak_vram_mb']}MB "
                  f"sec/upd={chosen['sec_per_update']}", flush=True)
    out = REPO / "runs/v8"; out.mkdir(parents=True, exist_ok=True)
    (out / "train_bench.json").write_text(
        json.dumps({"probes": results, "picks": picks}, indent=2), encoding="utf-8")
    print("wrote runs/v8/train_bench.json", flush=True)


if __name__ == "__main__":
    main()
