"""Eval for Style-Bridge post-trained arms (P1-P4).

Same protocol as eval_4070lp.py (partial-noise SDEdit, paired eps, S10 decode,
measure()-compatible payloads). For adapter arms (p3/p4) the content condition
is canon(z_src) with the frozen c1_gentle canonicalizer — matching training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/v15"
RAW = RUNS / "eval" / "raw"


def arm_path(name):
    for cand in (RUNS / name, RUNS / "scaling" / name):
        if (cand / "final.pt").exists():
            return cand
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--levels", default="0.4,0.7")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--style-shift", type=int, default=0)
    ap.add_argument("--uncond", action="store_true",
                    help="unconditional Style-ID control (all steps)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    levels = [float(x) for x in a.levels.split(",")]
    K = a.steps

    from src.utils import decode_s10, load_s10_decoder
    from src.model import canonicalizer
    from .model import V15UNet
    from .content_adapter import V15AdapterUNet

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    cache = torch.load(REPO / "runs/v9_local_4070/cache/n2c_z.pt",
                       map_location="cpu", weights_only=False)
    z_src_all = cache["z_src"]
    assert len(tasks) == len(z_src_all)

    s10 = load_s10_decoder(dev)
    canon = canonicalizer(dev)
    eps_by_level = {t0: torch.randn(len(tasks), 4, 64, 64,
                                    generator=torch.Generator().manual_seed(
                                        20261115 + int(t0 * 100)))
                    for t0 in levels}

    for name in a.arms.split(","):
        adir = arm_path(name.strip())
        ck = torch.load(adir / "final.pt", map_location="cpu",
                        weights_only=False)
        cfg = ck["config"]
        is_adapter = str(cfg.get("posttrain_v2", "")) in ("p3", "p4") or \
            any(k.startswith("adapter.") for k in ck["model"])
        base = V15UNet(base_dim=int(cfg["base_dim"]),
                       use_attention=bool(cfg["use_attention"]),
                       style_mode=str(cfg.get("style_mode", "adaln")),
                       n_experts=int(cfg.get("n_experts", 4))).to(dev)
        if is_adapter:
            model = V15AdapterUNet(base).to(dev)
            model.load_state_dict(ck["model"])
        else:
            model = base
            model.load_state_dict(ck["model"])
        model.eval()
        tag = f"{adir.name}_v2cfg{a.cfg:g}" + \
            (f"_shift{a.style_shift}" if a.style_shift else "") + \
            ("_uncond" if a.uncond else "")
        t0w = time.time()
        with torch.inference_mode():
            for lev in levels:
                ldir = RAW / tag / f"t0_{lev}"
                ldir.mkdir(parents=True, exist_ok=True)
                meta_batches = []
                eps = eps_by_level[lev]
                for bi in range(0, len(tasks), a.batch):
                    idx = list(range(bi, min(bi + a.batch, len(tasks))))
                    group = [tasks[i] for i in idx]
                    z_src = z_src_all[idx]
                    z = ((1 - lev) * eps[idx] + lev * z_src).to(dev)
                    sid = torch.tensor(
                        [(t["target_style_id"] + a.style_shift) % 5
                         for t in group], device=dev)
                    content = canon(z_src.float().to(dev),
                                    mode="c1_gentle", augment=False) \
                        if is_adapter else None
                    span = (1 - lev) / K
                    for j in range(K):
                        t = lev + span * j
                        tt = torch.full((len(idx),), t, device=dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=dev == "cuda"):
                            kw = {"content": content} if is_adapter else {}
                            if a.cfg == 1.0 and not a.uncond:
                                v = model(z, sid, tt,
                                          unconditional=False, **kw).float()
                            elif a.uncond:
                                v = model(z, sid, tt, unconditional=True,
                                          **kw).float()
                            else:
                                v_c = model(z, sid, tt,
                                            unconditional=False, **kw).float()
                                v_u = model(z, sid, tt, unconditional=True,
                                            **kw).float()
                                v = v_u + a.cfg * (v_c - v_u)
                        z = z + v * span
                    rgb = decode_s10(s10, z, batch=a.decode_batch)
                    u8 = rgb.mul(255).round().clamp_(0, 255).byte().cpu()
                    payload = {"ids": [t["task_id"] for t in group],
                               "rgb_u8": u8,
                               "z_final": z.detach().half().cpu()}
                    name_f = f"batch_{bi:05d}.pt"
                    tmp = ldir / (name_f + ".tmp")
                    torch.save(payload, tmp)
                    tmp.replace(ldir / name_f)
                    meta_batches.append({"offset": bi, "ids": payload["ids"],
                                         "file": name_f})
                    print(f"  {tag} L{lev} {bi+len(idx)}/{len(tasks)} "
                          f"{time.time()-t0w:.0f}s", flush=True)
                (ldir / "meta.json").write_text(json.dumps(
                    {"kind": "v15_probe_v2", "arm": tag, "level": lev,
                     "cfg": a.cfg, "style_shift": a.style_shift,
                     "k_steps": K, "n_tasks": len(tasks),
                     "adapter": is_adapter,
                     "batches": meta_batches}, indent=1))
        print(f"{tag} DONE {time.time()-t0w:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("EVAL_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
