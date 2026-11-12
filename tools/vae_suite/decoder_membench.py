"""Independent-process memory benchmark for the decoder pruning ladder.

Each (model, batch, mode) runs in its OWN subprocess so that no compiled graph,
inductor cache or CUDA-graph pool leaks between measurements.

Reports BOTH torch.cuda.max_memory_allocated (true tensor bytes) and
max_memory_reserved (allocator pool incl. CUDA Graph / workspace), plus the
real img/s of a full training update.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import torch

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"

MODELS = {
    "A0": None,                                        # 49.490M, [128,256,512,512]
    "p1_28m": [128, 224, 352, 384],                    # 27.799M
    "p2_18m": [96, 192, 256, 320],                     # 18.024M
    "p3_10m": [96, 160, 224, 224],                     # 10.667M
}


def run_one(model, bs, mode, warmup=3, iters=10):
    """Executed in a child process."""
    from diffusers import AutoencoderKL
    from tools.vae_suite.decoder_prune import build_decoder

    torch.set_num_threads(4)
    dev = "cuda"
    widths = MODELS[model]
    if widths is None:
        vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev)
        dec = vae.decoder
    else:
        dec = build_decoder(widths)
        ckf = OUT / model / "decoder.pt"
        if ckf.exists():
            ck = torch.load(ckf, map_location="cpu", weights_only=False)
            dec.load_state_dict(ck["state_dict"])
        dec = dec.to(dev)
    dec.train()
    params = sum(p.numel() for p in dec.parameters())

    if mode == "compile":
        dec.compile(dynamic=False)

    opt = torch.optim.AdamW(dec.parameters(), lr=1e-6)
    z = torch.randn(bs, 4, 64, 64, device=dev)
    x = torch.randn(bs, 3, 512, 512, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = dec(z)
            loss = (y.float() - x).abs().mean()
        loss.backward()
        opt.step()

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        import time
        t0 = time.time()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters

        res = {
            "model": model, "widths": widths, "params_M": round(params / 1e6, 3),
            "bs": bs, "mode": mode, "ok": True,
            "allocated_GB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "sec_per_update": round(dt, 3),
            "img_per_sec": round(bs / dt, 2),
        }
    except torch.cuda.OutOfMemoryError:
        res = {"model": model, "widths": widths, "params_M": round(params / 1e6, 3),
               "bs": bs, "mode": mode, "ok": False, "oom": True}
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default="")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    if args.child:
        m, bs, mode = args.child.split(":")
        run_one(m, int(bs), mode, iters=args.iters)
        return

    grid = []
    for m in ["A0", "p1_28m", "p2_18m", "p3_10m"]:
        for bs in [2, 4, 6, 8, 12]:
            grid.append((m, bs, "eager"))
    # one compile comparison at the largest fitting batch for the 10M model
    grid += [("p3_10m", 8, "compile"), ("p3_10m", 12, "compile"),
             ("p1_28m", 4, "compile")]

    results = []
    for m, bs, mode in grid:
        cmd = [sys.executable, __file__, "--child", f"{m}:{bs}:{mode}",
               "--iters", str(args.iters)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            r = {"model": m, "bs": bs, "mode": mode, "ok": False,
                 "error": (p.stderr or "")[-300:]}
            print(f"  {m:<8} bs={bs:<3} {mode:<8} ERROR", flush=True)
        else:
            r = json.loads(line[0][7:])
            if r.get("ok"):
                print(f"  {r['model']:<8} bs={r['bs']:<3} {r['mode']:<8} "
                      f"alloc={r['allocated_GB']:>6.2f}G res={r['reserved_GB']:>6.2f}G "
                      f"{r['img_per_sec']:>6.2f} img/s", flush=True)
            else:
                print(f"  {r['model']:<8} bs={bs:<3} {mode:<8} OOM", flush=True)
        results.append(r)

    (OUT / "memory_bench.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUT/'memory_bench.json'}")


if __name__ == "__main__":
    main()
