"""Quick CPU benchmark: VAE encoder throughput at various thread/batch/precision settings.

Runs entirely on CPU + RAM; does not touch the GPU (training keeps running).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
from diffusers import AutoencoderKL

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")


def main():
    man = json.loads(MANIFEST.read_text())
    mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    ev = torch.load("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/eval_pairs.pt",
                    weights_only=False)
    raws = ev["raw"][:16]
    x = torch.stack([
        torch.from_numpy(np.ascontiguousarray(mmap[r])).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        for r in raws
    ])
    print(f"[Input] {tuple(x.shape)} in RAM", flush=True)

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32)
    vae.eval()
    # strip decoder to save RAM (encoder-only eval is what AUROC needs)
    del vae.decoder
    del vae.post_quant_conv

    results = []
    for nthreads, bs, dtype in [(16, 4, "fp32"), (32, 4, "fp32"), (64, 4, "fp32"),
                                (32, 8, "fp32"), (32, 2, "fp32"),
                                (32, 4, "bf16"), (64, 8, "bf16")]:
        torch.set_num_threads(nthreads)
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                if dtype == "bf16":
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        for i in range(0, len(x), bs):
                            h = vae.encoder(x[i:i + bs])
                            h = vae.quant_conv(h)
                            _ = h[:, :4].float()
                else:
                    for i in range(0, len(x), bs):
                        h = vae.encoder(x[i:i + bs])
                        h = vae.quant_conv(h)
                        _ = h[:, :4]
            dt = time.perf_counter() - t0
            ips = len(x) / dt
            r = {"threads": nthreads, "batch": bs, "dtype": dtype,
                 "img_per_sec": round(ips, 2),
                 "sec_for_2400": round(2400 / ips, 1)}
            results.append(r)
            print(f"  threads={nthreads:>3} bs={bs} {dtype}: {ips:.2f} img/s "
                  f"-> 2400 imgs in {2400/ips/60:.1f} min", flush=True)
        except Exception as e:
            print(f"  threads={nthreads} bs={bs} {dtype}: ERROR {e}", flush=True)

    Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/cpu_bench.json").write_text(
        json.dumps(results, indent=2))
    print("[Saved] cpu_bench.json", flush=True)


if __name__ == "__main__":
    main()
