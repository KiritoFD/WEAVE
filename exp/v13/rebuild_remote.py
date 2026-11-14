"""V13 remote rebuild: cache -> LD4 -> decode (run on the 3060).

Rebuilds the whole LD4 chain from the remote's own N2C.e gen_latents — no
large uploads. Steps (each idempotent, resumable by marker files):
  1. cache:  exp.v9_local.cache_n2c  -> runs/v13_remote/n2c_z.pt
  2. ld4:    v9 LD4 optimization (750, deterministic given the cache)
             -> runs/v13_remote/ld4_lat/batch_*.pt
  3. decode: S10 -> runs/v13_remote/ld4_images/*.png (A0 base images)
Run: python rebuild_remote.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "I:/Latent_Style/WEAVE")
REPO = Path("I:/Latent_Style/WEAVE")
OUT = REPO / "runs/v13_remote"


def step_cache():
    marker = OUT / "n2c_z.pt"
    if marker.exists():
        print("[1] cache exists, skip", flush=True)
        return marker
    if not marker.exists():
        from exp.v9_local.cache_n2c import main as cache_main
        import sys as s
        s.argv = ["cache_n2c",
                  "--src", str(REPO / "runs/v8/N2C.e/gen_latents"),
                  "--out", str(marker)]
        cache_main()
    # ALWAYS ensure the canonical copy exists (run_gcl reads this path)
    canon = REPO / "runs/v9_local_4070/cache"
    canon.mkdir(parents=True, exist_ok=True)
    if not (canon / "n2c_z.pt").exists():
        (canon / "n2c_z.pt").write_bytes(marker.read_bytes())
    return marker


def step_ld4():
    lat_dir = OUT / "ld4_lat"
    lat_dir.mkdir(parents=True, exist_ok=True)
    done = lat_dir.parent / "ld4_done.json"
    if done.exists() and len(json.loads(done.read_text())) == 750:
        print("[2] LD4 latents complete, skip", flush=True)
        return
    from exp.v12_gcl.run_gcl import main as gcl_main
    import sys as s
    # LD4 base = G-machinery with no Query gate and base budget (eps .10)
    s.argv = ["run_gcl", "--arm", "G0R", "--split", "all"]
    gcl_main()


def step_decode():
    device = "cuda"
    img_dir = OUT / "ld4_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    import unicodedata
    from PIL import Image
    import torch.nn as nn
    from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
    ck = torch.load(str(REPO / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"),
                    map_location="cpu", weights_only=False)
    pqc = nn.Conv2d(4, 4, 1)
    pqc.load_state_dict(ck["post_quant_conv"])
    dec = build_decoder(ck["meta"]["widths_forward"])
    dec.load_state_dict(ck["state_dict"])
    s10 = StudentDecoder(pqc, dec).to(device).eval().half()

    def decode_s10(model, z, batch=1):
        outs = []
        with torch.inference_mode():
            for i in range(0, z.shape[0], batch):
                outs.append(model((z[i:i + batch] / 0.18215).half())
                            .float().mul(0.5).add(0.5).clamp(0, 1).cpu())
        return torch.cat(outs)
    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    batches = sorted((REPO / "runs/v12_gcl/all/G0R").glob("batch_*.pt"))
    seen = {}
    for p in batches:
        d = torch.load(p, map_location="cpu", weights_only=False)
        for i, tid in enumerate(d["ids"]):
            seen[unicodedata.normalize("NFC", tid.replace("/", "__"))] = \
                d["z_final"][i:i + 1]
    assert len(seen) == 750, f"expected 750 unique tasks, got {len(seen)}"
    t0 = time.time()
    for n, (key, zf) in enumerate(sorted(seen.items())):
        png = img_dir / f"{n:04d}_{hashlib.sha256(key.encode()).hexdigest()[:10]}.png"
        if png.exists():
            continue
        rgb = decode_s10(s10, zf.to("cuda"), batch=1)[0]
        arr = (rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        Image.fromarray(arr).save(png)
        if (n + 1) % 100 == 0:
            print(f"  decode {n+1}/750 {time.time()-t0:.0f}s", flush=True)
    print(f"[3] decoded {len(seen)} images", flush=True)


import hashlib  # noqa: E402
import time  # noqa: E402


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    step_cache()
    step_ld4()
    step_decode()
    print("REBUILD_COMPLETE")
