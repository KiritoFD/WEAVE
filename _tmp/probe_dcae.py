"""Probe whether the public DCAE checkpoint is obtainable on this machine."""
import json, os, sys
from pathlib import Path

out = {"hf_cache_dirs": [], "candidates": [], "download_test": None}

# 1. Look for any locally cached DCAE-like weights
roots = [
    Path("/mnt/i/Latent_Style/eval_cache/hf/hub"),
    Path("/mnt/i/Latent_Style/hf_cache"),
    Path(os.path.expanduser("~/.cache/huggingface/hub")),
    Path("/mnt/i/Latent_Style/WEAVE/weights"),
    Path("/mnt/i/Latent_Style/weights"),
]
for r in roots:
    if r.exists():
        subs = [p.name for p in r.iterdir() if p.is_dir()]
        out["hf_cache_dirs"].append({"root": str(r), "n": len(subs), "dc_like": [s for s in subs if "dc-ae" in s.lower() or "dcae" in s.lower()]})

# 2. Try loading from cache only
try:
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers", local_files_only=True)
    out["candidates"].append({"model": "mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers", "local": "OK",
                              "scaling_factor": float(ae.config.scaling_factor)})
except Exception as e:
    out["candidates"].append({"model": "mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers", "local": f"{type(e).__name__}: {str(e)[:200]}"})

# 3. Network probe (small HEAD request)
try:
    import urllib.request
    req = urllib.request.Request(
        "https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers/resolve/main/config.json",
        method="HEAD",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        out["download_test"] = f"HTTP {r.status}"
except Exception as e:
    out["download_test"] = f"{type(e).__name__}: {str(e)[:200]}"

print(json.dumps(out, indent=2))
