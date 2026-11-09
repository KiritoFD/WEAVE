"""Download the public DCAE checkpoint on the Windows host (WSL has no network).

Writes to I:\\Latent_Style\\weights\\dc-ae-f32c32-mix-1.0-diffusers so that WSL can
read it at /mnt/i/Latent_Style/weights/dc-ae-f32c32-mix-1.0-diffusers.
"""
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers"
DEST = Path(r"I:\Latent_Style\weights\dc-ae-f32c32-mix-1.0-diffusers")

# Only the model itself; skip demo gifs/images.
ALLOW = ["config.json", "diffusion_pytorch_model.safetensors", "LICENSE.txt"]

def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading {REPO_ID} -> {DEST}", flush=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(DEST),
        allow_patterns=ALLOW,
        max_workers=4,
    )
    dt = time.time() - t0
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"done in {dt:.1f}s, {len(files)} files, {total/1048576:.1f} MB", flush=True)
    for p in files:
        print(f"  {p.relative_to(DEST)}  {p.stat().st_size/1048576:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
