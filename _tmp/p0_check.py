"""P0 check: inspect dataset, packed latents, CSD model, VAE decoder, and memory."""
import os
import json
import torch
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"VRAM Free: {free_b / (1024**2):.1f} MiB / Total: {total_b / (1024**2):.1f} MiB")

    # Check packed latents
    packed_dir = Path("/mnt/i/Latent_Style/WEAVE/data/train/.latent_cache/packed/packed")
    if packed_dir.exists():
        for p in sorted(packed_dir.glob("*.pt")):
            try:
                data = torch.load(p, map_location="cpu")
                if isinstance(data, dict):
                    print(f"Packed {p.name}: dict with keys {list(data.keys())[:5]}")
                    for k in list(data.keys())[:3]:
                        v = data[k]
                        if torch.is_tensor(v):
                            print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
                elif torch.is_tensor(data):
                    print(f"Packed {p.name}: tensor shape {data.shape}, dtype {data.dtype}")
            except Exception as e:
                print(f"Error loading {p}: {e}")

    # Check test latents / test assets
    test_dirs = [
        "/mnt/i/Latent_Style/eval_assets/p2_render",
        "/mnt/i/Latent_Style/WEAVE/data/test",
        "/mnt/i/Latent_Style/WEAVE/runs/12h/anchors/base_ep10_750_512",
    ]
    for td in test_dirs:
        p = Path(td)
        print(f"Dir {p}: exists={p.exists()}")
        if p.exists():
            files = list(p.glob("*"))
            print(f"  contains {len(files)} items: {[f.name for f in files[:5]]}")

    # Check CSD model availability
    csd_path = Path("/mnt/i/Latent_Style/weights")
    print(f"Weights dir {csd_path}: exists={csd_path.exists()}")
    if csd_path.exists():
        print(f"  items: {[f.name for f in csd_path.glob('*')]}")

if __name__ == "__main__":
    main()
