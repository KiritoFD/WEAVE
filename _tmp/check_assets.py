import torch
from pathlib import Path

def main():
    p = Path("/mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt")
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=False)
        print("csd_proto.pt keys:", type(d), list(d.keys()) if isinstance(d, dict) else d.shape)
        if isinstance(d, dict):
            for k, v in d.items():
                if torch.is_tensor(v):
                    print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
                else:
                    print(f"  {k}: {type(v)}")

    lat_dir = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
    if lat_dir.exists():
        files = list(lat_dir.glob("*.pt"))
        print(f"orig_mean_fp16 files ({len(files)}): {[f.name for f in files]}")
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            if isinstance(d, dict):
                lat = d.get("latents")
                print(f"  {f.name}: latents shape {lat.shape if lat is not None else None}, num ids {len(d.get('ids', []))}")

if __name__ == "__main__":
    main()
