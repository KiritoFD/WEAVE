import torch
p = "/mnt/i/Latent_Style/WEAVE/data/train_d5/.latent_cache/packed/00_Early_Renaissance.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
print("type", type(d))
print("keys", list(d.keys()) if isinstance(d, dict) else None)
for k, v in d.items():
    if hasattr(v, "shape"):
        print("  ", k, tuple(v.shape), v.dtype)
    elif isinstance(v, list):
        print("  ", k, "len=", len(v), "first=", v[0] if v else None)
    else:
        print("  ", k, "=", v)
