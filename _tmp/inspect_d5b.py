import torch
p = "/mnt/i/Latent_Style/WEAVE/data/train_d5/.latent_cache/packed/00_Early_Renaissance.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
t = d["latents"]
print("shape", tuple(t.shape), "dtype", t.dtype)
print("nbytes", t.untyped_storage().nbytes(), "elem_expected", t.numel() * t.element_size())
print("ratio", t.untyped_storage().nbytes() / (t.numel() * t.element_size()))
