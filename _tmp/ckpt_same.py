import torch

a = torch.load("/mnt/i/Latent_Style/WEAVE/runs/v4/S1_A_P5/epoch_0008.pt", map_location="cpu", weights_only=False)
b = torch.load("/mnt/i/Latent_Style/WEAVE/runs/v4/S1_A_P5/step_000600.pt", map_location="cpu", weights_only=False)


def sd(x):
    for k in ("state_dict", "model_state_dict", "model", "params"):
        if isinstance(x, dict) and k in x and isinstance(x[k], dict):
            return x[k]
    return x


sa, sb = sd(a), sd(b)
print("keys_a", len(sa), "keys_b", len(sb))
common = [k for k in sa if k in sb and hasattr(sa[k], "shape")]
print("common", len(common))
maxdiff = 0.0
worst = ""
n_same = 0
for k in common:
    if sa[k].shape != sb[k].shape:
        print("shape mismatch", k, sa[k].shape, sb[k].shape)
        continue
    d = (sa[k].double() - sb[k].double()).abs().max().item()
    if d == 0.0:
        n_same += 1
    if d > maxdiff:
        maxdiff, worst = d, k
print("identical_tensors", n_same, "/", len(common), "max_abs_diff", maxdiff, "worst", worst)
print("epoch_field", a.get("epoch") if isinstance(a, dict) else None)
print("global_step_a", a.get("global_step") if isinstance(a, dict) else None,
      "global_step_b", b.get("global_step") if isinstance(b, dict) else None)
