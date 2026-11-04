import sys

import torch

from config_schema import compact_runtime_config, load_experiment_config

step = int(sys.argv[1])
src_path = f"runs/v2/V1/step_{step:06d}.pt"
out_path = f"runs/v2/V1/step_{step:06d}_bare.pt"
src = torch.load(src_path, map_location="cpu", weights_only=False)
raw = src["model"]
prefix = "model."
if all(k.startswith(prefix) for k in raw):
    bare = {k[len(prefix):]: v for k, v in raw.items()}
    stripped = len(bare)
else:
    bare = dict(raw)
    stripped = 0
config = compact_runtime_config(load_experiment_config("configs/v2/V1.json"))
payload = {
    "schema_version": "v2-bare-keys-v1",
    "variant": str(src.get("variant", "V1")),
    "step": int(src.get("step", step)),
    "model": bare,
    "model_state_dict": bare,
    "config": config,
}
torch.save(payload, out_path)
keys = list(bare.keys())
print(f"{out_path}: keys={len(keys)} stripped_prefix={stripped}")
print("first3=", keys[:3])
