import sys

import torch

from config_schema import compact_runtime_config, load_experiment_config

step = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
src_path = f"runs/v2/V1/step_{step:06d}.pt"
src = torch.load(src_path, map_location="cpu", weights_only=False)
state = src.get("model_state_dict", src["model"])
payload = {
    "epoch": int(step),
    "global_step": int(step),
    "model_state_dict": state,
    "config": compact_runtime_config(load_experiment_config("configs/v2/V1.json")),
}
out_path = f"runs/v2/V1/step_{step:06d}_eval.pt"
torch.save(payload, out_path)
print("exported", out_path, "keys=", len(state))
