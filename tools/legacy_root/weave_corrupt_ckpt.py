import torch

src = "runs/v2/V1/step_002500_bare.pt"
dst = "runs/v2/V1/_strict_check_badkey.pt"
payload = torch.load(src, map_location="cpu", weights_only=False)
state = payload["model_state_dict"]
first = next(iter(state))
state["bad_" + first] = state.pop(first)
torch.save(payload, dst)
print("corrupted key:", first, "->", "bad_" + first)
