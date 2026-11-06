"""Check the smoke checkpoint embeds the pgt config and that inference.json does not override it."""
from __future__ import annotations

import json

import torch

CKPT = "/mnt/i/Latent_Style/WEAVE/runs/pgt/_smoke/epoch_0001.pt"
INF = "/mnt/i/Latent_Style/WEAVE/inference.json"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("ckpt keys:", sorted(ck.keys()))
cfg = ck.get("config", {})
m = cfg.get("model", {})
b = cfg.get("bridge", {})
d = cfg.get("data", {})
t = cfg.get("training", {})
print("model.target_latent_hf_subband_fusion_enabled =", m.get("target_latent_hf_subband_fusion_enabled"))
print("model.target_latent_hf_head_fusion_gate_init =", m.get("target_latent_hf_head_fusion_gate_init"))
print("bridge.paired_latent_passthrough =", b.get("paired_latent_passthrough"))
print("data.paired_latent_cache_dir =", d.get("paired_latent_cache_dir"))
print("training.batch_size =", t.get("batch_size"), "cap =", t.get("stop_after_global_steps"))

inf = json.load(open(INF))
print("inference.json sections:", sorted(inf.keys()))
for sec in ("model", "bridge", "data"):
    if sec in inf:
        hit = {k: v for k, v in inf[sec].items()
               if "subband" in k or "passthrough" in k or "paired_latent" in k or "gate" in k}
        if hit:
            print(f"INF OVERRIDE {sec}: {hit}")
