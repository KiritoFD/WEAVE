#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import sys, torch, collections
sys.path.insert(0, ".")
from tools.v6.models_v6 import C3UNet

m = C3UNet(base_dim=48, num_styles=5, mode="direct")
tot = sum(p.numel() for p in m.parameters())
print(f"total params (base_dim=48, direct) = {tot:,}\n")

groups = collections.Counter()
for name, p in m.named_parameters():
    n = p.numel()
    if name.startswith("stem64"):
        k = "stem64 (12ch -> c1)"
    elif ".adaln." in name:
        k = "AdaLN linears (style+tau -> 5x channels)"
    elif ".conv1." in name or ".conv2." in name:
        k = "block conv3x3"
    elif ".content_proj." in name:
        k = "content injection 1x1"
    elif name.startswith("style_") or name.startswith("time_mlp"):
        k = "style_mlp + time_mlp"
    elif name.startswith("palette_head"):
        k = "StyleChannelPaletteHead"
    elif name.startswith("down") or name.startswith("up") or "proj" in name:
        k = "down/up/proj convs"
    elif name.startswith("out_"):
        k = "output head"
    else:
        k = "other"
    groups[k] += n

for k, v in groups.most_common():
    print(f"  {k:44s} {v:9,d}  ({v/tot*100:5.1f}%)")

print("\n--- provably redundant in DIRECT mode ---")
# 1) z_in == c, so [z_in, c, z_in-c] = [c, c, 0]: 4 of 12 input channels are zero
#    and the other 8 are duplicates.
stem = m.stem64
print(f"  stem64: {stem.weight.numel():,} params (12ch). A 4ch stem is exactly "
      f"equivalent: {4*stem.out_channels*9 + stem.out_channels:,} params")
# 2) tau is always 0.0 in direct mode -> time_mlp emits a constant
tm = sum(p.numel() for p in m.time_mlp.parameters())
print(f"  time_mlp: {tm:,} params emits a CONSTANT vector in direct mode "
      f"(tau is always 0.0) -> foldable into a bias")
# 3) make_coherent_real_target returns z_real when beta=0
print("  make_coherent_real_target: returns z_real unchanged when coherence_beta=0 "
      "(the best arm uses 0.0) -> dead in that config")
PY
