#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import sys, torch
sys.path.insert(0, ".")
from tools.v6.models_v6 import C3UNet

print("=== content_mode variants (params must be identical) ===")
outs = {}
for cm in ("full", "no_inject", "none"):
    torch.manual_seed(0)
    m = C3UNet(base_dim=32, num_styles=5, mode="direct", content_mode=cm).eval()
    z = torch.randn(2, 4, 64, 64); c = torch.randn(2, 4, 64, 64)
    s = torch.tensor([1, 3])
    with torch.no_grad():
        o = m(z, 0.0, c, s)
    outs[cm] = o
    print(f"  content_mode={cm:10s} params={sum(p.numel() for p in m.parameters()):,} "
          f"finite={bool(torch.isfinite(o).all())}")

def rand_head(m):
    """The output head is zero-initialised, so a fresh model always returns 0.
    Randomise it so content dependence is observable."""
    with torch.no_grad():
        for p in (m.out_conv64.weight, m.out_conv64.bias,
                  m.out_dwt32.weight, m.out_dwt32.bias,
                  m.out_adaln.weight, m.out_adaln.bias):
            p.normal_(0, 0.1)
    return m

# 'none' must ignore the content entirely: same output for different c
torch.manual_seed(0)
m = rand_head(C3UNet(base_dim=32, num_styles=5, mode="direct",
                     content_mode="none")).eval()
with torch.no_grad():
    o1 = m(torch.randn(2, 4, 64, 64), 0.0, torch.randn(2, 4, 64, 64), torch.tensor([1, 3]))
    o2 = m(torch.randn(2, 4, 64, 64), 0.0, torch.randn(2, 4, 64, 64), torch.tensor([1, 3]))
print(f"  content_mode=none ignores content: identical outputs = "
      f"{bool(torch.allclose(o1, o2))}")

# 'no_inject' must still depend on content (through the stem)
torch.manual_seed(0)
m = rand_head(C3UNet(base_dim=32, num_styles=5, mode="direct",
                     content_mode="no_inject")).eval()
with torch.no_grad():
    a = torch.randn(2, 4, 64, 64); b = torch.randn(2, 4, 64, 64)
    oa = m(a, 0.0, a, torch.tensor([1, 3]))
    ob = m(b, 0.0, b, torch.tensor([1, 3]))
print(f"  content_mode=no_inject still content-dependent: differs = "
      f"{not bool(torch.allclose(oa, ob))}")

# 'full' must depend on content too
torch.manual_seed(0)
m = rand_head(C3UNet(base_dim=32, num_styles=5, mode="direct",
                     content_mode="full")).eval()
with torch.no_grad():
    oa = m(a, 0.0, a, torch.tensor([1, 3]))
    ob = m(b, 0.0, b, torch.tensor([1, 3]))
print(f"  content_mode=full still content-dependent: differs = "
      f"{not bool(torch.allclose(oa, ob))}")
PY
