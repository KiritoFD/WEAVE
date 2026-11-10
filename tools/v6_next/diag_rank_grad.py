"""Minimal repro: why is grad(L_rank) w.r.t. the generator exactly zero?"""
import sys
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet

dev = "cuda"
torch.manual_seed(0)

gen = C3UNet(base_dim=16, num_styles=5, mode="direct").to(dev)
ck = torch.load(ROOT / "runs/v6_next/probes/r1_r1.pt", map_location="cpu", weights_only=False)
critic = LatentStyleNet(width=ck.get("width", 64)).to(dev).eval()
critic.load_state_dict(ck["model"])
for p in critic.parameters():
    p.requires_grad_(False)

c_in = torch.randn(8, 4, 64, 64, device=dev)
y_cross = torch.randint(0, 5, (8,), device=dev)

print("=== inside autocast (mirrors the training loop) ===")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    z_cross = gen(c_in, 0.0, c_in, y_cross)
    print("  z_cross.requires_grad =", z_cross.requires_grad, " dtype:", z_cross.dtype)
    q = critic(z_cross.float()).float()
    print("  q.requires_grad       =", q.requires_grad, " dtype:", q.dtype)
    oh = F.one_hot(y_cross, 5).bool()
    neg = torch.finfo(q.dtype).min
    m_t = q.gather(1, y_cross[:, None]).squeeze(1) - torch.logsumexp(
        q.masked_fill(oh, neg), dim=1)
    print("  m_t.requires_grad     =", m_t.requires_grad)
    l_rank = F.softplus(-m_t / 0.5).mean()
    print("  l_rank.requires_grad  =", l_rank.requires_grad, " value:", float(l_rank))
    l_style = z_cross.float().pow(2).mean()      # stand-in for the V6 term

g_rk = torch.autograd.grad(l_rank, list(gen.parameters()), retain_graph=True, allow_unused=True)
n_none = sum(1 for t in g_rk if t is None)
n_nonzero = sum(1 for t in g_rk if t is not None and float(t.abs().sum()) > 0)
print(f"\n  grad(l_rank): {len(g_rk)} tensors, None={n_none}, nonzero={n_nonzero}")
print("  norm =", float(torch.sqrt(sum((t ** 2).sum() for t in g_rk if t is not None))))

# order-of-call test: does consuming the V6 grad first kill the rank grad?
z2 = gen(c_in, 0.0, c_in, y_cross)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    l2 = z2.float().pow(2).mean()
    q2 = critic(z2.float()).float()
    m2 = q2.gather(1, y_cross[:, None]).squeeze(1) - torch.logsumexp(
        q2.masked_fill(F.one_hot(y_cross, 5).bool(), torch.finfo(q2.dtype).min), dim=1)
    r2 = F.softplus(-m2 / 0.5).mean()
g1 = torch.autograd.grad(l2, list(gen.parameters()), retain_graph=True, allow_unused=True)
g2 = torch.autograd.grad(r2, list(gen.parameters()), retain_graph=True, allow_unused=True)
print(f"\n  after consuming grad(l2) first: grad(r2) norm = "
      f"{float(torch.sqrt(sum((t ** 2).sum() for t in g2 if t is not None))):.6e}")
print(f"  grad(l2) norm = {float(torch.sqrt(sum((t ** 2).sum() for t in g1 if t is not None))):.6e}")

# is the critic's output actually differentiable w.r.t. its input?
zi = torch.randn(4, 4, 64, 64, device=dev, requires_grad=True)
qq = critic(zi.float())
print(f"\n  critic output requires_grad w.r.t. its input = {qq.requires_grad}")
