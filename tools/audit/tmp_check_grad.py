import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import latent_losses as L

z0 = torch.randn(2, 4, 64, 64, device="cuda")
zb = (z0 + 0.05 * torch.randn_like(z0)).clone().requires_grad_(True)
lc = L.content_structure(zb, z0)
g = torch.autograd.grad(lc, zb)[0]
print("structure", lc.item(), g.norm().item())

lc2 = L.content_ll(zb, z0)
g2 = torch.autograd.grad(lc2, zb)[0]
print("ll", lc2.item(), g2.norm().item())

lc3 = L.content_rms(zb, z0)
g3 = torch.autograd.grad(lc3, zb)[0]
print("rms", lc3.item(), g3.norm().item())
