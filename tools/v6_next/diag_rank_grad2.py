"""Isolate where the ranking gradient dies. Runs the same graph with and without autocast."""
import sys
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import V5_ROOT, V6_ROOT
from tools.v6_next.probe_style import LatentStyleNet

dev = "cuda"
ck = torch.load(ROOT / "runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt", map_location="cpu",
                weights_only=False)
a = ck.get("args", {})
gen = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5, mode=a.get("mode", "direct"),
             content_mode=a.get("content_mode", "full")).to(dev)
gen.load_state_dict(ck["model"])

ckC = torch.load(ROOT / "runs/v6_next/probes/r1_r1.pt", map_location="cpu", weights_only=False)
critic = LatentStyleNet(width=ckC.get("width", 64)).to(dev).eval()
critic.load_state_dict(ckC["model"])
for p in critic.parameters():
    p.requires_grad_(False)

canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                        map_location="cpu", weights_only=False)).to(dev)
lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
keys = sorted(lat)[:8]
z_real = torch.stack([lat[k] for k in keys]).to(dev).float()
with torch.no_grad():
    c_in = canon(z_real, mode="c1_gentle", augment=False)
y_cross = torch.tensor([1, 2, 3, 4, 0, 1, 2, 3], device=dev)


def rank_loss(z):
    q = critic(z.float()).float()
    oh = F.one_hot(y_cross, 5).bool()
    m = q.gather(1, y_cross[:, None]).squeeze(1) - torch.logsumexp(
        q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1)
    return F.softplus(-m / 0.5).mean(), q, m


def gnorm(gs):
    return float(torch.sqrt(sum((t ** 2).sum() for t in gs if t is not None)))


def nz(gs):
    return sum(1 for t in gs if t is not None and float(t.abs().max()) > 0)


print("=== A. no autocast: rank grad, computed first ===")
z2 = gen(c_in, 0.0, c_in, y_cross)
l2, _, _ = rank_loss(z2)
gp = torch.autograd.grad(l2, list(gen.parameters()), allow_unused=True)
print(f"  nonzero={nz(gp)}/{len(gp)}  norm={gnorm(gp):.6e}")

print("\n=== B. no autocast: V6 grad first, then rank grad (training order) ===")
z3 = gen(c_in, 0.0, c_in, y_cross)
lv6 = z3.float().pow(2).mean()
l3, _, _ = rank_loss(z3)
params = list(gen.parameters())
gv = torch.autograd.grad(lv6, params, retain_graph=True, allow_unused=True)
gr = torch.autograd.grad(l3, params, retain_graph=True, allow_unused=True)
print(f"  grad(V6)   norm={gnorm(gv):.6e}")
print(f"  grad(rank) nonzero={nz(gr)}/{len(gr)}  norm={gnorm(gr):.6e}")

print("\n=== C. INSIDE autocast fp16 ===")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    z4 = gen(c_in, 0.0, c_in, y_cross)
    l4, q4, m4 = rank_loss(z4)
    lv4 = z4.float().pow(2).mean()
print(f"  z4.dtype={z4.dtype}  l4.dtype={l4.dtype}  requires_grad={l4.requires_grad}")
p4 = list(gen.parameters())
gv4 = torch.autograd.grad(lv4, p4, retain_graph=True, allow_unused=True)
gr4 = torch.autograd.grad(l4, p4, retain_graph=True, allow_unused=True)
print(f"  grad(V6)   norm={gnorm(gv4):.6e}")
print(f"  grad(rank) nonzero={nz(gr4)}/{len(gr4)}  norm={gnorm(gr4):.6e}")

print("\n=== D. autocast, rank grad computed FIRST ===")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    z5 = gen(c_in, 0.0, c_in, y_cross)
    l5, _, _ = rank_loss(z5)
gr5 = torch.autograd.grad(l5, list(gen.parameters()), allow_unused=True)
print(f"  nonzero={nz(gr5)}/{len(gr5)}  norm={gnorm(gr5):.6e}")

print("\n=== E. autocast: grad of L_rank w.r.t. z_cross ===")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    z6 = gen(c_in, 0.0, c_in, y_cross).detach().requires_grad_(True)
    l6, _, _ = rank_loss(z6)
g6 = torch.autograd.grad(l6, z6)[0]
print(f"  max|grad z6|={float(g6.abs().max()):.3e}  norm={float(g6.norm()):.3e}")

print("\n=== F. autocast: are the critic params actually fp32? ===")
print(f"  critic dtype sample = {next(critic.parameters()).dtype}")
print(f"  gen dtype sample    = {next(gen.parameters()).dtype}")
