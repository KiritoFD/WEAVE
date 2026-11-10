#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
python3 - <<'PY'
import sys, torch
from pathlib import Path
ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(ROOT))
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet
from tools.v6.train_eval_v6 import V6_ROOT, STYLES

dev = "cuda"
ck = torch.load(ROOT / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt", map_location="cpu",
                weights_only=False)
a = ck.get("args", {})
gen = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5, mode="direct",
             content_mode=a.get("content_mode", "full")).to(dev)
gen.load_state_dict(ck["model"]); gen.eval()
canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                       map_location="cpu", weights_only=False)).to(dev)
cc = torch.load(ROOT / "runs/v6_next/probes/r1_r1.pt", map_location="cpu", weights_only=False)
crit = LatentStyleNet(width=cc.get("width", 64)).to(dev).eval()
crit.load_state_dict(cc["model"])
for p in crit.parameters():
    p.requires_grad_(False)

d = torch.load("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt",
               map_location="cpu", weights_only=False)
z = d["latents"].float()[-16:].to(dev)
t = torch.full((16,), 2, dtype=torch.long, device=dev)

with torch.inference_mode():
    c = canon(z, mode="c1_gentle", augment=False)
    zd = gen(c, 0.0, c, t)

def sq(x):
    return float(x.flatten(1).pow(2).sum(1).mean())

print(f"  ||z_src||^2          = {sq(z):12.3f}")
print(f"  ||c||^2 (canon)      = {sq(c):12.3f}")
print(f"  ||zd - c||^2 (Direct)= {sq(zd - c):12.3f}")
print(f"  ||zd||^2             = {sq(zd):12.3f}")

D = z[0].numel()
for rms in (0.25,):
    zr = zd.detach().clone().requires_grad_(True)
    q = crit(zr.float()).float()
    oh = torch.one_hot(t, 5).bool()
    m = q.gather(1, t[:, None]).squeeze(1) - torch.logsumexp(
        q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1)
    g = torch.autograd.grad(m.sum(), zr)[0]
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    step = rms * (g / (gn / D ** 0.5 + 1e-12))
    print(f"\n  nominal per-dim RMS  = {rms}")
    print(f"  ||g|| per sample     = {float(gn.mean()):12.4f}")
    print(f"  g per-dim RMS        = {float(gn.mean())/D**0.5:12.6f}")
    print(f"  ||step||^2 actual    = {sq(step):12.3f}   (expected {rms**2*D:.1f})")
    print(f"  step per-dim RMS     = {float(step.flatten(1).pow(2).mean().sqrt()):12.6f}")
    zT = (zd + step)
    print(f"  ||zT - c||^2         = {sq(zT - c):12.3f}")
PY
