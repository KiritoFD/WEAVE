"""Verify the transferred checkpoints actually load and forward on CPU.

A SHA match proves the bytes arrived intact; this proves they are the model they claim to be, by
reconstructing each one from its recorded args and running a forward pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
from tools.v6_next.probe_style import LatentStyleNet

CK = ROOT / "deliver" / "checkpoints"


def build_generator(a: dict) -> C3UNet:
    return C3UNet(base_dim=a.get("base_dim", 16), num_styles=5,
                  mode=a.get("mode", "direct"),
                  no_style=bool(a.get("no_style", False)),
                  content_mode=a.get("content_mode", "full"),
                  inject_64=not a.get("no_inject_64", False),
                  inject_32=not a.get("no_inject_32", False),
                  inject_16=not a.get("no_inject_16", False),
                  style_keys=bool(a.get("style_keys", False)),
                  style_keys_n=int(a.get("style_keys_n", 8) or 8),
                  style_keys_rank=int(a.get("style_keys_rank", 32) or 32))


def main() -> None:
    z = torch.randn(2, 4, 64, 64)
    t = torch.tensor([1, 3])
    bad = 0

    gens = sorted(p for p in CK.rglob("*.pt")
                  if "r1_r1" not in p.name and "score_diff" not in p.name)
    for p in gens:
        d = torch.load(p, map_location="cpu", weights_only=False)
        a = d.get("args", {})
        m = build_generator(a)
        m.load_state_dict(d["model"])
        m.eval()
        with torch.inference_mode():
            out = m(z, 0.0, z, t)
        n = sum(q.numel() for q in m.parameters())
        ok = tuple(out.shape) == (2, 4, 64, 64) and torch.isfinite(out).all()
        bad += int(not ok)
        # mechanism provenance lives in ckpt["args"], not at the top level
        prov = []
        if a.get("rank_lambda"):
            prov.append(f"rank={a['rank_lambda']}")
        if a.get("embed_style_lambda"):
            prov.append(f"embed={a['embed_style_lambda']}")
        if a.get("distill_lambda"):
            prov.append(f"distill={a['distill_lambda']}")
        if a.get("trust_lambda"):
            prov.append(f"trust={a['trust_lambda']}")
        print(f"  {'OK ' if ok else 'BAD'} {p.parent.name:22s} dim={a.get('base_dim'):>3} "
              f"params={n:>9,} out={tuple(out.shape)} "
              f"|out|={float(out.norm()):.2f} "
              f"{' '.join(prov) if prov else 'no auxiliary loss (control)'}")

    p = CK / "runs/v6_next/probes/r1_r1.pt"
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = LatentStyleNet(width=d.get("width", 64)).eval()
        m.load_state_dict(d["model"])
        with torch.inference_mode():
            logits = m(z)
        n = sum(q.numel() for q in m.parameters())
        ok = tuple(logits.shape) == (2, 5) and torch.isfinite(logits).all()
        bad += int(not ok)
        print(f"  {'OK ' if ok else 'BAD'} {'r1_r1 (critic)':22s} width={d.get('width')} "
              f"params={n:>9,} out={tuple(logits.shape)}")

    p = CK / "runs/v6/score_diff/score_diff_net.pt"
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = ConditionalScoreNet(base_dim=64, num_styles=5).eval()
        m.load_state_dict(d["model"])
        n = sum(q.numel() for q in m.parameters())
        with torch.inference_mode():
            try:
                out = m(z, torch.full((2,), 0.3), t)
                ok = torch.isfinite(out).all()
            except TypeError:
                out = m(z, t)
                ok = torch.isfinite(out).all()
        bad += int(not ok)
        print(f"  {'OK ' if ok else 'BAD'} {'score_diff (3.27M)':22s} params={n:>9,} "
              f"out={tuple(out.shape)}")

    print(f"\n  {'ALL LOAD AND FORWARD' if bad == 0 else f'{bad} FAILED'}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
