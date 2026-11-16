"""V17 DINO patch-token cache for student context conditioning (P2 S2+).

Uses the referee's own encode_dino (tools/pseudo_gt/matching.py) -> patch
[B,256,384] reshaped to (B,384,16,16), cached once per pool source stem.
Never computed for Test750 sources; CLS dropped; fp16.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .common import OUT


def encode_tokens_for_images(images, device='cuda', batch=8):
    """Inference-time tokens for arbitrary images (Test750 sources etc.)."""
    from src.utils._local_eval_refs import encode_dino, load_dino_model
    dino = load_dino_model(device=device)
    _, patch = encode_dino(dino, list(images), batch=batch, device=device)
    tok = patch.float()
    tok = tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return tok.reshape(len(images), 384, 16, 16)


def cache_tokens(pool_split: dict, out_path: Path | None = None,
                 device: str = 'cuda'):
    out_path = Path(out_path or (OUT / 'dino_cache' / 'pool_tokens.pt'))
    if out_path.exists():
        return out_path
    from src.utils import load_s10_decoder, decode_s10
    from src.utils._local_eval_refs import encode_dino, load_dino_model
    from PIL import Image
    dino = load_dino_model(device=device)
    dec = load_s10_decoder(device)
    store = {}
    for split, d in pool_split.items():
        tokens = []
        for st in range(0, len(d['ids']), 8):
            z = d['z'][st:st + 8].to(device)
            rgb = decode_s10(dec, z, batch=8)
            imgs = [Image.fromarray(x) for x in
                    rgb.mul(255).round().clamp(0, 255).byte().cpu()
                    .permute(0, 2, 3, 1).numpy()]
            _, patch = encode_dino(dino, imgs, batch=8, device=device)
            tok = patch.float()                          # (B,256,384)
            tok = tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            tokens.append(tok.reshape(len(imgs), 384, 16, 16).cpu().half())
        store[split] = {'ids': d['ids'], 'tokens': torch.cat(tokens)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'tokens': store,
                'contract': '(B,384,16,16) L2-normalised DINOv2-small patch '
                            'tokens, CLS dropped, fp16, pool sources only'},
               out_path)
    print(f"dino cache: {sum(len(v['ids']) for v in store.values())} images",
          flush=True)
    return out_path
