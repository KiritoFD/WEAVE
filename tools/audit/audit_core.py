"""Shared utilities for the WEAVE supervision audit (latent-only, no decoder backprop).

All losses / features in this module operate on the 4x64x64 VAE latent.  The
VAE decoder is used ONLY for offline evaluation (no gradients through it).

Conventions
-----------
* style order: STYLES (same as training / eval).
* diagnostic split: per style, ``n_ref`` reference images + ``n_query`` query
  images, disjoint, deterministic (seed 0).
* real style statistics for candidate losses come from the *ref* split only;
  query split is never used to fit anything.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paths import dino_cache_root, hf_cache_root  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SHORT = ["ER", "Imp", "Min", "Roc", "Uki"]
PACKED = ROOT / "data/train/.latent_cache/packed/packed"
DINO_CACHE = dino_cache_root("content_5k")
AUDIT_DIR = ROOT / "runs/audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    latents: torch.Tensor            # [5, N, 4, 64, 64] float32
    names: list[list[str]]           # per style
    artists: list[list[str]]         # per style

    def flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.latents.reshape(-1, *self.latents.shape[2:])
        y = torch.arange(5).repeat_interleave(self.latents.shape[1])
        return z, y


def _artist_of(name: str) -> str:
    base = Path(name).stem
    parts = base.split("__")
    if len(parts) < 2:
        return "unknown"
    tail = parts[1]
    # artist is the leading slug up to the first title token; WikiArt names use
    # style__artist_title, so take everything before the first "_" chunk that
    # looks like a title.  A robust proxy: artist = first two dash tokens.
    toks = tail.split("_")
    return toks[0]


def load_dataset() -> Dataset:
    lats, names, artists = [], [], []
    for i, style in enumerate(STYLES):
        p = PACKED / f"{i:02d}_{style}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        z = d["latents"].float()
        lats.append(z)
        fs = [Path(f).name for f in d["files"]]
        names.append(fs)
        artists.append([_artist_of(f) for f in fs])
    return Dataset(torch.stack(lats), names, artists)


@dataclass
class Split:
    ref: torch.Tensor    # [5, n_ref] long indices into per-style arrays
    query: torch.Tensor  # [5, n_query]


def make_split(n_per_style: int, *, n_ref: int = 800, n_query: int = 200, seed: int = 0) -> Split:
    g = torch.Generator().manual_seed(seed)
    ref, query = [], []
    for _ in range(5):
        perm = torch.randperm(n_per_style, generator=g)
        ref.append(perm[:n_ref].sort().values)
        query.append(perm[n_ref:n_ref + n_query].sort().values)
    return Split(torch.stack(ref), torch.stack(query))


def style_of(y: torch.Tensor) -> torch.Tensor:
    return y


# ---------------------------------------------------------------------------
# frozen latent representations (no parameters, no RGB)
# ---------------------------------------------------------------------------
def _chan_stats_maps(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Simple maps used by several feature builders."""
    m = x.mean(dim=1, keepdim=True)                     # [B,1,H,W]
    s = x.std(dim=1, keepdim=True)
    gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    g = (gx.abs() + gy.abs()).mean(dim=1, keepdim=True)
    return {"mean": m, "std": s, "grad": g}


def feat_global(z: torch.Tensor) -> torch.Tensor:
    """[B,4,64,64] -> [B,20]: per-channel mean/std/absmean/q10/q90."""
    flat = z.flatten(2)
    q = torch.quantile(flat, torch.tensor([0.1, 0.9], device=z.device), dim=2)
    out = [flat.mean(2), flat.std(2), flat.abs().mean(2), q[0], q[1]]
    return torch.cat(out, 1)


def feat_subband(z: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Wavelet subband statistics: per level, per band, per channel mean|.| & std."""
    from wavelet import dwt2_haar
    cur = z
    feats = []
    for _ in range(levels):
        ll, lh, hl, hh = dwt2_haar(cur)
        for b in (ll, lh, hl, hh):
            feats.append(b.abs().flatten(2).mean(2))
            feats.append(b.flatten(2).std(2))
        cur = ll
    return torch.cat(feats, 1)


def feat_lowpass(z: torch.Tensor, level: int = 2) -> torch.Tensor:
    """[B,4,64,64] -> low-pass latent statistics (per-channel mean/std/absmean)."""
    from wavelet import dwt2_haar_lowpass
    lp = dwt2_haar_lowpass(z, levels=level)
    return feat_global(lp)


def feat_patch(z: torch.Tensor, *, patch: int = 8, stride: int = 8) -> torch.Tensor:
    """[B,4,H,W] -> [B,N,12] local descriptors (mean/std/abs/grad per patch)."""
    m = _chan_stats_maps(z)
    outs = []
    for key in ("mean", "std", "grad"):
        outs.append(F.avg_pool2d(m[key], patch, stride))
    # also per-channel patch means/stds (4+4)
    per = F.avg_pool2d(z, patch, stride)
    per_sq = F.avg_pool2d(z * z, patch, stride)
    per_std = torch.sqrt(torch.clamp(per_sq - per * per, min=1e-6))
    cat = torch.cat(outs + [per, per_std], 1)  # [B, 1+1+1+4+4=11]
    b, c, h, w = cat.shape
    return cat.reshape(b, c, h * w).transpose(1, 2).contiguous()


def feat_chan_cov(z: torch.Tensor) -> torch.Tensor:
    """[B,4,64,64] -> [B,10] channel covariance upper triangle."""
    flat = z.flatten(2)
    flat = flat - flat.mean(2, keepdim=True)
    cov = flat @ flat.transpose(1, 2) / flat.shape[2]
    idx = torch.triu_indices(4, 4)
    return cov[:, idx[0], idx[1]]


def feat_local_cov(z: torch.Tensor, *, patch: int = 8, stride: int = 8) -> torch.Tensor:
    """[B,N,10] patch-wise covariance (then mean over patches by callers)."""
    b, c, h, w = z.shape
    p = F.unfold(z, kernel_size=patch, stride=stride)  # [B, C*ps*ps, N]
    p = p.transpose(1, 2)                              # [B,N,C*ps*ps]
    p = p - p.mean(2, keepdim=True)
    cov = p @ p.transpose(1, 2) / p.shape[2]           # [B,N,C,C]
    idx = torch.triu_indices(c, c)
    return cov[..., idx[0], idx[1]]


def feat_spectrum(z: torch.Tensor, *, res: int = 32, bins: int = 4) -> torch.Tensor:
    """[B,4,64,64] -> [B,4*bins*bins] radial angular power of the latent."""
    x = F.interpolate(z, size=(res, res), mode="area")
    spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)).abs()
    yy, xx = torch.meshgrid(torch.arange(res, device=z.device), torch.arange(res, device=z.device),
                            indexing="ij")
    cy = cx = (res - 1) / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (res / 2)
    rad = torch.clamp((r * bins).floor().long(), max=bins - 1)
    ang = torch.clamp(((torch.atan2(yy - cy, xx - cx) + math.pi) / (2 * math.pi) * bins).floor().long(),
                      max=bins - 1)
    idx = (ang * bins + rad).flatten()
    out = []
    for c in range(x.shape[1]):
        onehot = F.one_hot(idx, bins * bins).float()          # [res*res, bins^2]
        p = spec[:, c].reshape(x.shape[0], -1) @ onehot       # [B, bins^2]
        out.append(p / p.sum(1, keepdim=True).clamp_min(1e-8))
    return torch.cat(out, 1)


REPRESENTATIONS = {
    "raw_global": lambda z: feat_global(z),
    "subband": lambda z: feat_subband(z),
    "lowpass2": lambda z: feat_lowpass(z, 2),
    "lowpass4": lambda z: feat_lowpass(z, 4),
    "chan_cov": lambda z: feat_chan_cov(z),
    "spectrum": lambda z: feat_spectrum(z),
    "patch8": lambda z: feat_patch(z).mean(1),
    "patch4": lambda z: feat_patch(z, patch=4, stride=4).mean(1),
    "local_cov": lambda z: feat_local_cov(z).flatten(1),
}


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def _q_rows(sorted_x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    n = sorted_x.shape[0]
    pos = q * (n - 1)
    lo = torch.floor(pos).long().clamp(0, n - 1)
    hi = torch.ceil(pos).long().clamp(0, n - 1)
    frac = (pos - lo.to(pos.dtype)).unsqueeze(1)
    return sorted_x[lo] * (1 - frac) + sorted_x[hi] * frac


def swd2(a: torch.Tensor, b: torch.Tensor, *, n_q: int = 64) -> torch.Tensor:
    """Sliced-Wasserstein on coordinate axes, mean over dims (scale comparable)."""
    q = torch.linspace(0, 1, n_q, device=a.device, dtype=a.dtype)
    a_s = torch.sort(a, 0).values
    b_s = torch.sort(b.detach().float(), 0).values.to(a.dtype)
    return ((_q_rows(a_s, q) - _q_rows(b_s, q)) ** 2).mean(0).mean()


def swd2_proj(a: torch.Tensor, b: torch.Tensor, *, proj: torch.Tensor, n_q: int = 64) -> torch.Tensor:
    """SWD along FIXED random projections (proj: [D,P]); differentiable w.r.t. a."""
    q = torch.linspace(0, 1, n_q, device=a.device, dtype=a.dtype)
    pa = torch.sort(a @ proj, 0).values
    pb = torch.sort((b.detach().float() @ proj), 0).values.to(a.dtype)
    return ((_q_rows(pa, q) - _q_rows(pb, q)) ** 2).mean(0).mean()


def mmd2(a: torch.Tensor, b: torch.Tensor, *, sigma_mult: Sequence[float] = (0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    af, bf = a.float(), b.detach().float()
    both = torch.cat([af, bf], 0)
    d = torch.cdist(both, both)
    sigma0 = torch.median(d[d > 0]).clamp_min(1e-3)
    total = a.new_zeros(())
    for m in sigma_mult:
        s = float(sigma0) * m
        kaa = torch.exp(-torch.cdist(af, af) ** 2 / (2 * s * s)).mean()
        kbb = torch.exp(-torch.cdist(bf, bf) ** 2 / (2 * s * s)).mean()
        kab = torch.exp(-torch.cdist(af, bf) ** 2 / (2 * s * s)).mean()
        total = total + kaa + kbb - 2 * kab
    return total / len(sigma_mult)


# ---------------------------------------------------------------------------
# classification / retrieval
# ---------------------------------------------------------------------------
def standardize(fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = fit.mean(0)
    sd = fit.std(0).clamp_min(1e-6)
    return mu, sd


def nearest_centroid(train_x, train_y, test_x, *, k: int = 5):
    mu, sd = standardize(train_x)
    tr = (train_x - mu) / sd
    te = (test_x - mu) / sd
    cents = torch.stack([tr[train_y == c].mean(0) for c in range(k)])
    d = torch.cdist(te, cents)
    pred = d.argmin(1)
    return pred, d


def linear_probe(train_x, train_y, test_x, *, k: int = 5, ridge: float = 1e-2, iters: int = 200, lr: float = 0.1):
    """Multinomial logistic regression (torch, deterministic)."""
    mu, sd = standardize(train_x)
    tr, te = (train_x - mu) / sd, (test_x - mu) / sd
    d = tr.shape[1]
    W = torch.zeros(d, k, device=tr.device, requires_grad=True)
    b = torch.zeros(k, device=tr.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        logits = tr @ W + b
        loss = F.cross_entropy(logits, train_y) + ridge * W.pow(2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = te @ W + b
    return logits.argmax(1), logits


def confusion(pred: torch.Tensor, true: torch.Tensor, k: int = 5) -> list[list[int]]:
    m = torch.zeros(k, k, dtype=torch.long)
    for p, t in zip(pred, true):
        m[int(t), int(p)] += 1
    return m.tolist()


def auroc_ovr(scores: torch.Tensor, labels: torch.Tensor, c: int) -> float:
    """One-vs-rest AUROC from a softmax/logit score column (ties averaged)."""
    s = scores[:, c].float()
    pos = (labels == c)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    # average ranks for ties
    ranks_sorted = torch.arange(1, s.numel() + 1, dtype=torch.float32)
    ranks[order] = ranks_sorted
    # handle ties by grouping
    vals = s[order]
    i = 0
    while i < vals.numel():
        j = i + 1
        while j < vals.numel() and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top1_retrieval(x_q, y_q, x_g, y_g) -> float:
    """Top-1 style label accuracy: nearest gallery image (cosine on standardised feats)."""
    mu, sd = standardize(x_g)
    q = (x_q - mu) / sd
    g = (x_g - mu) / sd
    q = F.normalize(q, dim=1)
    g = F.normalize(g, dim=1)
    sim = q @ g.T
    idx = sim.argmax(1)
    return float((y_g[idx] == y_q).float().mean())


def cluster_bootstrap(values: torch.Tensor, groups: torch.Tensor, *, n_boot: int = 200, seed: int = 0):
    """Bootstrap mean over *groups* (content clusters), returning mean and 95% CI."""
    g = torch.Generator().manual_seed(seed)
    ug = torch.unique(groups)
    means = []
    for _ in range(n_boot):
        pick = ug[torch.randint(0, ug.numel(), (ug.numel(),), generator=g)]
        idx = torch.cat([torch.where(groups == u)[0] for u in pick])
        means.append(float(values[idx].mean()))
    means = torch.tensor(means)
    return {"mean": float(values.mean()), "lo": float(means.quantile(0.025)),
            "hi": float(means.quantile(0.975))}


# ---------------------------------------------------------------------------
# offline decode + metrics  (NO gradients through the decoder)
# ---------------------------------------------------------------------------
_METRIC_STATE: dict[str, Any] = {}


def load_metric_models(device: str = "cuda", need_clip: bool = True, need_lpips: bool = True, need_dino: bool = True):
    import os
    os.environ.setdefault("HF_HOME", str(hf_cache_root()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if _METRIC_STATE.get("loaded"):
        return _METRIC_STATE
    if need_clip:
        from transformers import CLIPModel, CLIPProcessor
        cdir = str(hf_cache_root("hub"))
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cdir,
                                          local_files_only=True).to(device).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cdir)
        _METRIC_STATE["clip"] = (model, proc)
        _METRIC_STATE["clip_size"] = 224
        _METRIC_STATE["clip_mean"] = (0.48145466, 0.4578275, 0.40821073)
        _METRIC_STATE["clip_std"] = (0.26862954, 0.26130258, 0.27577711)
    if need_lpips:
        import lpips
        _METRIC_STATE["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    if need_dino:
        from transformers import AutoModel
        # DINOv2-small matches the frozen eval protocol used for the 512 anchor.
        d = AutoModel.from_pretrained("facebook/dinov2-small",
                                      cache_dir=str(hf_cache_root("hub")),
                                      local_files_only=True).to(device).eval()
        _METRIC_STATE["dino"] = d
    from tools.legacy_utils.inference import load_vae
    _METRIC_STATE["vae"] = load_vae(device, model_id="ema",
                                    cache_dir=str(hf_cache_root()),
                                    compile_decoder=False)
    _METRIC_STATE["device"] = device
    _METRIC_STATE["loaded"] = True
    return _METRIC_STATE


@torch.inference_mode()
def decode(z: torch.Tensor, *, batch: int = 8, device: str = "cuda") -> torch.Tensor:
    """Offline latent -> [0,1] RGB.  Strictly no gradient."""
    from tools.legacy_utils.inference import decode_latent
    st = _METRIC_STATE.get("loaded") and _METRIC_STATE or load_metric_models(device)
    vae = st["vae"]
    out = []
    for i in range(0, z.shape[0], batch):
        out.append(decode_latent(vae, z[i:i + batch].float().to(device), device).float().cpu())
    return torch.cat(out)


_DINO_TF = None


@torch.inference_mode()
def dino_cls(img01: torch.Tensor, *, batch: int = 16, device: str = "cuda") -> torch.Tensor:
    """[N,3,H,W] in [0,1] -> normalised DINOv2-base CLS [N,768] (eval protocol 224 centre crop)."""
    global _DINO_TF
    st = load_metric_models(device)
    if _DINO_TF is None:
        import torchvision.transforms as T
        from PIL import Image
        # tensor CHW in [0,1] -> normalised 224 centre crop (no ToTensor)
        _DINO_TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                              T.CenterCrop(224),
                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = st["dino"]
    feats = []
    for i in range(0, img01.shape[0], batch):
        px = torch.stack([_DINO_TF(t) for t in img01[i:i + batch]]).to(device)
        out = model(px)
        feats.append(F.normalize(out.last_hidden_state[:, 0, :].float(), dim=-1).cpu())
    return torch.cat(feats)


@torch.inference_mode()
def clip_img(img01: torch.Tensor, *, batch: int = 32, device: str = "cuda") -> torch.Tensor:
    st = load_metric_models(device)
    model, _ = st["clip"]
    mean = torch.tensor(st["clip_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(st["clip_std"], device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, img01.shape[0], batch):
        px = F.interpolate(img01[i:i + batch].to(device), size=(224, 224), mode="bicubic",
                           align_corners=False, antialias=True)
        px = ((px - mean) / std).to(next(model.parameters()).dtype)
        out = model.get_image_features(pixel_values=px)
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            out = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            out = out.pooler_output
        feats.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(feats)


def quantize01(img: torch.Tensor) -> torch.Tensor:
    """Emulate PNG uint8 round-trip (eval_fast quantises before metrics)."""
    return (img.clamp(0, 1) * 255.0).round().div(255.0)


@torch.inference_mode()
def lpips_dist(gen01: torch.Tensor, src01: torch.Tensor, *, device: str = "cuda") -> torch.Tensor:
    st = load_metric_models(device)
    fn = st["lpips"]
    return fn(gen01.to(device) * 2 - 1, src01.to(device) * 2 - 1).flatten().cpu()


def save_json(name: str, payload: Any) -> Path:
    p = AUDIT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[audit] wrote {p}")
    return p


__all__ = [n for n in dir() if not n.startswith("_")]
