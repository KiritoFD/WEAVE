"""V14 scale scan: process ONE representation per invocation (avoids GPU OOM).

Usage: python scan_one.py --rep S3
Output: runs/v14/scan_S3.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

REPO = Path("I:/Latent_Style/WEAVE")
POOL = Path("I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
IMG_ROOT = Path("I:/Latent_Style/WEAVE/datasets/wikiart_new_512")
DINO_SNAP = Path("I:/Latent_Style/eval_cache/hf/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056")
OUT = Path("I:/Latent_Style/WEAVE/runs/v14")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
N_TRAIN, N_HOLD, N_BOOT = 200, 80, 1000
ALL_REPS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S8"]

_DINO = None
_CNN = None

def _h10(pid):
    import hashlib
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10

def load_rgb(path, size=512):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0

def to_gray(rgb):
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
    return (rgb * w).sum(1, keepdim=True).repeat(1, 3, 1, 1)

def sobel_mag(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                      device=x.device).view(1,1,3,3)
    ky = kx.transpose(2, 3)
    return (Fn.conv2d(x, kx, padding=1).square() + Fn.conv2d(x, ky, padding=1).square()).sqrt()

def dwt_levels(z):
    ll1 = Fn.avg_pool2d(z, 2)
    h1 = z - Fn.interpolate(ll1, size=z.shape[-2:], mode="nearest")
    ll2 = Fn.avg_pool2d(ll1, 2)
    h2 = ll1 - Fn.interpolate(ll2, size=ll1.shape[-2:], mode="nearest")
    e2 = sobel_mag(ll2.mean(1, keepdim=True))
    return {"H2": h2, "LL2": ll2, "H1": h1, "LL1": ll1, "edge2": e2}

# ---- extractors ----
def s0_latent_stats(z):
    ll32, ll16 = Fn.avg_pool2d(z, 2), Fn.avg_pool2d(z, 4)
    parts = [ll16.flatten(1), ll32.flatten(1)]
    for ll in (ll16, ll32):
        parts += [(ll[..., :, 1:] - ll[..., :, :-1]).flatten(1),
                  (ll[..., 1:, :] - ll[..., :-1, :]).flatten(1)]
    return torch.cat(parts, 1)

def s1_dwt_tokens(z):
    d = dwt_levels(z)
    t = torch.cat([Fn.adaptive_avg_pool2d(d["H2"], 8),
                   Fn.adaptive_avg_pool2d(d["LL2"], 8),
                   Fn.adaptive_avg_pool2d(d["edge2"], 8)], 1)[0]
    return t.reshape(t.shape[0], -1).T  # (64, C)

def s2_dwt_large(rgb):
    x = Fn.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
    z = x.mean(1, keepdim=True).repeat(1, 4, 1, 1)
    return s1_dwt_tokens(z)

def _cnn(device):
    global _CNN
    if _CNN is None:
        layers = []
        cin = 3
        for cout, stride in ((32,1),(64,2),(64,1),(128,2),(128,1)):
            layers += [torch.nn.Conv2d(cin, cout, 3, stride=stride, padding=1,
                                       bias=False), torch.nn.SiLU(True)]
            cin = cout
        _CNN = torch.nn.Sequential(*layers).to(device).eval()
    return _CNN

def _rgb_cnn(rgb, device):
    x = rgb if rgb.dim() == 4 else rgb.unsqueeze(0)
    return Fn.adaptive_avg_pool2d(_cnn(device)(x), 8).flatten(2).squeeze(0).T

def s3_rgb64(rgb, device):
    return _rgb_cnn(Fn.interpolate(rgb, size=(64, 64), mode="bilinear",
                                   align_corners=False), device)

def s4_rgb128(rgb, device):
    return _rgb_cnn(Fn.interpolate(rgb, size=(128, 128), mode="bilinear",
                                   align_corners=False), device)

def s5_rgb224(rgb, device):
    return _rgb_cnn(Fn.interpolate(rgb, size=(224, 224), mode="bilinear",
                                   align_corners=False), device)

_DINO = None
def _dino(device):
    global _DINO
    if _DINO is None:
        from transformers import AutoModel
        snap = Path("I:/Latent_Style/eval_cache/hf/hub/models--facebook--dinov2-small/"
                    "snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056")
        _DINO = AutoModel.from_pretrained(str(snap)).to(device).eval()
    return _DINO

def s6_dino(rgb, device):
    m = _dino(device)
    x = rgb if rgb.dim() == 4 else rgb.unsqueeze(0)
    x = Fn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        out = m(x, return_dict=True)
        tok = out.last_hidden_state[:, 1:, :]
        tok = tok.transpose(1, 2).reshape(1, -1, 16, 16)
    return Fn.adaptive_avg_pool2d(tok, 8).flatten(2).squeeze(0).T

def s8_multiscale(rgb, z, device):
    return torch.cat([s1_dwt_tokens(z), s6_dino(rgb, device)], 1)

# ---- grey variants ----
def s3_gray(rgb, device):
    return s3_rgb64(to_gray(rgb), device)
def s4_gray(rgb, device):
    return s4_rgb128(to_gray(rgb), device)
def s5_gray(rgb, device):
    return s5_rgb224(to_gray(rgb), device)
def s6_gray(rgb, device):
    return s6_dino(to_gray(rgb), device)


EXTRACT = {
    "S0": lambda rgb, z, dev: s0_latent_stats(z),
    "S1": lambda rgb, z, dev: s1_dwt_tokens(z),
    "S2": lambda rgb, z, dev: s2_dwt_large(rgb),
    "S3": lambda rgb, z, dev: s3_rgb64(rgb, dev),
    "S4": lambda rgb, z, dev: s4_rgb128(rgb, dev),
    "S5": lambda rgb, z, dev: s5_rgb224(rgb, dev),
    "S6": lambda rgb, z, dev: s6_dino(rgb, dev),
    "S8": lambda rgb, z, dev: s8_multiscale(rgb, z, dev),
}
GRAY_EXTRACT = {
    "S3": lambda rgb, dev: s3_gray(rgb, dev),
    "S4": lambda rgb, dev: s4_gray(rgb, dev),
    "S5": lambda rgb, dev: s5_gray(rgb, dev),
    "S6": lambda rgb, dev: s6_gray(rgb, dev),
}


def extract_all(rep, items, latents, device):
    """Extract features for one representation across all items."""
    fn = EXTRACT[rep]
    feats = []
    labels = []
    for n, (si, img_path, lat_idx) in enumerate(items):
        z = latents[STYLES[si]][lat_idx:lat_idx + 1].float().to(device)
        rgb = load_rgb(img_path, 512).to(device)
        f = fn(rgb, z, device)
        feats.append(f.half().cpu())
        labels.append(si)
        del f, rgb, z
        torch.cuda.empty_cache()
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    return torch.cat(feats).float(), torch.tensor(labels)


def ncm_acc(train_f, train_y, test_f, test_y):
    train_f = train_f.cpu()
    test_f = test_f.cpu()
    train_y = train_y.cpu()
    test_y = test_y.cpu()
    mu, sd = train_f.mean(0), train_f.std(0).clamp_min(1e-4)
    tr = (train_f - mu) / sd
    te = (test_f - mu) / sd
    means = torch.stack([tr[train_y == c].mean(0) for c in range(5)])
    pred = torch.cdist(te, means).argmin(1)
    acc = (pred == test_y).float().mean().item()
    # margin
    e = Fn.normalize(te, dim=1)
    mn = Fn.normalize(means, dim=1)
    sims = e @ mn.T
    correct = torch.gather(sims, 1, test_y[:, None].to(sims.device)).squeeze(1).cpu()
    wrong = sims.cpu().scatter(1, test_y[:, None].to(sims.device),
                               float("-inf")).max(1).values.cpu()
    margin = (correct - wrong).mean().item()
    g = torch.Generator().manual_seed(0)
    boots = [(correct[torch.randint(0, len(correct), (len(correct),), generator=g)] -
              wrong[torch.randint(0, len(wrong), (len(wrong),), generator=g)]).mean().item()
             for _ in range(N_BOOT)]
    ci_lo = float(np.quantile(boots, 0.025))
    ci_hi = float(np.quantile(boots, 0.975))
    return {"acc_painting": round(acc, 4), "margin": round(margin, 4),
            "ci95": [round(ci_lo, 4), round(ci_hi, 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rep = args.rep
    device = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()

    # pick paintings
    g = torch.Generator().manual_seed(0)
    pick = {"train": [], "holdout": [], "latents": {}}
    for si, s in enumerate(STYLES):
        d = torch.load(str(POOL / f"{si:02d}_{s}.pt"), map_location="cpu",
                       weights_only=False)
        tr, ho = [], []
        for i, pid in enumerate(d["ids"]):
            img = IMG_ROOT / s / Path(str(d["files"][i])).name
            if not img.exists():
                continue
            item = (si, str(img), i)
            (tr if _h10(str(pid)) < 9 else ho).append(item)
        take_tr = torch.randperm(len(tr), generator=g)[:N_TRAIN].tolist()
        take_ho = torch.randperm(len(ho), generator=g)[:N_HOLD].tolist()
        pick["train"] += [tr[i] for i in take_tr]
        pick["holdout"] += [ho[i] for i in take_ho]
        pick["latents"][s] = d["latents"]

    # train features
    fn = EXTRACT[rep]
    tr_f, tr_y = [], []
    for si, img_path, lat_idx in pick["train"]:
        z = pick["latents"][STYLES[si]][lat_idx:lat_idx + 1].float().to(device)
        rgb = load_rgb(img_path, 512).to(device)
        f = fn(rgb, z, device)
        tr_f.append(f.half().cpu())
        tr_y.append(si)
        del f, rgb, z
        torch.cuda.empty_cache()
    tr_f = torch.cat(tr_f).float()
    tr_y = torch.tensor(tr_y)
    print(f"[{rep}] train: {tr_f.shape} {time.time()-t0:.0f}s", flush=True)

    # holdout features
    ho_f, ho_y = [], []
    for si, img_path, lat_idx in pick["holdout"]:
        z = pick["latents"][STYLES[si]][lat_idx:lat_idx + 1].float().to(device)
        rgb = load_rgb(img_path, 512).to(device)
        f = fn(rgb, z, device)
        ho_f.append(f.half().cpu())
        ho_y.append(si)
        del f, rgb, z
        torch.cuda.empty_cache()
    ho_f = torch.cat(ho_f).float()
    ho_y = torch.tensor(ho_y)
    print(f"[{rep}] holdout: {ho_f.shape} {time.time()-t0:.0f}s", flush=True)

    # painting-level mean (64 patches per painting)
    n_tr_p = tr_f.shape[0] // 64
    n_ho_p = ho_f.shape[0] // 64
    tr_f = tr_f.reshape(n_tr_p, 64, -1).mean(1)
    ho_f = ho_f.reshape(n_ho_p, 64, -1).mean(1)
    tr_y = tr_y[:n_tr_p]
    ho_y = ho_y[:n_ho_p]

    result = ncm_acc(tr_f, tr_y, ho_f, ho_y)

    # gray retention (RGB spaces)
    if rep in GRAY_EXTRACT:
        gfn = GRAY_EXTRACT[rep]
        gray_accs = []
        for si, img_path, lat_idx in pick["holdout"]:
            rgb = load_rgb(img_path, 512).to(device)
            if rgb.dim() == 3:
                rgb = rgb.unsqueeze(0)
            f = gfn(rgb, device)
            gray_accs.append(f.cpu())
            del f, rgb
            torch.cuda.empty_cache()
        ho_g = torch.cat(gray_accs).float().cpu()
        n_ho_g = ho_g.shape[0] // 64
        ho_g = ho_g.reshape(n_ho_g, 64, -1).mean(1)
        g_result = ncm_acc(tr_f, tr_y, ho_g, ho_y)
        result["gray_acc"] = g_result["acc_painting"]
        result["gray_retention"] = round(
            g_result["acc_painting"] / max(result["acc_painting"], 1e-6), 3)

    print(f"[{rep}] {result}", flush=True)
    (OUT / f"scan_{rep}.json").write_text(json.dumps(result, indent=1))
    print(f"saved -> {OUT / f'scan_{rep}.json'}", flush=True)


if __name__ == "__main__":
    main()
