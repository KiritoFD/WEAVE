"""V14 scale scan — 3060 remote version (paths adapted, DINO via transformers)."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

REPO = Path("I:/Latent_Style/WEAVE")
POOL = Path("I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
IMG_ROOT = Path("I:/Latent_Style/WEAVE/datasets/wikiart_new_512")
DINO_SNAP = Path("I:/Latent_Style/eval_cache/hf/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056")
OUT = REPO / "runs/v14"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
N_TRAIN, N_HOLD, N_BOOT = 200, 80, 1000

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
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = kx.transpose(2, 3)
    return (Fn.conv2d(x, kx, padding=1).square() + Fn.conv2d(x, ky, padding=1).square()).sqrt()

def dwt_levels(z):
    ll1 = Fn.avg_pool2d(z, 2)
    h1 = z - Fn.interpolate(ll1, size=z.shape[-2:], mode="nearest")
    ll2 = Fn.avg_pool2d(ll1, 2)
    h2 = ll1 - Fn.interpolate(ll2, size=ll1.shape[-2:], mode="nearest")
    e2 = sobel_mag(ll2.mean(1, keepdim=True))
    return {"H2": h2, "LL2": ll2, "H1": h1, "LL1": ll1, "edge2": e2}

def s0_latent_stats(z):
    ll32, ll16 = Fn.avg_pool2d(z, 2), Fn.avg_pool2d(z, 4)
    parts = [ll16.flatten(1), ll32.flatten(1)]
    for ll in (ll16, ll32):
        parts += [(ll[..., :, 1:] - ll[..., :, :-1]).flatten(1),
                  (ll[..., 1:, :] - ll[..., :-1, :]).flatten(1)]
    return torch.cat(parts, 1)

def s1_dwt_tokens(z):
    d = dwt_levels(z)
    cat = torch.cat([Fn.adaptive_avg_pool2d(d["H2"], 8),
                     Fn.adaptive_avg_pool2d(d["LL2"], 8),
                     Fn.adaptive_avg_pool2d(d["edge2"], 8)], 1)  # (1,13,8,8)
    return cat[0].reshape(cat.shape[1], -1).T                    # (64,13)

def s2_dwt_large(rgb):
    x = Fn.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
    z = x.mean(1, keepdim=True).repeat(1, 4, 1, 1)
    return s1_dwt_tokens(z)

_CNN = None
def _cnn(device):
    global _CNN
    if _CNN is None:
        layers = []
        cin = 3
        for cout, stride in ((32,1),(64,2),(64,1),(128,2),(128,1)):
            layers += [torch.nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), torch.nn.SiLU(True)]
            cin = cout
        _CNN = torch.nn.Sequential(*layers).to(device).eval()
    return _CNN

def rgb_cnn(rgb, device):
    x = rgb if rgb.dim() == 4 else rgb.unsqueeze(0)
    return Fn.adaptive_avg_pool2d(_cnn(device)(x), 8).flatten(2).squeeze(0).T

def s3_rgb64(rgb, device):
    return rgb_cnn(Fn.interpolate(rgb, size=(64, 64), mode="bilinear", align_corners=False), device)

def s4_rgb128(rgb, device):
    return rgb_cnn(Fn.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False), device)

def s5_rgb224(rgb, device):
    return rgb_cnn(Fn.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False), device)

_DINO = None
def _dino(device):
    global _DINO
    if _DINO is None:
        from transformers import AutoModel
        _DINO = AutoModel.from_pretrained(str(DINO_SNAP)).to(device).eval()
    return _DINO

def s6_dino(rgb, device):
    m = _dino(device)
    x = Fn.interpolate(rgb if rgb.dim() == 4 else rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
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

def main():
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    pick = {"train": [], "holdout": [], "latents": {}}
    g = torch.Generator().manual_seed(0)
    for si, s in enumerate(STYLES):
        d = torch.load(str(POOL / f"{si:02d}_{s}.pt"), map_location="cpu", weights_only=False)
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
    print("counts:", len(pick["train"]), len(pick["holdout"]), flush=True)

    reps = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S8"]
    data = {}
    for split in ("train", "holdout"):
        items = pick[split]
        labels = []
        feats = {r: [] for r in reps}
        for n_done, (si, img_path, lat_idx) in enumerate(items):
            z = pick["latents"][STYLES[si]][lat_idx:lat_idx+1].float().to(device)
            rgb = load_rgb(img_path, 512).to(device)
            if rgb.dim() == 3:
                rgb = rgb.unsqueeze(0)
            for r in reps:
                if r == "S0": f = s0_latent_stats(z)
                elif r == "S1": f = s1_dwt_tokens(z)
                elif r == "S2": f = s2_dwt_large(rgb)
                elif r == "S3": f = s3_rgb64(rgb, device)
                elif r == "S4": f = s4_rgb128(rgb, device)
                elif r == "S5": f = s5_rgb224(rgb, device)
                elif r == "S6": f = s6_dino(rgb, device)
                elif r == "S8": f = s8_multiscale(rgb, z, device)
                feats[r].append(f.half().cpu())
                del f
                torch.cuda.empty_cache()
            labels.append(si)
            if (n_done + 1) % 50 == 0:
                print(f"  {split} {n_done+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        for r in reps:
            data[(split, r)] = torch.cat(feats[r]).float()
        data[(split, "labels")] = torch.tensor(labels)

    report = {"results": {}}
    for r in reps:
        tr, ho = data[("train", r)], data[("holdout", r)]
        tr_y, ho_y = data[("train", "labels")], data[("holdout", "labels")]
        mu, sd = tr.mean(0), tr.std(0).clamp_min(1e-4)
        tr_n, ho_n = (tr - mu) / sd, (ho - mu) / sd
        means = torch.stack([tr_n[tr_y == c].mean(0) for c in range(5)])
        pred = torch.cdist(ho_n, means).argmin(1)
        acc = (pred == ho_y).float().mean().item()
        e, mn = Fn.normalize(ho_n, dim=1), Fn.normalize(means, dim=1)
        sims = e @ mn.T
        correct = torch.gather(sims, 1, ho_y[:, None].to(sims.device)).squeeze(1).cpu()
        wrong = sims.cpu().scatter(1, ho_y[:, None].to(sims.device), float("-inf")).max(1).values.cpu()
        margin = (correct - wrong).mean().item()
        g2 = torch.Generator().manual_seed(0)
        boots = [(correct[torch.randint(0, len(correct), (len(correct),), generator=g2)] -
                  wrong[torch.randint(0, len(wrong), (len(wrong),), generator=g2)]).mean().item()
                 for _ in range(N_BOOT)]
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        report["results"][r] = {"acc_painting": round(acc, 4), "margin": round(margin, 4),
                                "ci95": [round(ci_lo, 4), round(ci_hi, 4)]}
        print(f"[scan] {r}: {report['results'][r]}", flush=True)

    report["gray_retention"] = {}
    for r in ("S3", "S4", "S5", "S6"):
        acc_orig = report["results"][r]["acc_painting"]
        gray_accs = []
        fn_map = {"S3": s3_rgb64, "S4": s4_rgb128, "S5": s5_rgb224, "S6": s6_dino}
        for split in ("train", "holdout"):
            items = pick[split]
            labs = data[(split, "labels")]
            gf = []
            for n_done, (si, img_path, lat_idx) in enumerate(items):
                rgb = load_rgb(img_path, 512).to(device)
            if rgb.dim() == 3:
                rgb = rgb.unsqueeze(0)
                g = to_gray(rgb)
                gf.append(fn_map[r](g, device).half().cpu())
            gft = torch.cat(gf).float()
            tr_r = data[(split, r)]
            mu, sd = tr_r.mean(0), tr_r.std(0).clamp_min(1e-4)
            gft_n = (gft - mu) / sd
            means = torch.stack([tr_r[tr_r_y := data[(split, "labels")] == c].mean(0)
                                 for c in range(5)])
            pred = torch.cdist(gft_n, means).argmin(1)
            gray_accs.append((pred.cpu() == labs).float().mean().item())
        report["gray_retention"][r] = round(
            sum(gray_accs) / len(gray_accs) / max(acc_orig, 1e-6), 3)
    print("[scan] gray_retention", report["gray_retention"], flush=True)

    r = report["results"]
    report["verdict"] = {
        "H1_scale": bool(r["S5"]["acc_painting"] > r["S1"]["acc_painting"] + 0.10),
        "H2_dwt_vs_rgb_same_size": bool(r["S3"]["acc_painting"] > r["S1"]["acc_painting"] + 0.10),
        "dino_ceiling": r["S6"]["acc_painting"],
        "S0_baseline": r["S0"]["acc_painting"],
    }
    print("[scan] verdict", json.dumps(report["verdict"], indent=1), flush=True)
    (OUT / "scale_scan.json").write_text(json.dumps(report, indent=1))
    print("saved ->", OUT / "scale_scan.json")

if __name__ == "__main__":
    main()
