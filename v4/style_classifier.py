"""噪声条件风格分类器 C_phi(z_t, t) -> p(s|z_t)。

用途有两个，且必须分开：
1. 训练一个 1–3M 参数的噪声条件 CNN（Style-ID 监督），供后面做 classifier guidance；
2. 另训一个**结构/初始化都不同**的独立评估器，避免生成器只学会骗指导分类器。

训练协议（与扩散臂一致）：真实 5 风格×1000 latent，输入 4×64×64，随机 t，
CE 损失，batch 256，五风格均衡采样。验证按噪声区间分桶报 Top-1，并给混淆矩阵、
训练/验证差距、按内容簇分组准确率。

用法（路径一律相对项目根）:
  python -m v4.style_classifier train --arch main --out runs/v4/style_clf/main.pt
  python -m v4.style_classifier train --arch alt  --out runs/v4/style_clf/alt.pt
  python -m v4.style_classifier eval  --ckpt runs/v4/style_clf/main.pt --out runs/v4/style_clf/main_val.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V4Config
from .data import STYLE_NAMES, load_bank
from .diffusion import Schedule, pred_to_x0_eps  # noqa: F401  (保持与环境一致的导入面)
from .model import timestep_embedding


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class TimeFiLM(nn.Module):
    def __init__(self, emb_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(emb_dim, channels * 2)
        # 小非零初始化：避免与扩散臂 A 同样的"零初始化死条件"，但也不让时间信号开局淹没风格信号
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class CondBlock(nn.Module):
    """两个 3×3 卷积 + 时间 FiLM，无残差（浅层分类器不需要深残差）。"""

    def __init__(self, c_in: int, c_out: int, emb_dim: int) -> None:
        super().__init__()
        g = 8 if c_out % 8 == 0 else 1
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, c_out)
        self.film1 = TimeFiLM(emb_dim, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, c_out)
        self.film2 = TimeFiLM(emb_dim, c_out)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.film1(F.silu(self.norm1(self.conv1(h))), emb)
        h = self.film2(F.silu(self.norm2(self.conv2(h))), emb)
        return h


class NoiseStyleCNN(nn.Module):
    """指导用主干：4 尺度，时间 FiLM 注入，约 2–3M 参数。"""

    arch = "main"

    def __init__(self, num_styles: int, timesteps: int = 1000,
                 channels: tuple[int, ...] = (48, 96, 192, 256), emb_dim: int = 64) -> None:
        super().__init__()
        self.num_styles = num_styles
        self.timesteps = timesteps
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d(4, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(channels):
            self.blocks.append(CondBlock(ch, ch, emb_dim))
            if i + 1 < len(channels):
                self.downs.append(nn.Conv2d(ch, channels[i + 1], 3, stride=2, padding=1))
        self.head_norm = nn.GroupNorm(8, channels[-1])
        self.head = nn.Linear(channels[-1], num_styles)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t.float(), self.emb_dim).to(z.dtype))
        x = self.stem(z)
        for i, blk in enumerate(self.blocks):
            x = blk(x, emb)
            if i < len(self.downs):
                x = self.downs[i](x)
        x = F.adaptive_avg_pool2d(F.silu(self.head_norm(x)), 1).flatten(1)
        return self.head(x)


class AltNoiseStyleCNN(nn.Module):
    """独立评估器：3 尺度、更浅、时间做成第 5 个输入通道，不用 FiLM。"""

    arch = "alt"

    def __init__(self, num_styles: int, timesteps: int = 1000,
                 channels: tuple[int, ...] = (64, 128, 256)) -> None:
        super().__init__()
        self.num_styles = num_styles
        self.timesteps = timesteps
        g = 8
        self.stem = nn.Conv2d(5, channels[0], 3, padding=1)
        self.stages = nn.ModuleList()
        for i, ch in enumerate(channels):
            self.stages.append(nn.Sequential(
                nn.GroupNorm(g, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                nn.GroupNorm(g, ch), nn.SiLU(), nn.MaxPool2d(2),
            ))
            if i + 1 < len(channels):
                self.stages.append(nn.Conv2d(ch, channels[i + 1], 3, padding=1))
        self.head = nn.Linear(channels[-1] * 2, num_styles)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = (t.float() / self.timesteps).view(-1, 1, 1, 1).expand(-1, 1, z.shape[-2], z.shape[-1])
        x = self.stem(torch.cat([z, tt.to(z.dtype)], dim=1))
        for st in self.stages:
            x = st(x)
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.head(torch.cat([avg, mx], dim=1))


ARCHS = {"main": NoiseStyleCNN, "alt": AltNoiseStyleCNN}


def build_arch(arch: str, cfg: V4Config) -> nn.Module:
    if arch == "main":
        return NoiseStyleCNN(cfg.num_styles, cfg.timesteps)
    if arch == "alt":
        return AltNoiseStyleCNN(cfg.num_styles, cfg.timesteps)
    raise SystemExit(f"unknown arch {arch!r}")


def balanced_labels(batch: int, num_styles: int, device: torch.device,
                    generator: torch.Generator) -> torch.Tensor:
    per = batch // num_styles
    base = torch.arange(num_styles, device=device).repeat(per)
    extra = batch - base.numel()
    if extra > 0:
        base = torch.cat([base, torch.randint(0, num_styles, (extra,), device=device,
                                              generator=generator)])
    return base[torch.randperm(base.numel(), device=device, generator=generator)]


def sample_batch(lat: torch.Tensor, sid: torch.Tensor, labels: torch.Tensor,
                 schedule: Schedule, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按给定类别标签等量取样本（每类内部随机），共用同一套 t/noise 采样。"""
    picks = []
    for s in range(int(sid.max().item()) + 1):
        cls_idx = torch.nonzero(sid == s, as_tuple=False).flatten()
        take = int((labels == s).sum())
        if take == 0:
            continue
        sel = torch.randint(0, cls_idx.numel(), (take,), device=lat.device, generator=generator)
        picks.append(cls_idx.index_select(0, sel))
    idx = torch.cat(picks)
    t = torch.randint(0, schedule.timesteps, (idx.numel(),), device=lat.device, generator=generator)
    noise = torch.randn((idx.numel(), *lat.shape[1:]), device=lat.device,
                        dtype=lat.dtype, generator=generator)
    zt = schedule.q_sample(lat.index_select(0, idx), t, noise)
    return zt, t, sid.index_select(0, idx)


@torch.no_grad()
def evaluate_clf(model: nn.Module, lat: torch.Tensor, sid: torch.Tensor, schedule: Schedule,
                 *, reps: int, batch: int, nb: int, seed: int,
                 clusters: np.ndarray | None = None) -> dict:
    """按噪声区间分桶 Top-1；混淆矩阵 / margin / 内容簇分组准确率。"""
    device = lat.device
    model.eval()
    n = lat.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    t_all = torch.randint(0, schedule.timesteps, (reps, n), device=device, generator=gen)
    correct = torch.zeros(reps, n, device=device)
    margins = torch.zeros(reps, n, device=device)
    confusion = torch.zeros(int(sid.max().item()) + 1, int(sid.max().item()) + 1,
                            dtype=torch.long, device=device)
    for r in range(reps):
        noise = torch.randn(lat.shape, device=device, dtype=lat.dtype, generator=gen)
        zt = schedule.q_sample(lat, t_all[r], noise)
        pred = torch.zeros(n, dtype=torch.long, device=device)
        marg = torch.zeros(n, device=device)
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(zt[sl], t_all[r][sl])
            logits = logits.float()
            pred[sl] = logits.argmax(1)
            prob = logits.softmax(1)
            true = sid[sl]
            marg[sl] = prob.gather(1, true[:, None]).squeeze(1) - \
                prob.masked_fill(
                    torch.nn.functional.one_hot(true, prob.shape[1]).bool(), -1.0).max(1).values
        correct[r] = (pred == sid).float()
        margins[r] = marg
        for a, b in zip(sid.cpu().numpy(), pred.cpu().numpy()):
            confusion[a, b] += 1

    t_np = t_all.cpu().numpy()
    bounds = [k * schedule.timesteps // nb for k in range(nb + 1)]
    buckets = []
    for k in range(nb):
        mask = (t_np >= bounds[k]) & (t_np < bounds[k + 1])
        if not mask.any():
            continue
        acc = float(correct.cpu().numpy()[mask].mean())
        mid = max(1, (bounds[k] + bounds[k + 1]) // 2)
        snr = max(float(schedule.snr(mid)), 1e-12)
        buckets.append({
            "t_lo": bounds[k], "t_hi": bounds[k + 1] - 1,
            "log10_snr": round(math.log10(snr), 3),
            "top1": round(acc, 4), "n": int(mask.sum()),
            "margin": round(float(margins.cpu().numpy()[mask].mean()), 4),
        })

    out = {
        "top1": round(float(correct.mean()), 4),
        "margin": round(float(margins.mean()), 4),
        "buckets": buckets,
        "confusion": confusion.cpu().tolist(),
        "pred_hist": confusion.sum(0).cpu().tolist(),
        "per_style_recall": [round(float(confusion[s].float().mean()), 4)
                            for s in range(confusion.shape[0])],
        "n": n, "reps": reps,
    }
    if clusters is not None:
        cl = np.asarray(clusters)
        rows = []
        for c in np.unique(cl):
            sel = cl == c
            rows.append({"cluster": int(c), "n": int(sel.sum()),
                         "top1": round(float(correct.cpu().numpy()[:, sel].mean()), 4)})
        rows.sort(key=lambda r: r["top1"])
        out["by_content_cluster"] = {"worst": rows[:3], "best": rows[-3:],
                                     "n_clusters": int(len(rows)),
                                     "mean": round(float(np.mean([r["top1"] for r in rows])), 4)}
    return out


@torch.no_grad()
def dataset_accuracy(model, lat, sid, schedule, *, n: int, reps: int, seed: int) -> float:
    gen = torch.Generator(device=lat.device).manual_seed(seed)
    idx = torch.randint(0, lat.shape[0], (min(n, lat.shape[0]),), device=lat.device, generator=gen)
    sub_lat = lat.index_select(0, idx)
    sub_sid = sid.index_select(0, idx)
    res = evaluate_clf(model, sub_lat, sub_sid, schedule, reps=reps, batch=128, nb=1,
                       seed=seed + 1)
    return float(res["top1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--arch", default="main", choices=sorted(ARCHS))
    tr.add_argument("--out", required=True)
    tr.add_argument("--indices", default="runs/v4/arms/B/indices.pt")
    tr.add_argument("--steps", type=int, default=3000)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log-every", type=int, default=100)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--val-reps", type=int, default=2)
    tr.add_argument("--bins", type=int, default=10)
    tr.add_argument("--device", default="auto")

    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--indices", default="runs/v4/arms/B/indices.pt")
    ev.add_argument("--val-reps", type=int, default=8)
    ev.add_argument("--bins", type=int, default=10)
    ev.add_argument("--device", default="auto")

    args = ap.parse_args()
    device = _device(args.device)
    cfg = V4Config()
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device.type)
    bank = load_bank(cfg.packed_dir)
    idx = torch.load(args.indices, map_location="cpu", weights_only=False)
    tr_idx = idx["train_idx"].long()
    va_idx = idx["val_idx"].long()
    train_lat = bank.latents.index_select(0, tr_idx).to(device)
    train_sid = bank.style_ids.index_select(0, tr_idx).to(device)
    val_lat = bank.latents.index_select(0, va_idx).to(device)
    val_sid = bank.style_ids.index_select(0, va_idx).to(device)
    val_clusters = idx["cluster"].numpy()[va_idx.numpy()] if "cluster" in idx else None

    if args.cmd == "eval":
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = build_arch(payload.get("arch", "main"), cfg).to(device)
        model.load_state_dict({k: v.float() for k, v in payload["state"].items()}, strict=True)
        rep = evaluate_clf(model, val_lat, val_sid, schedule, reps=args.val_reps,
                           batch=128, nb=args.bins, seed=1234, clusters=val_clusters)
        rep["ckpt"] = args.ckpt
        rep["arch"] = payload.get("arch")
        rep["train_metrics"] = payload.get("metrics")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[clf] {args.ckpt} arch={rep['arch']} val_top1={rep['top1']:.4f} "
              f"margin={rep['margin']:.3f}")
        for b in rep["buckets"]:
            print(f"  t={b['t_lo']:>3}-{b['t_hi']:<3} logSNR={b['log10_snr']:>6.2f} "
                  f"top1={b['top1']:.4f} margin={b['margin']:+.3f}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_arch(args.arch, cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[clf] arch={args.arch} params={params/1e6:.2f}M steps={args.steps} "
          f"batch={args.batch} lr={args.lr} train={train_lat.shape[0]} val={val_lat.shape[0]}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = {"val_top1": -1.0, "step": -1}
    t0 = time.perf_counter()
    hist = []
    for step in range(1, args.steps + 1):
        model.train()
        labels = balanced_labels(args.batch, cfg.num_styles, device, gen)
        zt, t, y = sample_batch(train_lat, train_sid, labels, schedule, gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(zt, t)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                acc = (logits.float().argmax(1) == y).float().mean()
            print(f"[clf] step={step} loss={float(loss.detach()):.4f} batch_acc={float(acc):.3f} "
                  f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate_clf(model, val_lat, val_sid, schedule, reps=args.val_reps,
                               batch=128, nb=args.bins, seed=args.seed + 99,
                               clusters=val_clusters)
            train_acc = dataset_accuracy(model, train_lat, train_sid, schedule,
                                         n=1500, reps=1, seed=args.seed + 77)
            rec = {"step": step, "loss": round(float(loss.detach()), 4),
                   "train_top1": round(train_acc, 4), **{k: val[k] for k in ("top1", "margin")},
                   "gap": round(train_acc - val["top1"], 4),
                   "buckets": val["buckets"], "per_style_recall": val["per_style_recall"],
                   "by_content_cluster": val.get("by_content_cluster")}
            hist.append(rec)
            print(f"[clf] step={step} val_top1={val['top1']:.4f} train_top1={train_acc:.4f} "
                  f"gap={rec['gap']:+.3f} margin={val['margin']:.3f}", flush=True)
            if val["top1"] > best["val_top1"]:
                best = {"val_top1": val["top1"], "step": step}
                torch.save({"arch": args.arch, "state": {k: v.half() for k, v in model.state_dict().items()},
                            "num_styles": cfg.num_styles, "timesteps": cfg.timesteps,
                            "steps": step, "params": params, "metrics": rec}, out_path)
        if step % 1000 == 0:
            (out_path.parent / (out_path.stem + "_history.json")).write_text(
                json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")

    final = hist[-1] if hist else {}
    print(json.dumps({"arch": args.arch, "params": params, "best": best,
                      "final": {k: final.get(k) for k in ("step", "train_top1", "top1", "gap")},
                      "out": str(out_path)}, indent=2, ensure_ascii=False))


def load_classifier(ckpt: str, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = V4Config()
    model = build_arch(payload.get("arch", "main"), cfg).to(device).eval()
    model.load_state_dict({k: v.float() for k, v in payload["state"].items()}, strict=True)
    return model, payload


def make_guidance_fn(ckpt: str, target_sid: int, device: torch.device,
                     *, t_min: int = 0, t_max: int = 999):
    """返回 ``guidance_fn(z, t) -> dCE/dz``；只在 [t_min, t_max] 内生效。"""
    model, _ = load_classifier(ckpt, device)

    def fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(z, t)
            loss = F.cross_entropy(logits.float(), torch.full((z.shape[0],), target_sid,
                                                              device=z.device, dtype=torch.long))
            grad, = torch.autograd.grad(loss, z)
        gate = ((t >= t_min) & (t <= t_max)).float().view(-1, 1, 1, 1)
        return grad * gate

    return fn


if __name__ == "__main__":
    main()
