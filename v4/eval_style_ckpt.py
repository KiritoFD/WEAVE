"""单个 checkpoint 的统一测试套件（不训练）。

每个 ckpt 都跑同一套口径，回答"Style-ID 到底有没有控制生成分布"：
1. 分桶去噪误差：真 / 空 / 错 ID（复用 audit_cond，eps 空间可比）；
2. 固定初始噪声、每种目标 ID 生成一批，用**独立**分类器测目标 Top-1 与 margin；
3. 交换同一内容 latent 的目标 ID（SDEdit），测输出差异与目标 Top-1；
4. 生成多样性 / 训练集最近邻（防记忆）；
5. 可选 DINO 风格选择性（需要 VAE 解码，慢，默认关闭）。

用法（路径一律相对项目根）:
  python -m v4.eval_style_ckpt --ckpt runs/v4/style_breakthrough/E1/step_04000.pt \
      --clf runs/v4/style_clf/alt.pt --out runs/v4/style_breakthrough/E1/eval_4000.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audit_cond import bucket_edges, load_model, loss_and_effect
from .config import V4Config
from .data import STYLE_NAMES, load_bank
from .diffusion import Schedule, ddim_sample, sdedit
from .style_classifier import load_classifier


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def classify_at_low_noise(clf, x0: torch.Tensor, schedule: Schedule, *,
                          t_hi: int = 128, reps: int = 2, seed: int = 7) -> dict:
    """把干净 latent 加噪到低噪声区间再分类（与分类器训练分布对齐）。"""
    device = x0.device
    gen = torch.Generator(device=device).manual_seed(seed)
    n = x0.shape[0]
    n_cls = int(schedule.timesteps)
    logits_all, labels_all = [], []
    for _ in range(reps):
        t = torch.randint(0, t_hi, (n,), device=device, generator=gen)
        noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=gen)
        zt = schedule.q_sample(x0, t, noise)
        logits = clf(zt, t).float()
        logits_all.append(logits)
    return {"logits": torch.cat(logits_all, 0)}


def clf_scores(logits: torch.Tensor, target: torch.Tensor) -> dict:
    prob = logits.softmax(1)
    n = target.numel()
    rep = logits.shape[0] // n
    target = target.repeat(rep)
    pred = logits.argmax(1)
    top1 = float((pred == target).float().mean())
    true_p = prob.gather(1, target[:, None]).squeeze(1).view(rep, n)
    other = prob.masked_fill(F.one_hot(target, prob.shape[1]).bool(), -1.0)
    other_p = other.max(1).values.view(rep, n)
    margin = float((true_p - other_p).mean())
    confusion = torch.zeros(prob.shape[1], prob.shape[1], dtype=torch.long)
    for a, b in zip(target.cpu(), pred.cpu()):
        confusion[a, b] += 1
    return {"top1": round(top1, 4), "margin": round(margin, 4),
            "pred_hist": confusion.sum(0).tolist(),
            "confusion": confusion.tolist()}


@torch.no_grad()
def diversity_and_nn(gen: torch.Tensor, sids: torch.Tensor, train_lat: torch.Tensor,
                     num_styles: int) -> dict:
    out = {}
    pair = []
    for s in range(num_styles):
        z = gen[sids == s].flatten(1)
        if z.shape[0] < 2:
            continue
        # 同风格内两两 RMS，抽样避免 O(n^2) 爆炸
        take = torch.randperm(z.shape[0])[: min(24, z.shape[0])]
        zz = z[take]
        d = torch.cdist(zz, zz)
        iu = torch.triu_indices(zz.shape[0], zz.shape[0], offset=1)
        pair.append(float(d[iu[0], iu[1]].mean()))
    out["within_style_pairwise_rms"] = round(float(np.mean(pair)), 4) if pair else None
    g = F.normalize(gen.flatten(1), dim=1)
    best = []
    for start in range(0, train_lat.shape[0], 512):
        chunk = F.normalize(train_lat[start:start + 512].flatten(1), dim=1)
        best.append((g @ chunk.T).max(1).values)
    sim = torch.cat(best)
    out["nn_cos_mean"] = round(float(sim.mean()), 4)
    out["nn_cos_p95"] = round(float(sim.quantile(0.95)), 4)
    out["frac_above_0.98"] = round(float((sim > 0.98).float().mean()), 4)
    return out


@torch.no_grad()
def decode_dino_style(gen: torch.Tensor, sids: torch.Tensor, ref_lat: torch.Tensor,
                      ref_sid: torch.Tensor, num_styles: int, *, cache_dir: str,
                      per_style_ref: int = 8) -> dict:
    """可选：VAE 解码 + DINOv2-small，测最近风格质心的 Top-1 与 margin。"""
    import os
    os.environ.setdefault("HF_HOME", cache_dir)
    from transformers import AutoModel
    from utils.inference import decode_latent, load_vae

    vae = load_vae("cuda", model_id="ema", cache_dir=cache_dir)
    dino = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()

    def embed(z: torch.Tensor) -> torch.Tensor:
        feats = []
        for start in range(0, z.shape[0], 8):
            imgs = decode_latent(vae, z[start:start + 8], "cuda")
            out = dino(pixel_values=imgs * 2.0 - 1.0)
            feats.append(F.normalize(out.last_hidden_state[:, 0].float(), dim=1))
        return torch.cat(feats, 0)

    idx_parts, ref_labels = [], []
    for s in range(num_styles):
        cand = torch.nonzero(ref_sid == s, as_tuple=False).flatten()[:per_style_ref]
        idx_parts.append(cand)
        ref_labels.append(torch.full((cand.numel(),), s, dtype=torch.long))
    ref_idx = torch.cat(idx_parts)
    ref_labels = torch.cat(ref_labels)
    ref = embed(ref_lat[ref_idx].to(gen.device))
    cents = torch.stack([ref[ref_labels == s].mean(0) for s in range(num_styles)])
    cents = F.normalize(cents, dim=1)
    emb = embed(gen)
    sim = emb @ cents.T
    pred = sim.argmax(1)
    true_sim = sim.gather(1, sids[:, None]).squeeze(1)
    other = sim.masked_fill(F.one_hot(sids, num_styles).bool(), -1.0).max(1).values
    return {"dino_style_top1": round(float((pred == sids).float().mean()), 4),
            "dino_style_margin": round(float((true_sim - other).mean()), 4),
            "dino_mean_true_sim": round(float(true_sim.mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clf", required=True, help="独立评估分类器（不要用指导分类器）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="runs/v4/arms/B/indices.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gen-per-style", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=20)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--swap-strength", type=float, default=0.45)
    ap.add_argument("--swap-content", type=int, default=8)
    ap.add_argument("--guide-clf", default="", help="swap 阶段可选的引导分类器")
    ap.add_argument("--guide-scale", type=float, default=0.0)
    ap.add_argument("--guide-t-max", type=int, default=700)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--vae-cache", default="eval_cache/hf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = _device(args.device)
    model, cfg, info = load_model(args.ckpt, "", device)
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device.type)
    idx = torch.load(args.indices, map_location="cpu", weights_only=False)
    bank = load_bank(cfg.packed_dir)
    val = bank.latents.index_select(0, idx["val_idx"].long())
    val_sid = bank.style_ids.index_select(0, idx["val_idx"].long())
    train_lat = bank.latents.index_select(0, idx["train_idx"].long())

    report = {"ckpt": args.ckpt, "step": info.get("step"), "params": info.get("params"),
              "prediction_type": cfg.prediction_type, "cond_mode": cfg.cond_mode,
              "clf": args.clf, "n_val": int(val.shape[0])}

    lt = val[: min(256, val.shape[0])].to(device)
    ls = val_sid[: min(256, val.shape[0])].to(device)
    report["denoise"] = loss_and_effect(model, cfg, schedule, lt, ls,
                                        reps=args.reps, edges=bucket_edges(cfg, args.bins),
                                        batch=128, seed=args.seed)
    report["denoise"]["verdict"] = {
        "buckets_with_repeatable_gain": [r["bucket"] for r in report["denoise"]["buckets"]
                                         if r["ci95_wrong"] > 0 and r["win_wrong"] >= 0.75
                                         and r["all_reps_positive_wrong"]
                                         and r["rel_wrong_pct"] >= 0.5]}

    sids = torch.arange(cfg.num_styles, device=device).repeat_interleave(args.gen_per_style)
    gen = torch.Generator(device=device).manual_seed(args.seed + 11)
    z = torch.randn(sids.numel(), cfg.latent_channels, 64, 64, device=device, generator=gen)
    out = ddim_sample(model, z, sids, schedule, t_start=cfg.timesteps - 1,
                      num_steps=args.ddim_steps, cfg_scale=args.cfg_scale,
                      prediction_type=cfg.prediction_type)
    clf, _ = load_classifier(args.clf, device)
    scores = clf_scores(classify_at_low_noise(clf, out, schedule, seed=args.seed)["logits"], sids)
    report["gen"] = {**scores,
                     **diversity_and_nn(out, sids, train_lat.to(device), cfg.num_styles)}
    report["gen"]["cfg_scale"] = args.cfg_scale
    report["gen"]["ddim_steps"] = args.ddim_steps
    report["gen"]["per_style"] = args.gen_per_style

    content = val[: args.swap_content].to(device)
    guide = {}
    if args.guide_clf and args.guide_scale != 0.0:
        from .style_classifier import make_guidance_fn
        guide = {sid: make_guidance_fn(args.guide_clf, sid, device, t_min=0, t_max=args.guide_t_max)
                 for sid in range(cfg.num_styles)}
    outs = []
    for sid in range(cfg.num_styles):
        g = torch.Generator(device=device).manual_seed(args.seed + 21)
        outs.append(sdedit(model, content, torch.full((content.shape[0],), sid, device=device),
                           schedule, strength=args.swap_strength,
                           num_steps=args.ddim_steps, cfg_scale=args.cfg_scale,
                           prediction_type=cfg.prediction_type,
                           guidance_fn=guide.get(sid), guidance_scale=args.guide_scale,
                           generator=g))
    outs = torch.stack(outs, 0)
    flat = outs.reshape(-1, *outs.shape[2:])
    target = torch.arange(cfg.num_styles, device=device).repeat_interleave(content.shape[0])
    swap_scores = clf_scores(classify_at_low_noise(clf, flat, schedule, seed=args.seed)["logits"],
                             target)
    pair = []
    for a in range(cfg.num_styles):
        for b in range(a + 1, cfg.num_styles):
            pair.append(float((outs[a] - outs[b]).pow(2).mean().sqrt()))
    report["swap"] = {**swap_scores,
                      "pairwise_style_rms": round(float(np.mean(pair)), 4),
                      "to_input_rms": round(float(np.mean(
                          [float((outs[s] - content).pow(2).mean().sqrt().item())
                           for s in range(cfg.num_styles)]
                      )), 4),
                      "strength": args.swap_strength,
                      "guide_scale": args.guide_scale}

    if args.decode:
        report["dino"] = decode_dino_style(out.cpu(), sids.cpu(),
                                           train_lat, bank.style_ids.index_select(
                                               0, idx["train_idx"].long()),
                                           cfg.num_styles, cache_dir=args.vae_cache)
        report["dino"]["note"] = "DINO-S 正式指标由最终评估脚本换算；这里只报选择性 Top-1/margin"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[eval] {args.ckpt} step={report['step']} pred={cfg.prediction_type} "
          f"cond={cfg.cond_mode}")
    print(f"  denoise d_wrong={report['denoise']['overall']['d_wrong']:+.5f} "
          f"({report['denoise']['overall']['rel_wrong_pct']:+.2f}%) "
          f"gain_buckets={report['denoise']['verdict']['buckets_with_repeatable_gain']}")
    print(f"  gen  top1={report['gen']['top1']:.4f} margin={report['gen']['margin']:+.3f} "
          f"pair_rms={report['gen']['within_style_pairwise_rms']} "
          f"nn_p95={report['gen']['nn_cos_p95']:.3f}")
    print(f"  swap top1={report['swap']['top1']:.4f} margin={report['swap']['margin']:+.3f} "
          f"pair_rms={report['swap']['pairwise_style_rms']:.3f}")
    if "dino" in report:
        print(f"  dino top1={report['dino']['dino_style_top1']:.4f} "
              f"margin={report['dino']['dino_style_margin']:+.3f}")


if __name__ == "__main__":
    main()
