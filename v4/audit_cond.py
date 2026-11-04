"""按噪声 / SNR 分桶的条件审计（只读，不训练）。

回答一个问题：Style-ID 在**哪些噪声区间**真的降低了去噪误差，且该收益可重复？

审计项：
1. 固定验证集 + 固定噪声，逐桶 ``L_true / L_null / L_wrong`` 与配对差 ``ΔL``；
   给出 95% CI、胜率、逐 rep 均值（可重复性），不报全局平均糊弄。
2. 每层 FiLM 调制幅度（scale/shift/delta_over_h），区分时间通路与风格通路。
3. 每个桶单独 backward：Style-ID embedding / null_style / 逐层 FiLM 的梯度范数。
4. 真/错 Style-ID 的输出差异（相对 RMS）。

判据：某桶 ``ΔL_wrong > 0``、CI 下界 > 0、胜率 ≥ 0.75 且相对收益 ≥ 0.5%，才算
"正确条件在该噪声区间带来可重复收益"。EPS 与 V 预测统一换算到 eps 空间比较。

用法（路径一律相对项目根）:
  python -m v4.audit_cond --ckpt runs/v4/arms/B/best.pt --out runs/v4/diag/audit_B.json
  python -m v4.audit_cond --ckpt "" --config v4/arms/E1.json --out runs/v4/diag/audit_E1_init.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import V4Config
from .data import STYLE_NAMES, load_bank
from .diffusion import Schedule, pred_to_x0_eps, v_target
from .model import FiLM, V4UNet, count_parameters


def _rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(ckpt: str, config_path: str, device: torch.device):
    info: dict = {"ckpt": ckpt or "fresh-init", "step": 0}
    if ckpt:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = V4Config.from_mapping(payload.get("v4_config"))
        state = payload["model"]
        info.update({"step": payload.get("step"), "val": payload.get("val"),
                     "prediction_type": payload.get("prediction_type", cfg.prediction_type)})
    else:
        payload = {}
        cfg = V4Config.from_mapping(json.loads(Path(config_path).read_text(encoding="utf-8"))
                                    if config_path else None)
        state = None
        info["prediction_type"] = cfg.prediction_type
    model = V4UNet(cfg).to(device).eval()
    if state is not None:
        model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    info["params"] = count_parameters(model)
    info["cond_mode"] = cfg.cond_mode
    return model, cfg, info


def resolve_indices(ckpt: str, explicit: str) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if ckpt:
        cands.append(Path(ckpt).parent / "indices.pt")
    cands.append(Path("runs/v4/arms/B/indices.pt"))
    for p in cands:
        if p.exists():
            return p
    raise SystemExit(f"找不到划分索引，候选={[str(p) for p in cands]}")


def bucket_edges(cfg: V4Config, nb: int) -> list[int]:
    return [k * cfg.timesteps // nb for k in range(nb + 1)]


def _bucket_label(schedule: Schedule, lo: int, hi: int) -> dict:
    mid = max(1, (lo + hi) // 2)
    snr = float(schedule.snr(mid))
    return {"t_lo": lo, "t_hi": hi - 1, "t_mid": mid,
            "snr_mid": round(snr, 4), "log10_snr_mid": round(math.log10(max(snr, 1e-12)), 3)}


@torch.no_grad()
def loss_and_effect(model, cfg: V4Config, schedule: Schedule, lat, sid, *,
                    reps: int, edges: list[int], batch: int, seed: int) -> dict:
    """逐桶 L_true/L_null/L_wrong + 真错输出差异；固定验证集，噪声按 (sample, rep) 固定。"""
    device = lat.device
    n = lat.shape[0]
    nb = len(edges) - 1
    names = ("true", "null", "wrong")
    per = {k: torch.zeros(reps, n, device=device) for k in names}
    diff = {k: torch.zeros(reps, n, device=device) for k in names}
    ts = torch.zeros(reps, n, device=device, dtype=torch.long)

    gen = torch.Generator(device=device).manual_seed(seed)
    for r in range(reps):
        t = torch.randint(0, schedule.timesteps, (n,), device=device, generator=gen)
        ts[r] = t
        noise = torch.randn(lat.shape, device=device, dtype=lat.dtype, generator=gen)
        zt = schedule.q_sample(lat, t, noise)
        wrong = (sid + 1 + r) % len(STYLE_NAMES)
        conds = {"true": sid, "null": torch.full_like(sid, -1), "wrong": wrong}
        outs = {}
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            zt_b, t_b, noise_b = zt[sl], t[sl], noise[sl]
            for key in names:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.use_bf16):
                    pred = model(zt_b, t_b.float(), conds[key][sl])
                _, eps = pred_to_x0_eps(pred.float(), zt_b, t_b, schedule, cfg.prediction_type)
                per[key][r, sl] = ((eps - noise_b) ** 2).flatten(1).mean(1)
                outs.setdefault(key, []).append(eps)
        out_true = torch.cat(outs["true"])
        for key in ("null", "wrong"):
            out_ref = torch.cat(outs[key])
            diff[key][r] = ((out_true - out_ref) ** 2).flatten(1).mean(1).sqrt() / \
                out_true.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)

    rows = []
    ts_np = ts.cpu().numpy()
    for k in range(nb):
        lo, hi = edges[k], edges[k + 1]
        mask_np = (ts_np >= lo) & (ts_np < hi)          # [reps, n]，逐 rep 各自分桶
        if not mask_np.any():
            continue
        mask = torch.from_numpy(mask_np).to(device)
        row = {"bucket": k, **_bucket_label(schedule, lo, hi), "n_total": int(mask_np.sum())}
        for key in names:
            row[f"L_{key}"] = round(float(per[key][mask].mean()), 5)
        for key in ("null", "wrong"):
            d = per[key][mask] - per["true"][mask]
            flat = d.reshape(-1)
            row[f"d_{key}"] = round(float(flat.mean()), 6)
            row[f"ci95_{key}"] = round(float(1.96 * flat.std(unbiased=True) /
                                            math.sqrt(max(flat.numel(), 1))), 6)
            row[f"rel_{key}_pct"] = round(100 * float(flat.mean()) /
                                          max(float(per["true"][mask].mean()), 1e-8), 3)
            row[f"win_{key}"] = round(float((d > 0).float().mean()), 3)
            rep_means = [float((per[key][i][mask[i]] - per["true"][i][mask[i]]).mean())
                         for i in range(reps) if bool(mask[i].any())]
            row[f"rep_means_{key}"] = [round(v, 6) for v in rep_means]
            row[f"all_reps_positive_{key}"] = bool(rep_means and all(v > 0 for v in rep_means))
            row[f"out_rel_{key}"] = round(float(diff[key][mask].mean()), 5)
        rows.append(row)

    overall = {}
    for key in names:
        overall[f"L_{key}"] = round(float(per[key].mean()), 5)
    for key in ("null", "wrong"):
        d = (per[key] - per["true"]).reshape(-1)
        overall[f"d_{key}"] = round(float(d.mean()), 6)
        overall[f"rel_{key}_pct"] = round(100 * float(d.mean()) /
                                           max(float(per["true"].mean()), 1e-8), 3)
        overall[f"win_{key}"] = round(float((d > 0).float().mean()), 3)

    # 按"错误风格"细分：排除是某一个错误 ID 撑起全部收益的可能
    by_wrong_style = {}
    for s in range(len(STYLE_NAMES)):
        acc = []
        for r in range(reps):
            ws = (sid + 1 + r) % len(STYLE_NAMES)
            sel = ws == s
            if bool(sel.any()):
                acc.append(float((per["wrong"][r, sel] - per["true"][r, sel]).mean()))
        by_wrong_style[str(s)] = {
            "style": STYLE_NAMES[s],
            "d_wrong": round(float(np.mean(acc)), 6) if acc else None,
            "reps": len(acc),
        }
    return {"buckets": rows, "overall": overall, "by_wrong_style": by_wrong_style}


@torch.no_grad()
def film_audit(model: V4UNet, cfg: V4Config, schedule: Schedule, edges: list[int],
               *, n: int, seed: int) -> list[dict]:
    """每个桶取代表 t，测量逐层 FiLM 的调制幅度与"真 vs 错"的层间差异。"""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, cfg.latent_channels, 64, 64, device=device, generator=gen)
    layers = [(nm, mod) for nm, mod in model.named_modules() if isinstance(mod, FiLM)]
    rec: dict[str, dict[str, dict]] = {}
    handles = []

    def make_hook(key, name):
        def hook(mod, inp, out):
            h, cond = inp[0], inp[1]
            delta = (out - h).detach()
            sc, sh = mod.proj(cond).chunk(2, dim=-1)
            rec.setdefault(key, {})[name] = {
                "delta": delta.detach(),
                "delta_over_h": _rms(delta) / max(_rms(h), 1e-8),
                "scale_abs": float(sc.abs().mean()),
                "shift_abs": float(sh.abs().mean()),
            }
        return hook

    rows = []
    for k in range(len(edges) - 1):
        t_val = float(max(1, (edges[k] + edges[k + 1]) // 2))
        rec.clear()
        for key in ("true", "wrong", "null"):
            for nm, mod in layers:
                handles.append(mod.register_forward_hook(make_hook(key, nm)))
            sid = {"true": 0, "wrong": 1, "null": -1}[key]
            model(z, torch.full((n,), t_val, device=device),
                  torch.full((n,), sid, dtype=torch.long, device=device))
            for hnd in handles:
                hnd.remove()
            handles = []

        by_stage: dict[str, dict] = {}
        per_layer = []
        for nm, _ in layers:
            stage = nm.split(".")[0]
            kind = "style" if "style_film" in nm else "time"
            rt, rw = rec["true"][nm], rec["wrong"][nm]
            eff = _rms(rt["delta"] - rw["delta"]) / max(_rms(rt["delta"]), 1e-8)
            entry = {"layer": nm, "kind": kind,
                     "delta_over_h": round(rt["delta_over_h"], 5),
                     "scale_abs": round(rt["scale_abs"], 6),
                     "shift_abs": round(rt["shift_abs"], 6),
                     "style_specific_rel": round(eff, 5)}
            per_layer.append(entry)
            agg = by_stage.setdefault(f"{stage}/{kind}", {"delta_over_h": [], "style_specific_rel": []})
            agg["delta_over_h"].append(entry["delta_over_h"])
            agg["style_specific_rel"].append(entry["style_specific_rel"])
        stage_agg = {key: {kk: round(float(np.mean(vv)), 5) for kk, vv in v.items()}
                     for key, v in by_stage.items()}
        rows.append({"bucket": k, **_bucket_label(schedule, edges[k], edges[k + 1]),
                     "layers": per_layer, "by_stage": stage_agg})
    return rows


def grad_audit(model: V4UNet, cfg: V4Config, schedule: Schedule, lat, sid,
               edges: list[int], *, n: int, seed: int) -> list[dict]:
    """每个桶单独 backward：条件通路梯度是否到达各尺度。"""
    device = lat.device
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randint(0, lat.shape[0], (n,), device=device, generator=gen)
    z0, s0 = lat.index_select(0, idx), sid.index_select(0, idx)
    noise = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=gen)
    rows = []
    for k in range(len(edges) - 1):
        t_val = max(0, (edges[k] + edges[k + 1]) // 2)
        t_idx = torch.full((n,), t_val, dtype=torch.long, device=device)
        t = t_idx.float()
        model.train()
        model.zero_grad(set_to_none=True)
        zt = schedule.q_sample(z0, t_idx, noise)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.use_bf16):
            pred = model(zt, t, s0)
        target = v_target(z0, noise, t, schedule) if cfg.prediction_type == "v" else noise
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        g = {nm: float(p.grad.norm()) if p.grad is not None else 0.0
             for nm, p in model.named_parameters()}
        # 空条件分支：单独 backward 一次，保证 null_style 一定被命中
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.use_bf16):
            pred_n = model(zt, t, torch.full_like(s0, -1))
        ((pred_n - target) ** 2).mean().backward()
        g_null = {nm: float(p.grad.norm()) if p.grad is not None else 0.0
                  for nm, p in model.named_parameters()}

        def mean_of(pattern: str, null: bool = False) -> float:
            src = g_null if null else g
            vals = [v for nm, v in src.items() if pattern in nm]
            return round(float(np.mean(vals)), 8) if vals else 0.0

        def stage_sum(stage: str) -> float:
            vals = [v for nm, v in g.items() if nm.startswith(stage + ".")]
            return round(float(sum(vals)), 8) if vals else 0.0

        rows.append({
            "bucket": k, **_bucket_label(schedule, edges[k], edges[k + 1]),
            "loss": round(float(loss.detach()), 5),
            "style_embed": round(g.get("style_embed.weight", 0.0), 8),
            "null_style": round(g_null.get("null_style", 0.0), 8),
            "style_proj": mean_of("style_proj"),
            "style_film_proj": mean_of("style_film_proj"),
            "film_time_mean": round(float(np.mean(
                [v for nm, v in g.items() if "film" in nm and "style_film" not in nm] or [0.0])), 8),
            "film_style_mean": mean_of("style_film"),
            "time_embed": mean_of("time_embed"),
            "input_conv": mean_of("input_conv"),
            "out_conv": mean_of("out_conv"),
            "stage0": stage_sum("stage0"), "stage1": stage_sum("stage1"),
            "stage2": stage_sum("stage2"), "stage3": stage_sum("stage3"),
            "mid": stage_sum("mid"), "up3_res": stage_sum("up3_res"),
            "up2_res": stage_sum("up2_res"), "up1_res": stage_sum("up1_res"),
            "zero_grad_params": int(sum(1 for v in g.values() if v == 0.0)),
            "total_params": len(g),
        })
    model.eval()
    return rows


def verdict(report: dict) -> dict:
    good_w, good_n = [], []
    for row in report["cond_loss"]["buckets"]:
        if (row["ci95_wrong"] > 0 and row["win_wrong"] >= 0.75
                and row["all_reps_positive_wrong"] and row["rel_wrong_pct"] >= 0.5):
            good_w.append(row["bucket"])
        if (row["ci95_null"] > 0 and row["win_null"] >= 0.75
                and row["all_reps_positive_null"] and row["rel_null_pct"] >= 0.5):
            good_n.append(row["bucket"])
    return {
        "effective_vs_wrong_buckets": good_w,
        "effective_vs_null_buckets": good_n,
        "pass": bool(good_w),
        "criterion": "ΔL>0, CI95下界>0, 胜率≥0.75, 所有rep均值>0, 相对收益≥0.5%",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v4/arms/B/best.pt")
    ap.add_argument("--config", default="", help="--ckpt 为空时用于构造新初始化")
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--grad-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-mapping", action="store_true")
    args = ap.parse_args()

    device = _device(args.device)
    model, cfg, info = load_model(args.ckpt, args.config, device)
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end,
                               device=device.type)
    edges = bucket_edges(cfg, args.bins)

    report: dict = {"info": info, "v4_config": cfg.to_dict(), "device": str(device)}
    if not args.skip_mapping:
        from .cond_diag import check_mapping
        report["mapping"] = check_mapping(cfg)

    indices = resolve_indices(args.ckpt, args.indices)
    idx = torch.load(indices, map_location="cpu", weights_only=False)
    report["indices"] = str(indices)
    bank = load_bank(cfg.packed_dir)
    val_idx = idx["val_idx"].long()
    lat = bank.latents.index_select(0, val_idx).to(device)
    sid = bank.style_ids.index_select(0, val_idx).to(device)
    if lat.shape[0] > args.n:
        lat, sid = lat[:args.n], sid[:args.n]
    report["n"] = int(lat.shape[0])

    report["cond_loss"] = loss_and_effect(model, cfg, schedule, lat, sid, reps=args.reps,
                                          edges=edges, batch=args.batch, seed=args.seed)
    report["film"] = film_audit(model, cfg, schedule, edges, n=min(8, args.n), seed=args.seed + 1)
    report["grad"] = grad_audit(model, cfg, schedule, lat, sid, edges,
                                n=min(args.grad_n, args.n), seed=args.seed + 2)
    report["verdict"] = verdict(report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[audit] ckpt={info['ckpt']} step={info['step']} params={info['params']/1e6:.2f}M "
          f"pred={cfg.prediction_type} cond={cfg.cond_mode} n={report['n']} reps={args.reps}")
    if "mapping" in report:
        m = report["mapping"]
        print(f"[audit] mapping spot_all_equal={m['spot_all_equal']} "
              f"counts={m['bank_counts']} test_dirs_match={m['test_dirs_match']}")
    print("\n[bucket] t-range   logSNR   L_true   d_wrong(±CI)      win   rel%   allReps  d_null")
    for row in report["cond_loss"]["buckets"]:
        print(f"{row['bucket']:>2}  {row['t_lo']:>3}-{row['t_hi']:<3} {row['log10_snr_mid']:>7.2f} "
              f"{row['L_true']:.5f}  {row['d_wrong']:+.5f}±{row['ci95_wrong']:.5f}  "
              f"{row['win_wrong']:.2f}  {row['rel_wrong_pct']:+.2f}  "
              f"{str(row['all_reps_positive_wrong']):<5}  {row['d_null']:+.5f}")
    print("\n[grad] bucket  style_embed  null_style  style_film  time_film  out_conv  zero_grad")
    for row in report["grad"]:
        print(f"{row['bucket']:>2}     {row['style_embed']:.2e}  {row['null_style']:.2e}  "
              f"{row['film_style_mean']:.2e}  {row['film_time_mean']:.2e}  "
              f"{row['out_conv']:.2e}  {row['zero_grad_params']}/{row['total_params']}")
    print("\n[verdict]", json.dumps(report["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
