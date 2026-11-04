"""Style-ID 条件诊断：映射一致性、FiLM 调制幅度、梯度、按噪声区间的 L_true/L_null/L_wrong。

用法:
  python3 -m v4.cond_diag --ckpt runs/v4/prior/best.pt --out runs/v4/diag/best.json
  python3 -m v4.cond_diag --ckpt "" --out runs/v4/diag/init.json   # 未训练初始化
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
from .diffusion import Schedule, training_loss
from .model import FiLM, V4UNet, count_parameters


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def build_model(ckpt_path: str) -> tuple[V4UNet, V4Config, Schedule, dict]:
    info = {}
    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = V4Config.from_mapping(payload["v4_config"])
        sd = payload["model"]
        info = {"ckpt": ckpt_path, "step": payload.get("step"), "val": payload.get("val")}
    else:
        cfg = V4Config()
        sd = None
        info = {"ckpt": "fresh-init", "step": 0}
    model = V4UNet(cfg).cuda().eval()
    if sd is not None:
        model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    sched = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")
    info["params"] = count_parameters(model)
    return model, cfg, sched, info


def check_mapping(cfg: V4Config) -> dict:
    bank = load_bank(cfg.packed_dir)
    counts = [int((bank.style_ids == i).sum()) for i in range(len(STYLE_NAMES))]
    test_dirs = sorted(p.name for p in Path("data/test").iterdir() if p.is_dir())

    spot = []
    for sid in range(len(STYLE_NAMES)):
        prefix = STYLE_NAMES[sid] + "__"
        names = [Path(str(f).replace("\\", "/")).name for f in bank.files
                 if Path(str(f).replace("\\", "/")).name.startswith(prefix)][:2]
        for name in names:
            path = Path("data/train") / STYLE_NAMES[sid] / name
            single = torch.load(path, map_location="cpu", weights_only=False)
            z = single["latent"] if isinstance(single, dict) and "latent" in single else single
            rows = [i for i, f in enumerate(bank.files)
                    if Path(str(f).replace("\\", "/")).name == name]
            if not rows:
                spot.append({"style": STYLE_NAMES[sid], "file": name, "row": -1, "max_abs_diff": 999})
                continue
            row = rows[0]
            diff = float((z.float() - bank.latents[row]).abs().max())
            spot.append({"style": STYLE_NAMES[sid], "file": name, "row": row, "max_abs_diff": diff})

    per_style_mean = [round(float(bank.latents[bank.style_ids == i].mean()), 4) for i in range(5)]
    per_style_std = [round(float(bank.latents[bank.style_ids == i].std()), 4) for i in range(5)]
    return {
        "style_names": list(STYLE_NAMES),
        "test_dirs_sorted": test_dirs,
        "test_dirs_match": test_dirs == list(STYLE_NAMES),
        "bank_counts": counts,
        "latent_pack_order_matches_names": counts == [1000] * len(STYLE_NAMES),
        "per_style_latent_mean": per_style_mean,
        "per_style_latent_std": per_style_std,
        "spot_checks": spot,
        "spot_all_equal": all(s["max_abs_diff"] < 1e-6 for s in spot),
    }


@torch.no_grad()
def film_diag(model: V4UNet, cfg: V4Config, sched: Schedule, n: int = 8) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(1234)
    z = torch.randn(n, cfg.latent_channels, 64, 64, device="cuda", generator=gen)
    layers = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, FiLM)]
    rec: dict[str, list] = {"true": [], "null": [], "wrong": []}

    def make_hook(key):
        def hook(mod, inp, out):
            h = inp[0]
            delta = (out - h).detach()
            sc, sh = mod.proj(inp[1]).chunk(2, -1)
            rec[key].append({"delta_over_h": rms(delta) / max(rms(h), 1e-8),
                             "delta": delta,
                             "scale_abs": float(sc.abs().mean()),
                             "shift_abs": float(sh.abs().mean())})
        return hook

    handles = []
    for key in ("true", "null", "wrong"):
        for name, mod in layers:
            handles.append(mod.register_forward_hook(make_hook(key)))
        sid = torch.zeros(n, dtype=torch.long, device="cuda")
        if key == "null":
            sid = torch.full((n,), -1, dtype=torch.long, device="cuda")
        elif key == "wrong":
            sid = torch.ones(n, dtype=torch.long, device="cuda")
        model(z, torch.full((n,), 500.0, device="cuda"), sid)
        for hnd in handles:
            hnd.remove()
        handles = []

    groups: dict[str, list] = {}
    for name, _ in layers:
        stage = name.split(".")[0] if "." in name else name
        groups.setdefault(stage, []).append(name)

    def agg(key):
        vals = [r["delta_over_h"] for r in rec[key]]
        return {"mean": round(sum(vals) / len(vals), 5), "max": round(max(vals), 5)}

    def by_stage(key):
        out = {}
        for stage, names in groups.items():
            idxs = [i for i, (nm, _) in enumerate(layers) if nm in names]
            out[stage] = round(sum(rec[key][i]["delta_over_h"] for i in idxs) / len(idxs), 5)
        return out

    # cond 向量自身的可分性
    conds = {}
    for sid in range(len(STYLE_NAMES)):
        conds[sid] = model._cond(torch.full((4,), 500.0, device="cuda"),
                                 torch.full((4,), sid, dtype=torch.long, device="cuda"))[0].float()
    null_c = model._cond(torch.full((4,), 500.0, device="cuda"),
                         torch.full((4,), -1, dtype=torch.long, device="cuda"))[0].float()
    base = float(conds[0].norm())
    pairwise = [float((conds[i] - conds[j]).norm()) / base for i in range(5) for j in range(i + 1, 5)]
    def style_effect(ref_key: str) -> dict:
        out = {}
        for stage, names in groups.items():
            idxs = [i for i, (nm, _) in enumerate(layers) if nm in names]
            rel = [rms(rec["true"][i]["delta"] - rec[ref_key][i]["delta"])
                   / max(rms(rec["true"][i]["delta"]), 1e-8) for i in idxs]
            out[stage] = round(sum(rel) / len(rel), 5)
        return out

    return {
        "modulation_rms_ratio": {"true": agg("true"), "null": agg("null"), "wrong": agg("wrong")},
        "modulation_by_stage_true": by_stage("true"),
        "modulation_by_stage_null": by_stage("null"),
        "modulation_by_stage_wrong": by_stage("wrong"),
        "style_effect_vs_null_by_stage": style_effect("null"),
        "style_effect_vs_wrong_by_stage": style_effect("wrong"),
        "cond_true_vs_null_rel": round(float((conds[0] - null_c).norm()) / base, 4),
        "cond_pairwise_rel_mean": round(sum(pairwise) / len(pairwise), 4),
        "cond_pairwise_rel_min": round(min(pairwise), 4),
        "n_layers": len(layers),
    }


def grad_diag(model: V4UNet, cfg: V4Config, sched: Schedule, bank, n: int = 16) -> dict:
    model.train()
    gen = torch.Generator(device="cuda").manual_seed(7)
    idx = torch.randint(0, len(bank), (n,), device="cuda", generator=gen)
    model.zero_grad(set_to_none=True)
    loss, _ = training_loss(model, bank.latents.cuda().index_select(0, idx),
                            bank.style_ids.cuda().index_select(0, idx), sched, cfg_dropout=0.0,
                            generator=gen)
    loss.backward()
    g = {k: float(v.grad.norm()) if v.grad is not None else 0.0 for k, v in model.named_parameters()}

    def group(prefixes, contains: bool = False):
        vals = [v for k, v in g.items()
                if any((p in k) if contains else k.startswith(p) for p in prefixes)]
        return round(sum(vals) / max(len(vals), 1), 6)

    out = {
        "loss": round(float(loss.detach()), 4),
        "style_embed_grad": round(g.get("style_embed.weight", 0.0), 6),
        "null_style_grad": round(g.get("null_style", 0.0), 6),
        "style_proj_grad_mean": group(["style_proj."]),
        "film_proj_grad_mean": group(["film1.proj", "film2.proj"], contains=True),
        "film_proj_grad_max": round(max([v for k, v in g.items() if "film" in k], default=0.0), 6),
        "input_conv_grad": group(["input_conv."]),
        "out_conv_grad": group(["out_conv."]),
        "time_embed_grad_mean": group(["time_embed."]),
        "zero_grad_params": int(sum(1 for v in g.values() if v == 0.0)),
        "total_params": len(g),
        "param_norms": {
            "style_embed": round(float(model.style_embed.weight.norm()), 4),
            "null_style": round(float(model.null_style.norm()), 4),
            "film_proj_mean": round(sum(float(p.norm()) for n, p in model.named_parameters()
                                        if "film1.proj" in n or "film2.proj" in n)
                                 / max(1, len([1 for n, _ in model.named_parameters()
                                               if "film1.proj" in n or "film2.proj" in n])), 4),
        },
    }
    model.eval()
    return out


@torch.no_grad()
def cond_loss_report(model: V4UNet, cfg: V4Config, sched: Schedule, bank, idx, *,
                     reps: int = 4, n: int = 512, seed: int = 99) -> dict:
    """固定样本、固定噪声与时间步，比较 L_true / L_null / L_wrong（按 t 分桶）。"""
    model.eval()
    lat = bank.latents.cuda().index_select(0, idx.cuda())
    sid = bank.style_ids.cuda().index_select(0, idx.cuda())
    n = min(n, lat.shape[0])
    lat, sid = lat[:n], sid[:n]
    wrong = (sid + 1) % len(STYLE_NAMES)
    neg = torch.full_like(sid, -1)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t = torch.randint(0, cfg.timesteps, (reps, n), device="cuda", generator=gen)
    noise = torch.randn((reps, n, *lat.shape[1:]), device="cuda", dtype=lat.dtype, generator=gen)

    per = {"true": [], "null": [], "wrong": []}
    for r in range(reps):
        zt = sched.q_sample(lat, t[r], noise[r])
        for key, cond in (("true", sid), ("null", neg), ("wrong", wrong)):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                pred = model(zt, t[r].float(), cond)
            per[key].append(((pred.float() - noise[r]) ** 2).flatten(1).mean(1))
    err = {k: torch.stack(v, 0) for k, v in per.items()}   # [reps, n]

    nb = 6
    bounds = [k * cfg.timesteps // nb for k in range(nb + 1)]
    rows = []
    for k in range(nb):
        sel = (t[0] >= bounds[k]) & (t[0] < bounds[k + 1])
        if not bool(sel.any()):
            continue
        row = {"t": f"{bounds[k]}-{bounds[k+1] - 1}", "n": int(sel.sum())}
        for key in ("true", "null", "wrong"):
            row[f"L_{key}"] = round(float(err[key][:, sel].mean()), 5)
        d_null = (err["null"][:, sel] - err["true"][:, sel]).reshape(-1)
        d_wrong = (err["wrong"][:, sel] - err["true"][:, sel]).reshape(-1)
        row["d_null"] = round(float(d_null.mean()), 5)
        row["d_wrong"] = round(float(d_wrong.mean()), 5)
        row["ci95_null"] = round(float(1.96 * d_null.std(unbiased=True) / math.sqrt(d_null.numel())), 5)
        row["ci95_wrong"] = round(float(1.96 * d_wrong.std(unbiased=True) / math.sqrt(d_wrong.numel())), 5)
        base = float(err["true"][:, sel].mean())
        row["rel_null_pct"] = round(100 * row["d_null"] / base, 3)
        row["rel_wrong_pct"] = round(100 * row["d_wrong"] / base, 3)
        rows.append(row)
    all_true = float(err["true"].mean())
    return {
        "n_samples": n, "reps": reps, "buckets": rows,
        "overall": {
            "L_true": round(all_true, 5),
            "d_null": round(float((err["null"] - err["true"]).mean()), 5),
            "d_wrong": round(float((err["wrong"] - err["true"]).mean()), 5),
        },
        "mid_snr_note": "中 SNR = t 分桶 2/3（334-666），低噪声自带风格信息、高噪声预测偏向输入",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v4/prior/best.pt")
    ap.add_argument("--out", default="runs/v4/diag/best.json")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--skip-mapping", action="store_true")
    args = ap.parse_args()

    model, cfg, sched, info = build_model(args.ckpt)
    bank = load_bank(cfg.packed_dir)
    idx_path = Path(args.ckpt).parent / "indices.pt" if args.ckpt else Path("runs/v4/prior/indices.pt")
    if not idx_path.exists():
        idx_path = Path("runs/v4/prior/indices.pt")
    val_idx = torch.load(idx_path, map_location="cpu", weights_only=False)["val_idx"]

    out = {"info": info, "v4_config": cfg.to_dict()}
    if not args.skip_mapping:
        out["mapping"] = check_mapping(cfg)
    out["film"] = film_diag(model, cfg, sched)
    out["grad"] = grad_diag(model, cfg, sched, bank)
    out["cond_loss"] = cond_loss_report(model, cfg, sched, bank, val_idx,
                                        reps=args.reps, n=args.n)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"info": info, "mapping": out.get("mapping", {}).get("spot_all_equal"),
                      "film": out["film"]["modulation_rms_ratio"],
                      "film_by_stage_true": out["film"]["modulation_by_stage_true"],
                      "style_effect_vs_null": out["film"]["style_effect_vs_null_by_stage"],
                      "style_effect_vs_wrong": out["film"]["style_effect_vs_wrong_by_stage"],
                      "cond_pairwise_rel_mean": out["film"]["cond_pairwise_rel_mean"],
                      "grad": out["grad"]}, indent=2, ensure_ascii=False))
    print("\n[t bucket]      L_true   L_null   L_wrong   d_null(±95%)       d_wrong(±95%)")
    for r in out["cond_loss"]["buckets"]:
        print(f"{r['t']:<14} {r['L_true']:.5f}  {r['L_null']:.5f}  {r['L_wrong']:.5f}  "
              f"{r['d_null']:+.5f}±{r['ci95_null']:.5f}  {r['d_wrong']:+.5f}±{r['ci95_wrong']:.5f}  "
              f"({r['rel_null_pct']:+.2f}% / {r['rel_wrong_pct']:+.2f}%)")
    print("overall:", json.dumps(out["cond_loss"]["overall"]))


if __name__ == "__main__":
    main()

