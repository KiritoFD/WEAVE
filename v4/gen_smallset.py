"""固定小集生成：给定 ckpt，用固定源图 + 固定种子 + 固定采样步数做风格转换，
产出 PNG 与 ``metrics.csv``（供 ``tools/12h/v4_refpool_eval.py`` 算 DINO-S/C、CLIP-S）。

约定：
- 源图取 ``data/test/<style>`` 每风格按文件名排序前 ``--src-per-style`` 张；
- 每张源图对所有 5 个目标风格各转一次（同风格对同一风格也包含，便于算 s/x 拆分）；
- 噪声种子 = ``--seed * 1000 + pair_index``，同一 ckpt 内可复现；
- LPIPS 可选（``--lpips``），默认 alex 主干，逐对计算后写回 metrics.csv。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paths import hf_cache_root, resolve
from v4.config import V4Config
from v4.data import STYLE_NAMES
from v4.diffusion import Schedule, sdedit
from v4.model import V4UNet

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image(path: Path, size: int = 512) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype="uint8").copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="--lpips-only 时可为空")
    ap.add_argument("--out", required=True, help="输出目录（PNG + metrics.csv + gen.json）")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--src-per-style", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vae-roundtrip", action="store_true",
                    help="控制实验：只做 encode→decode，不跑扩散，用于分离 VAE 往返损伤")
    ap.add_argument("--guide-clf", default="", help="噪声条件风格分类器 ckpt；空=不加引导")
    ap.add_argument("--guide-scale", type=float, default=0.0)
    ap.add_argument("--guide-t-min", type=int, default=0)
    ap.add_argument("--guide-t-max", type=int, default=700)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--lpips-only", action="store_true", help="不重跑生成，只对已有 metrics.csv 补算 LPIPS")
    ap.add_argument("--lpips-net", default="alex")
    return ap.parse_args()


def compute_lpips(gen_x: torch.Tensor, src_x: torch.Tensor, net: str) -> torch.Tensor:
    import lpips
    from utils.run_evaluation import _lpips_forward_safe

    loss_fn = lpips.LPIPS(net=net, verbose=False).to("cuda").eval()
    return _lpips_forward_safe(loss_fn, gen_x, src_x, device="cuda", chunk_size=4,
                               cpu_fallback=True, tag="smallset-lpips")


def lpips_only(out: Path, args: argparse.Namespace) -> int:
    rows = list(csv.DictReader((out / "metrics.csv").open(newline="", encoding="utf-8")))
    test_dir = resolve(args.test_dir)
    gen = torch.cat([load_image(out / r["gen_image"]).unsqueeze(0) for r in rows], 0).cuda()
    src = torch.cat([load_image(test_dir / r["src_style"] / r["src_image"]).unsqueeze(0) for r in rows], 0).cuda()
    d = compute_lpips(gen, src, args.lpips_net)
    for row, val in zip(rows, d.tolist()):
        row["content_lpips"] = round(float(val), 5)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    path = out / "gen.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary.pop("lpips_error", None)
    summary["content_lpips"] = round(float(np.mean([r["content_lpips"] for r in rows])), 5)
    summary["lpips_net"] = args.lpips_net
    summary["content_lpips_by_tgt"] = {
        name: round(float(np.mean([r["content_lpips"] for r in rows if r["tgt_style"] == name])), 5)
        for name in STYLE_NAMES
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()
    out = resolve(args.out)
    if args.lpips_only:
        return lpips_only(out, args)
    if not args.ckpt:
        raise SystemExit("--ckpt 必填（除非 --lpips-only）")
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    if str(payload.get("variant")) != "V4DIFF":
        raise SystemExit(f"unexpected variant {payload.get('variant')!r}")
    cfg = V4Config.from_mapping(payload["v4_config"])
    model = V4UNet(cfg).cuda()
    model.load_state_dict({k: v.float() for k, v in payload["model"].items()}, strict=True)
    model.eval()
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")

    from utils.inference import decode_latent, encode_image, load_vae
    vae = load_vae("cuda", model_id="ema", cache_dir=str(hf_cache_root()))

    test_dir = resolve(args.test_dir)
    sources: list[tuple[int, Path]] = []
    for sid, name in enumerate(STYLE_NAMES):
        files = sorted([p for p in (test_dir / name).iterdir() if p.suffix.lower() in IMAGE_EXTS])[: args.src_per_style]
        if not files:
            raise SystemExit(f"没有源图: {test_dir / name}")
        sources.extend((sid, p) for p in files)

    guidance = {}
    if args.guide_clf and args.guide_scale != 0.0:
        from v4.style_classifier import make_guidance_fn
        clf_path = str(resolve(args.guide_clf))
        guidance = {sid: make_guidance_fn(clf_path, sid, torch.device("cuda"),
                                           t_min=args.guide_t_min, t_max=args.guide_t_max)
                    for sid in range(cfg.num_styles)}

    # 源图编码只做一次
    src_cache: dict[Path, torch.Tensor] = {}
    for _, path in sources:
        src_cache[path] = encode_image(vae, load_image(path).unsqueeze(0).cuda()).float()

    rows: list[dict] = []
    pairs: list[tuple[int, Path, int, torch.Tensor]] = []
    pair_idx = 0
    for src_sid, path in sources:
        targets = [src_sid] if args.vae_roundtrip else list(range(cfg.num_styles))
        for tgt_sid in targets:
            gen = torch.Generator(device="cuda").manual_seed(args.seed * 1000 + pair_idx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                if args.vae_roundtrip:
                    z = src_cache[path]
                else:
                    z = sdedit(model, src_cache[path], torch.tensor([tgt_sid], device="cuda"),
                               schedule, strength=args.strength, num_steps=args.steps,
                               cfg_scale=args.cfg, prediction_type=cfg.prediction_type,
                               guidance_fn=guidance.get(tgt_sid),
                               guidance_scale=args.guide_scale, generator=gen)
            img = decode_latent(vae, z.float())[0].float().cpu().clamp(0, 1)
            rel_dir = STYLE_NAMES[tgt_sid]
            (img_dir / rel_dir).mkdir(parents=True, exist_ok=True)
            name = f"{pair_idx:03d}_{STYLE_NAMES[src_sid]}_{path.stem}.png"
            arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(img_dir / rel_dir / name)
            rows.append({
                "gen_image": (Path("images") / rel_dir / name).as_posix(),
                "src_style": STYLE_NAMES[src_sid],
                "src_image": path.name,
                "tgt_style": STYLE_NAMES[tgt_sid],
                "_src_tensor": load_image(path).unsqueeze(0),
            })
            pairs.append((src_sid, path, tgt_sid, img.unsqueeze(0)))
            pair_idx += 1
            print(f"[gen] {pair_idx}/{len(sources) * cfg.num_styles} {STYLE_NAMES[src_sid]}->{STYLE_NAMES[tgt_sid]} {name}", flush=True)

    src_x = torch.cat([r.pop("_src_tensor") for r in rows], 0)
    summary = {
        "ckpt": str(args.ckpt), "step": payload.get("step"), "n": len(rows),
        "steps": args.steps, "strength": args.strength, "cfg": args.cfg, "seed": args.seed,
        "src_per_style": args.src_per_style, "prediction_type": cfg.prediction_type,
        "mode": ("vae_roundtrip" if args.vae_roundtrip else "sdedit"),
        "guide_clf": (args.guide_clf or None), "guide_scale": args.guide_scale,
        "guide_t_min": args.guide_t_min, "guide_t_max": args.guide_t_max, "content_lpips": None,
    }
    rc = 0
    if args.lpips:
        try:
            # _lpips_forward_safe 不改输入设备，输入必须与 loss_fn 同设备
            gen_x = torch.cat([p[3] for p in pairs], 0).cuda()
            d = compute_lpips(gen_x, src_x.cuda(), args.lpips_net)
            for row, val in zip(rows, d.tolist()):
                row["content_lpips"] = round(float(val), 5)
            summary["content_lpips"] = round(float(np.mean([r["content_lpips"] for r in rows])), 5)
            summary["lpips_net"] = args.lpips_net
            by_tgt = {}
            for name in STYLE_NAMES:
                vals = [r["content_lpips"] for r in rows if r["tgt_style"] == name]
                by_tgt[name] = round(float(np.mean(vals)), 5)
            summary["content_lpips_by_tgt"] = by_tgt
        except Exception as exc:  # 失败必须留痕，不静默
            import traceback
            traceback.print_exc()
            summary["lpips_error"] = f"{type(exc).__name__}: {exc}"
            rc = 3

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "gen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
