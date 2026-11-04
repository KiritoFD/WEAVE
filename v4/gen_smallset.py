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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="输出目录（PNG + metrics.csv + gen.json）")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--src-per-style", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--lpips-net", default="alex")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = resolve(args.out)
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

    # 源图编码只做一次
    src_cache: dict[Path, torch.Tensor] = {}
    for _, path in sources:
        src_cache[path] = encode_image(vae, load_image(path).unsqueeze(0).cuda()).float()

    rows: list[dict] = []
    pairs: list[tuple[int, Path, int, torch.Tensor]] = []
    pair_idx = 0
    for src_sid, path in sources:
        for tgt_sid in range(cfg.num_styles):
            gen = torch.Generator(device="cuda").manual_seed(args.seed * 1000 + pair_idx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                z = sdedit(model, src_cache[path], torch.tensor([tgt_sid], device="cuda"),
                           schedule, strength=args.strength, num_steps=args.steps,
                           cfg_scale=args.cfg, prediction_type=cfg.prediction_type,
                           generator=gen)
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

    if args.lpips:
        import lpips
        from utils.run_evaluation import _lpips_forward_safe

        loss_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).to("cuda").eval()
        gen_x = torch.cat([p[3] for p in pairs], 0)
        src_x = torch.cat([r["_src_tensor"] for r in rows], 0)
        d = _lpips_forward_safe(loss_fn, gen_x, src_x, device="cuda", chunk_size=4,
                                cpu_fallback=True, tag="smallset-lpips")
        for row, val in zip(rows, d.tolist()):
            row["content_lpips"] = round(float(val), 5)

    for row in rows:
        row.pop("_src_tensor", None)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ckpt": str(args.ckpt), "step": payload.get("step"), "n": len(rows),
        "steps": args.steps, "strength": args.strength, "cfg": args.cfg, "seed": args.seed,
        "src_per_style": args.src_per_style, "prediction_type": cfg.prediction_type,
        "content_lpips": None,
    }
    if args.lpips:
        summary["content_lpips"] = round(float(np.mean([r["content_lpips"] for r in rows])), 5)
        summary["lpips_net"] = args.lpips_net
        by_tgt = {}
        for sid, name in enumerate(STYLE_NAMES):
            vals = [r["content_lpips"] for r in rows if r["tgt_style"] == name]
            by_tgt[name] = round(float(np.mean(vals)), 5)
        summary["content_lpips_by_tgt"] = by_tgt
    (out / "gen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
