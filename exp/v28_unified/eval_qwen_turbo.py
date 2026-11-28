"""Evaluate Qwen-Image-2.1-Turbo on WEAVE Cross600 (quantized for 12GB RTX 3060).

Supports:
1. Offline component-by-component bitsandbytes quantization (`--quantize-to`) so
   even a 16GB+ text encoder and 14GB transformer quantize safely on <=25GB RAM.
2. Resumable inference over Cross600 (or full 750) using the frozen 1-ref-per-style
   manifest (`sg750_manifest.csv`), recording per-image latency and peak VRAM.
3. Atomic `RawWriter`-compatible `batch_*.pt` + `meta.json` emission for
   process-isolated scoring via `python -m src.eval.measure`.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PROMPT = (
    "Redraw image 1 in the {target_style} artistic style, brushwork, and color "
    "palette of image 2, keeping the scene and subjects of image 1."
)


def _resolve_weave_path(raw_path: str) -> Path:
    """Resolve a manifest path across Windows/WSL/Linux WEAVE roots."""
    p = Path(raw_path)
    if p.is_file():
        return p
    parts = p.parts
    for marker in ("WEAVE", "weave"):
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)
            cand = REPO_ROOT.joinpath(*parts[idx + 1 :])
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"Cannot resolve WEAVE path: {raw_path}")


def _load_cross_tasks(
    manifest_csv: Path, *, include_same: bool = False, same_only: bool = False, limit: int = 0
) -> list[dict]:
    """Load frozen tasks paired with sg750_manifest reference image paths."""
    tasks_json = REPO_ROOT / "eval_protocol" / "tasks_750.json"
    tasks = json.loads(tasks_json.read_text(encoding="utf-8"))["tasks"]
    with manifest_csv.open(encoding="utf-8") as f:
        sg_by_id = {r["task_id"]: r for r in csv.DictReader(f)}

    selected = []
    for t in tasks:
        if same_only:
            if t["is_cross"]:
                continue
        elif not include_same and not t["is_cross"]:
            continue
        tid = t["task_id"]
        if tid not in sg_by_id:
            raise KeyError(f"Missing task_id in manifest: {tid}")
        src_path = REPO_ROOT / t["source_rel_path"]
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")
        ref_raw = sg_by_id[tid]["style_paths"].split(";")[0].strip()
        ref_path = _resolve_weave_path(ref_raw)
        selected.append({
            "task_id": tid,
            "source_style": t["source_style"],
            "target_style": t["target_style"],
            "is_cross": bool(t["is_cross"]),
            "source_path": src_path,
            "ref_path": ref_path,
        })
    if limit > 0:
        selected = selected[:limit]
    return selected


def quantize_and_save(src_dir: Path, dst_dir: Path, *, transformer_bits: int = 4) -> None:
    """Quantize text_encoder (4-bit NF4) and transformer (4-bit NF4 or 8-bit) sequentially."""
    from diffusers import BitsAndBytesConfig as DfBnb, QwenImage21Transformer2DModel
    from transformers import BitsAndBytesConfig as TfBnb, Qwen3VLForConditionalGeneration

    dst_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("processor", "scheduler", "vae"):
        s = src_dir / sub
        d = dst_dir / sub
        if s.is_dir() and not d.exists():
            shutil.copytree(s, d)
    for fname in ("model_index.json", "LICENSE", "README.md"):
        s = src_dir / fname
        d = dst_dir / fname
        if s.is_file() and not d.exists():
            shutil.copy2(s, d)

    te_dst = dst_dir / "text_encoder"
    if not (te_dst / "config.json").exists():
        print("[quantize] Loading & quantizing text_encoder (4-bit NF4)...", flush=True)
        te_cfg = TfBnb(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            str(src_dir / "text_encoder"),
            quantization_config=te_cfg,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        te.save_pretrained(str(te_dst))
        del te
        gc.collect()
        torch.cuda.empty_cache()
        print("[quantize] Saved 4-bit text_encoder.", flush=True)

    tr_dst = dst_dir / "transformer"
    if not (tr_dst / "config.json").exists():
        print(f"[quantize] Loading & quantizing transformer ({transformer_bits}-bit)...", flush=True)
        if transformer_bits == 4:
            tr_cfg = DfBnb(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif transformer_bits == 8:
            tr_cfg = DfBnb(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported transformer_bits={transformer_bits}")
        tr = QwenImage21Transformer2DModel.from_pretrained(
            str(src_dir / "transformer"),
            quantization_config=tr_cfg,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        tr.save_pretrained(str(tr_dst))
        del tr
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[quantize] Saved {transformer_bits}-bit transformer.", flush=True)


def load_pipeline(model_dir: Path, *, quant_mode: str = "auto", offload: str = "model"):
    """Load QwenImage21Pipeline with optional on-the-fly quantization and CPU offload."""
    from diffusers import (
        BitsAndBytesConfig as DfBnb,
        PipelineQuantizationConfig,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import BitsAndBytesConfig as TfBnb, Qwen3VLForConditionalGeneration

    qcfg = None
    if quant_mode == "bnb4":
        qcfg = PipelineQuantizationConfig(
            quant_mapping={
                "text_encoder": TfBnb(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                "transformer": DfBnb(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    elif quant_mode == "bnb4_t8":
        qcfg = PipelineQuantizationConfig(
            quant_mapping={
                "text_encoder": TfBnb(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                "transformer": DfBnb(load_in_8bit=True),
            }
        )
    elif quant_mode not in ("auto", "none"):
        raise ValueError(f"Unknown quant_mode={quant_mode!r}")

    kwargs = {"torch_dtype": torch.bfloat16}
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    elif quant_mode == "auto":
        # Pre-quantized bnb checkpoints load onto cuda:0 by default; stage one component
        # at a time onto CPU so peak VRAM during loading never exceeds one component.
        te_cfg_path = model_dir / "text_encoder" / "config.json"
        tr_cfg_path = model_dir / "transformer" / "config.json"
        if te_cfg_path.is_file() and "quantization_config" in json.loads(te_cfg_path.read_text(encoding="utf-8")):
            te = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_dir / "text_encoder"), torch_dtype=torch.bfloat16
            )
            te.to("cpu")
            torch.cuda.empty_cache()
            kwargs["text_encoder"] = te
        if tr_cfg_path.is_file() and "quantization_config" in json.loads(tr_cfg_path.read_text(encoding="utf-8")):
            tr = QwenImage21Transformer2DModel.from_pretrained(
                str(model_dir / "transformer"), torch_dtype=torch.bfloat16
            )
            tr.to("cpu")
            torch.cuda.empty_cache()
            kwargs["transformer"] = tr

    pipe = QwenImage21Pipeline.from_pretrained(str(model_dir), **kwargs)
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "cuda":
        pipe.to("cuda")
    else:
        raise ValueError(f"Unknown offload={offload!r}")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _pil_to_u8_chw(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _save_batch_atomic(raw_dir: Path, offset: int, ids: list[str], imgs_u8: list[torch.Tensor]) -> dict:
    fname = f"batch_{offset:05d}.pt"
    tmp = raw_dir / (fname + ".tmp")
    payload = {"ids": list(ids), "rgb_u8": torch.stack(imgs_u8, dim=0)}
    torch.save(payload, tmp)
    tmp.replace(raw_dir / fname)
    return {"offset": offset, "file": fname, "ids": list(ids)}


def _load_existing_batches(raw_dir: Path) -> tuple[list[dict], set[str], int]:
    """Scan existing batch_*.pt files to support safe resume."""
    done_ids: set[str] = set()
    batches: list[dict] = []
    offset = 0
    for bf in sorted(raw_dir.glob("batch_*.pt")):
        payload = torch.load(bf, map_location="cpu", weights_only=False)
        bids = list(payload["ids"])
        batches.append({"offset": offset, "file": bf.name, "ids": bids})
        done_ids.update(bids)
        offset += len(bids)
    return batches, done_ids, offset


def run_eval(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.diffusers_src:
        sys.path.insert(0, str(Path(args.diffusers_src)))

    model_dir = Path(args.model_dir)
    if args.quantize_to:
        qdir = Path(args.quantize_to)
        quantize_and_save(model_dir, qdir, transformer_bits=args.transformer_bits)
        model_dir = qdir
        if args.quantize_only:
            return

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    png_dir = out_dir / "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_cross_tasks(
        Path(args.manifest),
        include_same=args.include_same,
        same_only=args.same_only,
        limit=args.limit,
    )
    batches, done_ids, offset = _load_existing_batches(raw_dir) if args.resume else ([], set(), 0)

    timing_log_path = out_dir / "per_image_timing.jsonl"
    prior_times: list[float] = []
    if args.resume and timing_log_path.is_file():
        for line in timing_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("task_id") in done_ids and not rec.get("is_warmup"):
                    prior_times.append(float(rec["ms"]))

    remaining = [t for t in tasks if t["task_id"] not in done_ids]
    print(
        f"[qwen-eval] total={len(tasks)} done={len(done_ids)} remaining={len(remaining)} "
        f"model={model_dir} quant={args.quant} gen={args.gen_res} cond={args.cond_res}",
        flush=True,
    )

    if remaining:
        pipe = load_pipeline(model_dir, quant_mode=args.quant, offload=args.offload)
        buf_ids: list[str] = []
        buf_u8: list[torch.Tensor] = []
        peak_vram_mb = 0.0
        t_start_wall = time.perf_counter()

        with timing_log_path.open("a" if args.resume else "w", encoding="utf-8") as tlog:
            for idx, task in enumerate(remaining):
                tid = task["task_id"]
                src_img = Image.open(task["source_path"]).convert("RGB")
                ref_img = Image.open(task["ref_path"]).convert("RGB")
                prompt = args.prompt.format(
                    source_style=task["source_style"].replace("_", " "),
                    target_style=task["target_style"].replace("_", " "),
                )
                gen = torch.Generator("cpu").manual_seed(args.seed + offset + len(buf_ids))

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = pipe(
                    prompt=prompt,
                    image=[src_img, ref_img],
                    width=args.gen_res,
                    height=args.gen_res,
                    output_resolution=args.cond_res,
                    true_cfg_scale=1.0,
                    use_kv_cache=not args.no_kv_cache,
                    generator=gen,
                ).images[0]
                torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_mb = max(peak_vram_mb, vram_mb)

                out_rgb = out.convert("RGB")
                if out_rgb.size != (512, 512):
                    out_512 = out_rgb.resize((512, 512), Image.Resampling.LANCZOS)
                else:
                    out_512 = out_rgb

                if args.save_png:
                    fname = tid.replace("/", "__") + ".png"
                    out_512.save(png_dir / fname, format="PNG", compress_level=1)

                buf_ids.append(tid)
                buf_u8.append(_pil_to_u8_chw(out_512))

                is_warmup = (len(done_ids) == 0 and idx == 0)
                if not is_warmup:
                    prior_times.append(dt_ms)
                rec = {
                    "idx": offset + len(buf_ids) - 1,
                    "task_id": tid,
                    "ms": round(dt_ms, 2),
                    "peak_vram_mb": round(vram_mb, 1),
                    "is_warmup": is_warmup,
                }
                tlog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tlog.flush()

                if len(buf_u8) >= args.batch_save or idx == len(remaining) - 1:
                    batches.append(_save_batch_atomic(raw_dir, offset, buf_ids, buf_u8))
                    offset += len(buf_ids)
                    buf_ids.clear()
                    buf_u8.clear()
                    elapsed = time.perf_counter() - t_start_wall
                    avg_ms = statistics.mean(prior_times) if prior_times else dt_ms
                    eta_min = (len(remaining) - 1 - idx) * (avg_ms / 1000.0) / 60.0
                    print(
                        f"  [{offset}/{len(tasks)}] last={dt_ms/1000:.2f}s avg={avg_ms/1000:.2f}s "
                        f"peak_vram={vram_mb:.0f}MB elapsed={elapsed/60:.1f}m eta={eta_min:.1f}m",
                        flush=True,
                    )

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    meta = {
        "kind": "qwen_image_21_turbo",
        "arm": args.arm_name,
        "n_tasks": offset,
        "gen_res": args.gen_res,
        "cond_res": args.cond_res,
        "quant": args.quant,
        "offload": args.offload,
        "use_kv_cache": not args.no_kv_cache,
        "batches": batches,
    }
    (raw_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    timing_summary = {
        "arm": args.arm_name,
        "n_measured": len(prior_times),
        "mean_ms_per_img": round(statistics.mean(prior_times), 2) if prior_times else None,
        "median_ms_per_img": round(statistics.median(prior_times), 2) if prior_times else None,
        "p95_ms_per_img": round(sorted(prior_times)[int(0.95 * (len(prior_times) - 1))], 2) if prior_times else None,
        "img_per_sec": round(1000.0 / statistics.mean(prior_times), 4) if prior_times else None,
        "gen_res": args.gen_res,
        "cond_res": args.cond_res,
        "quant": args.quant,
        "offload": args.offload,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "timing.json").write_text(json.dumps(timing_summary, indent=2), encoding="utf-8")
    print("[qwen-eval] Timing summary:", json.dumps(timing_summary), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-name", default="Qwen-Image-2.1-Turbo")
    ap.add_argument("--diffusers-src", default="")
    ap.add_argument("--quantize-to", default="", help="Optional dir to save pre-quantized checkpoint")
    ap.add_argument("--quantize-only", action="store_true")
    ap.add_argument("--transformer-bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--quant", default="auto", choices=["auto", "none", "bnb4", "bnb4_t8"])
    ap.add_argument("--offload", default="model", choices=["model", "sequential", "cuda"])
    ap.add_argument("--gen-res", type=int, default=512)
    ap.add_argument("--cond-res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-save", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-same", action="store_true")
    ap.add_argument("--same-only", action="store_true")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--no-kv-cache", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
