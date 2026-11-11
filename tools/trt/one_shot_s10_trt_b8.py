"""One-shot: c0 S10 decoder -> fp16 ONNX -> TRT b8 engine -> benchmark.

Reuses the repo's own to_fp16()/build() helpers so the engine is built with the
exact same STRONGLY_TYPED + ws=2GB settings as the paper baseline engine
(vae_decoder_512_b8_fp16.engine). Then times BOTH engines on the same GPU for a
fair decoder-only comparison.

3060 Windows python:
  python tools/trt/one_shot_s10_trt_b8.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.trt.convert_onnx_fp16 import infer, to_fp16
import tensorrt as trt
import tools.trt.build_trt_engines as _b
_b.logger = trt.Logger(trt.Logger.WARNING)  # module uses a global 'logger' that is never defined
build = _b.build
from tools.trt.bench_s10_trt import run_engine

ONNX_DIR = ROOT / "runs/trt/onnx"
ENG_DIR = ROOT / "runs/trt/engines_s10new"
SRC = ONNX_DIR / "s10new_c0_decoder_512.onnx"
DST = ONNX_DIR / "s10new_c0_decoder_512_fp16.onnx"
ENGINE = ENG_DIR / "s10new_c0_decoder_512_b8_fp16.engine"
OFFICIAL = ROOT / "runs/trt/engines" / "vae_decoder_512_b8_fp16.engine"


def main() -> None:
    import onnx
    # 1) fp32 -> uniform fp16 ONNX (same converter as paper)
    if not DST.exists():
        model, stats = to_fp16(infer(onnx.load(str(SRC))))
        onnx.save(model, str(DST))
        print(json.dumps({"step": "fp16_onnx", "mb": round(DST.stat().st_size / 1024 / 1024, 1),
                          **stats}), flush=True)
    else:
        print(f"[skip] {DST} exists", flush=True)

    # 2) build ONE fixed-shape b8 fp16 engine (STRONGLY_TYPED, ws=2GB)
    if not ENGINE.exists():
        row = build(DST, 512, 8, precision="fp16", workspace_mb=2048,
                    engine_dir=ENG_DIR)
        # build() names it vae_decoder_512_b8_fp16.engine; rename to our target
        built = Path(row["engine"])
        if built != ENGINE:
            built.replace(ENGINE)
        row["engine"] = str(ENGINE)
        print(json.dumps({"step": "build_engine", **row}), flush=True)
    else:
        print(f"[skip] {ENGINE} exists", flush=True)

    # 3) benchmark BOTH engines, same GPU, same timing code
    rows = []
    for name, p in (("official_49M", OFFICIAL), ("s10_c0_10_66M", ENGINE)):
        if not p.exists():
            print(f"skip {name}: {p} missing", flush=True)
            continue
        r = run_engine(p, 8, iters=50, warmup=10)
        r["variant"] = name
        rows.append(r)
        print(json.dumps({"step": "bench", **r}), flush=True)

    if len(rows) == 2:
        base = rows[0]["ms_per_img"]
        rows[1]["speedup_vs_official"] = round(base / rows[1]["ms_per_img"], 3)
        rows[1]["official_ms_per_img"] = round(base, 3)
    out = ENG_DIR / "one_shot_s10_trt_b8.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
