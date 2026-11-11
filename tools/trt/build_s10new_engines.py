"""Build TRT FP16 engines for the new decoders + official b12 (3060).

Official b8 engine already exists in runs/trt/engines (cached by the paper run).
New engines go to runs/trt/engines_s10new to avoid overwriting baseline assets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(ROOT))

from tools.trt.build_trt_engines import build

ONNX = ROOT / "runs/trt/onnx"
ENG_NEW = ROOT / "runs/trt/engines_s10new"

rows = []
# official baseline at batch 12 (b8 already exists)
rows.append(build(ONNX / "vae_decoder_512_fp16.onnx", 512, 12,
                  precision="fp16", workspace_mb=2048, engine_dir=ENG_NEW))
print(json.dumps(rows[-1]), flush=True)
for v in ("old_s10", "c0", "c1"):
    onnx = ONNX / f"s10new_{v}_decoder_512.onnx"
    for b in (8, 12):
        rows.append(build(onnx, 512, b, precision="fp16",
                          workspace_mb=2048, engine_dir=ENG_NEW))
        print(json.dumps(rows[-1]), flush=True)

(ENG_NEW / "build_s10new_report.json").write_text(json.dumps(rows, indent=1),
                                                  encoding="utf-8")
print("wrote build report with", len(rows), "engines", flush=True)
