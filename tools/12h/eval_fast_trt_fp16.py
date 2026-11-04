"""eval_fast wrapper: force the TensorRT VAE decoder to the fp16 engine.

`eval_fast.py` derives the engine precision from `--vae-dtype`, but the only
built engine is fp16.  This wrapper pins the decoder precision to fp16 so the
torch-vs-TRT delta isolates the decoder backend instead of changing the VAE
compute dtype for the whole pipeline.

Usage (identical CLI to eval_fast.py):
  python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint ... \
      --decoder trt --vae-dtype fp32 --batch 8 --decode-batch 8
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(p))

import trt_vae_decoder  # noqa: E402

_orig_init = trt_vae_decoder.TrtVaeDecoder.__init__


def _init_fp16(self, engine_dir, vae, *, precision="fp16", device="cuda"):
    _orig_init(self, engine_dir, vae, precision="fp16", device=device)


trt_vae_decoder.TrtVaeDecoder.__init__ = _init_fp16

import eval_fast  # noqa: E402

if __name__ == "__main__":
    eval_fast.main()

