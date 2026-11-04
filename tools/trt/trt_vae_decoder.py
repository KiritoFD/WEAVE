"""Optional TensorRT backend for the SD1.5 VAE decoder (eval-side only).

Engines are fixed-shape per (size, batch).  Any miss (no engine / different
batch) falls back to the stock torch path, so behaviour is unchanged unless an
engine actually exists.  Never used unless ``--decoder trt`` is passed.

Engine lookup order:
  vae_decoder_{size}_b{batch}_{precision}.engine
"""
from __future__ import annotations

from pathlib import Path

import torch

from utils.inference import decode_latent


class TrtVaeDecoder:
    def __init__(self, engine_dir: str | Path, vae, *, precision: str = "fp16",
                 device: str = "cuda") -> None:
        self.engine_dir = Path(engine_dir)
        self.vae = vae
        self.precision = precision
        self.device = device
        self._cache: dict[tuple[int, int], tuple[object, torch.dtype] | None] = {}
        self.hits = 0
        self.fallbacks = 0
        self._warned = False

    def _load(self, size: int, batch: int):
        key = (size, batch)
        if key in self._cache:
            return self._cache[key]
        path = self.engine_dir / f"vae_decoder_{size}_b{batch}_{self.precision}.engine"
        if not path.exists():
            self._cache[key] = None
            return None
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            self._cache[key] = None
            return None
        ctx = engine.create_execution_context()
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        self._cache[key] = (ctx, dtype)
        print(f"[trt] loaded {path.name}")
        return self._cache[key]

    def usable(self, batch: int, size: int) -> bool:
        return self._load(size, batch) is not None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch, _, lat, _ = z.shape
        size = lat * 8
        entry = self._load(size, batch)
        if entry is None:
            self.fallbacks += 1
            if not self._warned:
                print(f"[trt] no engine for size={size} batch={batch} ({self.precision}); "
                      "falling back to torch")
                self._warned = True
            return decode_latent(self.vae, z, self.device)

        ctx, dtype = entry
        zc = z.contiguous().to(dtype=dtype)
        out = torch.empty((batch, 3, size, size), device=zc.device, dtype=dtype)
        ctx.set_tensor_address("z", zc.data_ptr())
        ctx.set_tensor_address("image", out.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        self.hits += 1
        return out

    def summary(self) -> dict:
        return {
            "decoder": "trt",
            "precision": self.precision,
            "engine_dir": str(self.engine_dir),
            "engine_hits": self.hits,
            "torch_fallbacks": self.fallbacks,
        }

