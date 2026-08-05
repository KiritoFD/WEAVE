# ONNX export and TensorRT compilation (VAE only)

The reported **73.0 s / 750** generation-only timing uses an accelerated VAE path.
ONNX graphs are **not** shipped in the minimal `code.zip` (each graph is ~65–99 MB).
Rebuild them from the public SD1.5 EMA VAE as below.

## Environment
- GPU: NVIDIA RTX 3060 12 GB (or similar)
- PyTorch with CUDA, ONNX Runtime, TensorRT 10.x
- Resolution: 512×512, precision FP16

## Export (fixed batch 8)

Export **encoder** and **decoder** separately. WEAVE remains PyTorch; only the frozen VAE is exported.

| File | Approx. size | Role |
|------|--------------|------|
| `encoder_b8.onnx` | ~68 MB | VAE encode, batch=8 |
| `decoder_b8.onnx` | ~99 MB | VAE decode, batch=8 |

Notes:
1. Apply decoder `scaling_factor = 0.18215` **outside** the ONNX graph.
2. Prefer fixed batch size (enables stronger TRT tactics) over fully dynamic axes.
3. Place graphs under `data/onnx_exported/` relative to the package root.

In the full project tree, decoder export helpers live under `experiments/inference_speed/`
(e.g. `export_decoder.py`). Encoder export uses `torch.onnx.export` on the SD1.5 EMA VAE encode path.

## TensorRT compile

| Setting | Recommended |
|---------|-------------|
| Batch | **8** |
| Precision | FP16 |
| Workspace | **2 GB** |
| Decoder build time | ~15 min (first time; then cache) |

On RTX 3060 12 GB:
- batch **16** often OOMs during TRT build (~10.7 GB requested) and can be **slower** after fallback.
- workspace 4 GB improves decode by only ~0.6%.

## Audited pipeline timing (batch 8, TRT ws=2 GB)

```
image → [VAE Encoder ONNX] → latent → [WEAVE PyTorch, K=8] → latent → [VAE Decoder TRT] → image
```

| Stage | Wall (750) | Per image |
|-------|------------|-----------|
| Encoder ONNX + WEAVE | 14.0 s | 18.7 ms |
| Decoder TRT | 59.0 s | 78.6 ms |
| **Total** | **73.0 s** | **97.3 ms** |

See `docs/generation_only_timing_summary.json` for the ledger.
