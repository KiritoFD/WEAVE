# Installation

This guide covers system requirements, environment creation, dependency
installation, and verification for the WEAVE package. WEAVE is a
run-from-source project (no `setup.py` / `pyproject.toml`), so "installation"
means setting up the Python environment and confirming that the package
imports cleanly.

---

## 1. System Requirements

| Component | Minimum | Recommended / Tested |
|-----------|---------|----------------------|
| **OS**        | Linux (Ubuntu 20.04+) or Windows 10/11            | Ubuntu 22.04, Windows 11 |
| **Python**    | 3.10                                              | 3.10 / 3.11              |
| **GPU**       | NVIDIA GPU, 12 GB VRAM                            | RTX 3060 12 GB           |
| **CUDA**      | 11.8                                              | 11.8 / 12.1              |
| **PyTorch**   | 2.0+                                              | 2.1+                     |
| **Disk**      | ~50 GB (datasets + latent caches + checkpoints)   | SSD recommended          |
| **RAM**       | 16 GB                                             | 32 GB                    |

> ℹ️ **Single-GPU friendly.** The canonical 15-epoch run trains from scratch on
> a single RTX 3060 12 GB with `batch_size=8` and `bf16` mixed precision.
> Multi-GPU is not required.

---

## 2. Create the Python Environment

A dedicated environment is strongly recommended to avoid dependency conflicts
with the metric backbones (DINO, CLIP, LPIPS).

### Option A — conda (recommended)

```bash
conda create -n weave python=3.10 -y
conda activate weave
```

### Option B — venv

```bash
python3.10 -m venv .venv
# Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install PyTorch with the Correct CUDA Build

PyTorch is **not** pinned in `requirements.txt` because the right wheel depends
on your CUDA version. Install it explicitly first so that subsequent `pip
install -r requirements.txt` resolves the metric libraries against your local
Torch build.

```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## 4. Install Runtime Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

For development and running the test suite:

```bash
pip install -r requirements-dev.txt
```

---

## 5. Verify the Installation

WEAVE is imported as a top-level package named `WEAVE` (uppercase). Run from the
repository root:

```bash
# Smoke test: imports and model factory
python -c "from WEAVE import WEAVE, build_model_from_config; print('WEAVE package OK')"

# Build the canonical model from config.json (does not load weights)
python -c "from WEAVE import build_model_from_config; import json; c=json.load(open('config.json')); m=build_model_from_config(c); print('params (M):', sum(p.numel() for p in m.parameters())/1e6)"
```

Run the test suite (optional but recommended):

```bash
pytest -q
```

---

## 6. Hardware Notes (RTX 3060 12 GB, Batch Size 8)

The canonical configuration was validated on a single RTX 3060 12 GB under the
following settings — all defined in `config.json` and `inference.json`:

| Setting                      | Value      | Where           |
|------------------------------|------------|-----------------|
| `training.batch_size`        | 8          | `config.json`   |
| `training.use_amp`           | `true`     | `config.json`   |
| `training.amp_dtype`         | `bf16`     | `config.json`   |
| `training.grad_clip_norm`    | 1.0        | `config.json`   |
| `training.fused_adamw`       | `true`     | `config.json`   |
| `full_eval.batch_size`       | 16         | `inference.json`|
| `full_eval.num_steps`        | 8          | `inference.json`|
| `model.endpoint_adain_scale` | 1.0 → 2.0  | train → eval    |

Peak VRAM during training is ~10.5 GB (well within the 12 GB budget). Evaluation
peaks lower thanks to no-gradient inference, but VAE decoding of large batches
can spike; `full_eval.vae_decode_batch_size` (default 16) can be reduced if you
hit OOM during evaluation.

---

## 7. Troubleshooting

### CUDA version mismatch

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

The installed PyTorch wheel does not match your driver's CUDA version. Reinstall
PyTorch with the correct `--index-url` (see §3). Check with
`nvidia-smi` (driver CUDA) vs `torch.version.cuda` (PyTorch CUDA).

### Out-of-memory (OOM) during training

- Reduce `training.batch_size` in `config.json` (8 → 4 → 2). Gradient
  accumulation is not enabled by default; prefer keeping `batch_size` a power
  of two.
- Confirm `training.use_amp` is `true` and `training.amp_dtype` is `bf16`.
- Enable `training.fused_adamw: true` (default) — it lowers memory and is faster
  on Ampere and newer.
- Clear the latent cache (`runs/cache/`) if it has grown stale; it is
  safe to delete and will be rebuilt.

### Out-of-memory (OOM) during evaluation

- Lower `full_eval.batch_size` and `full_eval.vae_decode_batch_size` in
  `inference.json` (e.g. 16 → 8 → 4).
- Set `full_eval.lpips_chunk_size` (default 4) lower if LPIPS computation OOMs.
- `full_eval.generation_batch_size` / `metric_batch_size` set to `0` auto-tune;
  set them explicitly to a small value if auto-tuning overshoots.

### ONNX / TensorRT VAE decoder (optional)

For faster VAE decoding at evaluation time, WEAVE supports an ONNX and TensorRT
backend. These are **optional** and not required for any reported result.

| Field (in `inference.json`)        | Default | Purpose                              |
|------------------------------------|---------|--------------------------------------|
| `full_eval.vae_onnx_decoder`       | `""`    | Path to an ONNX VAE decoder file     |
| `full_eval.vae_onnx_tensorrt`      | `false` | Enable TensorRT acceleration for it  |
| `full_eval.vae_compile_decoder`    | `false` | `torch.compile` the VAE decoder      |

Leave these at their defaults unless you have a pre-exported ONNX decoder.

### `import weave` fails

The package is named `WEAVE` (uppercase). Use `from WEAVE import ...` or
`import WEAVE`, not `import weave`. The directory name matches the import name
on case-sensitive filesystems (Linux), so preserve the casing when cloning.

### Missing latent cache on first run

The first training/evaluation run builds latent caches under `runs/cache/`.
This is expected and can take a while for large datasets. The cache is safe to
delete and rebuild; nothing in `runs/` is version-controlled.

---

## 8. Next Steps

- Configure datasets — see [DATA.md](DATA.md) (forthcoming).
- Reproduce the paper tables — see [REPRODUCE.md](REPRODUCE.md) (forthcoming).
- Browse experiments — see [EXPERIMENTS.md](EXPERIMENTS.md) (forthcoming).

For the release history, see [CHANGELOG.md](CHANGELOG.md).
