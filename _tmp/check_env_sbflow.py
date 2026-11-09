import sys, importlib, json
out = {}
for m in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"):
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        out[m] = f"MISSING: {type(e).__name__}"
try:
    from diffusers import AutoencoderDC
    out["AutoencoderDC"] = "OK"
except Exception as e:
    out["AutoencoderDC"] = f"FAIL: {type(e).__name__}: {e}"
try:
    import torch
    out["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
except Exception as e:
    out["cuda"] = f"err {e}"
print(json.dumps(out, indent=2))
