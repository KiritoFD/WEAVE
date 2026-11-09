import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    caches = []
    for i, s in enumerate(STYLES):
        g = torch.Generator().manual_seed(500 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(40, 32, 16, 16, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(40, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(40)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.4552, "image_size": 512}, p)
        caches.append(p)

    out = tmp / "run_cov"
    cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", "cov",
           "--caches", *[str(c) for c in caches], "--output", str(out),
           "--steps", "10", "--batch", "16", "--width", "32", "--depth", "1",
           "--heads", "2", "--log-interval", "5", "--save-interval", "10",
           "--warmup", "2", "--online", "sde", "--online-steps", "2",
           "--online-warmup", "2", "--online-dynamic-cov"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(1)
    s = json.loads((out / "train_summary.json").read_text())
    hist = s["loss_history"]
    covs = [h.get("cross_cov_mean") for h in hist if "cross_cov_mean" in h]
    print(f"PASS arm31 dynamic cross covariance: steps={len(hist)} "
          f"cov_values={[round(c, 4) for c in covs]}")
