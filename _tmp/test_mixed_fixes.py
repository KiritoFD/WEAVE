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
        g = torch.Generator().manual_seed(700 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(40, 32, 16, 16, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(40, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(40)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.4552, "image_size": 512}, p)
        caches.append(p)

    # 1) mixed + dist_loss must actually contribute
    out = tmp / "run_dist"
    cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", "distcheck",
           "--caches", *[str(c) for c in caches], "--output", str(out),
           "--steps", "6", "--batch", "40", "--width", "32", "--depth", "1",
           "--heads", "2", "--log-interval", "3", "--save-interval", "6",
           "--warmup", "2", "--mixed-batch", "--dist-loss", "global",
           "--lambda-dist", "0.5", "--style-enc-epochs", "2", "--ode-steps-train", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    s = json.loads((out / "train_summary.json").read_text())
    dists = [h.get("dist") for h in s["loss_history"] if "dist" in h]
    assert dists and all(d is not None and d != 0.0 for d in dists), f"dist not applied: {dists}"
    print(f"PASS mixed+dist_loss contributes: dist values = {[round(d, 6) for d in dists]}")

    # 2) all per-sample objectives must run through the mixed sampler
    for obj, extra in [("single_velocity", []), ("score_current", []),
                       ("bidir", ["--target", "endpoint"]), ("bidir_joint", [])]:
        o = tmp / f"run_{obj}"
        cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", obj,
               "--caches", *[str(c) for c in caches], "--output", str(o),
               "--steps", "6", "--batch", "40", "--width", "32", "--depth", "1",
               "--heads", "2", "--log-interval", "3", "--save-interval", "6",
               "--warmup", "2", "--mixed-batch", "--objective", obj, *extra]
        rr = subprocess.run(cmd, capture_output=True, text=True)
        assert rr.returncode == 0, f"{obj} failed:\n{rr.stdout[-1500:]}\n{rr.stderr[-2000:]}"
        assert "mixed" in rr.stdout, f"{obj} did not use the mixed path"
        ss = json.loads((o / "train_summary.json").read_text())
        print(f"PASS mixed sampler used by {obj:16s} "
              f"losses={['%.3f' % h['loss'] for h in ss['loss_history']]}")
    print("MIXED-BATCH FIX VERIFICATION PASSED")
