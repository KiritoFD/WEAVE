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
        g = torch.Generator().manual_seed(400 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(40, 32, 16, 16, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(40, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(40)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.4552, "image_size": 512}, p)
        caches.append(p)

    cases = [
        ("pivot_cfm", ["--pivot", "--objective", "single_velocity", "--eps", "0.0"]),
        ("pivot_bridge", ["--pivot"]),
        ("pivot_dist", ["--pivot", "--dist-loss", "global", "--lambda-dist", "0.05",
                        "--style-enc-epochs", "2", "--ode-steps-train", "1"]),
        ("pivot_online", ["--pivot", "--online", "sde", "--online-steps", "2",
                          "--online-warmup", "2"]),
    ]
    ok = 0
    for name, extra in cases:
        out = tmp / f"run_{name}"
        cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", name,
               "--caches", *[str(c) for c in caches], "--output", str(out),
               "--steps", "10", "--batch", "16", "--width", "32", "--depth", "1",
               "--heads", "2", "--log-interval", "5", "--save-interval", "10",
               "--warmup", "2", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name} FAILED")
            print(r.stdout[-2000:])
            print(r.stderr[-2500:])
            break
        s = json.loads((out / "train_summary.json").read_text())
        ck = torch.load(out / "model_step10.pt", map_location="cpu", weights_only=False)
        assert ck["model_kwargs"]["num_styles"] == 6, ck["model_kwargs"]
        losses = ["%.3f" % h["loss"] for h in s["loss_history"]]
        print(f"PASS {name:14s} pairs={len(s['unordered_pairs'])} losses={losses}")
        ok += 1

    # pivot inference composition
    if ok == len(cases):
        from sbflow.infer5 import load_model5
        from sbflow.pivot import pivot_transfer
        _, model, stats = load_model5(str(tmp / "run_pivot_bridge" / "model_step10.pt"), "cpu")
        z = torch.randn(3, 32, 16, 16)
        for (i, j) in [(0, 4), (4, 0), (1, 3)]:
            o = pivot_transfer(model, z, i, j, 5, stats, 0.04, steps=2)
            assert torch.isfinite(o).all(), (i, j)
        print("PASS pivot_transfer composition (finite for 3 directions)")
    print(f"PIVOT SMOKE: {ok}/{len(cases)} passed")
