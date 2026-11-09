import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

CASES = [
    ("dist_global", ["--dist-loss", "global", "--lambda-dist", "0.05",
                     "--style-enc-epochs", "3", "--ode-steps-train", "1"]),
    ("dist_local_fixed", ["--dist-loss", "local", "--style-enc", "fixed",
                          "--lambda-dist", "0.05", "--ode-steps-train", "1"]),
    ("dist_both", ["--dist-loss", "both", "--lambda-dist", "0.05",
                   "--style-enc-epochs", "3", "--ode-steps-train", "1"]),
    ("online_sde", ["--online", "sde", "--online-steps", "2", "--online-warmup", "2"]),
    ("online_ode", ["--online", "ode", "--online-steps", "2", "--online-warmup", "2"]),
    ("online_mix", ["--online", "sde", "--online-steps", "2", "--online-mix", "0.5",
                    "--online-warmup", "2"]),
    ("online_replay", ["--online", "sde", "--online-steps", "2", "--online-replay",
                       "--online-warmup", "2"]),
]

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    caches = []
    for i, s in enumerate(STYLES):
        g = torch.Generator().manual_seed(300 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(40, 32, 16, 16, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(40, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(40)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.4552, "image_size": 512}, p)
        caches.append(p)

    ok = 0
    for name, extra in CASES:
        out = tmp / f"run_{name}"
        cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", name,
               "--caches", *[str(c) for c in caches], "--output", str(out),
               "--steps", "5", "--batch", "40", "--width", "32", "--depth", "1",
               "--heads", "2", "--log-interval", "5", "--save-interval", "5",
               "--warmup", "2", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name} FAILED")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            break
        s = json.loads((out / "train_summary.json").read_text())
        losses = ["%.3f" % h["loss"] for h in s["loss_history"]]
        print(f"PASS {name:18s} losses={losses}")
        ok += 1
    print(f"DE SMOKE: {ok}/{len(CASES)} passed")
