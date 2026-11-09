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
        g = torch.Generator().manual_seed(600 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(40, 32, 16, 16, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(40, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(40)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.4552, "image_size": 512}, p)
        caches.append(p)

    cases = [
        ("mixed_default", []),
        ("mixed_explicit", ["--mixed-batch"]),
        ("single_pair_legacy", ["--single-pair"]),
        ("mixed_freq_stats", ["--frequency-stats"]),
        ("mixed_joint", ["--mixed-batch", "--objective", "bidir_joint"]),
        ("mixed_heads2", ["--mixed-batch", "--heads-out", "2"]),
        ("mixed_pairemb", ["--mixed-batch", "--style-cond", "pair"]),
        ("mixed_dist", ["--mixed-batch", "--dist-loss", "global", "--lambda-dist", "0.05",
                        "--style-enc-epochs", "2", "--ode-steps-train", "1"]),
        ("mixed_ot", ["--mixed-batch", "--ot-ratio", "0.5"]),
    ]
    ok = 0
    for name, extra in cases:
        out = tmp / f"run_{name}"
        cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", name,
               "--caches", *[str(c) for c in caches], "--output", str(out),
               "--steps", "12", "--batch", "40", "--width", "32", "--depth", "1",
               "--heads", "2", "--log-interval", "6", "--save-interval", "12",
               "--warmup", "2", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name} FAILED")
            print(r.stdout[-2500:])
            print(r.stderr[-3000:])
            break
        s = json.loads((out / "train_summary.json").read_text())
        losses = ["%.3f" % h["loss"] for h in s["loss_history"]]
        print(f"PASS {name:20s} losses={losses}")
        ok += 1

    # batch 40 with 10 pairs x 2 directions = 2 per cell; verify the mix really happens
    if ok == len(cases):
        from sbflow.train5 import _mixed_pair_ids
        pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
        g = torch.Generator().manual_seed(0)
        pa, pb, d = _mixed_pair_ids(pairs, 40, "cpu", g)
        from collections import Counter
        cells = Counter(zip(pa.tolist(), pb.tolist(), d.tolist()))
        print(f"  mixed cells: {len(cells)} distinct (pair, direction), "
              f"counts={sorted(set(cells.values()))}")
        assert len(cells) == 20 and set(cells.values()) == {2}, cells
        print("PASS mixed batch covers all 20 (pair, direction) cells equally")
    print(f"MIXED BATCH SMOKE: {ok}/{len(cases)} passed")
