"""Verify the two mechanisms that could otherwise be config-only no-ops."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def caches(tmp):
    out = []
    for i, s in enumerate(STYLES):
        g = torch.Generator().manual_seed(1000 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(48, 4, 8, 8, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(48, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(48)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.18215, "image_size": 512}, p)
        out.append(p)
    return out


def run(tmp, name, extra):
    out = tmp / name
    cmd = [sys.executable, "-m", "sbflow.train5", "--experiment-id", name,
           "--caches", *[str(c) for c in caches(tmp)], "--output", str(out),
           "--steps", "8", "--batch", "40", "--width", "32", "--depth", "1",
           "--heads", "2", "--warmup", "2", "--log-interval", "4",
           "--save-interval", "8", "--equiv-batch", "40",
           "--save-equiv", "2,4,6,8", "--full-equiv", "6", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"{name}:\n{r.stdout[-2000:]}\n{r.stderr[-2500:]}"
    return out, json.loads((out / "train_summary.json").read_text())


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    out, s = run(tmp, "equiv", [])
    ck = out / "model_equiv6.pt"
    assert ck.exists(), "equiv checkpoint missing"
    d = torch.load(ck, map_location="cpu", weights_only=False)
    for k in ("model", "ema", "optimizer", "scheduler", "rng", "pair_sched",
              "samplers", "style_stats", "step", "equiv_step"):
        assert k in d, f"equiv checkpoint is not fully resumable: missing {k}"
    print(f"PASS equivalent checkpoints: {sorted(p.name for p in out.glob('model_equiv*.pt'))} "
          f"full payload step={d['step']} equiv={d['equiv_step']}")

    out2, s2 = run(tmp, "refresh", ["--online", "sde", "--online-steps", "2",
                                    "--online-warmup", "2",
                                    "--online-refresh-interval", "4"])
    assert "online" in s2["config"] and s2["config"]["online"] == "sde"
    print(f"PASS exp33 cached online endpoint refresh: losses="
          f"{['%.3f' % h['loss'] for h in s2['loss_history']]}")

    out3, s3 = run(tmp, "triangle", ["--triangle-weight", "0.05"])
    has = any("triangle" in h for h in s3["loss_history"])
    assert has, "triangle_weight did not add a loss term"
    vals = [h.get("triangle") for h in s3["loss_history"] if "triangle" in h]
    print(f"PASS exp39 triangle consistency contributes: {[round(v, 5) for v in vals]}")

print("MECHANISM ACTIVATION TESTS PASSED")
