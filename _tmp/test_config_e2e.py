"""End-to-end config-driven short training + resume test on synthetic caches."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def make(tmp):
    out = []
    for i, s in enumerate(STYLES):
        g = torch.Generator().manual_seed(1100 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(48, 4, 8, 8, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(48, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(48)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.18215, "image_size": 512}, p)
        out.append(p)
    return out


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    caches = make(tmp)
    cfg = json.loads(Path("configs/sbflow/exp01.json").read_text())
    cfg.update(experiment_id="cfgtrain", output=str(tmp / "run"), steps=6,
               batch=40, width=32, depth=1, heads=2, warmup=2,
               log_interval=3, save_interval=6, equiv_batch=40,
               save_equiv="2,4,6", full_equiv=4, lr_cycle_steps=6,
               equiv_steps=6)
    cp = tmp / "config.json"
    cp.write_text(json.dumps(cfg))
    cmd = [sys.executable, "-m", "sbflow.train5", "--config", str(cp),
           "--caches", *[str(c) for c in caches]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    out = tmp / "run"
    files = sorted(p.name for p in out.glob("*.pt"))
    assert "model_equiv4.pt" in files, files
    ck = torch.load(out / "model_equiv4.pt", map_location="cpu", weights_only=False)
    assert ck["equiv_step"] == 4 and ck["step"] == 4
    print(f"PASS config-driven train: {files}")

    cfg2 = dict(cfg)
    cfg2.update(experiment_id="cfgfork", output=str(tmp / "fork"), steps=2,
                resume=str(out / "model_equiv4.pt"), save_equiv="2,4,6", full_equiv=4)
    cp2 = tmp / "fork.json"
    cp2.write_text(json.dumps(cfg2))
    r2 = subprocess.run([sys.executable, "-m", "sbflow.train5", "--config", str(cp2),
                         "--caches", *[str(c) for c in caches]],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout[-2500:] + r2.stderr[-2500:]
    assert (tmp / "fork" / "model_step6.pt").exists(), "fork must end at global step 6"
    print("PASS config-driven fork: equiv4 + 2 -> global step 6")
print("CONFIG TRAIN/RESUME E2E PASSED")
