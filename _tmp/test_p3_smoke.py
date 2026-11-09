"""Smoke-test the P3 path: pivot training + unconditional sampling + scoring.

Pivot mode is a different code path from the cross-domain arms (pairs become
{(style, pivot)} and an extra Gaussian "style" is appended), so it is worth verifying
end to end on synthetic data before spending 4000 real updates on it.
"""
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
        g = torch.Generator().manual_seed(7100 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(48, 4, 8, 8, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(48, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(48)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.18215, "image_size": 512}, p)
        out.append(p)
    return out


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    cs = caches(tmp)
    cfg = json.loads(Path("configs/v11/p3.json").read_text())
    cfg.update(experiment_id="p3smoke", output=str(tmp / "run"), steps=4, batch=40,
               width=32, depth=1, heads=2, warmup=2, log_interval=1,
               save_interval=4, save_equiv="", full_equiv=0, lr_cycle_steps=4)
    cp = tmp / "p3.json"
    cp.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "sbflow.train5", "--config", str(cp),
                        "--caches", *[str(c) for c in cs]],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert "pivot mode" in r.stdout, "pivot mode was not engaged"
    ck = tmp / "run" / "model_step4.pt"
    assert ck.exists(), sorted(p.name for p in (tmp / "run").glob("*.pt"))
    d = torch.load(ck, map_location="cpu", weights_only=False)
    assert d["config"]["pivot"] is True
    assert d["style_stats"]["style_mean"].shape[0] == 6, "pivot style was not appended"
    print(f"PASS pivot training: pairs={d['config'].get('pivot')} "
          f"styles={d['style_stats']['style_mean'].shape[0]} (5 real + 1 pivot)")
    print("P3 TRAINING PATH OK")
    print("NOTE: gen_uncond/score_uncond need SD-VAE + CSD, so they are exercised on the"
          " GPU box rather than here.")
