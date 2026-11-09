"""End-to-end: a real training run with L_dist must report a non-zero dist_grad_norm.

This is the check V10 lacked. The V10 code produced a non-zero L_dist *value* with a
zero gradient, so `float(ld) != 0.0` passed while the distribution loss trained nothing.
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
        g = torch.Generator().manual_seed(4200 + i)
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
           "--steps", "3", "--batch", "40", "--width", "32", "--depth", "1",
           "--heads", "2", "--warmup", "2", "--log-interval", "1",
           "--save-interval", "100", "--equiv-batch", "40", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r, out


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    for mode in ("global", "both"):
        r, out = run(tmp, f"dist_{mode}",
                     ["--dist-loss", mode, "--style-enc", "fixed",
                      "--lambda-dist", "0.1", "--ode-steps-train", "1"])
        assert r.returncode == 0, f"{mode}:\n{r.stdout[-2500:]}\n{r.stderr[-2500:]}"
        s = json.loads((out / "train_summary.json").read_text())
        hist = s["loss_history"]
        gn = [h.get("dist_grad_norm") for h in hist if h.get("dist_grad_norm") is not None]
        assert gn, f"dist_loss={mode} never recorded dist_grad_norm"
        assert gn[0] > 1e-12, f"dist_loss={mode} gradient is zero ({gn[0]})"
        rho = [h.get("rho_swd_over_bridge") for h in hist
               if h.get("rho_swd_over_bridge") is not None]
        bg = [h.get("bridge_grad_norm") for h in hist
              if h.get("bridge_grad_norm") is not None]
        assert rho and bg, f"dist_loss={mode} did not record rho / bridge_grad_norm"
        assert rho[0] > 0, f"rho is zero for {mode}"
        print(f"PASS dist_loss={mode}: dist_grad_norm={gn[0]:.3e}, "
              f"bridge_grad_norm={bg[0]:.3e}, rho={rho[0]:.4f}, "
              f"losses={[round(h['loss'], 3) for h in hist]}")

    # A run WITHOUT dist loss must not require the gradient check.
    r, out = run(tmp, "nodist", [])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    print("PASS training without L_dist still works")

    # With grad microbatching (the heavy-arm setting), the diagnostic must still be
    # recorded. V11 bug: `rec = r` is overwritten by every microbatch, so the step-1
    # gradient numbers were silently dropped whenever grad_microbatch > 0.
    r, out = run(tmp, "dist_mb",
                 ["--dist-loss", "both", "--style-enc", "fixed", "--lambda-dist", "0.1",
                  "--ode-steps-train", "1", "--grad-microbatch", "20"])
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    s = json.loads((out / "train_summary.json").read_text())
    hist = s["loss_history"]
    rho = [h.get("rho_swd_over_bridge") for h in hist
           if h.get("rho_swd_over_bridge") is not None]
    assert rho, ("grad_microbatch>0 dropped rho from the history; the step-1 diagnostic "
                 "must survive microbatching")
    print(f"PASS dist_loss with grad_microbatch=20 keeps rho={rho[0]:.4f}")

print("DIST GRADIENT END-TO-END PASSED")
