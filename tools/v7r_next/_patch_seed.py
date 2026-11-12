"""Patch arms.py probe_step random branch to honour V7R_NEXT_SEED.

P0.1 requires >=5 independent random-direction seeds. The original uses the
global torch RNG; we set it per eval process from an env var so seeds are
reproducible AND isolated.
"""
import io
from pathlib import Path
P = Path("/mnt/i/Latent_Style/WEAVE/tools/v7r/arms.py")
s = io.open(P, encoding="utf-8").read()
old = """            if cfg["random"]:
                g = torch.randn(z0.shape, device=z0.device,
                                dtype=torch.float32)
                g = rms_normalize(g)"""
new = """            if cfg["random"]:
                import os
                _sd = int(os.environ.get("V7R_NEXT_SEED", "0"))
                torch.manual_seed(_sd)
                g = torch.randn(z0.shape, device=z0.device,
                                dtype=torch.float32)
                g = rms_normalize(g)"""
assert old in s, "probe_step anchor not found"
io.open(P, "w", encoding="utf-8").write(s.replace(old, new))
print("patched arms.py random seed isolation")
