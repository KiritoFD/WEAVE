"""Add 'v7r_next' to --gen choices (check the choices line specifically)."""
import io
from pathlib import Path
P = Path("/mnt/i/Latent_Style/WEAVE/tools/v6_next/r2_eval.py")
s = io.open(P, encoding="utf-8").read()
old = '                 "saved_images", "relative", "v7", "v7r"],'
new = '                 "saved_images", "relative", "v7", "v7r", "v7r_next"],'
if new in s:
    print("choices already patched")
elif old in s:
    io.open(P, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print("patched --gen choices")
else:
    # show the real line for diagnosis
    for l in s.splitlines():
        if '"saved_images", "relative"' in l:
            print("FOUND:", repr(l))
    raise SystemExit("choices anchor not found")
