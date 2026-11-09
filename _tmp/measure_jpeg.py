import io
from pathlib import Path

from PIL import Image

SRC = Path("/mnt/i/Latent_Style/WEAVE/results/D5-512/cut")
files = sorted(SRC.glob("*.png"))[:8]
for q in (85, 90, 92):
    total_png = 0
    total_jpg = 0
    for p in files:
        total_png += p.stat().st_size
        im = Image.open(p).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        total_jpg += buf.tell()
    n = len(files)
    print(f"q={q}: png {total_png/n/1024:.0f} KB/img, jpg {total_jpg/n/1024:.0f} KB/img, "
          f"ratio {total_png/total_jpg:.2f}x, size={Image.open(files[0]).size}")
