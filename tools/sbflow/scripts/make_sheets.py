"""Build source | DCAE no-op | generated contact sheets for any eval directory."""
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path("/mnt/i/Latent_Style/WEAVE")


def main(eval_name: str, n: int = 6) -> None:
    man = REPO / "runs/blog_sbflow" / eval_name / "manifest.json"
    out = man.parent / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads(man.read_text(encoding="utf-8"))
    tasks = d["tasks"]
    a = [t for t in tasks if t["source_style"] == "Impressionism"][: n // 2]
    b = [t for t in tasks if t["source_style"] == "Ukiyo_e"][: n // 2]
    for i, t in enumerate(a + b):
        src = Image.open(REPO / t["source_rel_path"]).convert("RGB").resize((256, 256), Image.BICUBIC)
        noop = Image.open(t["dcae_noop"]).convert("RGB").resize((256, 256), Image.BICUBIC)
        gen = Image.open(t["generated"]).convert("RGB").resize((256, 256), Image.BICUBIC)
        sheet = Image.new("RGB", (768, 256))
        sheet.paste(src, (0, 0))
        sheet.paste(noop, (256, 0))
        sheet.paste(gen, (512, 0))
        name = f"{i:02d}_{t['source_style']}_to_{t['target_style']}.png"
        sheet.save(out / name)
        print(name)
    print("sheets ->", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
