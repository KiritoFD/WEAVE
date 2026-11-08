"""Build side-by-side comparison sheets: source | DCAE no-op | generated."""
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path("/mnt/i/Latent_Style/WEAVE")
MAN = REPO / "runs/blog_sbflow/eval_ImpUki_noOT/manifest.json"
OUT = REPO / "runs/blog_sbflow/eval_ImpUki_noOT/sheets"
OUT.mkdir(parents=True, exist_ok=True)


def main(n: int = 6) -> None:
    d = json.loads(MAN.read_text(encoding="utf-8"))
    tasks = d["tasks"]
    # pick a spread: first n/2 A->B, last n/2 B->A
    a = [t for t in tasks if t["source_style"] == "Impressionism"][: n // 2]
    b = [t for t in tasks if t["source_style"] == "Ukiyo_e"][: n // 2]
    picked = a + b
    for i, t in enumerate(picked):
        src = Image.open(REPO / t["source_rel_path"]).convert("RGB").resize((256, 256), Image.BICUBIC)
        noop = Image.open(t["dcae_noop"]).convert("RGB").resize((256, 256), Image.BICUBIC)
        gen = Image.open(t["generated"]).convert("RGB").resize((256, 256), Image.BICUBIC)
        sheet = Image.new("RGB", (256 * 3, 256))
        sheet.paste(src, (0, 0))
        sheet.paste(noop, (256, 0))
        sheet.paste(gen, (512, 0))
        name = f"{i:02d}_{t['source_style']}_to_{t['target_style']}.png"
        sheet.save(OUT / name)
        print(f"{name}  <- {t['task_id']}")
    print("sheets in", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
