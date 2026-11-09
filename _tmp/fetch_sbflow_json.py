"""Fetch only the evaluation JSONs from 3060 into the local runs/blog_sbflow tree."""
import subprocess
from pathlib import Path

REMOTE = "/mnt/i/Latent_Style/WEAVE/runs/blog_sbflow"
LOCAL = Path("runs/blog_sbflow")
FILES = [
    "eval_ImpUki_noOT/manifest.json",
    "eval_ImpUki_noOT_step2000/manifest.json",
    "eval_ImpUki_noOT_step4000/manifest.json",
    "eval_ImpUki_noOT_step6000/manifest.json",
    "eval_ImpUki_noOT_step8000/manifest.json",
    "eval_ImpUki_OT0.5_step2000/manifest.json",
    "eval_ImpUki_OT0.5_step4000/manifest.json",
    "eval_ImpUki_OT0.5_step6000/manifest.json",
    "eval_ImpUki_OT0.5_step8000/manifest.json",
]
SCORES = [
    "eval_ImpUki_noOT_step2000/style_scores.json",
    "eval_ImpUki_noOT_step4000/style_scores.json",
    "eval_ImpUki_noOT_step6000/style_scores.json",
    "eval_ImpUki_noOT_step8000/style_scores.json",
    "eval_ImpUki_OT0.5_step2000/style_scores.json",
    "eval_ImpUki_OT0.5_step4000/style_scores.json",
    "eval_ImpUki_OT0.5_step6000/style_scores.json",
    "eval_ImpUki_OT0.5_step8000/style_scores.json",
]


def pull(rel: str) -> None:
    dst = LOCAL / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith("manifest.json"):
        src = f"{REMOTE}/{rel}"
    else:
        src = f"{REMOTE}/{rel}"
    with dst.open("wb") as f:
        subprocess.run(
            ["ssh", "3060", f"wsl cat {src}"],
            stdout=f,
            check=True,
        )
    print(f"{rel}  {dst.stat().st_size/1024:.0f} KB")


def main() -> None:
    for rel in FILES + SCORES:
        pull(rel)


if __name__ == "__main__":
    main()
