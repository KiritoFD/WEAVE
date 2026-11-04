"""Three-way eval-path comparison (old / fast-torch / fast-TRT) with deltas.

Usage: python3 tools/12h/compare_paths.py --dirs old=<dir> torch=<dir> trt=<dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ("dino_s", "dino_c", "clip_s", "lpips")
SPLITS = ("main_cross_style", "identity_same_style", "all")


def load(d: Path) -> dict:
    split = json.loads((d / "results_split.json").read_text())
    dino = json.loads((d / "dino_summary.json").read_text())
    out = {
        "main_cross_style": {k: v for k, v in split["main_cross_style"].items() if not k.endswith("_n")},
        "identity_same_style": {k: v for k, v in split["identity_same_style"].items() if not k.endswith("_n")},
        "all": {k.replace("all_", ""): dino[k] for k in
                ("all_dino_s", "all_dino_c", "all_clip_s", "all_lpips")},
        "protocol": split.get("protocol", {}),
        "decoder": split.get("decoder"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="name=path")
    args = ap.parse_args()
    data = {}
    for spec in args.dirs:
        name, path = spec.split("=", 1)
        data[name] = load(Path(path))
    names = list(data)
    base = names[0]
    for split in SPLITS:
        print(f"## {split}")
        print("| path | " + " | ".join(KEYS) + " |")
        print("|---|" + "---|" * len(KEYS))
        for n in names:
            print(f"| {n} | " + " | ".join(f"{data[n][split][k]:.6f}" for k in KEYS) + " |")
        print("| Δ vs " + base + " | " + " | ".join(
            f"{data[n][split][k] - data[base][split][k]:+.6f}" for k in KEYS
            for n in [names[-1]]) + " |")
        print()
    for n in names:
        print(n, "decoder=", data[n]["decoder"], "backend=", data[n]["protocol"].get("backbone"),
              "batch=", data[n]["protocol"].get("eval_batch"), "vae_dtype=", data[n]["protocol"].get("vae_dtype"))


if __name__ == "__main__":
    main()

