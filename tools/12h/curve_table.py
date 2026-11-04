"""Build the loss-quality curve table from training logs + eval_curve outputs.

For each (arm, step) the training loss is the last logged epoch line at that
step (epoch == global step for these runs, 1 step/epoch).  Quality is the
600 cross-style split of the SAME fast eval path.

Usage: python3 tools/12h/curve_table.py --root runs/12h
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EPOCH_RE = re.compile(
    r"Epoch (\d+)/\d+ \| loss=([\d.eE+-]+).*? ll=([\d.eE+-]+) lh=([\d.eE+-]+) hl=([\d.eE+-]+)"
)
STEPS = (200, 400, 600, 1000, 2000)
ARMS = ("A1", "A2")


def parse_training(log_path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        out[epoch] = {
            "loss": float(m.group(2)),
            "ll": float(m.group(3)),
            "lh": float(m.group(4)),
            "hl": float(m.group(5)),
        }
    return out


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sx = sum((v - mx) ** 2 for v in x) ** 0.5
    sy = sum((v - my) ** 2 for v in y) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def steps_per_epoch(arm: str, train: dict[int, dict], root: Path) -> int:
    """A1 = 1 step/epoch, A2 = 4 steps/epoch (2000 steps / 500 epochs)."""
    max_epoch = max(train) if train else 0
    cfg_path = root.parent.parent / "configs" / "12h" / f"{arm}.json"
    stop = 0
    try:
        cfg = json.loads(cfg_path.read_text())
        stop = int(cfg.get("training", {}).get("stop_after_global_steps", 0) or 0)
    except Exception:
        pass
    if max_epoch and stop:
        return max(1, round(stop / max_epoch))
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/12h")
    ap.add_argument("--out", default="runs/12h/eval_curve/curve_table.json")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for arm in ARMS:
        train = parse_training(root / f"{arm}.log")
        spe = steps_per_epoch(arm, train, root)
        for step in STEPS:
            d = root / "eval_curve" / f"{arm}_{step}"
            epoch = step // spe
            row = {"arm": arm, "step": step, "epoch": epoch, "steps_per_epoch": spe,
                   "train": train.get(epoch)}
            if (d / "dino_summary.json").exists():
                dino = json.loads((d / "dino_summary.json").read_text())
                split = json.loads((d / "results_split.json").read_text())
                cross = {k: v for k, v in split["main_cross_style"].items() if not k.endswith("_n")}
                row.update({
                    "dino_s": cross["dino_s"],
                    "dino_c": cross["dino_c"],
                    "clip_s": cross["clip_s"],
                    "lpips": cross["lpips"],
                    "all750": {k: dino.get(k) for k in
                               ("all_dino_s", "all_dino_c", "all_clip_s", "all_lpips")},
                    "eval_dir": str(d),
                })
            else:
                row["status"] = "missing"
            rows.append(row)

    payload = {"rows": rows, "corr": {}}
    for arm in ARMS + ("pooled",):
        sel = [r for r in rows if (arm == "pooled" or r["arm"] == arm)
               and r.get("train") and "dino_c" in r]
        if len(sel) >= 3:
            payload["corr"][arm] = {
                "n": len(sel),
                "loss_vs_dino_c": pearson([r["train"]["loss"] for r in sel], [r["dino_c"] for r in sel]),
                "loss_vs_lpips": pearson([r["train"]["loss"] for r in sel], [r["lpips"] for r in sel]),
                "loss_vs_dino_s": pearson([r["train"]["loss"] for r in sel], [r["dino_s"] for r in sel]),
            }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    for r in rows:
        t = r.get("train") or {}
        print(f"{r['arm']} {r['step']:>5} loss={t.get('loss')} ll={t.get('ll')} lh={t.get('lh')} "
              f"hl={t.get('hl')} dino_s={r.get('dino_s')} dino_c={r.get('dino_c')} "
              f"clip_s={r.get('clip_s')} lpips={r.get('lpips')} {r.get('status','')}")
    print(json.dumps(payload["corr"], indent=1))


if __name__ == "__main__":
    main()

