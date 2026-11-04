#!/usr/bin/env python3
"""汇总 checkpoint 诊断：把 eval_<step>/eval.json 拼成一张里程碑表。"""
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads((path / "eval.json").read_text(encoding="utf-8"))
    step = payload.get("step")
    row = {"ckpt": Path(payload["ckpt"]).name, "step": step}
    for split in ("train_denoise", "val_denoise"):
        prefix = "train" if split.startswith("train") else "val"
        row[prefix + "_loss_fresh"] = round(payload[split]["loss_fresh_noise"], 4)
        row[prefix + "_cond_gain"] = round(payload[split].get("cond_gain", float("nan")), 4)
        row[prefix + "_bins"] = [round(v, 3) for v in payload[split]["bins_fresh_noise"]]
    row["probe_val_acc"] = round(payload["style_probe_val_acc"], 4)
    row["gen_probe_acc"] = round(payload["gen"]["probe_acc"], 4)
    row["gen_nn_p95"] = round(payload["gen"]["memorization"]["nn_cos_p95"], 4)
    row["gen_nn_frac098"] = round(payload["gen"]["memorization"]["frac_above_0.98"], 4)
    row["swap_style_rms"] = round(payload["swap"]["pairwise_style_rms"], 4)
    row["swap_to_input_rms"] = round(payload["swap"]["to_input_rms"], 4)
    row["gen_confusion"] = payload["gen"]["confusion"]
    return row


def main() -> None:
    root = Path(sys.argv[1])
    rows = [load(d) for d in sorted(root.glob("eval_step_*"))] + \
           [load(d) for d in sorted(root.glob("eval_best")) if (d / "eval.json").exists()]
    print(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
