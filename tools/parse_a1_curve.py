"""Extract per-epoch loss curves (loss / ll / lh / hl) from A1 training logs."""
import json
import os
import re
import sys

root = sys.argv[1]
pat = re.compile(
    r"Epoch (\d+)/(\d+) \| loss=([0-9.]+) flow=([0-9.]+) ll=([0-9.]+) lh=([0-9.]+) hl=([0-9.]+)"
)
steps_per_epoch = {"a1_32": 1, "a1_128": 4, "a1_512": 16, "a1_128_noise": 4}
out = {}
for tag, spe in steps_per_epoch.items():
    log = os.path.join(root, tag + ".log")
    if not os.path.exists(log):
        continue
    rows = []
    for line in open(log, errors="ignore"):
        m = pat.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        rows.append({
            "step": ep * spe,
            "epoch": ep,
            "loss": float(m.group(3)),
            "flow": float(m.group(4)),
            "ll": float(m.group(5)),
            "lh": float(m.group(6)),
            "hl": float(m.group(7)),
        })
    out[tag] = rows
    if rows:
        first, last = rows[0], rows[-1]
        print("%-14s steps=%d loss %.4f -> %.4f | ll %.4f -> %.4f | lh %.4f -> %.4f | hl %.4f -> %.4f"
              % (tag, last["step"], first["loss"], last["loss"], first["ll"], last["ll"],
                 first["lh"], last["lh"], first["hl"], last["hl"]))
    else:
        print("%-14s no epoch lines" % tag)
json.dump(out, open(os.path.join(root, "a1_curves.json"), "w"), indent=2)
print("WROTE", os.path.join(root, "a1_curves.json"))

