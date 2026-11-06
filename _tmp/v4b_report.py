"""Per-arm V4 report: three-caliber metrics, gate status, timing/VRAM."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
RR = Path("/mnt/i/Latent_Style/pseudo_gt_24h/eval_render")
ANCHOR = ROOT / "runs/12h/anchors/base_ep10_750_512/results_split.json"

name = __import__("sys").argv[1]
d = ROOT / "runs/v4" / name
cell = RR / f"{name}_final"

eval_p = cell / "v4_eval750.json"
if not eval_p.exists():
    eval_p = d / "v4_eval750.json"
data = json.loads(eval_p.read_text())

split_keys = [("all750", "750"), ("cross600", "600"), ("same150", "150")]


def anchor(key, metric):
    if not ANCHOR.exists():
        return None
    a = json.loads(ANCHOR.read_text())
    node = a.get(key, a)
    if isinstance(node, dict):
        for k in (metric, metric.lower(), metric.replace("DINO_", "dino_")):
            if k in node:
                return float(node[k])
    return None


print(f"=== {name} ===")
print(f"{'口径':<5} {'n':>4} {'DINO_S':>9} {'ΔS vs 锚':>10} {'DINO_C':>8} {'CLIP_S':>8} {'LPIPS':>7} "
      f"{'CSD_dT':>9} {'CSD_dS':>9} {'CSD_dM':>9} {'DINO_dM':>9} {'SWD':>8}")
for key, label in split_keys:
    m = data.get(key, {})
    an = anchor(key, "DINO_S")
    ds = f"{m['DINO_S'] - an:+.6f}" if an is not None else "n/a"
    print(f"{label:<5} {m.get('n', 0):>4} {m.get('DINO_S', 0):>9.6f} {ds:>10} "
          f"{m.get('DINO_C', 0):>8.4f} {m.get('CLIP_S', 0):>8.4f} {m.get('LPIPS', 0):>7.4f} "
          f"{m.get('CSD_dT', 0):>9.5f} {m.get('CSD_dS', 0):>9.5f} {m.get('CSD_dM', 0):>9.5f} "
          f"{m.get('DINO_dM', 0):>9.5f} {m.get('PatchSWD', 0):>8.4f}")
print("CI600 dT", data["cross600"].get("CSD_dT_ci"), "dS", data["cross600"].get("CSD_dS_ci"))
print("SWD_sd600", data["cross600"].get("PatchSWD_sd"))

es = json.loads((d / "internal_dynamics_summary.json").read_text())
print("GATE armed=%s fired=%s would_fire=%s reason=%s step=%s" % (
    es.get("armed"), es.get("fired"), es.get("would_fire"), es.get("stop_reason"), es.get("global_step")))
print("  gate_mean:", es.get("gate_mean_series"))
print("  rho      :", es.get("rho_series"))
print("  rho_ratio:", [round(es["rho_series"][i] / es["rho_series"][i - 1], 4)
                      for i in range(1, len(es.get("rho_series", [])))])

tf = d / "train_facts.txt"
if tf.exists():
    el, pk, rk = tf.read_text().split()
    print(f"TRAIN elapsed={el}s peak_vram={pk}mib peak_rss={rk}mib")
ck = sorted([p for p in d.glob("step_*.pt")] + [p for p in d.glob("epoch_*.pt")])
if ck:
    print("last_ckpt", ck[-1].name)

rl = (d / "render.log").read_text(errors="ignore") if (d / "render.log").exists() else ""
mo = re.search(r'"num_images": (\d+), "wall_sec_render": ([0-9.]+), "wall_sec_total": ([0-9.]+)', rl)
if mo:
    n, wr, wt = int(mo.group(1)), float(mo.group(2)), float(mo.group(3))
    print(f"RENDER n={n} render={wr}s total={wt}s imgs/s={n / wr:.3f}")
mf = cell / "render_manifest.json"
if mf.exists():
    print("render_manifest", mf.read_text()[:400])
