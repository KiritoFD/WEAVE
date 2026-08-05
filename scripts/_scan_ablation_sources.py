from pathlib import Path
import json, csv

ROOT = Path(r"G:/GitHub/Latent_Style/WEAVE")
paper = {
    "WEAVE": (0.4916, 0.7128, 0.2593, 0.8103),
    "w/o AdaIN": (0.4821, 0.7005, 0.2258, 0.8455),
    "final-only AdaIN": (0.4836, 0.7293, 0.3954, 0.6660),
    "w/o HF conditioning": (0.4825, 0.7157, 0.2837, 0.8125),
    "lambda_LL=1.0": (0.4910, 0.7180, 0.2626, 0.7874),
    "direct endpoint": (0.4894, 0.7172, 0.3138, 0.7966),
    "learned HH head": (0.4930, 0.7164, 0.2670, 0.8061),
    "zero-velocity": (0.4752, 0.7028, 0.2565, 0.8558),
}

print("=== dino_summary.json ===")
for p in sorted(ROOT.rglob("dino_summary.json")):
    s = str(p).lower()
    if "archive" in s or "legacy" in s:
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    ds, cs, lp, dc = d.get("all_dino_s"), d.get("all_clip_s"), d.get("all_lpips"), d.get("all_dino_c")
    if ds is None:
        continue
    off = d.get("off_dino_s")
    rel = p.relative_to(ROOT)
    mark = ""
    for name, (tds, tcs, tlp, tdc) in paper.items():
        if abs(ds - tds) < 0.001 and (cs is None or abs(float(cs) - tcs) < 0.002):
            mark = f"  << {name}"
    print(f"{ds:.4f} CS={cs} LP={lp} DC={dc} OFF_DS={off} | {rel}{mark}")

print("\n=== metrics with dino + paper CS match ===")
for p in sorted(ROOT.rglob("metrics.csv")):
    s = str(p).lower()
    if "archive" in s or "legacy" in s:
        continue
    try:
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    except Exception:
        continue
    if not rows:
        continue
    keys = rows[0].keys()
    if "clip_style" not in keys:
        continue
    has_dino = "dino_style" in keys or "dino_s" in keys
    try:
        cs = sum(float(r["clip_style"]) for r in rows) / len(rows)
        lp = sum(float(r["content_lpips"]) for r in rows) / len(rows)
    except Exception:
        continue
    for name, (tds, tcs, tlp, tdc) in paper.items():
        if abs(cs - tcs) < 0.0008 and abs(lp - tlp) < 0.003:
            ds = dc = None
            if has_dino:
                kds = "dino_style" if "dino_style" in keys else "dino_s"
                kdc = "dino_content" if "dino_content" in keys else "dino_c"
                ds = sum(float(r[kds]) for r in rows) / len(rows)
                dc = sum(float(r[kdc]) for r in rows) / len(rows)
            print(f"{name}: CS={cs:.4f} LP={lp:.4f} DS={ds} DC={dc} dino={has_dino} n={len(rows)} | {p.relative_to(ROOT)}")
