import csv
from pathlib import Path
from collections import defaultdict

p = Path(r"G:/GitHub/Latent_Style/WEAVE/results/cross_protocol/D5-512/stylealigned_metrics.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
print("n", len(rows), "cols", list(rows[0].keys())[:15])

def fmean(rs, k):
    xs = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    return sum(xs) / len(xs) if xs else None

idt = [r for r in rows if r["src_style"] == r["tgt_style"]]
cross = [r for r in rows if r["src_style"] != r["tgt_style"]]
print("n_idt", len(idt), "n_cross", len(cross))
for k in ["dino_style", "clip_style", "content_lpips", "dino_content"]:
    if k in rows[0]:
        print(f"{k}: all={fmean(rows,k):.6f} idt={fmean(idt,k):.6f} cross={fmean(cross,k):.6f}")

print("\nby tgt (idt vs cross dino_style):")
for tgt in sorted(set(r["tgt_style"] for r in rows)):
    i = [r for r in idt if r["tgt_style"] == tgt]
    c = [r for r in cross if r["tgt_style"] == tgt]
    print(f"  {tgt:20s} idt_n={len(i):3d} DS={fmean(i,'dino_style'):.4f}  cross_n={len(c):3d} DS={fmean(c,'dino_style'):.4f}")

print("\nby src->tgt matrix mean dino_style:")
styles = sorted(set(r["src_style"] for r in rows))
print(" " * 18 + "  ".join(f"{s[:6]:>6s}" for s in styles))
for src in styles:
    cells = []
    for tgt in styles:
        rs = [r for r in rows if r["src_style"] == src and r["tgt_style"] == tgt]
        cells.append(f"{fmean(rs,'dino_style'):.3f}" if rs else "  NA ")
    print(f"{src[:16]:16s} " + "  ".join(cells))

print("\nhighest 8 dino_style:")
for r in sorted(rows, key=lambda x: float(x["dino_style"]), reverse=True)[:8]:
    print(
        f"  {r['src_style'][:12]}->{r['tgt_style'][:12]} DS={float(r['dino_style']):.4f} "
        f"DC={float(r.get('dino_content',0)):.4f} LP={float(r.get('content_lpips',0)):.4f}"
    )
print("lowest 8 dino_style:")
for r in sorted(rows, key=lambda x: float(x["dino_style"]))[:8]:
    print(
        f"  {r['src_style'][:12]}->{r['tgt_style'][:12]} DS={float(r['dino_style']):.4f} "
        f"DC={float(r.get('dino_content',0)):.4f} LP={float(r.get('content_lpips',0)):.4f}"
    )

# Is identity DS surprisingly LOW for StyleAligned?
print("\nidentity DS histogram buckets:")
vals = [float(r["dino_style"]) for r in idt]
print(" min/med/max", min(vals), sorted(vals)[len(vals)//2], max(vals))
# content collapse on idt?
print("identity DC mean", fmean(idt, "dino_content"), "LP mean", fmean(idt, "content_lpips"))
print("cross DC mean", fmean(cross, "dino_content"), "LP mean", fmean(cross, "content_lpips"))
