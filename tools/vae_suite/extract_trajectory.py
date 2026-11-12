"""Extract full loss trajectories from the aborted Style-VAE run."""
import json
import re
from pathlib import Path

LOG = Path("/root/Workspace/WEAVE_VAE/runs_vae/style_vae/train.log")

pat = re.compile(
    r"\[S(\d)\]\s+upd\s+(\d+)\s+\|\s+rec=([\d.]+)\s+kl=([\d.]+)\s+csd=([\d.]+)\s+sup=([\d.]+)"
    r"(?:\s+swap=([\d.]+)\s+cyc=([\d.]+))?\s+\|\s+g=([\d.]+)\s+\|\s+([\d.]+)s"
)

rows = []
for line in LOG.read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        rows.append({
            "stage": int(m.group(1)), "upd": int(m.group(2)),
            "rec": float(m.group(3)), "kl": float(m.group(4)),
            "csd": float(m.group(5)), "sup": float(m.group(6)),
            "g": float(m.group(9)), "sec": float(m.group(10)),
            "cos_teacher": round(1.0 - float(m.group(5)), 4),
        })

print(f"parsed {len(rows)} log points, updates {rows[0]['upd']}..{rows[-1]['upd']}")
print(f"\n{'upd':>5}{'rec':>9}{'kl':>8}{'L_csd':>9}{'cos_t':>8}{'L_sup':>8}{'g':>7}{'s':>7}")
print("-" * 61)
for r in rows:
    print(f"{r['upd']:>5}{r['rec']:>9.4f}{r['kl']:>8.2f}{r['csd']:>9.4f}"
          f"{r['cos_teacher']:>8.4f}{r['sup']:>8.3f}{r['g']:>7.2f}{r['sec']:>7.2f}")

# bucket averages
print("\n=== 100-update bucket means ===")
print(f"{'bucket':>10}{'rec':>9}{'kl':>8}{'L_csd':>9}{'cos_t':>8}{'L_sup':>8}")
print("-" * 53)
for lo in range(0, rows[-1]["upd"] + 1, 100):
    b = [r for r in rows if lo <= r["upd"] < lo + 100]
    if not b:
        continue
    f = lambda k: sum(r[k] for r in b) / len(b)
    print(f"{lo:>4}-{lo+99:<5}{f('rec'):>9.4f}{f('kl'):>8.2f}{f('csd'):>9.4f}"
          f"{f('cos_teacher'):>8.4f}{f('sup'):>8.3f}")

Path("/root/Workspace/WEAVE_VAE/runs_vae/style_vae/trajectory.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8")
print("\n[saved] trajectory.json")
