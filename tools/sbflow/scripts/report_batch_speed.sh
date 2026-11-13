#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== actual training runs (wall clock from train_summary.json) ==="
python3 - <<'PY'
import glob, json
rows = []
for f in sorted(glob.glob("runs/blog_sbflow/v10/exp_*/train_summary.json")):
    d = json.loads(open(f).read())
    tag = f.split("/")[-2].replace("exp_", "")
    rows.append((tag, d.get("batch"), d.get("updates"), d.get("sec_per_update"),
                 d.get("peak_vram_mib"), d.get("n_params")))
print(f"{'exp':8s} {'batch':>6s} {'updates':>8s} {'s/upd':>8s} {'samples/s':>10s} {'peak MiB':>9s}")
for t, b, u, s, p, n in rows:
    thr = (b / s) if (b and s) else float("nan")
    print(f"{t:8s} {b:6d} {u:8d} {s:8.4f} {thr:10.0f} {p:9.1f}")
PY
echo
echo "=== bench_components (per-component cuda sync inflates this) ==="
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("runs/blog_sbflow/bench/*.json")):
    d = json.loads(open(f).read())
    if "sec_per_update" not in d:
        continue
    print(f"  {d['tag']:26s} params={d['n_params']:>9d} mb={d['microbatch']:>4d}x{d['accum']} "
          f"s/upd={d['sec_per_update']:.4f} peak={d['peak_vram_mib']:8.1f} MiB")
PY
echo
echo "=== patch/batch probe (real SD-VAE caches) ==="
python3 - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("runs/blog_sbflow/v10").glob("_bench_pb_*/train_summary.json")):
    d = json.loads(p.read_text())
    cfg = d.get("config", {})
    print(f"  patch={cfg.get('patch')} tokens={(64//int(cfg.get('patch',1)))**2:5d} "
          f"batch={d.get('batch'):5d} s/upd={d.get('sec_per_update'):.4f} "
          f"peak={d.get('peak_vram_mib'):8.1f} MiB")
PY
