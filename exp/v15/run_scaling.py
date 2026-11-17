r"""V15 scaling sequencer v2: 7 tiers (dim128-512) + dim96 8k control, serial.

Run on the 3060 inside tmux:
  tmux new-session -d -s scaling cmd /c _tmp/run_v15_scaling.bat
2026-11-15 user revision: ALL arms 8k updates; micro re-tuned to fill memory
toward ~10GB (capped by micro<=128 divisor constraint of eff. batch 128);
two new tiers dim384/dim512.  Pure conv, no attention, no where, global
AdaLN, effective batch 128, seed 0.
Status: runs/v15/scaling_status.json, logs: runs/v15/logs/<tag>.log
OOM backstop: retry once with micro halved / accum doubled (eff. batch kept).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL = str(REPO.parent / "pseudo_gt_24h/orig_mean_fp16")
OUT = REPO / "runs/v15"
LOGS = OUT / "logs"
PYTHON = sys.executable
TIMEOUT_S = 24 * 3600   # dim512 at 8k is the whale; generous ceiling

# (tag, base_dim, updates, micro, accum) — micros re-tuned to SMOKE-MEASURED
# peaks (activation memory grows super-linearly in dim): all arms <= ~10GB.
ARMS = [
    ("S1_dim128", 128, 8000, 128, 1),   # 9.48GB measured @128
    ("S2_dim160", 160, 8000, 64, 2),    # @128=11.63GB unsafe -> 64
    ("S3_dim192", 192, 8000, 64, 2),    # 7.59GB measured
    ("S4_dim256", 256, 8000, 64, 2),    # 9.82GB measured
    ("S5_dim320", 320, 8000, 32, 4),    # @64=12.08GB unsafe -> 32
    ("S6_dim384", 384, 8000, 32, 4),    # 8.30GB measured
    ("S7_dim512", 512, 8000, 16, 8),    # @32=11.02GB unsafe -> 16
    ("B8k_dim96", 96, 8000, 128, 1),    # 6.5GB; vs Group A 4k: length axis
]


def main():
    (OUT / "scaling").mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    status_file = OUT / "scaling_status.json"
    results = {}
    if status_file.exists():
        results = json.loads(status_file.read_text())
    t0 = time.time()
    for tag, dim, updates, micro, accum in ARMS:
        if results.get(tag, {}).get("status") == "done":
            print(f"[skip] {tag} done", flush=True)
            continue
        out_dir = OUT / "scaling" / tag
        log_path = LOGS / f"{tag}.log"
        print(f"[start] {tag} dim={dim} upd={updates} micro={micro} "
              f"accum={accum}", flush=True)
        t_arm = time.time()
        status, micro_used = "done", micro
        for attempt in range(3):
            micro_used = micro
            args = [PYTHON, "-m", "exp.v15.train",
                    "--base-dim", str(dim), "--updates", str(updates),
                    "--micro", str(micro), "--accum", str(accum),
                    "--seed", "0", "--pool", POOL, "--out", str(out_dir)]
            try:
                with open(log_path, "w") as lf:
                    r = subprocess.run(args, cwd=str(REPO), timeout=TIMEOUT_S,
                                       stdout=lf, stderr=subprocess.STDOUT)
                if r.returncode == 0:
                    status = "done"
                    break
                status = f"error:{r.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                break
            if attempt < 2 and "OutOfMemoryError" in \
                    log_path.read_text(errors="replace"):
                micro, accum = max(1, micro // 2), accum * 2
                print(f"  OOM -> retry micro={micro} accum={accum}",
                      flush=True)
                continue
            break
        results[tag] = {"status": status, "micro": micro_used,
                        "elapsed_min": round((time.time() - t_arm) / 60, 1)}
        status_file.write_text(json.dumps(results, indent=1))
        print(f"[done] {tag} {status} "
              f"({results[tag]['elapsed_min']}min) "
              f"total {time.time()-t0:.0f}s", flush=True)
    print("SCALING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
