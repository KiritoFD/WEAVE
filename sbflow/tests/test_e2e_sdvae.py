"""Real-data end-to-end check on the GPU box: SD-VAE caches -> train5 -> gen5 -> PNG manifest.

Runs a very short training (so it finishes in minutes) and generates a handful of
tasks to prove the SD-VAE encode/decode path, the canonical pair ordering and the
manifest plumbing all work on the real 15,545-image dataset.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/i/Latent_Style/WEAVE")
CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]
OUT = REPO / "runs/blog_sbflow/v10/_e2e_sdvae"


def run(cmd):
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT\n", r.stdout[-3000:])
        print("STDERR\n", r.stderr[-3000:])
        raise SystemExit("FAILED: " + " ".join(map(str, cmd[:8])))
    return r


def main():
    caches = [f"{CACHE_ROOT}/{f}" for f in FILES]
    run([sys.executable, "-m", "sbflow.train5", "--experiment-id", "e2e_sdvae",
         "--caches", *caches, "--output", str(OUT / "train"),
         "--steps", "20", "--batch", "256", "--patch", "8",
         "--width", "128", "--depth", "2", "--heads", "4",
         "--warmup", "4", "--log-interval", "10", "--save-interval", "20"])
    s = json.loads((OUT / "train" / "train_summary.json").read_text())
    print(f"[e2e] trained {s['updates']} updates, {len(s['unordered_pairs'])} pairs, "
          f"counts={s['counts']}, {s['sec_per_update']}s/update, "
          f"peak={s['peak_vram_mib']} MiB")

    ck = OUT / "train" / "model_step20.pt"
    run([sys.executable, "-m", "sbflow.gen5", "--repo-root", ".",
         "--ckpt", str(ck), "--out", str(OUT / "gen"),
         "--subset", "quick200", "--steps", "2", "--batch", "8", "--limit", "12"])
    man = json.loads((OUT / "gen" / "manifest.json").read_text())
    info, tasks = man["info"], man["tasks"]
    assert len(tasks) == 12, len(tasks)
    dirs = sorted({(t["source_style"], t["target_style"]) for t in tasks})
    print(f"[e2e] generated {len(tasks)} images for {len(dirs)} directed pairs "
          f"in {info['total_sec']}s, decode={info['timing_sec']['decode']}s")
    for t in tasks[:4]:
        p = OUT / "gen" / t["image"]
        assert p.exists() and p.stat().st_size > 1000, p
    print("[e2e] PNGs written and non-empty")
    print("E2E SD-VAE PASSED")


if __name__ == "__main__":
    main()
