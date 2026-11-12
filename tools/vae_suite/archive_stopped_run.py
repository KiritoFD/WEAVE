"""Archive the stopped run: last durable checkpoint + all logs/metrics."""
import json
import shutil
import time
from pathlib import Path

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
ARCH = OUT / "archive_stopped_run"
ARCH.mkdir(parents=True, exist_ok=True)

# last durable checkpoint
ck = sorted((OUT / "checkpoints").glob("ckpt_u*.pt"))
last = ck[-1]
print(f"[Archive] last durable checkpoint: {last.name}")

# reconstruct final_diffusers_vae from EMA weights of last checkpoint
import torch
from diffusers import AutoencoderKL
A0 = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
d = torch.load(last, map_location="cpu", weights_only=False)
vae = AutoencoderKL.from_pretrained(A0)
vae.load_state_dict(d.get("ema", d["model"]))
fd = OUT / "final_diffusers_vae"
fd.mkdir(parents=True, exist_ok=True)
vae.save_pretrained(str(fd))
if "m_center" in d:
    torch.save(d["m_center"], fd / "m_center.pt")
torch.save(d["m_center"] if "m_center" in d else torch.zeros(768), OUT / "m_center.pt")
print(f"[Archive] exported {fd}")

# copy logs/metrics
for name in ["train.log", "train_metrics.jsonl", "calibration.json",
             "chain_11h.log", "cpu_eval_full.log", "cpu_eval_500.log",
             "cpu_eval_full.json", "cpu_eval_500.json", "coverage_check.json"]:
    p = OUT / name
    if p.exists():
        shutil.copy2(p, ARCH / name)

# summary
rows = []
for l in (OUT / "train_metrics.jsonl").read_text().splitlines():
    if not l.strip().startswith("{"):
        continue
    d2 = json.loads(l)
    if "upd" in d2 and "rec" in d2:
        rows.append(d2)
last_upd = max(r["upd"] for r in rows)
summary = {
    "stopped_at_update": last_upd,
    "stopped_at_hours": rows[-1]["elapsed_h"],
    "stopped_at_epoch": rows[-1]["epoch"],
    "last_durable_checkpoint": last.name,
    "checkpoint_update": d.get("upd"),
    "updates_lost_since_last_ckpt": last_upd - d.get("upd", 0),
    "reason": "no measurable Content-Matched AUROC improvement at 1500-1930 steps",
    "archived_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(ARCH / "stop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
