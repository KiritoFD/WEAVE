"""Append one row per run to /mnt/i/Latent_Style/runs/registry.jsonl."""
import argparse, hashlib, json, re, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--route", required=True)
ap.add_argument("--note", default="")
ap.add_argument("--valid", default="auto", choices=["auto", "yes", "no"])
ap.add_argument("--invalid-reason", default="")
a = ap.parse_args()

run_dir, eval_dir = Path(a.run_dir), Path(a.eval_dir)
reg = Path("/mnt/i/Latent_Style/runs/registry.jsonl")

cfg = run_dir / "config.json"
cfg_text = cfg.read_text() if cfg.exists() else ""
cfg_hash = hashlib.sha1(cfg_text.encode()).hexdigest()[:12] if cfg_text else ""
cfg_payload = json.loads(cfg_text) if cfg_text else {}
train = cfg_payload.get("training", {}) if isinstance(cfg_payload, dict) else {}

log = (run_dir / "train.log").read_text(errors="ignore") if (run_dir / "train.log").exists() else ""
m = re.findall(r"Epoch (\d+)/(\d+) .*?comp=([0-9.]+)s epoch=([0-9.]+)s sps=([0-9.]+) bstep=([0-9.]+)s.*?peak=([0-9.]+)/", log)
epochs = [{"epoch": int(x[0]), "comp_s": float(x[2]), "epoch_s": float(x[3]), "sps": float(x[4]), "peak_gb": float(x[6])} for x in m]

report = {}
rp = eval_dir / "results_split.json"
if rp.exists():
    r = json.loads(rp.read_text())
    report = {"cross600": {k: round(v, 4) for k, v in r["main_cross_style"].items() if not k.endswith("_n")},
              "same150": {k: round(v, 4) for k, v in r["identity_same_style"].items() if not k.endswith("_n")}}

gate = json.loads((run_dir / "gate.json").read_text()) if (run_dir / "gate.json").exists() else {}
summary_candidates = sorted(
    run_dir.rglob("internal_dynamics_summary.json"),
    key=lambda path: path.stat().st_mtime,
)
ids = json.loads(summary_candidates[-1].read_text()) if summary_candidates else {}
budget = json.loads((run_dir / "vram_budget.json").read_text()) if (run_dir / "vram_budget.json").exists() else {}
route = str(a.route)
if route.upper() == "C":
    gate_keys = ("activation_log", "nce_nonzero", "grad_nonzero", "loss_declining", "nce_min", "nce_max", "grad_abs_max")
else:
    gate_keys = ("activation_log", "style_loss_nonzero", "content_loss_nonzero", "grad_nonzero",
                 "loss_declining", "style_loss_min", "grad_abs_max")
valid = bool(report) and bool(gate.get("valid")) and bool(gate.get("loss_declining"))
if a.valid == "yes":
    valid = True
elif a.valid == "no":
    valid = False

if a.invalid_reason:
    invalid_reason = a.invalid_reason
elif valid:
    invalid_reason = ""
else:
    check_keys = (("activation_log", "nce_nonzero", "grad_nonzero", "loss_declining")
                  if route.upper() == "C" and gate.get("route") != "CTRL" else
                  ("mechanism_off", "grad_nonzero", "loss_declining")
                  if route.upper() == "C" else
                  ("activation_log", "grad_nonzero", "loss_declining")
                  if route.upper() == "D" else
                  ("activation_log", "style_loss_nonzero", "content_loss_nonzero", "grad_nonzero", "loss_declining"))
    failed = [k for k in check_keys if not gate.get(k)]
    invalid_reason = "gate: " + ",".join(failed) if failed else "no eval report"

row = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "run_id": run_dir.name,
    "route": a.route,
    "config_hash": cfg_hash,
    "seed": train.get("seed"),
    "epochs": epochs,
    "batch": train.get("batch_size"),
    "microbatch": (cfg_payload.get("bridge", {}) or {}).get("output_supervision_microbatch"),
    "peak_gb": max((e["peak_gb"] for e in epochs), default=None),
    "vram_budget_batch": budget.get("batch"),
    "free_at_start_mib": budget.get("free_at_start_mib"),
    "peak_vram_gb": budget.get("peak_vram_gb") or max((e["peak_gb"] for e in epochs), default=None),
    "vram_guard_failed": bool(budget.get("guard_failed", False)),
    "sps": max((e["sps"] for e in epochs), default=None),
    "train_s": sum(e["epoch_s"] for e in epochs) or None,
    "metrics_600": report.get("cross600"),
    "metrics_150": report.get("same150"),
    "gate": {k: gate.get(k) for k in gate_keys},
    "early_stop_armed": bool(ids.get("armed", False)),
    "early_stop_record_only": bool(ids.get("record_only", False)),
    "fired": bool(ids.get("fired", False)),
    "would_fire": bool(ids.get("would_fire", False)),
    "fired_epoch": ids.get("fired_epoch"),
    "rho_series": ids.get("rho_series"),
    "gate_mean_series": ids.get("gate_mean_series"),
    "stop_reason": ids.get("stop_reason"),
    "valid": valid,
    "invalid_reason": invalid_reason,
    "note": a.note,
}
reg.parent.mkdir(parents=True, exist_ok=True)
with reg.open("a") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print("REGISTERED", json.dumps(row, ensure_ascii=False)[:400])
