"""One-off 2026-11-04: read-only registry inventory for handoff."""
import json
from pathlib import Path

REG = Path("/mnt/i/Latent_Style/runs/registry.jsonl")
rows = []
for line in REG.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        rows.append({"run_id": "<parse-error>", "raw": line[:80]})

print("total", len(rows))
print("--- invalid rows ---")
for r in rows:
    if r.get("valid") is False:
        print(
            "%-24s route=%-3s batch=%-4s status=%-8s steps=%s reason=%s"
            % (
                str(r.get("run_id"))[:24],
                str(r.get("route")),
                str(r.get("batch")),
                str(r.get("status")),
                r.get("steps_done", "-"),
                str(r.get("invalid_reason"))[:90],
            )
        )
print("--- valid rows (compact) ---")
for r in rows:
    if r.get("valid") is True:
        m6 = r.get("metrics_600") or {}
        print(
            "%-24s route=%-2s b=%-4s dino_s=%s clip_s=%s"
            % (
                str(r.get("run_id"))[:24],
                str(r.get("route")),
                str(r.get("batch")),
                m6.get("dino_s", "-"),
                m6.get("clip_s", "-"),
            )
        )
