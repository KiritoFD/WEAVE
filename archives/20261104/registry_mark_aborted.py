"""One-off 2026-11-04: mark legacy invalid registry rows as aborted (no deletion).

Rewrites /mnt/i/Latent_Style/runs/registry.jsonl in place; a timestamped backup is
written next to it first. Only adds/overwrites the "status" field on rows whose
valid is False and that have no status yet.
"""
import json
import shutil
import time
from pathlib import Path

REG = Path("/mnt/i/Latent_Style/runs/registry.jsonl")

rows = []
changed = 0
for line in REG.read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("valid") is False and not rec.get("status"):
        rec["status"] = "aborted"
        changed += 1
    rows.append(rec)

backup = REG.with_name("registry.jsonl.bak-20261104-status")
shutil.copy2(REG, backup)
REG.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print("backup:", backup)
print("rows:", len(rows), "status added:", changed)
