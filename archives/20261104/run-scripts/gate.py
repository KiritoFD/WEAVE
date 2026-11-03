"""Anti-fake-conclusion gate for a WEAVE run.

usage: gate.py <run_dir> [route]
  route=A (default): requires Route-A output supervision activation + non-zero
                     style/content losses + non-zero grads.
  route=C: requires Route-C reference conditioning activation + non-zero
           InfoNCE + non-zero grads (Route-A losses are expected to be zero).
  route=CTRL: control arm of Route C (Style-ID path). Requires the reference
              mechanism to stay OFF, plus non-zero grads and a declining loss.
  route=D: requires the Route-D texture refiner to be ON, non-zero grads and a
           declining loss.
Both routes also report whether the total loss is declining.
"""
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
route = (sys.argv[2] if len(sys.argv) > 2 else "A").upper()
log = run_dir / "train.log"
dbg = run_dir / "ckpt" / "numeric_debug.jsonl"
gate = {"run_dir": str(run_dir), "route": route}

text = log.read_text(errors="ignore") if log.exists() else ""
gate["activation_log"] = (
    "Route C reference conditioning ON" in text
    if route == "C"
    else ("Route A output supervision ON" in text if route == "A"
          else "Route C reference conditioning ON" in text)
)
if route == "D":
    gate["activation_log"] = "Route D texture refiner ON" in text

fwd, bwd = [], []
if dbg.exists():
    for line in dbg.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("stage") == "forward":
            fwd.append(rec)
        elif rec.get("stage") == "backward":
            bwd.append(rec)


def _vals(recs, key):
    return [r["metrics"][key] for r in recs if key in r.get("metrics", {})]


ls = _vals(fwd, "out_loss_style_out")
lc = _vals(fwd, "out_loss_content_out")
nce = _vals(fwd, "loss_reference_nce")
tot = _vals(fwd, "total") or _vals(fwd, "loss")
grads = [r.get("extra", {}).get("grad_abs_max", 0.0) for r in bwd]
gate.update({
    "numeric_debug_records": len(fwd) + len(bwd),
    "style_loss_n": len(ls),
    "style_loss_min": min(ls) if ls else None,
    "content_loss_min": min(lc) if lc else None,
    "nce_n": len(nce),
    "nce_min": min(nce) if nce else None,
    "nce_max": max(nce) if nce else None,
    "grad_abs_max": max(grads) if grads else None,
    "loss_first": tot[0] if tot else None,
    "loss_last": tot[-1] if tot else None,
})
gate["style_loss_nonzero"] = bool(ls) and gate["style_loss_min"] > 0.0
gate["content_loss_nonzero"] = bool(lc) and gate["content_loss_min"] > 0.0
gate["nce_nonzero"] = bool(nce) and gate["nce_max"] > 0.0
gate["grad_nonzero"] = bool(grads) and max(grads) > 0.0
half = max(1, len(tot) // 2)
gate["loss_first_half_mean"] = statistics.fmean(tot[:half]) if tot else None
gate["loss_second_half_mean"] = statistics.fmean(tot[half:]) if tot else None
gate["loss_declining"] = (
    (gate["loss_first_half_mean"] > gate["loss_second_half_mean"]) if len(tot) >= 4 else None
)
if route == "C":
    gate["valid"] = bool(gate["activation_log"] and gate["nce_nonzero"] and gate["grad_nonzero"])
elif route == "CTRL":
    gate["mechanism_off"] = not gate["activation_log"]
    gate["valid"] = bool(gate["mechanism_off"] and gate["grad_nonzero"] and gate["loss_declining"])
elif route == "D":
    gate["valid"] = bool(gate["activation_log"] and gate["grad_nonzero"] and gate["loss_declining"])
else:
    gate["valid"] = bool(gate["activation_log"] and gate["style_loss_nonzero"]
                         and gate["content_loss_nonzero"] and gate["grad_nonzero"])

(run_dir / "gate.json").write_text(json.dumps(gate, indent=2))
print("GATE", json.dumps(gate))
