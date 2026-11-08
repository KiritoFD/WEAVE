"""Split raw-vs-EMA comparison output into per-cell eval files for the summarizer."""
from __future__ import annotations

import json
from pathlib import Path

V8 = Path(__file__).resolve().parents[2] / "runs" / "v8"


def main() -> None:
    src_path = V8 / "p0_raw_vs_ema_750.json"
    src = json.loads(src_path.read_text(encoding="utf-8"))
    for key, entry in src["methods"].items():
        rows = entry.get("rows", [])
        out = {
            "protocol": {"tasks": "test750", "rgb_reference": "paper_rgb_v1",
                         "source": src_path.name, "nfe": 4, "weight": key.split("_")[-1]},
            "methods": {"euler4": {"cross": entry["cross"], "same": entry["same"], "all": entry["cross"]}},
            "rows": {"euler4": rows},
        }
        p = V8 / f"{key}_test750_eval.json"
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {p.name} rows={len(rows)}")


if __name__ == "__main__":
    main()
