"""Merge V8 evaluator shard JSONs into one evaluation JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / "runs" / "v8"


def main(name: str, tasks: str, suffix: str = "") -> None:
    out = None
    for p in sorted(V8.glob(f"{name}_{tasks}_eval_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if out is None:
            out = d
        else:
            out["methods"].update(d["methods"])
            out["rows"].update(d["rows"])
    if out is None:
        raise RuntimeError(f"No shards found for {name}_{tasks}")
    out_path = V8 / f"{name}_{tasks}_eval{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(list(V8.glob(f'{name}_{tasks}_eval_*.json')))} shards -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
