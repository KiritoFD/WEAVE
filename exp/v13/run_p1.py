"""V13 P1 runner: train E_s -> build memory -> Sim gate (one shot).

Works both locally (Windows python) and on the 3060 (WSL python3):
run from the repo root with `python -m exp.v13.run_p1` or `python3 -m exp.v13.run_p1`.
"""
from __future__ import annotations

import json
from pathlib import Path

from exp.v13.stroke_encoder import main as train_main
from exp.v13.memory import main as memory_main

REPO = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    print("=== P1 step 1: train stroke encoder ===", flush=True)
    import sys
    sys.argv = ["stroke_encoder", "--updates", "4000"]
    train_main()

    print("=== P1 step 2: build memory + Sim gate ===", flush=True)
    memory_main()

    gate = json.loads((REPO / "runs/v13/p1_gate.json").read_text())
    print("=== P1 VERDICT:", "PASS" if gate["gate_pass"] else "FAIL",
          "===", flush=True)
