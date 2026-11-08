"""V8: run all local math unit tests in one command."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
scripts = [
    "tools/v8/tests/test_bridge.py",
    "tools/v8/tests/test_time_reverse.py",
    "tools/v8/tests/test_gaussian.py",
    "tools/v8/tests/test_ode.py",
]
# test_data_split.py imports full Evaluator V2 / VAE stack and is run on 3060.
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
for s in scripts:
    print(f"\n--- {s} ---", flush=True)
    subprocess.run([sys.executable, s], cwd=ROOT, env=env, check=True)
print("\nALL V8 LOCAL MATH TESTS PASS")
