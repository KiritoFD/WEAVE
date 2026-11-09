"""Local smoke test for suite40 config resolution and dry command construction.

Does not launch training or require remote caches; it only verifies the bookkeeping
that decides which checkpoint each fork loads and which final file the runner expects.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from sbflow.suite40 import CORE, resolve, select, write_results  # noqa: E402


def main():
    cfg_dir = Path("configs/sbflow")
    assert len(list(cfg_dir.glob("exp*.json"))) == 40
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for aid in ("01", "04", "05", "06", "24", "27", "32", "35", "37", "39"):
            cfg = json.loads((cfg_dir / f"exp{aid}.json").read_text())
            r = resolve(cfg, root)
            assert r["experiment_id"] == f"exp{aid}"
            assert r["output"] == str(root / f"exp{aid}")
            if aid == "05":
                assert r["mixed_batch"] is False and r["single_pair"] is True
            if aid == "06":
                assert r["equal_weight_stats"] is False and r["frequency_stats"] is True
            if cfg["fork_from"]:
                assert ("model_equiv3000.pt" in r["resume"]
                        or "model_step3000.pt" in r["resume"]), r["resume"]
                assert "exp01" in r["resume"] or "exp08" in r["resume"] or "exp11" in r["resume"] \
                    or "exp35" in r["resume"]
            print(f"  {aid}: steps={r['steps']} fork={r.get('resume', '-')} "
                  f"core={r['priority']}")
        assert select("core") == CORE
        assert select("01-03,08") == ["01", "02", "03", "08"]
        state = {"01": {"status": "done", "eval": {
            "target_top1": 0.2, "target_margin": -0.1, "csd_dT": 0.01,
            "csd_dS": -0.3, "csd_swd": 0.02, "csd_mmd2_unbiased": 0.2,
            "dino_c": 0.7, "n": 200}, "train_seconds": 12}}
        write_results(state, root)
        assert (root / "results.json").exists() and (root / "summary.md").exists()
    print("SUITE40 DRY TEST PASSED")


if __name__ == "__main__":
    main()
