#!/usr/bin/env bash
# Pre-flight: verify every module the unattended drivers will invoke actually imports on this
# machine. A missing import would only surface after the queue releases the GPU.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
python3 - <<'PY'
import importlib, traceback

mods = [
    "tools.v6_24h.stats",
    "tools.v6_24h.eval_tiers",
    "tools.v6_24h.m0_alpha",
    "tools.v6_24h.csd_manifold_check",
    "tools.v6_24h.p5_final",
    "tools.v6_24h.tier_select",
    "tools.v6_24h.win_rate",
    "tools.v6_24h.confusion_analysis",
    "tools.v6_24h.dT_vs_top1",
    "tools.v6_24h.inject_decomposition",
    "tools.v6_24h.build_eval_tiers",
    "tools.v6_24h.build_screen50",
    "tools.v6_24h.test_row_schema",
    "tools.v6_24h.smoke16",
    "tools.v6_24h.queue",
]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        bad.append(m)
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
        traceback.print_exc()
print()
print(f"{len(mods)-len(bad)}/{len(mods)} import cleanly")
raise SystemExit(1 if bad else 0)
PY
