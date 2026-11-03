#!/usr/bin/env bash
set -e
S=/mnt/i/Latent_Style/_sync
W=/mnt/i/Latent_Style/WEAVE
mv "$S/config_schema.py" "$S/internal_dynamics.py" "$S/trainer.py" "$S/run.py" "$W/"
mv "$S/test_internal_dynamics.py" "$S/test_nextgen_skeleton.py" "$W/tests/"
mv "$S/append_registry.py" "$S/p0_curve.sh" "$S/routeC_run.sh" "$S/routeD_run.sh" \
   "$S/p0_capacity_es.sh" "$S/es_smoke.sh" "$S/es_fire_demo.sh" "$S/infra_loop.sh" \
   "$S/parity_equal_samples.sh" "$S/infra_bench.sh" "$S/infra_all.sh" "$W/runs/"
mv "$S/hf_oriented_internal_early_stop.json" "$W/configs/"
mv "$S/Exp.md" "$S/Plan.md" "$W/"
echo SYNC_OK
