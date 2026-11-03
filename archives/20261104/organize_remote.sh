#!/usr/bin/env bash
# One-off 2026-11-04 handoff: move (never delete) one-off scripts/logs/backups
# into archives/20261104 and keep per-run artifacts in place.
# Safe to re-run: mv is a no-op (reported) when the source is already gone.
set -u

ROOT=/mnt/i/Latent_Style
A=$ROOT/archives/20261104
W=$ROOT/WEAVE

echo "== 1. training-root runs: one-off scripts -> archives =="
cd "$ROOT/runs" || exit 1
for f in batch_parity.sh dl_clip.sh dl_weights.sh dino_cache.sh eval_full.sh eval_smoke.sh \
         gate.py p0_adain.sh p0_all.sh p0_capacity.sh p0_curve.sh \
         ramp.sh ramp2.sh ramp_mem.sh ramp_mem2.sh ramp_mem3.sh \
         routeA_scan.sh routeA_scan_1b.sh routeC_run.sh routeC_smoke.sh routeD_run.sh \
         sweep_queue.sh summarize_report.py; do
  [ -e "$f" ] && mv -v "$f" "$A/run-scripts/" || echo "skip (gone): $f"
done

echo "== 2. training-root runs: logs -> archives =="
for f in clip_download.log dino_cache.log dino_cache_content.log weights_download.log \
         nextgen_assets.log nextgen_eval_quick.log routeA_scan_stdout.log routeA_scan_1b_stdout.log; do
  [ -e "$f" ] && mv -v "$f" "$A/run-logs/" || echo "skip (gone): $f"
done

echo "== 3. repo runs: one-off experiment scripts -> archives =="
cd "$W/runs" || exit 1
for f in a1_overfit.sh b160_retest.sh baseline_nohh600.sh es_fire_demo.sh es_smoke.sh \
         eval_parity.sh flow_vram_slope.sh infra_all.sh infra_bench.sh infra_loop.sh \
         mem_probe.sh p0_capacity_es.sh p0_curve.sh p6_bench.sh parity_equal_samples.sh \
         probe_smoke.sh routeC_run.sh routeC_smoke.sh routeD_run.sh; do
  [ -e "$f" ] && mv -v "$f" "$W/archives/20261104/run-scripts/" || echo "skip (gone): $f"
done

echo "== 4. repo runs: legacy artifacts -> archives =="
for f in endpoint_adain_1p5.csv endpoint_adain_2p0.csv endpoint_adain_2p5.csv equiv_old.pt equiv_root.pt; do
  [ -e "$f" ] && mv -v "$f" "$W/archives/20261104/legacy-artifacts/" || echo "skip (gone): $f"
done

echo "== 5. repo runs: one-off analysis helpers -> tools =="
for f in check_dinov2_attn.py compare_eval_parity.py parse_a1_curve.py; do
  [ -e "$f" ] && mv -v "$f" "$W/tools/" || echo "skip (gone): $f"
done

echo "== 6. repo root: drafts/backups -> archives =="
cd "$W" || exit 1
for f in _tmp_cs.py config_schema.py.bak-item2 config_schema.py.bak-p6p7 flow.py.bak-item2 \
         run.py.bak-item2 run.py.bak-p6p7 trainer.py.bak-p6; do
  [ -e "$f" ] && mv -v "$f" "$W/archives/20261104/repo-backups/" || echo "skip (gone): $f"
done

echo "== 7. per-run config.json next to eval output =="
cd "$ROOT/runs" || exit 1
for pair in \
  "p0_diag/baseline_nohh600/config.json:p0_diag/baseline_nohh600_eval/baseline_nohh600/config.json" \
  "p0_diag/baseline_nohh600/config.json:p0_diag/baseline_nohh600_eval2/baseline_nohh600/config.json"; do
  src=${pair%%:*}; dst=${pair##*:}
  if [ -f "$src" ] && [ -d "$(dirname "$dst")" ] && [ ! -f "$dst" ]; then
    cp -v "$src" "$dst"
  fi
done

echo "== DONE =="
