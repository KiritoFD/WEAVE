#!/usr/bin/env bash
# ==============================================================================
# WEAVE Art-VAE Suite: Deferred Evaluation Runner for A0, A1, A2
# Runs P0 Hard Gate & P1 Style Probes on the completed VAE checkpoints.
# Can be executed on GPU or CPU without interrupting training.
# ==============================================================================
set -e

VAL_MANIFEST="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
B0_BENCHMARK="/root/Workspace/WEAVE_VAE/b0_benchmark_full.json"
RUNS_DIR="/root/Workspace/WEAVE_VAE/runs_vae"
PYTHON="/opt/conda/envs/cu121/bin/python"

echo "=== [$(date)] Starting Deferred Evaluation for A0, A1, A2 ==="

MODES=("A0" "A1" "A2")

for MODE in "${MODES[@]}"; do
    MODEL_DIR="$RUNS_DIR/$MODE"
    FINAL_VAE_DIR="$MODEL_DIR/final_diffusers_vae"
    P0_REPORT="$MODEL_DIR/p0_eval_report.json"
    PROBE_REPORT="$MODEL_DIR/probe_eval_report.json"

    if [ ! -d "$FINAL_VAE_DIR" ]; then
        echo "[WARN] Final model not found for $MODE at $FINAL_VAE_DIR. Skipping."
        continue
    fi

    echo ""
    echo "=== Running P0 Hard Gate for $MODE ==="
    $PYTHON -u /root/Workspace/WEAVE_VAE/tools/vae_suite/eval_b0_p0.py \
        --vae-path "$FINAL_VAE_DIR" \
        --val-manifest "$VAL_MANIFEST" \
        --b0-benchmark-json "$B0_BENCHMARK" \
        --output-json "$P0_REPORT"

    echo ""
    echo "=== Running P1 Style Probes for $MODE ==="
    $PYTHON -u /root/Workspace/WEAVE_VAE/tools/vae_suite/probes.py \
        --vae-path "$FINAL_VAE_DIR" \
        --val-manifest "$VAL_MANIFEST" \
        --output-json "$PROBE_REPORT"
done

echo ""
echo "=== [$(date)] All Deferred Evaluations Completed! ==="
