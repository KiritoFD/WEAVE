#!/usr/bin/env bash
# ==============================================================================
# WEAVE Art-VAE Master Pipeline: Full Sequential Execution of A0, A1, A2
# ==============================================================================
# Strict Protocol Requirements:
# 1. Independent initialization: Each model starts from raw sd-vae-ft-ema.
# 2. Unified exposure: Exactly identical update steps N=2100, effective_batch=24, seed=42.
# 3. High-speed zero-decode RAM dataset: 75GB wikiart_512_raw_manifest.json.
# 4. Immediate Automated Validation: Runs P0 Hard Gate & P1 Probes on 2048 val set.
# 5. Fault-tolerant & Idempotent: Automatically skips completed phases on resume.
# ==============================================================================
set -e

STEPS=${1:-2100} # Default 2100 updates (~3.4 hours/model, total ~10.5h for all 3)
MICROBATCH=6
EFFECTIVE_BATCH=24
NUM_WORKERS=4

VAE_INIT="/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
DATA_MANIFEST="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
VAL_MANIFEST="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
B0_BENCHMARK="/root/Workspace/WEAVE_VAE/b0_benchmark_full.json"
RUNS_DIR="/root/Workspace/WEAVE_VAE/runs_vae"
PYTHON="/opt/conda/envs/cu121/bin/python"

mkdir -p "$RUNS_DIR"
LOG_FILE="$RUNS_DIR/pipeline_master.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "=== [$(date)] Master VAE Sequential Pipeline Started ==="
echo "=== Target Steps N: $STEPS | microbatch: $MICROBATCH | effective_batch: $EFFECTIVE_BATCH ==="
echo "=============================================================================="

MODES=("A0" "A1" "A2")

for MODE in "${MODES[@]}"; do
    MODEL_DIR="$RUNS_DIR/$MODE"
    FINAL_VAE_DIR="$MODEL_DIR/final_diffusers_vae"
    P0_REPORT="$MODEL_DIR/p0_eval_report.json"
    PROBE_REPORT="$MODEL_DIR/probe_eval_report.json"

    echo ""
    echo "##############################################################################"
    echo "### [$(date)] Pipeline Phase: Mode $MODE"
    echo "### Output Directory: $MODEL_DIR"
    echo "##############################################################################"

    mkdir -p "$MODEL_DIR"

    # 1. Training Phase
    if [ -d "$FINAL_VAE_DIR" ] && [ -f "$FINAL_VAE_DIR/diffusion_pytorch_model.safetensors" ]; then
        echo "[$(date)] Mode $MODE final model already exists at $FINAL_VAE_DIR. Skipping training."
    else
        echo "[$(date)] Starting training for $MODE (independent from original VAE)..."
        $PYTHON -u /root/Workspace/WEAVE_VAE/tools/vae_suite/train.py \
            --mode "$MODE" \
            --vae-init-path "$VAE_INIT" \
            --raw-manifest "$DATA_MANIFEST" \
            --output-dir "$MODEL_DIR" \
            --steps "$STEPS" \
            --microbatch "$MICROBATCH" \
            --effective-batch "$EFFECTIVE_BATCH" \
            --num-workers "$NUM_WORKERS" \
            --save-every 500 \
            --seed 42
        echo "[$(date)] Mode $MODE training finished successfully."
    fi

    # Flush GPU cache before evaluation
    $PYTHON -c "import torch; torch.cuda.empty_cache()"
    sleep 5

    # 2. P0 Hard Gate Evaluation Phase (2048 Independent Val Images)
    if [ -f "$P0_REPORT" ]; then
        echo "[$(date)] P0 report already exists at $P0_REPORT. Skipping."
    else
        echo "[$(date)] Running P0 Hard Gate Evaluation for $MODE..."
        $PYTHON -u /root/Workspace/WEAVE_VAE/tools/vae_suite/eval_b0_p0.py \
            --vae-path "$FINAL_VAE_DIR" \
            --val-manifest "$VAL_MANIFEST" \
            --b0-benchmark-json "$B0_BENCHMARK" \
            --output-json "$P0_REPORT"
        echo "[$(date)] P0 evaluation for $MODE finished."
    fi

    # 3. P1 Style Probes Evaluation Phase (3 Independent Probes, 136 Classes, Unseen Artists)
    if [ -f "$PROBE_REPORT" ]; then
        echo "[$(date)] P1 probe report already exists at $PROBE_REPORT. Skipping."
    else
        echo "[$(date)] Running P1 Style Probes Evaluation for $MODE..."
        $PYTHON -u /root/Workspace/WEAVE_VAE/tools/vae_suite/probes.py \
            --vae-path "$FINAL_VAE_DIR" \
            --val-manifest "$VAL_MANIFEST" \
            --output-json "$PROBE_REPORT"
        echo "[$(date)] P1 probe evaluation for $MODE finished."
    fi

    echo "[$(date)] Completed full cycle for Mode $MODE."
    $PYTHON -c "import torch; torch.cuda.empty_cache()"
    sleep 10
done

echo ""
echo "=============================================================================="
echo "=== [$(date)] All Sequential Training and Evaluations (A0, A1, A2) Finished! ==="
echo "=============================================================================="
