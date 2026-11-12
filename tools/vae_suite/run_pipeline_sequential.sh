#!/usr/bin/env bash
# ==============================================================================
# WEAVE Art-VAE Pure Master Training Pipeline: Sequential A0 -> A1 -> A2
# ==============================================================================
# Strict Protocol Requirements:
# 1. Zero Evaluation Blocking: Dedicated 100% GPU to training. Evaluation is deferred.
# 2. Independent initialization: Each model starts strictly from raw sd-vae-ft-ema.
# 3. Unified exposure: Exactly identical update steps N=2100, effective_batch=24, seed=42.
# 4. High-speed zero-decode RAM dataset: 75GB wikiart_512_raw_manifest.json.
# 5. Fault-tolerant & Idempotent: Automatically skips completed phases on resume.
# ==============================================================================
set -e

# Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

STEPS=${1:-530} # Unified updates N=530 with large effective_batch=96 (total 50,880 images/model, ~3.46h)
MICROBATCH=6
EFFECTIVE_BATCH=96
NUM_WORKERS=4

VAE_INIT="/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
DATA_MANIFEST="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
RUNS_DIR="/root/Workspace/WEAVE_VAE/runs_vae"
PYTHON="/opt/conda/envs/cu121/bin/python"

mkdir -p "$RUNS_DIR"
LOG_FILE="$RUNS_DIR/pipeline_master.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "=== [$(date)] WEAVE Art-VAE Pure Training Pipeline (A0 -> A1 -> A2) Started ==="
echo "=== Target Steps N: $STEPS | microbatch: $MICROBATCH | effective_batch: $EFFECTIVE_BATCH ==="
echo "=== Protocol: Pure Training Only (Evaluations Deferred Post-Training) ==="
echo "=============================================================================="

MODES=("A0" "A1" "A2")

for MODE in "${MODES[@]}"; do
    MODEL_DIR="$RUNS_DIR/$MODE"
    FINAL_VAE_DIR="$MODEL_DIR/final_diffusers_vae"

    echo ""
    echo "##############################################################################"
    echo "### [$(date)] Pure Training Phase: Mode $MODE"
    echo "### Output Directory: $MODEL_DIR"
    echo "##############################################################################"

    mkdir -p "$MODEL_DIR"

    # Training Phase
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

    # Flush GPU cache completely before moving to next mode
    $PYTHON -c "import torch; torch.cuda.empty_cache()"
    echo "[$(date)] Mode $MODE finished. Memory flushed. Preparing next mode..."
    sleep 10
done

echo ""
echo "=============================================================================="
echo "=== [$(date)] All Sequential Training (A0, A1, A2) Finished Successfully! ==="
echo "=============================================================================="
