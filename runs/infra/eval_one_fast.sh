#!/usr/bin/env bash
# P8 fast eval chain: same numbers as eval_one.sh, less wall time.
#   image_save_workers 1 -> 4, ref_feature_batch 16 -> 32, dino batch 32 -> 64,
#   summary_grid disabled unless SAVE_GRID=true (formal figure runs only).
# usage: eval_one_fast.sh <run_name> <checkpoint> [out_root] [config_path] [reference_tokens] [test_dir]
set -u
name="$1"; ckpt="$2"; root="${3:-/mnt/i/Latent_Style/runs/routeA_scan_eval}"
cfg="${4:-/mnt/i/Latent_Style/WEAVE/config.json}"
reftok="${5:-}"
test_dir="${6:-data/test}"
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/i/Latent_Style/WEAVE || exit 1
out="$root/$name"; mkdir -p "$out"
LOG="$out/eval.log"
CLIP=/mnt/i/Latent_Style/weights/AI-ModelScope__clip-vit-large-patch14
extra=()
if [ -n "$reftok" ]; then extra=(--reference_tokens "$reftok"); fi
EVAL_BATCH="${EVAL_BATCH:-32}"
DINO_BATCH="${DINO_BATCH:-64}"
REF_FEATURE_BATCH="${REF_FEATURE_BATCH:-32}"
IMAGE_SAVE_WORKERS="${IMAGE_SAVE_WORKERS:-1}"
SAVE_GRID="${SAVE_GRID:-false}"
SUBSET="${SUBSET:-full}"
SUBSET_N="${SUBSET_N:-0}"
REF_POOL="${REF_POOL:-default}"
grid=()
if [ "$SAVE_GRID" != "true" ]; then grid=(--no-save_summary_grid); fi
echo "===== $(date '+%F %T') EVAL_FAST $name bs=$EVAL_BATCH dino_bs=$DINO_BATCH ref_bs=$REF_FEATURE_BATCH workers=$IMAGE_SAVE_WORKERS grid=$SAVE_GRID subset=$SUBSET n=$SUBSET_N ref_pool=$REF_POOL =====" | tee -a "$LOG"
python3 utils/run_evaluation.py --checkpoint "$ckpt" --output "$out" --test_dir "$test_dir" \
  --config "$cfg" --batch_size "$EVAL_BATCH" --image_save_workers "$IMAGE_SAVE_WORKERS" \
  --ref_feature_batch_size "$REF_FEATURE_BATCH" --clip_model_name "$CLIP" "${grid[@]}" "${extra[@]}" \
  --subset "$SUBSET" --n "$SUBSET_N" --ref_pool "$REF_POOL" \
  --save_generated_images \
  --clip_hf_cache_dir /mnt/i/Latent_Style/eval_cache/hf >> "$LOG" 2>&1
echo "[rc=$?] run_evaluation $name" | tee -a "$LOG"
python3 utils/compute_dino_metrics.py --eval_dir "$out" --test_dir "$test_dir" \
  --cache_dir /mnt/i/Latent_Style/eval_cache/hf --batch_size "$DINO_BATCH" \
  --output_csv "$out/dino_metrics.csv" --output_json "$out/dino_summary.json" >> "$LOG" 2>&1
python3 tools/merge_eval_metrics.py --metrics "$out/metrics.csv" --dino "$out/dino_metrics.csv" \
  --protocol "$out/eval_protocol.json" \
  --out "$out/results.json" >> "$LOG" 2>&1
if [ "$REF_POOL" = "strict" ]; then
  ref_policy_name="reserved_first_image_of_target_style_test_dir"
else
  ref_policy_name="first_image_of_target_style_test_dir"
fi
python3 tools/run_eval_split.py --input "$out/results.json" \
  --reference_policy "$ref_policy_name" \
  --out "$out/results_split.json" >> "$LOG" 2>&1
python3 tools/finalize_results.py --eval-dir "$out" | tee -a "$LOG"
python3 /mnt/i/Latent_Style/runs/summarize_report.py "$out/results_split.json" | tee -a "$LOG"
echo "EVAL_FAST_DONE $name" | tee -a "$LOG"
