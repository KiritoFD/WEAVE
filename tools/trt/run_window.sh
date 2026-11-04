#!/usr/bin/env bash
# One-shot GPU window for the TensorRT SD1.5 VAE decoder.  Self-contained, <=10 min.
#
#   bash tools/trt/run_window.sh          # normal run
#   TRT_WINDOW_B1=1 bash tools/trt/run_window.sh   # also try torch.compile (B1)
#
# Steps: nvidia-smi precheck -> build 64_b8 + 512_b8 fp16 engines -> B0 vs B2 parity
#        and B0/B2/B3 timing (B1 only if the env var is set and time remains).
#
# Exit codes: 0 ok | 3 GPU busy (python process or >500 MiB used) | 4 lock held | 124 timeout
set -uo pipefail

ROOT=/mnt/i/Latent_Style/WEAVE
LOCK=/tmp/trt_window.lock
BUDGET=${TRT_WINDOW_BUDGET:-600}
VAE_CACHE=/mnt/i/Latent_Style/eval_cache/hf
[[ -d /tmp/vae_cache/modelscope ]] && VAE_CACHE=/tmp/vae_cache
PY=python3

# ---- self-timeout: re-exec under `timeout` -----------------------------------
if [[ "${1:-}" != "--inner" ]]; then
  exec timeout "${BUDGET}" bash "$0" --inner
fi

# ---- lock -------------------------------------------------------------------
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "[window] busy: another run holds ${LOCK}"
  exit 4
fi
trap 'rm -f "${LOCK}"' EXIT

T0=$(date +%s)
log() { echo "[window +$(( $(date +%s) - T0 ))s] $*"; }
left() { echo $(( BUDGET - ($(date +%s) - T0) )); }

# ---- 1. precheck ------------------------------------------------------------
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \r')
if [[ -n "${apps}" ]]; then
  log "ABORT: GPU already has compute processes: ${apps}"
  exit 3
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
if (( used > 500 )); then
  log "ABORT: ${used} MiB already in use (> 500 MiB)"
  exit 3
fi
log "precheck ok: ${used} MiB used, no compute apps"

cd "${ROOT}" || exit 1

# ---- 2. engines (fp16, fixed batch 8) ---------------------------------------
mkdir -p runs/trt/engines runs/trt/bench
for size in 64 512; do
  if [[ -f "runs/trt/engines/vae_decoder_${size}_b8_fp16.engine" ]]; then
    log "engine ${size}_b8 already present, skip build"
    continue
  fi
  log "building engine ${size}_b8 fp16 (remaining $(left)s)"
  timeout 240 ${PY} tools/trt/build_trt_engines.py --sizes "${size}" --batches 8 --precision fp16
  rc=$?
  if (( rc != 0 )); then
    log "build ${size}_b8 failed rc=${rc}; continuing with whatever exists"
  fi
done

# ---- 3a. parity + timing B0 / B2 / B3 ---------------------------------------
log "bench B0 vs B2/B3 at batch 8 (remaining $(left)s)"
timeout 180 ${PY} tools/trt/bench_vae_decoder.py \
  --sizes 64,512 --batches 8 --precision fp16 \
  --vae_cache "${VAE_CACHE}" \
  --backends b0,b2,b3 --warmup 20 --iters 50
rc=$?
log "bench b0,b2,b3 rc=${rc}"

# ---- 3b. optional torch.compile ---------------------------------------------
if [[ "${TRT_WINDOW_B1:-0}" == "1" ]] && (( $(left) > 150 )); then
  log "bench B1 torch.compile (remaining $(left)s)"
  timeout 150 ${PY} tools/trt/bench_vae_decoder.py \
    --sizes 64,512 --batches 8 --precision fp16 \
    --vae_cache "${VAE_CACHE}" \
    --backends b1 --compile-modes reduce-overhead,max-autotune --warmup 20 --iters 50
  log "bench b1 rc=$?"
else
  log "skip B1 (TRT_WINDOW_B1=${TRT_WINDOW_B1:-0}, remaining $(left)s)"
fi

log "done in $(( $(date +%s) - T0 ))s; results in runs/trt/bench/"
ls -1 runs/trt/engines/*.engine 2>/dev/null

