#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RES="$ROOT/resources"
LOG="$ROOT/logs"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zhouyingchengliao/miniconda3/envs/motion-matching/bin/torchrun}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
NITER_SELECTOR="${NITER_SELECTOR:-60000}"
NITER_RESIDUAL="${NITER_RESIDUAL:-60000}"
SELECTOR_BATCHSIZE="${SELECTOR_BATCHSIZE:-4096}"
RESIDUAL_BATCHSIZE="${RESIDUAL_BATCHSIZE:-2048}"
SELECTOR_GPUS="${SELECTOR_GPUS:-0,1,2}"
RESIDUAL_GPUS="${RESIDUAL_GPUS:-3,4,5}"

gpu_count() {
  local value="$1"
  python - <<PY
value = "${value}".strip()
print(len([x for x in value.split(",") if x.strip()]))
PY
}

SELECTOR_NPROC="$(gpu_count "$SELECTOR_GPUS")"
RESIDUAL_NPROC="$(gpu_count "$RESIDUAL_GPUS")"

cd "$RES"
test -f ./hybrid_retrieval_targets.npz

echo "[6/6] Training selector + residual stepper..." | tee "$LOG/hybrid_phase6_only.log"

CUDA_VISIBLE_DEVICES="$SELECTOR_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node "$SELECTOR_NPROC" ./train_selector.py \
  --batchsize "$SELECTOR_BATCHSIZE" \
  --lr 0.0005 \
  --niter "$NITER_SELECTOR" \
  --save-every "$SAVE_EVERY" \
  --targets ./hybrid_retrieval_targets.npz \
  > "$LOG/selector.log" 2>&1 &
SELECTOR_PID=$!

CUDA_VISIBLE_DEVICES="$RESIDUAL_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node "$RESIDUAL_NPROC" ./train_residual_stepper.py \
  --batchsize "$RESIDUAL_BATCHSIZE" \
  --lr 0.0005 \
  --niter "$NITER_RESIDUAL" \
  --save-every "$SAVE_EVERY" \
  --frame-mask ./frame_mask.bin \
  --targets ./hybrid_retrieval_targets.npz \
  > "$LOG/residual_stepper.log" 2>&1 &
RESIDUAL_PID=$!

wait "$SELECTOR_PID"
wait "$RESIDUAL_PID"
touch "$LOG/hybrid_phase6_only.done"
echo "done" | tee -a "$LOG/hybrid_phase6_only.log"
