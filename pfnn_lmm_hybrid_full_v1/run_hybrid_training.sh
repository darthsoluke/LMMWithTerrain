#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MM_ROOT="$(cd "$ROOT/.." && pwd)"
PURE_ROOT="$MM_ROOT/pfnn_lmm_aligned_full_v3"
RES="$ROOT/resources"
LOG="$ROOT/logs"
BASE="$MM_ROOT/resources"
BUILD_FEATURES="$MM_ROOT/build_features"
PYTHON_BIN="${PYTHON_BIN:-/home/zhouyingchengliao/miniconda3/envs/motion-matching/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zhouyingchengliao/miniconda3/envs/motion-matching/bin/torchrun}"
NITER_DECOMP="${NITER_DECOMP:-100000}"
NITER_SELECTOR="${NITER_SELECTOR:-60000}"
NITER_RESIDUAL="${NITER_RESIDUAL:-60000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
DECOMP_BATCHSIZE="${DECOMP_BATCHSIZE:-1024}"
SELECTOR_BATCHSIZE="${SELECTOR_BATCHSIZE:-4096}"
RESIDUAL_BATCHSIZE="${RESIDUAL_BATCHSIZE:-2048}"
DECOMP_GPUS="${DECOMP_GPUS:-0,1,2,3,4,5}"
SELECTOR_GPUS="${SELECTOR_GPUS:-0,1,2}"
RESIDUAL_GPUS="${RESIDUAL_GPUS:-3,4,5}"
TERRAIN_CONFIG="$BASE/terrain_sampling_config.txt"

gpu_count() {
  local value="$1"
  python - <<PY
value = "${value}".strip()
print(len([x for x in value.split(",") if x.strip()]))
PY
}

DECOMP_NPROC="$(gpu_count "$DECOMP_GPUS")"
SELECTOR_NPROC="$(gpu_count "$SELECTOR_GPUS")"
RESIDUAL_NPROC="$(gpu_count "$RESIDUAL_GPUS")"

mkdir -p "$RES" "$LOG"
rm -f "$LOG"/*.log "$LOG"/*.done

ln -sf "$PURE_ROOT/resources/database.bin" "$RES/database.bin"

for name in bvh.py quat.py tquat.py txform.py train_common.py train_decompressor.py train_selector.py train_residual_stepper.py build_hybrid_retrieval_dataset.py generate_terrain_assets.py audit_database_frames.py terrain_sampling_config.txt; do
  rm -f "$RES/$name"
  ln -s "$BASE/$name" "$RES/$name"
done

echo "PYTHON_BIN=$PYTHON_BIN" > "$LOG/hybrid_train.log"
echo "TORCHRUN_BIN=$TORCHRUN_BIN" >> "$LOG/hybrid_train.log"
echo "NITER_DECOMP=$NITER_DECOMP NITER_SELECTOR=$NITER_SELECTOR NITER_RESIDUAL=$NITER_RESIDUAL SAVE_EVERY=$SAVE_EVERY" >> "$LOG/hybrid_train.log"
echo "DECOMP_BATCHSIZE=$DECOMP_BATCHSIZE SELECTOR_BATCHSIZE=$SELECTOR_BATCHSIZE RESIDUAL_BATCHSIZE=$RESIDUAL_BATCHSIZE" >> "$LOG/hybrid_train.log"
echo "DECOMP_GPUS=$DECOMP_GPUS SELECTOR_GPUS=$SELECTOR_GPUS RESIDUAL_GPUS=$RESIDUAL_GPUS" >> "$LOG/hybrid_train.log"

echo "[1/6] Generating terrain+SDF environment features..." | tee -a "$LOG/hybrid_train.log"
"$PYTHON_BIN" -u "$BASE/generate_terrain_assets.py" \
  --database "$RES/database.bin" \
  --output-dir "$RES" \
  --boxes-file "$BASE/environment_boxes.txt" \
  --terrain-config "$TERRAIN_CONFIG" \
  > "$LOG/generate_environment.log" 2>&1

echo "[2/6] Building 61D matching features..." | tee -a "$LOG/hybrid_train.log"
"$BUILD_FEATURES" "$RES/database.bin" "$RES/terrain_features.bin" "$RES/features.bin" \
  > "$LOG/build_features.log" 2>&1

"$PYTHON_BIN" - "$RES" <<'PY' > "$LOG/feature_headers.log"
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rel in ("database.bin", "terrain_features.bin", "features.bin"):
    with (root / rel).open("rb") as f:
        print(rel, struct.unpack("II", f.read(8)))
PY

echo "[3/6] Auditing frames..." | tee -a "$LOG/hybrid_train.log"
"$PYTHON_BIN" -u "$BASE/audit_database_frames.py" \
  --database "$RES/database.bin" \
  --features "$RES/features.bin" \
  --terrain-features "$RES/terrain_features.bin" \
  --output-dir "$RES" \
  > "$LOG/frame_audit.log" 2>&1

echo "[4/6] Training terrain-contact decompressor..." | tee -a "$LOG/hybrid_train.log"
cd "$RES"
CUDA_VISIBLE_DEVICES="$DECOMP_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node "$DECOMP_NPROC" ./train_decompressor.py \
  --batchsize "$DECOMP_BATCHSIZE" \
  --lr 0.0003 \
  --niter "$NITER_DECOMP" \
  --save-every "$SAVE_EVERY" \
  --frame-mask ./frame_mask.bin \
  --terrain-grid ./pfnn_terrain_rocky_grid.bin \
  > "$LOG/decompressor.log" 2>&1

echo "[5/6] Building hybrid retrieval targets..." | tee -a "$LOG/hybrid_train.log"
"$PYTHON_BIN" -u ./build_hybrid_retrieval_dataset.py \
  --database ./database.bin \
  --features ./features.bin \
  --frame-mask ./frame_mask.bin \
  --output ./hybrid_retrieval_targets.npz \
  --stats-out ./hybrid_retrieval_stats.json \
  --valid-spans-out ./hybrid_valid_spans.json \
  > "$LOG/retrieval_targets.log" 2>&1

echo "[6/6] Training selector + residual stepper..." | tee -a "$LOG/hybrid_train.log"
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
touch "$LOG/hybrid_train.done"
echo "done" | tee -a "$LOG/hybrid_train.log"
