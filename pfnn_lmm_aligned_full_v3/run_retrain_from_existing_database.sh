#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MM_ROOT="$(cd "$ROOT/.." && pwd)"
RES="$ROOT/resources"
LOG="$ROOT/logs"
BASE="$MM_ROOT/resources"
BUILD_FEATURES="$MM_ROOT/build_features"
PYTHON_BIN="${PYTHON_BIN:-/home/zhouyingchengliao/miniconda3/envs/motion-matching/bin/python}"
NITER="${NITER:-100000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
DECOMP_GPU="${DECOMP_GPU:-0}"
STEPPER_GPU="${STEPPER_GPU:-1}"
PROJECTOR_GPU="${PROJECTOR_GPU:-2}"
PROJECTOR_NN_CHUNK="${PROJECTOR_NN_CHUNK:-65536}"
TERRAIN_CONFIG="$BASE/terrain_sampling_config.txt"

mkdir -p "$RES" "$LOG"
rm -f "$LOG"/generate_environment.log "$LOG"/build_features.log "$LOG"/feature_headers.log \
      "$LOG"/frame_audit.log "$LOG"/decompressor.log "$LOG"/stepper.log "$LOG"/projector.log \
      "$LOG"/retrain_existing.done "$LOG"/retrain_existing.log

if [ ! -f "$RES/database.bin" ]; then
  echo "Missing existing database at $RES/database.bin" >&2
  exit 1
fi

echo "PYTHON_BIN=$PYTHON_BIN" > "$LOG/retrain_existing.log"
echo "NITER=$NITER SAVE_EVERY=$SAVE_EVERY" >> "$LOG/retrain_existing.log"
echo "DECOMP_GPU=$DECOMP_GPU STEPPER_GPU=$STEPPER_GPU PROJECTOR_GPU=$PROJECTOR_GPU" >> "$LOG/retrain_existing.log"

for name in bvh.py quat.py tquat.py txform.py train_common.py train_decompressor.py train_stepper.py train_projector.py terrain_sampling_config.txt; do
  rm -f "$RES/$name"
  ln -s "$BASE/$name" "$RES/$name"
done

echo "[1/5] Generating terrain+SDF environment features..." | tee -a "$LOG/retrain_existing.log"
"$PYTHON_BIN" -u "$BASE/generate_terrain_assets.py" \
  --database "$RES/database.bin" \
  --output-dir "$RES" \
  --boxes-file "$BASE/environment_boxes.txt" \
  --terrain-config "$TERRAIN_CONFIG" \
  > "$LOG/generate_environment.log" 2>&1

echo "[2/5] Building 57D matching features..." | tee -a "$LOG/retrain_existing.log"
"$BUILD_FEATURES" "$RES/database.bin" "$RES/terrain_features.bin" "$RES/features.bin" \
  > "$LOG/build_features.log" 2>&1

"$PYTHON_BIN" - "$RES" <<'PY' > "$LOG/feature_headers.log"
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rel in ("database.bin", "terrain_features.bin", "features.bin"):
    path = root / rel
    with path.open("rb") as f:
        print(rel, struct.unpack("II", f.read(8)))
PY

echo "[3/5] Auditing frames and building invalid-frame mask..." | tee -a "$LOG/retrain_existing.log"
"$PYTHON_BIN" -u "$BASE/audit_database_frames.py" \
  --database "$RES/database.bin" \
  --features "$RES/features.bin" \
  --terrain-features "$RES/terrain_features.bin" \
  --output-dir "$RES" \
  > "$LOG/frame_audit.log" 2>&1

echo "[4/5] Training decompressor..." | tee -a "$LOG/retrain_existing.log"
cd "$RES"
CUDA_VISIBLE_DEVICES="$DECOMP_GPU" "$PYTHON_BIN" -u ./train_decompressor.py \
  --device cuda:0 \
  --batchsize 32 \
  --lr 0.0003 \
  --niter "$NITER" \
  --save-every "$SAVE_EVERY" \
  --frame-mask ./frame_mask.bin \
  --terrain-grid ./pfnn_terrain_rocky_grid.bin \
  > "$LOG/decompressor.log" 2>&1

echo "[5/5] Training stepper + projector..." | tee -a "$LOG/retrain_existing.log"
CUDA_VISIBLE_DEVICES="$STEPPER_GPU" "$PYTHON_BIN" -u ./train_stepper.py \
  --device cuda:0 \
  --batchsize 64 \
  --lr 0.0005 \
  --niter "$NITER" \
  --save-every "$SAVE_EVERY" \
  --frame-mask ./frame_mask.bin \
  > "$LOG/stepper.log" 2>&1 &
STEPPER_PID=$!

CUDA_VISIBLE_DEVICES="$PROJECTOR_GPU" "$PYTHON_BIN" -u ./train_projector.py \
  --device cuda:0 \
  --batchsize 256 \
  --lr 0.0005 \
  --niter "$NITER" \
  --save-every "$SAVE_EVERY" \
  --nn-chunk "$PROJECTOR_NN_CHUNK" \
  --frame-mask ./frame_mask.bin \
  > "$LOG/projector.log" 2>&1 &
PROJECTOR_PID=$!

wait "$STEPPER_PID"
wait "$PROJECTOR_PID"
touch "$LOG/retrain_existing.done"
echo "done" | tee -a "$LOG/retrain_existing.log"
