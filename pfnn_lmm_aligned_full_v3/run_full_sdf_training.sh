#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MM_ROOT="$(cd "$ROOT/.." && pwd)"
PFNN_ROOT="$(cd "$MM_ROOT/../PFNN" && pwd)"
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

mkdir -p "$RES" "$LOG"
rm -f "$LOG"/*.log "$LOG"/*.done

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing python executable at $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -x "$BUILD_FEATURES" ]; then
  echo "Missing build_features executable at $BUILD_FEATURES" >&2
  exit 1
fi

echo "PYTHON_BIN=$PYTHON_BIN" > "$LOG/pipeline.log"
echo "NITER=$NITER SAVE_EVERY=$SAVE_EVERY" >> "$LOG/pipeline.log"
echo "DECOMP_GPU=$DECOMP_GPU STEPPER_GPU=$STEPPER_GPU PROJECTOR_GPU=$PROJECTOR_GPU" >> "$LOG/pipeline.log"

echo "[1/5] Exporting aligned PFNN database..." | tee -a "$LOG/pipeline.log"
(
  cd "$PFNN_ROOT"
  "$PYTHON_BIN" -u "$PFNN_ROOT/export_lmm_database.py" --output-dir "$RES"
) \
  > "$LOG/export.log" 2>&1
touch "$LOG/export.done"

echo "[2/5] Linking training scripts..." | tee -a "$LOG/pipeline.log"
for name in bvh.py quat.py tquat.py txform.py train_common.py train_decompressor.py train_stepper.py train_projector.py; do
  rm -f "$RES/$name"
  ln -s "$BASE/$name" "$RES/$name"
done

echo "[3/5] Generating terrain+SDF environment features..." | tee -a "$LOG/pipeline.log"
"$PYTHON_BIN" -u "$BASE/generate_terrain_assets.py" \
  --database "$RES/database.bin" \
  --output-dir "$RES" \
  --boxes-file "$BASE/environment_boxes.txt" \
  > "$LOG/generate_environment.log" 2>&1

echo "[4/5] Building 45D matching features..." | tee -a "$LOG/pipeline.log"
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

echo "[5/5] Training Learned Motion Matching networks..." | tee -a "$LOG/pipeline.log"
cd "$RES"

CUDA_VISIBLE_DEVICES="$DECOMP_GPU" "$PYTHON_BIN" -u ./train_decompressor.py \
  --device cuda:0 \
  --batchsize 32 \
  --lr 0.0003 \
  --niter "$NITER" \
  --save-every "$SAVE_EVERY" \
  > "$LOG/decompressor.log" 2>&1

CUDA_VISIBLE_DEVICES="$STEPPER_GPU" "$PYTHON_BIN" -u ./train_stepper.py \
  --device cuda:0 \
  --batchsize 64 \
  --lr 0.0005 \
  --niter "$NITER" \
  --save-every "$SAVE_EVERY" \
  > "$LOG/stepper.log" 2>&1 &
STEPPER_PID=$!

CUDA_VISIBLE_DEVICES="$PROJECTOR_GPU" "$PYTHON_BIN" -u ./train_projector.py \
  --device cuda:0 \
  --batchsize 256 \
  --lr 0.0005 \
  --niter "$NITER" \
  --save-every "$SAVE_EVERY" \
  --nn-chunk "$PROJECTOR_NN_CHUNK" \
  > "$LOG/projector.log" 2>&1 &
PROJECTOR_PID=$!

wait "$STEPPER_PID"
wait "$PROJECTOR_PID"
touch "$LOG/pipeline.done"
echo "done" | tee -a "$LOG/pipeline.log"
