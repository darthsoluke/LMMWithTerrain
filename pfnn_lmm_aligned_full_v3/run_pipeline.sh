#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MM_ROOT="$(cd "$ROOT/.." && pwd)"
RES="$ROOT/resources"
LOG="$ROOT/logs"
BASE="$MM_ROOT/resources"
BUILD_FEATURES="$MM_ROOT/build_features"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-$("$PYTHON_BIN" - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda:0")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)}"
NITER="${NITER:-100000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
PROJECTOR_NN_CHUNK="${PROJECTOR_NN_CHUNK:-65536}"
EXPORT_CMD="export_lmm_database.py --output-dir $RES"
mkdir -p "$LOG" "$RES"

while pgrep -f "$EXPORT_CMD" >/dev/null; do
  sleep 30
done

for name in bvh.py quat.py tquat.py txform.py train_common.py train_decompressor.py train_stepper.py train_projector.py simulation_run.bin simulation_walk.bin character.bin character.fs character.vs character_330.fs character_330.vs checkerboard.fs checkerboard.vs checkerboard_330.fs checkerboard_330.vs pfnn_terrain_rocky_grid.bin; do
  rm -f "$RES/$name"
  ln -s "$BASE/$name" "$RES/$name"
done

if [ ! -x "$BUILD_FEATURES" ]; then
  echo "Missing build_features executable at $BUILD_FEATURES" >&2
  exit 1
fi

echo "Using python: $PYTHON_BIN" > "$LOG/run_pipeline.log"
echo "Using device: $TRAIN_DEVICE" >> "$LOG/run_pipeline.log"
echo "Iterations: $NITER" >> "$LOG/run_pipeline.log"
echo "Save every: $SAVE_EVERY" >> "$LOG/run_pipeline.log"

"$PYTHON_BIN" -u "$BASE/generate_terrain_assets.py" --database "$RES/database.bin" --output-dir "$RES" --boxes-file "$BASE/environment_boxes.txt" \
  > "$LOG/generate_environment.log" 2>&1

"$BUILD_FEATURES" "$RES/database.bin" "$RES/terrain_features.bin" "$RES/features.bin" > "$LOG/build_features.log" 2>&1
cd "$RES"

"$PYTHON_BIN" -u "$RES/train_decompressor.py" --device "$TRAIN_DEVICE" --batchsize 32 --lr 0.0003 --niter "$NITER" --save-every "$SAVE_EVERY" \
  > "$LOG/decompressor.log" 2>&1

"$PYTHON_BIN" -u "$RES/train_stepper.py" --device "$TRAIN_DEVICE" --batchsize 64 --lr 0.0005 --niter "$NITER" --save-every "$SAVE_EVERY" \
  > "$LOG/stepper.log" 2>&1

"$PYTHON_BIN" -u "$RES/train_projector.py" --device "$TRAIN_DEVICE" --batchsize 256 --lr 0.0005 --niter "$NITER" --save-every "$SAVE_EVERY" --nn-chunk "$PROJECTOR_NN_CHUNK" \
  > "$LOG/projector.log" 2>&1
