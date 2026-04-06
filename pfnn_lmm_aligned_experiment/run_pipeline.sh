#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MM_ROOT="$(cd "$ROOT/.." && pwd)"
RES="$ROOT/resources"
LOG="$ROOT/logs"
BASE="$MM_ROOT/resources"
BUILD_FEATURES="$MM_ROOT/build_features"
mkdir -p "$LOG" "$RES"

for name in bvh.py quat.py tquat.py txform.py train_common.py train_decompressor.py train_stepper.py train_projector.py simulation_run.bin simulation_walk.bin character.bin character.fs character.vs character_330.fs character_330.vs checkerboard.fs checkerboard.vs checkerboard_330.fs checkerboard_330.vs pfnn_terrain_rocky_grid.bin; do
  rm -f "$RES/$name"
  ln -s "$BASE/$name" "$RES/$name"
done

if [ ! -x "$BUILD_FEATURES" ]; then
  echo "Missing build_features executable at $BUILD_FEATURES" >&2
  exit 1
fi

# Wait for aligned export to finish writing database+terrain.
while pgrep -f "export_lmm_database.py --output-dir $RES" >/dev/null; do
  sleep 30
done

# Build environment-aware features from the shared runtime scene.
python -u "$BASE/generate_terrain_assets.py" --database "$RES/database.bin" --output-dir "$RES" --boxes-file "$BASE/environment_boxes.txt" \
  > "$LOG/generate_environment.log" 2>&1

# Build matching features.
"$BUILD_FEATURES" "$RES/database.bin" "$RES/terrain_features.bin" "$RES/features.bin" > "$LOG/build_features.log" 2>&1

cd "$RES"

# Start decompressor on GPU0 and wait for latent checkpoint.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n motion-matching \
  python -u train_decompressor.py --device cuda:0 --batchsize 32 --lr 0.0003 --niter 10000 --save-every 1000 \
  > "$LOG/decompressor_gpu0.log" 2>&1

# Then run stepper/projector in parallel.
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n motion-matching \
  python -u train_stepper.py --device cuda:0 --batchsize 64 --lr 0.0005 --niter 10000 --save-every 1000 \
  > "$LOG/stepper_gpu1.log" 2>&1 &
STEPPER_PID=$!

CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n motion-matching \
  python -u train_projector.py --device cuda:0 --batchsize 256 --lr 0.0005 --niter 10000 --save-every 1000 --nn-chunk 65536 \
  > "$LOG/projector_gpu2.log" 2>&1 &
PROJECTOR_PID=$!

wait "$STEPPER_PID"
wait "$PROJECTOR_PID"
