#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MM_ROOT="$(cd "$ROOT/.." && pwd)"
OUT_DIR="${1:-$ROOT/eval}"

mkdir -p "$OUT_DIR"

python "$MM_ROOT/resources/evaluate_terrain_models.py" \
  --controller "$ROOT/run_controller.sh" \
  --workdir "$ROOT" \
  --output-dir "$OUT_DIR" \
  --terrain-grid "$ROOT/resources/pfnn_terrain_rocky_grid.bin" \
  --boxes-file "$ROOT/resources/environment_boxes.txt"
