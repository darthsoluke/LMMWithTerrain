#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-$ROOT/captures/$STAMP}"

if [ $# -gt 0 ]; then
  shift
fi

mkdir -p "$OUT_DIR"

echo "capture output: $OUT_DIR"

exec "$ROOT/../controller" \
  --capture-trace "$OUT_DIR/runtime_trace.csv" \
  --capture-tag "interactive_$STAMP" \
  --capture-failure-dump-dir "$OUT_DIR" \
  "$@" 2>&1 | tee "$OUT_DIR/controller_stdout.log"
