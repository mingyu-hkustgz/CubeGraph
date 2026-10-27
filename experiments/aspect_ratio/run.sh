#!/usr/bin/env bash
# Run from the CubeGraph repository. No construction or index overwrite.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo 'usage: run.sh DATA_ROOT DATASET INDEX_PATH OUTPUT_DIR SHARED_LOCK' >&2
  exit 2
fi
DATA_ROOT=$(realpath "$1")
DATASET=$2
INDEX_PATH=$(realpath "$3")
OUTPUT_DIR=$(realpath -m "$4")
SHARED_LOCK=$5
case "$DATASET" in sift) META=uniform_2d;; yfcc) META=real_2d;; *) exit 2;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# Existing artifacts remain untouched; a .complete marker skips only audited jobs.
mkdir -p "$OUTPUT_DIR"/{bin,logs,results,provenance}
exec 9>"$SHARED_LOCK"
echo "WAIT_LOCK $(date -Is)"
flock 9
echo "LOCKED $(date -Is)"
trap 'rc=$?; printf "exit=%s time=%s\n" "$rc" "$(date -Is)" > "$OUTPUT_DIR/state.txt"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run(){ timeout --signal=TERM --kill-after=30s 10h "$@"; }
freeze_hashes(){
  local destination=$1
  shift
  sha256sum "$@" > "$destination.current"
  if [ -f "$destination" ]; then
    cmp "$destination" "$destination.current" || {
      echo "Provenance changed: $destination; use a new output directory" >&2
      return 1
    }
    rm "$destination.current"
  else
    mv "$destination.current" "$destination"
  fi
}
# This is query-only. Do not share the lock with a second active measurement.
g++ -std=c++17 -O3 -march=core-avx2 -mavx512f -fopenmp \
  -I"$REPO/include" -I"$REPO/third_party" "$REPO/experiments/bench_bottom_up.cpp" \
  -o "$OUTPUT_DIR/bin/bench"
freeze_hashes "$OUTPUT_DIR/provenance/code.sha256" \
  "$REPO/include/IndexCube.h" "$REPO/third_party/hnsw-cube.h" \
  "$REPO/experiments/bench_bottom_up.cpp" "$OUTPUT_DIR/bin/bench" \
  "$REPO/experiments/audit_box.py" "$REPO/experiments/aspect_ratio/generate.py" \
  "$REPO/experiments/aspect_ratio/verify_workloads.py" "$REPO/experiments/aspect_ratio/run.sh"
freeze_hashes "$OUTPUT_DIR/provenance/index.sha256" "$INDEX_PATH" "$INDEX_PATH".layer_*
freeze_hashes "$OUTPUT_DIR/provenance/inputs.sha256" \
  "$DATA_ROOT/$DATASET/${DATASET}_metadata_${META}.bin" \
  "$DATA_ROOT/$DATASET/${DATASET}_base.fvecs" "$DATA_ROOT/$DATASET/${DATASET}_query.fvecs"
if [ ! -f "$OUTPUT_DIR/workloads/manifest.json" ]; then
  # A partial generation must be inspected and moved aside before retrying.
  run python3 "$REPO/experiments/aspect_ratio/generate.py" \
    "$DATA_ROOT" "$DATASET" "$OUTPUT_DIR/workloads" > "$OUTPUT_DIR/logs/generate.log" 2>&1
fi
if [ ! -f "$OUTPUT_DIR/workloads/verification.json" ]; then
  run python3 "$REPO/experiments/aspect_ratio/verify_workloads.py" \
    "$DATA_ROOT" "$OUTPUT_DIR/workloads" > "$OUTPUT_DIR/logs/workload-audit.log" 2>&1
fi
freeze_hashes "$OUTPUT_DIR/provenance/workloads.sha256" \
  "$OUTPUT_DIR/workloads/manifest.json" "$OUTPUT_DIR/workloads/verification.json" \
  "$OUTPUT_DIR/workloads/"a*/*.bin
for ALPHA in 1 4 16 32; do
  for MODE in pred fly; do
    CSV="$OUTPUT_DIR/results/a${ALPHA}_${MODE}.csv"
    if [ -f "$CSV.complete" ]; then
      python3 - "$CSV" <<'PY'
import hashlib, json, sys
from pathlib import Path
p=Path(sys.argv[1]); receipt=json.loads(Path(str(p)+'.audit.json').read_text())
assert receipt['status']=='PASS'
assert receipt['csv_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
PY
      python3 "$REPO/experiments/audit_box.py" "$CSV" "$DATASET" "$META" \
        "$OUTPUT_DIR/workloads/a$ALPHA" "$DATA_ROOT" cube --ratio 100
      continue
    fi
    echo "QUERY alpha=$ALPHA mode=$MODE $(date -Is)"
    CG_SEARCH_MODE="$MODE" CG_RATIO_FILTER=0100 run "$OUTPUT_DIR/bin/bench" query \
      "$DATA_ROOT" "$DATASET" "$META" multi-cross2 "$INDEX_PATH" \
      "$OUTPUT_DIR/workloads/a$ALPHA" "$CSV" > "$OUTPUT_DIR/logs/a${ALPHA}_${MODE}.log" 2>&1
    python3 "$REPO/experiments/audit_box.py" "$CSV" "$DATASET" "$META" \
      "$OUTPUT_DIR/workloads/a$ALPHA" "$DATA_ROOT" cube --ratio 100 \
      >> "$OUTPUT_DIR/logs/a${ALPHA}_${MODE}.log" 2>&1
    touch "$CSV.complete"
  done
done
sha256sum "$OUTPUT_DIR/results/"*.csv > "$OUTPUT_DIR/provenance/results.sha256"
echo "COMPLETE $(date -Is)"
