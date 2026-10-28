#!/usr/bin/env bash
# Frozen-index 0.5% extension. Every completed measurement is audited before skip.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$REPO/experiments/selectivity"
ROOT=/mnt/ssd/ymy/CubeGraph/new-host-20260919
BASE=/mnt/ssd/ymy/CubeGraph
OFF="$ROOT/official"
AC="$BASE/external/acorn-source"
OUT=$(realpath "$1")
cd "$OUT"
mkdir -p bin logs provenance workloads converted results radius/workloads radius/results
trap 'rc=$?; printf "exit=%s time=%s\n" "$rc" "$(date -Is)" > state.txt' EXIT
exec 9>"$ROOT/logs/runner.lock"
echo "WAIT_LOCK $(date -Is)"; flock 9; echo "LOCKED $(date -Is)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run(){ timeout --signal=TERM --kill-after=30s 10h "$@"; }
stage(){
 local tag=$1; shift
 if [ -f "logs/$tag.complete" ]; then return; fi
 if [ -e "logs/$tag.log" ]; then echo "INCOMPLETE $tag: preserve and inspect the prior attempt before retry" >&2; return 1; fi
 echo "START $tag $(date -Is)"
 run "$@" >"logs/$tag.log" 2>&1
 touch "logs/$tag.complete"
 echo "DONE $tag $(date -Is)"
}
python3 "$OFF/verify_sources.py" >provenance/author-source-verification.txt
python3 - "$REPO" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1])
for f,h in [('include/IndexCube.h','340381c15d04037f34547288bcff318d5445ceca5bc4bec7ba1b1bdc880a3977'),('third_party/hnsw-cube.h','3edc62462112a07c0269bdfa6a4797681d2f7bc4daba9643008b299e5d00df07')]:
 assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,f
print('CORE_UNCHANGED')
PY
flags=(-std=c++17 -O3 -march=core-avx2 -mavx512f -fopenmp)
if [ ! -f provenance/compiled.complete ]; then
 g++ "${flags[@]}" "$SRC/generate_boxes.cpp" -o bin/generate_boxes
 g++ "${flags[@]}" -I"$REPO/include" -I"$REPO/third_party" "$REPO/experiments/bench_bottom_up.cpp" -o bin/cg
 g++ "${flags[@]}" -I"$REPO/include" -I"$REPO/third_party" "$SRC/bench_post.cpp" -o bin/post
 g++ -O3 -march=native -std=c++17 -pthread -I"$OFF/KHI" -I"$OFF/KHI/include" "$SRC/bench_khi.cpp" "$OFF/KHI/data.cpp" "$OFF/KHI/kd_tree.cpp" "$OFF/KHI/ma_graph.cpp" "$OFF/KHI/query.cpp" -o bin/khi
 g++ -O3 -march=native -std=c++17 -pthread -I"$OFF/iRangeGraph/include" "$SRC/bench_irange.cpp" -o bin/irange
 g++ "${flags[@]}" -I"$AC" "$SRC/bench_acorn.cpp" "$AC/build/faiss/libfaiss.a" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl -lm -o bin/acorn
 g++ "${flags[@]}" -ffp-contract=off "$SRC/generate_radius.cpp" -o bin/generate_radius
 g++ "${flags[@]}" -ffp-contract=off -I"$REPO/include" -I"$REPO/third_party" "$SRC/bench_radius.cpp" -o bin/radius
 g++ "${flags[@]}" -ffp-contract=off -DACORN_DRIVER -I"$AC" "$SRC/bench_radius.cpp" "$AC/build/faiss/libfaiss.a" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl -lm -o bin/radius_acorn
 sha256sum bin/* "$SRC"/*.{cpp,py,sh,h} "$REPO/experiments/bench_bottom_up.cpp" "$REPO/experiments/audit_box.py" >provenance/compiled.sha256
 touch provenance/compiled.complete
fi
sha256sum -c provenance/compiled.sha256 >logs/compiled-verification.log
if [ ! -f provenance/compiled-v2.complete ]; then
 g++ "${flags[@]}" -I"$AC" "$SRC/bench_acorn.cpp" "$AC/build/faiss/libfaiss.a" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl -lm -o bin/acorn-v2
 g++ "${flags[@]}" -ffp-contract=off -DACORN_DRIVER -I"$AC" "$SRC/bench_radius.cpp" "$AC/build/faiss/libfaiss.a" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl -lm -o bin/radius_acorn-v2
 sha256sum bin/*-v2 "$SRC"/*.{cpp,py,sh,h} "$REPO/experiments/bench_bottom_up.cpp" "$REPO/experiments/audit_box.py" >provenance/compiled-v2.sha256
 touch provenance/compiled-v2.complete
fi
sha256sum -c provenance/compiled-v2.sha256 >logs/compiled-v2-verification.log
python3 "$SRC/audit_validity.py" >logs/validity-selftest.log
{ date -Is; uname -a; g++ --version; lscpu; free -h; } >provenance/environment.txt
stage preflight python3 "$SRC/preflight.py" "$OUT"
configs=('sift uniform_2d' 'sift uniform_3d' 'sift uniform_4d' 'yfcc real_2d' 'yfcc real_3d' 'wikitext1m real_2d' 'wikitext1m real_3d' 'msmarc10m uniform_2d' 'msmarc10m uniform_3d' 'msmarc10m uniform_4d')
for cfg in "${configs[@]}"; do
 read -r ds meta <<< "$cfg"; tag=${ds}_${meta}
 data="$BASE/DATA"; cvroot="$OFF/converted"; ixroot="$OFF/indexes"; acix="$ROOT/acorn-rerun/indexes/acorn"
 if [ "$ds" = wikitext1m ]; then data="$BASE/DATA-dedup"; cvroot="$OFF/converted-dedup"; ixroot="$OFF/indexes-dedup"; acix="$ROOT/acorn-rerun/indexes/acorn-dedup"; fi
 wl="$OUT/workloads/$tag"; cv="$OUT/converted/$tag"; mkdir -p "$wl"
 if [ ! -f "provenance/input_$tag.sha256" ]; then
  sha256sum "$data/$ds/${ds}_base.fvecs" "$data/$ds/${ds}_query.fvecs" "$data/$ds/${ds}_metadata_$meta.bin" >"provenance/input_$tag.sha256"
 fi
 stage "generate_$tag" env OMP_NUM_THREADS=32 bin/generate_boxes --data-root "$data" --dataset "$ds" --metadata "$meta" --output-dir "$wl" --queries 1000 --k 20 --threads 32 --seed 42 --mode fixed-cardinality
 stage "gt_$tag" python3 "$SRC/verify_boxes.py" "$data" "$ds" "$meta" "$wl/${tag}_r0050.bin" --workers 8
 stage "convert_$tag" python3 "$SRC/convert.py" "$data" "$ds" "$meta" "$wl/${tag}_r0050.bin" "$cvroot/$tag" "$cv"
 cgix="$ROOT/bottom-up-20261015-final/indexes/$tag.cube"
 if [ ! -f "provenance/index_$tag.sha256" ]; then
  sha256sum "$cgix" "$cgix".layer_* "$ixroot/khi_${tag}_tree.bin" "$ixroot/khi_${tag}_graph.bin" "$data/$ds/${meta}_${ds}.post" "$acix/${ds}_M32_g12_mb64.index" >"provenance/index_$tag.sha256"
  if [[ "$meta" = *_2d ]]; then sha256sum "$ixroot/irange_$tag" >>"provenance/index_$tag.sha256"; fi
 fi
 out="$OUT/results/cg_$tag.csv"
 stage "query_cg_$tag" env CG_SEARCH_MODE=pred CG_RATIO_FILTER=0050 bin/cg query "$data" "$ds" "$meta" multi-cross2 "$cgix" "$wl" "$out"
 run python3 "$REPO/experiments/audit_box.py" "$out" "$ds" "$meta" "$wl" "$data" cube --ratio 50 >"logs/audit_cg_$tag.log" 2>&1
 out="$OUT/results/post_$tag.csv"
 stage "query_post_$tag" bin/post post "$data" "$ds" "$meta" "$data/$ds/${meta}_${ds}.post" "$wl" "$out"
 run python3 "$SRC/audit_post.py" "$out" "$ds" "$meta" "$wl" "$data" cube --ratio 50 >"logs/audit_post_$tag.log" 2>&1
 out="$OUT/results/khi_$tag.csv"
 stage "query_khi_$tag" bin/khi "$cv" "$ixroot/khi_$tag" "$out"
 run python3 "$SRC/audit_native.py" "$cv" "$out" >"logs/audit_khi_$tag.log" 2>&1
 if [[ "$meta" = *_2d ]]; then
  out="$OUT/results/irange_$tag.csv"
  stage "query_irange_$tag" bin/irange "$cv" "$ixroot/irange_$tag" "$out"
  run python3 "$SRC/audit_native.py" "$cv" "$out" >"logs/audit_irange_$tag.log" 2>&1
 fi
 for rep in 1 2 3; do
  dir="$OUT/results/acorn_${tag}_rep$rep"; mkdir -p "$dir"
  stage "query_acorn_${tag}_rep$rep" env OMP_NUM_THREADS=32 bin/acorn-v2 --dataset "$ds" --metadata "$meta" --cube-data-root "$data" --index-dir "$acix" --output-dir "$dir" --workload-dir "$wl" --M 32 --gamma 12 --M-beta 64 --queries 1000 --gt-threads 32 --ratios 0.005 --ef-values 20,32,64,128,256,512,1024,2048,4096
  run python3 "$SRC/audit_acorn.py" "$dir/${ds}_M32_g12_mb64_${meta}.csv" "$ds" "$meta" "$wl" "$data" acorn --ratio 50 >"logs/audit_acorn_${tag}_rep$rep.log" 2>&1
 done
 # Extend the matched-index search-mode comparison on its two original datasets.
 if [ "$meta" = uniform_2d ] && { [ "$ds" = sift ] || [ "$ds" = msmarc10m ]; }; then
  out="$OUT/results/fly_$tag.csv"
  stage "query_fly_$tag" env CG_SEARCH_MODE=fly CG_RATIO_FILTER=0050 bin/cg query "$data" "$ds" "$meta" multi-cross2 "$cgix" "$wl" "$out"
  run python3 "$REPO/experiments/audit_box.py" "$out" "$ds" "$meta" "$wl" "$data" cube --ratio 50 >"logs/audit_fly_$tag.log" 2>&1
 fi
 echo "DATASET_COMPLETE $tag $(date -Is)"
done
for d in 2 3 4; do stage "generate_radius$d" env OMP_NUM_THREADS=16 bin/generate_radius "$BASE/DATA/sift" "$d" 0 1000 "$OUT/radius/workloads"; done
stage generate_compose2 env OMP_NUM_THREADS=16 bin/generate_radius "$BASE/DATA/sift" 2 1 1000 "$OUT/radius/workloads"
stage gt_radius python3 "$SRC/verify_radius.py" "$BASE/DATA/sift" "$OUT/radius/workloads" "$OUT/radius/results" --workers 8 --gt-only
for w in "$OUT/radius/workloads/"*.bin; do
 name=$(basename "$w" .bin); case "$name" in radius3d*) d=3;; radius4d*) d=4;; *) d=2;; esac
 for method in cg post acorn; do
  binary=bin/radius; ix="$ROOT/bottom-up-20261015-final/indexes/sift_uniform_${d}d.cube"
  if [ "$method" = post ]; then ix="$BASE/DATA/sift/uniform_${d}d_sift.post"; fi
  if [ "$method" = acorn ]; then binary=bin/radius_acorn-v2; ix="$ROOT/acorn-rerun/indexes/acorn/sift_M32_g12_mb64.index"; fi
  stage "query_${method}_$name" "$binary" "$method" "$BASE/DATA/sift" "$ix" "$w" "$OUT/radius/results/${method}_$name.csv" 0
 done
done
run python3 "$SRC/verify_radius.py" "$BASE/DATA/sift" "$OUT/radius/workloads" "$OUT/radius/results" --workers 8 >logs/radius-final-audit.log 2>&1
sha256sum results/*.csv results/*/*.csv radius/results/*.csv >provenance/results.sha256
echo "COMPLETE $(date -Is)"
