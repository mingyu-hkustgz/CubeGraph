#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/mnt/ssd/ymy/CubeGraph/new-host-20260919
BASE=/mnt/ssd/ymy/CubeGraph
DATA="$BASE/DATA"
WL="$BASE/external/codex-workloads/fixedcard-v1"
trap 'rc=$?; printf "exit=%s time=%s\n" "$rc" "$(date -Is)" > state.txt' EXIT
exec 9>"$ROOT/logs/runner.lock"
echo "WAIT_LOCK $(date -Is)"; flock 9; echo "LOCKED $(date -Is)"
mkdir -p provenance results/ablation results/layers results/distributions results/radius logs indexes subsets updates/results/dynamic-update
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CXXFLAGS=(-std=c++17 -O3 -march=core-avx2 -mavx512f -fopenmp -Iinclude -Ithird_party)
g++ "${CXXFLAGS[@]}" bench.cpp -o bin/bench
g++ "${CXXFLAGS[@]}" -DABLATION_METRICS bench.cpp -o bin/bench_metrics
g++ "${CXXFLAGS[@]}" radius/benchmark.cpp -o bin/radius
g++ "${CXXFLAGS[@]}" dynamic.cpp -o bin/dynamic
sha256sum include/* third_party/*.h third_party/hnswlib/*.h *.cpp *.py *.sh bin/* > provenance/code.sha256
{ date -Is; uname -a; lscpu; free -h; uptime; g++ --version; } > provenance/environment.txt
run(){ timeout --signal=TERM --kill-after=30s 10h "$@"; }
set_input(){
 ds=$1; meta=$2; data="$DATA"; wl="$WL"
 if [ "$ds" = wikitext1m ]; then data="$BASE/DATA-dedup"; wl="$BASE/external/codex-workloads-dedup/fixedcard-v1"; fi
}
build(){
 local tag=$1 variant=$2
 local ix="$PWD/indexes/$tag.cube"
 if [ -f "$ix.complete" ]; then return; fi
 echo "BUILD $tag $(date -Is)"
 OMP_NUM_THREADS=128 run /usr/bin/time -v bin/bench build "$data" "$ds" "$meta" "$variant" "$ix" > "logs/build_$tag.log" 2>&1
 sha256sum "$ix" "$ix".layer_* > "logs/build_$tag.sha256"
 touch "$ix.complete"
}
query(){
 local tag=$1 variant=$2 ix=$3 out=$4 mode=$5 binary=${6:-bin/bench}
 if [ -f "$out.complete" ]; then return; fi
 echo "QUERY $tag $mode $(date -Is)"
 CG_SEARCH_MODE="$mode" OMP_NUM_THREADS=1 run /usr/bin/time -v "$binary" query "$data" "$ds" "$meta" "$variant" "$ix" "$wl" "$out" > "logs/query_$tag.log" 2>&1
 python3 audit.py "$out" "$ds" "$meta" "$wl" "$data" cube >> "logs/query_$tag.log" 2>&1
 touch "$out.complete"
}
# Formal box comparison uses predetermined search for every dataset.
for cfg in 'sift uniform_2d' 'sift uniform_3d' 'sift uniform_4d' 'yfcc real_2d' 'yfcc real_3d' 'wikitext1m real_2d' 'wikitext1m real_3d' 'msmarc10m uniform_2d' 'msmarc10m uniform_3d' 'msmarc10m uniform_4d'; do
 read -r d m <<< "$cfg"; set_input "$d" "$m"; tag=${ds}_${meta}
 build "$tag" multi-cross2
 query "$tag" multi-cross2 "$PWD/indexes/$tag.cube" "$PWD/results/$tag.csv" pred
 sha256sum "$data/$ds/${ds}_metadata_${meta}.bin" "$wl/${ds}_${meta}_r"*.bin >> provenance/inputs.sha256
 done
# Radius/composition reuse frozen predicates and ground truth.
for w in "$ROOT/radius-compose-20261010/workloads/"*.bin; do
 name=$(basename "$w" .bin); case "$name" in radius3d*) dim=3;; radius4d*) dim=4;; *) dim=2;; esac
 out="$PWD/results/radius/cg_$name.csv"
 if [ ! -f "$out.complete" ]; then
  echo "RADIUS $name $(date -Is)"
  OMP_NUM_THREADS=1 run bin/radius cg "$DATA/sift" "$PWD/indexes/sift_uniform_${dim}d.cube" "$w" "$out" 0 > "logs/radius_$name.log" 2>&1
  test "$(wc -l < "$out")" -eq 25; touch "$out.complete"
 fi
 done
# The independent auditor checks unchanged baseline outputs alongside new CG.
for p in "$ROOT/radius-compose-20261010/results/"{post,acorn}_*; do
 case "$p" in *.csv|*.ids.bin) ln -s "$p" "results/radius/$(basename "$p")" 2>/dev/null || true;; esac
 done
OMP_NUM_THREADS=1 run python3 radius/verify_radius_compose.py "$DATA/sift" "$ROOT/radius-compose-20261010/workloads" results/radius --workers 16 > logs/radius_audit.log 2>&1
# Fixed-resolution layer sensitivity uses the native on-the-fly selector.
for d in sift msmarc10m; do
 set_input "$d" uniform_2d
 python3 slice_index.py "indexes/${ds}_${meta}.cube" "subsets/$ds"
 for L in 6 5 4 3 2 1; do
  query "${ds}_L$L" multi-cross2 "$PWD/subsets/$ds/L$L.cube" "$PWD/results/layers/${ds}_L$L.csv" fly
 done
 done
# Physical ablations: three independent builds; timing and counters separate.
for cfg in 'sift uniform_2d' 'sift uniform_3d' 'yfcc real_2d'; do
 read -r d m <<< "$cfg"; set_input "$d" "$m"
 for variant in multi-cross2 multi-cross0 singlefinest-cross2 singlefinest-cross0; do
  tag=${ds}_${meta}_${variant}
  for rep in 1 2 3; do build "${tag}_rep$rep" "$variant"; done
  ix="$PWD/indexes/${tag}_rep1.cube"
  query "${tag}_timing" "$variant" "$ix" "$PWD/results/ablation/$tag.timing.csv" fly
  query "${tag}_metrics" "$variant" "$ix" "$PWD/results/ablation/$tag.metrics.csv" fly bin/bench_metrics
 done
 done
for m in normal_2d clustered_2d skewed_2d hollow_2d; do
 set_input sift "$m"; tag=${ds}_${meta}; build "$tag" multi-cross2
 query "$tag" multi-cross2 "$PWD/indexes/$tag.cube" "$PWD/results/distributions/$tag.csv" fly
 done
for m in uniform_6d uniform_8d; do set_input sift "$m"; build "${ds}_${meta}" multi-cross2; done
# Insert-only dynamic maintenance: same frozen workload and original driver.
for cfg in 'sift uniform_2d' 'yfcc real_2d'; do
 read -r d m <<< "$cfg"
 for ratio in 0.1 0.3 0.5; do
  tag=${d}_r${ratio}
  [ -f "updates/$tag.complete" ] && continue
  echo "UPDATE $tag $(date -Is)"
  (cd updates; OMP_NUM_THREADS=32 run ../bin/dynamic --dataset "$d" --source "$DATA/$d/" --meta "$m" --k 20 --threads 32 --filter-ratio 0.1 --ratio "$ratio") > "logs/update_$tag.log" 2>&1
  touch "updates/$tag.complete"
 done
 done
sha256sum results/*.csv results/*/*.csv > provenance/results.sha256
printf 'COMPLETE %s\n' "$(date -Is)"
