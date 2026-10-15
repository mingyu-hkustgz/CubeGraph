# Reproducing a box experiment

Build on Linux/x86 with GCC and OpenMP from the repository root:

```sh
mkdir -p build
c++ -std=c++17 -O3 -march=core-avx2 -mavx512f -fopenmp \
  -Iinclude -Ithird_party experiments/bench_bottom_up.cpp -o build/bench
OMP_NUM_THREADS=128 build/bench build "$DATA_ROOT" sift uniform_2d multi-cross2 "$INDEX_PATH"
CG_SEARCH_MODE=pred OMP_NUM_THREADS=1 build/bench query \
  "$DATA_ROOT" sift uniform_2d multi-cross2 "$INDEX_PATH" "$WORKLOAD_DIR" "$OUTPUT_CSV"
python3 experiments/audit_box.py "$OUTPUT_CSV" sift uniform_2d "$WORKLOAD_DIR" "$DATA_ROOT" cube
```

`DATA_ROOT/<dataset>/` contains the fvecs base/query vectors and the binary
metadata file named `<dataset>_metadata_<metadata>.bin`. Metadata begins with
uint64 point count and dimension, followed by row-major float32 attributes.

The frozen box workloads are named `<dataset>_<metadata>_r0100.bin`, `r0200`,
`r0500`, and `r1000`. Each starts with uint64 query count, dimension and k;
float32 lower and upper bounds follow for each query, then row-major int64
exact top-k IDs. The auditor requires 1,000 queries, k=20, four selectivities,
eight search widths and three repeats, and independently recomputes returned
ID eligibility, uniqueness and recall. NumPy is required for the auditor.

Query timing excludes result auditing and ID-file writes. Construction timing
includes input loading and excludes serialization. Each parent graph uses one
refinement round with beam min(ef_construction, 4*M); the finest graph uses
M=16, ef_construction=200. Cross-edge searches use beam20 and retain two
neighbors per adjacent cube. The default query driver uses on-the-fly search;
set CG_SEARCH_MODE=pred for the main axis-aligned-box comparison. Radius and
composition use on-the-fly search.

The four physical variants are multi-cross2, multi-cross0,
singlefinest-cross2 and singlefinest-cross0. The single-layer variants keep the
same finest geometric resolution. Build separate artifacts for all variants;
never reuse an index merely because its filename exists.
