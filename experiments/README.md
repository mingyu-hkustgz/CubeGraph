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

## Layer retention, radius, and updates

`slice_index.py INDEX_PATH OUTPUT_DIR` retains the finest 1 through 6 layers of
an existing six-layer 2D index. It preserves geometric resolutions and graph
files, and records logical file sizes. Query these subset directories with the
box driver using its default on-the-fly mode.

Build the radius/composition driver with `-ffp-contract=off` to preserve the
float32 predicate semantics of the frozen workloads:

```sh
c++ -std=c++17 -O3 -march=core-avx2 -mavx512f -fopenmp \
  -ffp-contract=off -Iinclude -Ithird_party \
  experiments/radius/benchmark.cpp -o build/radius
OMP_NUM_THREADS=1 build/radius cg "$DATA_ROOT/sift" "$INDEX_PATH" \
  "$RADIUS_WORKLOAD" "$OUTPUT_CSV" 0
```

The last argument is 0 for the full sweep and 1 for the two-width edge-case
sweep. `radius/verify_radius_compose.py` independently checks frozen predicates,
exact ground truth, returned IDs, recall, and wall-clock QPS using NumPy.
Run it with the SIFT data directory, workload directory and result directory;
the complete auditor expects CG, ACORN and POST files for all 16 workloads.

```sh
c++ -std=c++17 -O3 -march=core-avx2 -mavx512f -fopenmp \
  -Iinclude -Ithird_party experiments/bench_dynamic.cpp -o build/dynamic
OMP_NUM_THREADS=32 build/dynamic --dataset sift --source "$DATA_ROOT/sift/" \
  --meta uniform_2d --k 20 --threads 32 --filter-ratio 0.1 --ratio 0.1
```

Dynamic experiments use a 200,000-vector pool, 100,000 initial vectors and
insert-only batches of 10%, 30% or 50% of the initial cardinality. Run in a fresh
working directory: summary files append measurements. Per-width search logs
are retained under `results/recall@20/`; summary points select the measured
setting closest to 95% recall and report achieved recall explicitly. These are
not the main box experiment's common 99% recall operating points. Query timing
uses a steady wall clock with traversal counters disabled; ID uniqueness and
predicate membership are checked outside timing. Missing output files or an
empty search summary are fatal errors.

## Search-mode and shape studies

See [search_modes](search_modes/README.md) for the audited comparison on
identical six-layer graphs and [aspect_ratio](aspect_ratio/README.md) for the
completed count-calibrated high-aspect-ratio study and its audited results.
`CG_RATIO_FILTER=0100` optionally restricts the box driver to the 1% workload;
without it, the original four-fraction grid is unchanged. For this explicit
subset only, pass `--ratio 100` to `audit_box.py`; its default still requires
all 96 measurements.
