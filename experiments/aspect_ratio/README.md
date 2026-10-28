# High-aspect-ratio experiment

Status (2026-10-28): **complete and audited**. All 16 dataset/aspect/mode
combinations finished on October 27; the complete archive was collected and
verified on October 28. All 384 measurement rows and 384 original returned-ID
files pass their audits and local hash checks; independent exhaustive checks
cover all 8,000 ground-truth queries. Every combination reaches 99% recall.
No legacy plotted coordinates are used as measurements.

The protocol freezes 1,000 queries at **1% actual qualifying
count**, k=20, on SIFT1M and YFCC 2D. Each query uses the same vector and sampled
anchor across aspect ratios 1/4/16/32. Aspect ratios are defined after per-attribute min/max normalization; stored
predicates remain in original metadata coordinates. Long-axis orientation alternates across
queries. Rectangles are translated when needed to stay inside the domain.
Anchors are sampled uniformly over data objects and rejected jointly if any
aspect ratio is infeasible or violates the count tolerance. Stored float32 predicates must satisfy a relative normalized-aspect tolerance of 1e-4. The manifest reports attempted/accepted anchors,
actual counts, predicates, input hashes and the ground-truth distance rule.
This conditions the workload on joint feasibility and does not represent all
possible query locations. Count tolerance is at most max(2, 0.1% of target).

Both discovery modes reuse the same current six-layer index; no index rebuild
is required. Each mode sweeps widths 20/32/64/128/256/512/1024/2048 with three
repeats, one query thread, and untimed warm-ups. Audits check exact ground truth,
predicate eligibility, uniqueness, recall and source hashes. Each operating
point must reach 99% recall in every repeat. These sequential passes are not
interleaved statistical trials. This is a CG shape/mode sensitivity study,
not a KHI or ACORN comparison.

From the repository root, on the existing Linux/x86 evaluation server:

```sh
bash experiments/aspect_ratio/run.sh "$DATA_ROOT" sift "$SIFT_INDEX" \
  "$SIFT_OUTPUT" "$SHARED_MEASUREMENT_LOCK" > "$SIFT_QUEUE_LOG" 2>&1
bash experiments/aspect_ratio/run.sh "$DATA_ROOT" yfcc "$YFCC_INDEX" \
  "$YFCC_OUTPUT" "$SHARED_MEASUREMENT_LOCK" > "$YFCC_QUEUE_LOG" 2>&1
```

Use the same shared lock as other experiments. Each output directory is
separate from the original index/results archive. Jobs with audited `.complete`
markers are skipped; existing incomplete CSVs are never overwritten and must
be inspected/moved aside before retrying. Inspect `state.txt`, `logs/`, workload
verification, and all eight CSV audit files before collecting results.

Local validation covers closed-form interval inclusion versus an independent
boundary calculation, uniform and skewed metadata fixtures, all four aspect
ratios, count/geometry checks, and independently recomputed exact ground truth.
The benchmark and optional ratio selector compiled and completed the formal
Linux run. Default four-fraction runs are unchanged.

## Results and evidence

The table gives median QPS [smallest width] with recall at least 99% in every
repeat. All eight widths, including lower-recall and slower settings, remain
in the CSVs and paper curves.

| Dataset | Aspect | Predetermined | On-the-fly | Pred/fly |
|---|---:|---:|---:|---:|
| SIFT1M | 1 | 2,894.6 [64] | 2,840.5 [64] | 1.019 |
| SIFT1M | 4 | 1,588.8 [64] | 1,516.0 [64] | 1.048 |
| SIFT1M | 16 | 1,019.3 [32] | 975.8 [32] | 1.045 |
| SIFT1M | 32 | 341.0 [32] | 325.3 [32] | 1.048 |
| YFCC | 1 | 688.1 [64] | 772.4 [64] | 0.891 |
| YFCC | 4 | 515.7 [64] | 560.5 [64] | 0.920 |
| YFCC | 16 | 277.0 [64] | 293.8 [64] | 0.943 |
| YFCC | 32 | 204.2 [64] | 212.3 [64] | 0.962 |

Elongation reduces throughput on both datasets; neither search mode dominates
both datasets. The sequential repeats do not establish statistical significance.
Changing aspect also changes the eligible vectors, even with a shared anchor
and qualifying count. Absolute QPS is not a cross-batch speedup over the
separate box workload in `search_modes/`.

SIFT accepts 1,000 of 1,000 anchors, with 10,000--10,002 qualifying objects.
YFCC accepts 1,000 of 4,512 anchors (22.2%), with 9,990--10,008 qualifying objects
for each aspect. Its rejected anchors comprise 3,349 normalized count/feasibility
failures, 111 original-coordinate count failures and 52 stored-geometry failures.
This joint rejection is part of the sampling protocol and limits generalization.

`results/` retains all CSVs, original audit receipts, frozen predicates and GT,
manifests, source/input/index hashes, completion states, the summary and the
full-collection verification receipt. Raw returned-ID files and logs are in the
full archive identified by SHA-256 in `results/batch.json`; they are not included
in this compact public copy. Recompute the public operating points without
overwriting the full-collection receipt:

```sh
python3 experiments/aspect_ratio/summarize.py experiments/aspect_ratio/results \
  --receipts-only --output-dir /tmp/cubegraph-aspect-check
cmp experiments/aspect_ratio/results/summary.csv /tmp/cubegraph-aspect-check/summary.csv
```

With the extracted full archive available, omit `--receipts-only` and pass its
root to verify every original returned-ID hash as well.

## Execution provenance

SIFT used commit `6afa68d`; YFCC used `121c766`. The index/search core is identical
to the audited `5a30277` implementation in both runs. The first YFCC generation
failed strict stored-geometry verification before any timing measurements:
float32 coordinate conversion distorted some thin boxes. Commit `121c766` adds
joint rejection after that conversion, using the unchanged 1e-4 audit tolerance.
The failed workload is preserved as `yfcc-failed-geometry-v1` in the full archive;
only the regenerated and fully verified YFCC workload appears in the results.
The per-dataset execution sources were compared byte-for-byte with their commits.
