# High-aspect-ratio experiment

Status (2026-10-27): **prepared, not yet measured**. The school SSH route was
unreachable from the authoring machine. No legacy throughput coordinates are
used as results for the current implementation.

The archived paper swept aspect ratios 1/4/16/32 on SIFT and YFCC, but the
local archive contains only plotted coordinates, without the raw query/output
logs needed for the current protocol. Its nominal 10% filter setting and
32-thread timings are not comparable to the current single-threaded,
count-calibrated experiment. In a normalized 2D domain, a contained rectangle
with area 0.1 and aspect ratio 32 cannot exist (its long side would exceed 1).

The replacement protocol freezes 1,000 queries at **1% actual qualifying
count**, k=20, on SIFT1M and YFCC 2D. Each query uses the same vector and sampled
anchor across aspect ratios 1/4/16/32. Aspect ratios are defined after per-attribute min/max normalization; stored
predicates remain in original metadata coordinates. Long-axis orientation alternates across
queries. Rectangles are translated when needed to stay inside the domain.
Anchors are sampled uniformly over data objects and rejected jointly if any
aspect ratio is infeasible. The manifest reports attempted/accepted anchors,
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
The benchmark's new optional ratio selector still needs compilation and formal
execution on the evaluation server. Default four-fraction runs are unchanged.
