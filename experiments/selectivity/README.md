# 0.5% selectivity extension

Prepared October 28, 2026. Formal measurement and verification are pending;
no result or speedup is claimed by this protocol document.

This adds a more selective workload to the existing four-fraction study. It
reuses every frozen index and leaves all 1/2/5/10% measurements untouched.
The intended main-paper grid is 0.5/1/2/5%; the 10% curves remain in the
supplementary report so broader-filter reversals are still visible.

## Fixed protocol

- Ten box configurations: SIFT 2D/3D/4D, YFCC 2D/3D, deduplicated WIKI
  2D/3D, and MSMARCO10M 2D/3D/4D. CG, KHI, ACORN gamma=12 and POST run on
  all ten; iRangeGraph participates in its four native 2D configurations.
- The box generator preserves the original seed-42 object-anchor sampling,
  domain-clipped normalized maximum-distance calibration, first 1,000 query
  vectors, float32 stored bounds and k=20. This extension targets 0.5% and
  computes exhaustive float64 squared-L2 GT with object-ID tie breaking.
  `verify_boxes.py` independently recomputes every query's membership and GT,
  reports boundary-tie count deviations, and rejects a grossly incorrect
  realized fraction. It does not select queries based on method performance.
- Native KHI/iRangeGraph indexes use the existing vector and attribute order.
  `convert.py` verifies all frozen native vectors, queries and metadata before
  creating fresh 0.5% constraints. Float bounds map exactly to inclusive rank
  bounds for iRangeGraph. Author algorithm sources remain unchanged.
- Three single-threaded repetitions, 20 warm-ups, widths 20/32/64/128/256/512/
  1024/2048 (ACORN additionally 4096). The operating point is the smallest
  tested width with valid predicate-satisfying output and 99% recall in all three repeats; QPS is their median.
  Non-reaching settings remain reported. ACORN mask preparation is untimed.
- POST keeps its existing candidate-budget search and timing boundary,
  including the prior lightweight recall bookkeeping. Returned IDs are now
  retained for independent auditing; artifact writes are outside timing.
- SIFT1M and MSMARCO10M 2D additionally run on-the-fly CG on the same indexes
  and queries, extending the matched-index mode comparison.
- Four complex workloads: SIFT 2D/3D/4D balls and 2D box-minus-circle, with
  the original calibration and float32 predicate semantics. All three
  supported methods (CG on-the-fly, ACORN, POST) share predicates and exact
  GT; independent audits check all 4,000 queries and every returned ID.

The full extension has 14,000 independently checked GT queries, 78 result
CSVs and 1,422 measurement cells. These are planned counts, not a completion
receipt. No index construction, high-aspect-ratio sweep, or old measurement is
rerun. New experiment outputs must be in a separate directory.

## Execution

On the original Linux/x86 evaluation server, after packaging this repository:

```sh
mkdir /path/to/new-output
bash experiments/selectivity/run.sh /path/to/new-output
```

The runner uses the original shared measurement lock and serial execution;
each command has a 10-hour bound. It checks pinned author/core sources,
records input/index/binary hashes and audits outputs. A failed stage must be
inspected and preserved before resuming; existing complete stages are skipped
and their measurement outputs re-audited. Do not run other work on the server
while measurements are active.

## Invalid baseline outputs

The first 0.5% SIFT 2D ACORN query at width 20 stopped the original strict
harness. A preserved diagnostic reproduced query 145 returning object 118295
outside its box. This is consistent with the previously documented unfiltered
initial-result insertion in the pinned ACORN implementation. The library and
index are unchanged. Instrumentation v2 saves every raw ID and records predicate
violation counts outside timing, allowing later widths to be measured. An
independent auditor labels those cells `FAIL_PREDICATE`; collection may verify
that failure evidence, but such cells are never valid performance points.
Structural output errors (range, negative IDs other than -1, duplicates) still
abort. Curves and operating points require all three repetitions to be valid.
The initial stopped attempt and diagnostic are retained with their original
sources and binaries; already completed other-method measurements are reused.
