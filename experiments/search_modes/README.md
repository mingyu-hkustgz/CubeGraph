# Predetermined versus on-the-fly

These are current bottom-up implementation measurements from the audited
2026-10-15 batch, not the old paper's curves. `sources.json` identifies every
CSV and its source path/hash. Each accompanying audit receipt records returned
ID hashes and workload hashes. The original raw returned IDs remain in the
full experiment archive; they are not duplicated in this Git repository.

`provenance/run_all.sh` records that predetermined search used the original
six-layer index; on-the-fly used the six-layer retention artifact.
`provenance/slice_index.py` asserts byte identity of the full six-layer index
directory and links the same graph files. Vector/query/workload inputs and
widths are identical. Measurements are separate sequential timing passes.

```sh
python3 experiments/search_modes/summarize.py
# Additionally verify all 384 raw returned-ID files from the original archive:
python3 experiments/search_modes/summarize.py --ids-root "$FORMAL_RESULTS"
```

The operating point is the smallest width whose recall is at least 0.99 in
all three repeats; QPS is the median of those repeats. At qualifying fractions
1/2/5/10%, predetermined/on-the-fly ratios are 1.047/1.076/1.094/1.069 on
SIFT1M 2D and 0.997/0.998/1.005/1.001 on MSMARCO10M 2D. The selected widths
coincide. Small timing differences are descriptive, not a statistical
significance claim. These measurements do not compare modes on polygons.

To rerun both modes, invoke `bench_bottom_up.cpp` twice on the **same** index
and workload directory, using `CG_SEARCH_MODE=pred` and `CG_SEARCH_MODE=fly`
and distinct output paths. Audit both outputs with `audit_box.py`.
