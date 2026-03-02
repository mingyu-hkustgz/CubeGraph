# Query Filter and Groundtruth Generation

Scripts for generating query filters with specific selectivity and computing groundtruth for filtered vector search.

## Overview

This workflow generates:
1. **Query Filters**: Multi-dimensional range or radius filters with target selectivity (1%, 5%, 10%, 30%)
2. **Groundtruth**: K nearest neighbors for each filtered query using brute-force search

## Scripts

### Python Scripts

#### `generate_filters.py`
Generates query filters with specific selectivity levels.

**Usage:**
```bash
python3 scripts/generate_filters.py \
    --metadata DATA/sift_metadata_uniform_2d.bin \
    --query DATA/sift_query.fvecs \
    --output DATA/filters/sift_uniform_range_1pct.json \
    --selectivity 0.01 \
    --filter-type range \
    --seed 42
```

**Arguments:**
- `--metadata`: Path to metadata .bin file
- `--query`: Path to query .fvecs file
- `--output`: Output filter file (.json)
- `--selectivity`: Target selectivity (e.g., 0.01 for 1%)
- `--filter-type`: Filter type (`range` or `radius`)
- `--seed`: Random seed for reproducibility

**Filter Types:**
- **Range Filter**: Multi-dimensional bounding box with min/max bounds for each dimension
- **Radius Filter**: Sphere with center point and radius

**Output Format (JSON):**
```json
[
  {
    "query_id": 0,
    "filter_type": "range",
    "params": {
      "min_bounds": [10.5, 20.3],
      "max_bounds": [30.2, 45.7]
    },
    "target_selectivity": 0.01,
    "actual_selectivity": 0.0098
  },
  ...
]
```

#### `generate_groundtruth.py`
Computes groundtruth (K nearest neighbors) for filtered queries.

**Usage:**
```bash
python3 scripts/generate_groundtruth.py \
    --base DATA/sift_base.fvecs \
    --query DATA/sift_query.fvecs \
    --metadata DATA/sift_metadata_uniform_2d.bin \
    --filters DATA/filters/sift_uniform_range_1pct.json \
    --output DATA/groundtruth/sift_uniform_range_1pct_gt.ivecs \
    --k 100
```

**Arguments:**
- `--base`: Path to base vectors .fvecs file
- `--query`: Path to query .fvecs file
- `--metadata`: Path to metadata .bin file
- `--filters`: Path to filters .json file
- `--output`: Output groundtruth .ivecs file
- `--k`: Number of nearest neighbors

**Output Format (.ivecs):**
For each query: `[k: int32] [id_0: int32] [id_1: int32] ... [id_{k-1}: int32]`

### Shell Scripts

#### `generate_all_filters.sh`
Batch generates filters for all combinations of:
- Distributions: uniform, normal, clustered, skewed
- Filter types: range, radius
- Selectivity levels: 1%, 5%, 10%, 30%

**Usage:**
```bash
./scripts/generate_all_filters.sh
```

**Output:**
Creates filter files in `DATA/filters/`:
- `sift_uniform_range_1pct.json`
- `sift_uniform_range_5pct.json`
- `sift_uniform_radius_1pct.json`
- ... (32 files total: 4 distributions × 2 filter types × 4 selectivity levels)

#### `generate_all_groundtruth.sh`
Batch generates groundtruth for all filter files.

**Usage:**
```bash
./scripts/generate_all_groundtruth.sh
```

**Output:**
Creates groundtruth files in `DATA/groundtruth/`:
- `sift_uniform_range_1pct_gt.ivecs`
- `sift_uniform_range_5pct_gt.ivecs`
- ... (32 files total)

## Complete Workflow

### Step 1: Generate Metadata
```bash
# Generate metadata for all distributions
./scripts/generate_all_metadata.sh
```

This creates:
- `DATA/sift_metadata_uniform_2d.bin`
- `DATA/sift_metadata_normal_2d.bin`
- `DATA/sift_metadata_clustered_2d.bin`
- `DATA/sift_metadata_skewed_2d.bin`

### Step 2: Generate Filters
```bash
# Generate filters for all combinations
./scripts/generate_all_filters.sh
```

This creates 32 filter files in `DATA/filters/`.

### Step 3: Generate Groundtruth
```bash
# Generate groundtruth for all filters
./scripts/generate_all_groundtruth.sh
```

This creates 32 groundtruth files in `DATA/groundtruth/`.

### Step 4: Visualize (Optional)
```bash
# Visualize metadata distributions
./scripts/visualize_all_metadata.sh
```

## File Formats

### Metadata (.bin)
```
[n: uint64] [d: uint64] [vector_0: float32[d]] [vector_1: float32[d]] ...
```

### Query Vectors (.fvecs)
```
[dim: int32] [vector_0: float32[dim]] [dim: int32] [vector_1: float32[dim]] ...
```

### Filters (.json)
```json
[
  {
    "query_id": int,
    "filter_type": "range" | "radius",
    "params": {
      // For range: "min_bounds": [float, ...], "max_bounds": [float, ...]
      // For radius: "center": [float, ...], "radius": float
    },
    "target_selectivity": float,
    "actual_selectivity": float
  },
  ...
]
```

### Groundtruth (.ivecs)
```
[k: int32] [id_0: int32] ... [id_{k-1}: int32]  // Query 0
[k: int32] [id_0: int32] ... [id_{k-1}: int32]  // Query 1
...
```

## Selectivity

**Selectivity** is the fraction of points that pass the filter:
- 1% selectivity: ~10,000 points (out of 1M)
- 5% selectivity: ~50,000 points
- 10% selectivity: ~100,000 points
- 30% selectivity: ~300,000 points

The scripts use binary search to find filter parameters that achieve the target selectivity.

## Examples

### Generate a single filter
```bash
python3 scripts/generate_filters.py \
    --metadata DATA/sift_metadata_uniform_2d.bin \
    --query DATA/sift_query.fvecs \
    --output my_filter.json \
    --selectivity 0.05 \
    --filter-type radius
```

### Generate groundtruth for a single filter
```bash
python3 scripts/generate_groundtruth.py \
    --base DATA/sift_base.fvecs \
    --query DATA/sift_query.fvecs \
    --metadata DATA/sift_metadata_uniform_2d.bin \
    --filters my_filter.json \
    --output my_groundtruth.ivecs \
    --k 100
```

## Performance

- **Filter generation**: ~1-2 seconds per query (100 queries in ~2 minutes)
- **Groundtruth computation**: Depends on selectivity
  - 1% selectivity: ~10 seconds per query
  - 30% selectivity: ~300 seconds per query
  - For 100 queries with 1% selectivity: ~15-20 minutes

## Notes

- Query vectors remain unchanged (use existing `sift_query.fvecs`)
- Filters are generated independently for each query
- Groundtruth is computed using brute-force L2 distance
- The scripts support any attribute dimension (default: 2D)
- Random seeds ensure reproducibility
