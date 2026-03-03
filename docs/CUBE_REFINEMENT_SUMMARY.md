# Cube-Based Filtered Vector Search - Implementation Summary

## Date: 2026-03-02

## Overview

This document summarizes the work done to implement and refine a cube-based spatial partitioning approach for filtered vector search using HNSW graphs with cross-cube edges.

## 1. Project Structure

```
CubeGraph/
├── include/
│   ├── IndexCube.h          # Main cube-based index implementation
│   ├── Index2D.h            # 1D segment tree approach
│   ├── IndexGridND.h        # Hierarchical grid for N-dimensional attributes
│   └── IndexRTreeND.h      # R-tree with STR bulk loading
├── src/
│   └── test_cube_index.cpp  # Test program for cube index
├── third_party/
│   └── hnswlib/
│       ├── hnsw-cube.h      # Extended HNSW with cross-cube edges
│       └── hnsw-static.h    # Static HNSW implementation
├── scripts/
│   ├── visualize_cube_index.py  # Visualization script
│   └── ...
└── docs/
    └── CUBE_IMPLEMENTATION_SUMMARY.md
```

## 2. Key Implementation Details

### 2.1 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| HNSW_CUBE_M | 32 | Internal edges per node (intra-cube) |
| HNSW_CUBE_M_CROSS | 16 | Total cross-cube edges (distributed across 2*d directions) |
| HNSW_CUBE_efConstruction | 100 | HNSW construction parameter |
| CUBE_THRESHOLD | 50000 | Minimum vectors per leaf cube |

### 2.2 Cross-Cube Edge Calculation

For a d-dimensional attribute space:
- Number of neighbors: 2*d (e.g., 2D: 4 neighbors, 3D: 6 neighbors)
- Cross-cube edges per direction: M_cross / (2*d)
- For 2D: 16 / 4 = 4 edges per direction
- For 3D: 16 / 6 ≈ 2 edges per direction

### 2.3 Memory Layout

Each node at level 0 stores:
- Intra-cube links: M0 = M * 2 = 64 edges
- Cross-cube links: 2*d * M_cross_per_dir edges
- For 2D: 4 * 4 = 16 cross-cube edges
- Total: 80 edges per node

## 3. Bug Fixes

### 3.1 Filter Functionality Fix

**Problem**: The filter was not being applied correctly to the entry point node during HNSW search.

**Solution**: Added filter check for the entry point in `searchBaseLayerWithCrossCubes`:
```cpp
labeltype ep_label = this->getExternalLabel(ep_id);
bool ep_allowed = (isIdAllowed == nullptr) || (*isIdAllowed)(ep_label);
```

### 3.2 Label Handling Fix

**Problem**: Incorrect logic for converting internal IDs to external labels.

**Solution**: Simplified to always use `getExternalLabel`:
```cpp
labeltype label = this->getExternalLabel(pair.second);
```

### 3.3 Memory Layout Fix (Previous Work)

- Changed M from 16 to 32 internal edges
- Dynamic calculation of M_cross_per_dir based on dimension
- Added OpenMP support in IndexCube.h

## 4. Test Results

### 4.1 Small Dataset (10K vectors)

| Metric | Value |
|--------|-------|
| Build time | ~2.3 seconds |
| Leaf cubes | 1 (threshold: 50000) |
| Search time | 75-115 μs |
| Results | Correctly filtered |

### 4.2 SIFT Dataset (1M vectors)

| Metric | Value |
|--------|-------|
| Build time | ~98 seconds |
| Leaf cubes | 64 |
| Range query search time | ~2-3 ms |
| Radius query search time | ~32 ms |
| Overlapping cubes | 36 (for filter [20,80]×[20,80]) |

### 4.3 Filter Verification

Debug output confirmed correct filtering:
```
Filter: bbox=[20,20] to [80,80]
Result metadata: ID=887782:(22.01,78.46) ID=871280:(21.96,22.15) ID=208384:(20.53,78.78)
```
All results are within the filter bounding box.

## 5. Known Limitations

1. **Cross-cube edge building**: The ANN-based cross-cube edge building causes segmentation faults with large datasets. Currently using a simplified validation-only approach.

2. **Radius filter performance**: Takes ~32ms vs ~3ms for range filter - may need optimization.

3. **Duplicate results**: Results may contain duplicates from multiple overlapping cubes.

## 6. Files Modified

1. `include/IndexCube.h` - Main index implementation
   - Updated parameters
   - Added OpenMP include
   - Fixed filter integration
   - Added debug output (later removed)

2. `third_party/hnswlib/hnsw-cube.h` - Extended HNSW
   - Updated M_cross calculation
   - Fixed entry point filtering
   - Fixed label handling
   - Simplified cross-cube edge building

3. `src/test_cube_index.cpp` - Test program

4. `scripts/generate_metadata.py` - Metadata generation

## 7. Usage

### Build
```bash
mkdir -p build && cd build
cmake ..
make test_cube_index
```

### Run Test
```bash
./build/src/test_cube_index \
    DATA/sift/sift_base.fvecs \
    DATA/sift/sift_metadata_uniform_2d.bin \
    DATA/sift/sift_query.fvecs \
    DATA/sift/sift_cube.index \
    2
```

### Generate Metadata
```bash
python scripts/generate_metadata.py \
    --data DATA/sift/sift_base.fvecs \
    --output-dir DATA/sift \
    --attr-dim 2 \
    --distributions uniform
```

## 8. Future Work

1. Fix cross-cube edge building for large datasets
2. Optimize radius filter performance
3. Add result deduplication
4. Implement parallel cross-cube edge building with proper synchronization
5. Test with 3D metadata
6. Benchmark recall against groundtruth
