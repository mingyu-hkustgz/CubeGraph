# Cube-Based Filtered Vector Search

## Overview

This implementation provides a cube-based spatial partitioning approach for filtered vector search using HNSW graphs with cross-cube edges.

## Key Components

### 1. `hnsw-cube.h` - Extended HNSW with Cross-Cube Edges

Located in `third_party/hnswlib/hnsw-cube.h`

**Key Features:**
- Extends `HierarchicalNSWStatic` with cross-cube edge support
- Each cube maintains connections to neighboring cubes in the spatial hierarchy
- Cross-cube edges enable efficient traversal across cube boundaries

**Cross-Cube Edge Structure:**
- For d-dimensional attribute space: 2*d neighbors per cube
  - 2D: 4 neighbors (left, right, top, bottom)
  - 3D: 6 neighbors (left, right, top, bottom, front, back)
- Each neighbor connection adds M_cross edges (default: 8)
- Total edges per node at level 0: M0 (intra-cube) + 2*d*M_cross (cross-cube)

**Example for 2D:**
```
Regular HNSW edges: M0 = 32
Cross-cube edges: 4 directions × 8 edges = 32
Total edges: 64
```

**Example for 3D:**
```
Regular HNSW edges: M0 = 32
Cross-cube edges: 6 directions × 8 edges = 48
Total edges: 80
```

### 2. `IndexCube.h` - Cube-Based Spatial Index

Located in `include/IndexCube.h`

**Key Features:**
- Hierarchical spatial partitioning using cubes
- Automatic cube subdivision based on point density
- Cross-cube edge construction for adjacent cubes
- Efficient filtered search by identifying overlapping cubes

**Cube Structure:**
- Root cube covers entire attribute space
- Recursively subdivides into 2^d children until reaching threshold
- Leaf cubes contain HNSW indices with cross-cube edges
- Adjacent cubes are connected via cross-cube edges

**Search Process:**
1. Given query filter (range or radius), find overlapping cubes
2. Search within each overlapping cube using HNSW with cross-cube edges
3. Cross-cube edges allow traversal to neighboring cubes
4. Merge and rank results from all cubes

## Architecture

### Hierarchical Cube Structure

```
                    Root Cube
                   /    |    \
                  /     |     \
            Cube 0   Cube 1   Cube 2
            /  \      /  \      /  \
          C00 C01   C10 C11   C20 C21

Each leaf cube contains:
- HNSW graph for intra-cube search
- Cross-cube edges to adjacent cubes
```

### Cross-Cube Edge Encoding

Cross-cube edges are encoded as:
```
[direction:4 bits][neighbor_idx:4 bits][internal_id:24 bits]
```

- **direction**: Which neighbor (0-5 for 3D)
- **neighbor_idx**: Index in neighbor list
- **internal_id**: Node ID within neighbor cube

### Memory Layout

Each node at level 0 stores:
```
[Regular HNSW links][Cross-cube links][Label]
|<-- M0 edges -->|<-- 2*d*M_cross -->|
```

## Usage

### Building the Index

```cpp
#include "IndexCube.h"

// Create index
IndexCube index(attr_dim, vec_dim);

// Build from data
index.build_index(
    "data/sift_base.fvecs",      // Base vectors
    "data/sift_metadata_2d.bin",  // Metadata
    "data/sift_cube.index"        // Output index
);
```

### Searching with Filters

**Range Query:**
```cpp
// Create range filter
CubeBoundingBox filter_bbox(2);  // 2D
filter_bbox.min_bounds = {20.0f, 30.0f};
filter_bbox.max_bounds = {80.0f, 90.0f};

CubeQuery query_filter(filter_bbox, query_vector);

// Search
auto results = index.search(query_vector, query_filter, k);
```

**Radius Query:**
```cpp
// Create radius filter
std::vector<float> center = {50.0f, 50.0f};
float radius = 30.0f;

CubeQuery query_filter(center, radius, query_vector);

// Search
auto results = index.search(query_vector, query_filter, k);
```

## Parameters

### Cube Parameters
- `CUBE_THRESHOLD`: Minimum points per cube (default: 4096)
- `HNSW_CUBE_M`: Intra-cube edges (default: 16)
- `HNSW_CUBE_M_CROSS`: Cross-cube edges per direction (default: 8)
- `HNSW_CUBE_efConstruction`: Construction parameter (default: 200)

### Tuning Guidelines

**For high selectivity (>10%):**
- Increase `M_cross` for better cross-cube connectivity
- Decrease `CUBE_THRESHOLD` for finer partitioning

**For low selectivity (<1%):**
- Decrease `M_cross` to reduce overhead
- Increase `CUBE_THRESHOLD` for coarser partitioning

**For high-dimensional attribute space (d>3):**
- Increase `M_cross` as number of neighbors grows (2*d)
- Consider adaptive M_cross per dimension

## Compilation

Add to your CMakeLists.txt:

```cmake
add_executable(test_cube_index src/test_cube_index.cpp)
target_include_directories(test_cube_index PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)
```

Build:
```bash
mkdir -p build
cd build
cmake ..
make test_cube_index
```

## Testing

```bash
# Run test
./build/test_cube_index \
    DATA/sift_base.fvecs \
    DATA/sift_metadata_uniform_2d.bin \
    DATA/sift_query.fvecs \
    DATA/sift_cube.index \
    2
```

## Performance Characteristics

### Time Complexity

**Build:**
- Cube partitioning: O(n log n)
- HNSW construction per cube: O(n/c * log(n/c)) where c = number of cubes
- Cross-cube edge construction: O(n * M_cross * 2*d)
- Total: O(n log n + n * M_cross * d)

**Search:**
- Find overlapping cubes: O(log c)
- Search per cube: O(log(n/c) + M * ef)
- Cross-cube traversal: O(M_cross * 2*d)
- Total: O(log c + k * (log(n/c) + M * ef + M_cross * d))

### Space Complexity

**Per node:**
- Regular HNSW: M0 * sizeof(tableint) ≈ 128 bytes (M0=32)
- Cross-cube edges: 2*d*M_cross * sizeof(tableint)
  - 2D: 4*8*4 = 128 bytes
  - 3D: 6*8*4 = 192 bytes
- Total: 256 bytes (2D), 320 bytes (3D)

**Overhead vs regular HNSW:**
- 2D: 2× memory
- 3D: 2.5× memory

## Advantages

1. **Efficient Filtered Search**: Only search cubes overlapping with filter
2. **Cross-Cube Connectivity**: Seamless traversal across cube boundaries
3. **Scalability**: Hierarchical structure scales to large datasets
4. **Flexibility**: Supports both range and radius queries
5. **Adaptivity**: Cube size adapts to data distribution

## Limitations

1. **Memory Overhead**: 2-2.5× memory vs regular HNSW
2. **Build Time**: Additional time for cross-cube edge construction
3. **Dimension Sensitivity**: Cross-cube edges grow with 2*d
4. **Boundary Effects**: Points near cube boundaries may have suboptimal connectivity

## Comparison with Other Approaches

### vs. Index2D (Segment Tree)
- **Pros**: Better for multi-dimensional queries, more flexible
- **Cons**: Higher memory overhead, more complex

### vs. IndexGridND (Hierarchical Grid)
- **Pros**: Cross-cube edges improve connectivity
- **Cons**: Similar memory, slightly slower build

### vs. IndexRTreeND (R-tree)
- **Pros**: Better for uniform data, simpler structure
- **Cons**: Less adaptive to data distribution

## Future Improvements

1. **Adaptive M_cross**: Adjust cross-cube edges based on cube density
2. **Lazy Cross-Cube Construction**: Build cross-cube edges on-demand
3. **Compression**: Compress cross-cube edge encoding
4. **Parallel Construction**: Parallelize cube building and edge construction
5. **Dynamic Updates**: Support insertion/deletion with cube rebalancing

## References

- HNSW: Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs.
- Spatial Indexing: Samet, H. (2006). Foundations of multidimensional and metric data structures.

## Example Output

```
========================================
Cube-Based Filtered Vector Search Test
========================================
Base vectors: DATA/sift_base.fvecs
Metadata: DATA/sift_metadata_uniform_2d.bin
Query vectors: DATA/sift_query.fvecs
Attribute dimension: 2

Building cube index...
  Vectors: 1000000
  Vector dim: 128
  Attribute dim: 2
  Total leaf cubes: 243
  Cross-cube edges built
Build time: 45230 ms

Testing queries...
Query 0:
  Found 12 overlapping cubes
  Search time: 234 μs
  Results: 10 neighbors
    0: ID=12345, dist=123.45
    1: ID=67890, dist=145.67
    ...

========================================
Test completed successfully!
========================================
```
