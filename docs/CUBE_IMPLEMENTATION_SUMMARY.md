# Cube-Based Index Implementation Summary

## Files Created

### 1. Core Implementation Files

#### `third_party/hnswlib/hnsw-cube.h` (16KB)
Extended HNSW graph structure with cross-cube edges.

**Key Classes:**
- `HierarchicalNSWCube<dist_t>`: Extends `HierarchicalNSWStatic` with cross-cube connectivity

**Key Features:**
- Cross-cube edge storage and management
- Neighbor cube tracking
- Extended search with cross-cube traversal
- Configurable M_cross parameter (edges per neighbor direction)

**Memory Layout:**
```
Node at Level 0:
[Intra-cube edges: M0 × 4 bytes]
[Cross-cube edges: 2*d*M_cross × 4 bytes]
[Label: 4 bytes]

Example (2D, M0=32, M_cross=8):
[128 bytes][128 bytes][4 bytes] = 260 bytes per node
```

#### `include/IndexCube.h` (16KB)
Hierarchical cube-based spatial index.

**Key Classes:**
- `IndexCube`: Main index class with hierarchical cube structure
- `CubeNode`: Individual cube with HNSW index and neighbor connections
- `CubeBoundingBox`: Multi-dimensional bounding box
- `CubeQuery`: Query structure supporting range and radius filters
- `CubeFilter`: Filter functor for metadata-based filtering

**Key Features:**
- Recursive cube subdivision (2^d children per node)
- Automatic neighbor detection and cross-cube edge construction
- Efficient filtered search by identifying overlapping cubes
- Support for both range and radius queries

### 2. Test and Documentation

#### `src/test_cube_index.cpp` (6.3KB)
Test program demonstrating cube-based filtered search.

**Features:**
- Index building from base vectors and metadata
- Range query testing
- Radius query testing
- Performance measurement

#### `docs/CUBE_INDEX.md` (7.9KB)
Comprehensive documentation.

**Contents:**
- Architecture overview
- Usage examples
- Parameter tuning guidelines
- Performance characteristics
- Comparison with other approaches

## Architecture Overview

### Hierarchical Cube Structure

```
                    Root Cube [0, 100] × [0, 100]
                   /           |           \
                  /            |            \
        [0,50]×[0,50]   [50,100]×[0,50]   [0,50]×[50,100]  ...
            /  \            /  \              /  \
         Leaf Cubes with HNSW indices
```

### Cross-Cube Edge Connectivity (2D Example)

```
    Cube A          Cube B
    +-----+         +-----+
    |  *  |-------->|  *  |  Cross-cube edges
    |  *  |<--------|  *  |  (M_cross = 8 per direction)
    +-----+         +-----+
       ↑               ↑
       |               |
    Cube C          Cube D
    +-----+         +-----+
    |  *  |-------->|  *  |
    |  *  |<--------|  *  |
    +-----+         +-----+

Each cube maintains:
- Intra-cube HNSW edges (M0 = 32)
- Cross-cube edges to 4 neighbors (4 × 8 = 32)
- Total: 64 edges per node
```

### Search Process

```
1. Query with filter: Range [20, 80] × [30, 90]

2. Find overlapping cubes:
   +---+---+---+---+
   |   | X | X |   |  X = overlapping cubes
   +---+---+---+---+
   |   | X | X | X |
   +---+---+---+---+
   |   |   | X |   |
   +---+---+---+---+

3. Search in each overlapping cube:
   - Start from entry point in cube
   - Traverse intra-cube edges
   - Follow cross-cube edges to neighbors
   - Apply metadata filter

4. Merge results from all cubes
```

## Key Differences from hnsw-static.h

### Extended Data Structure

**hnsw-static.h:**
```cpp
// Level 0 node:
[M0 edges][Label]
```

**hnsw-cube.h:**
```cpp
// Level 0 node:
[M0 edges][2*d*M_cross cross-cube edges][Label]

// Additional members:
std::vector<NeighborCube> neighbor_cubes_;
size_t M_cross_;
size_t attr_dim_;
```

### Extended Search Algorithm

**hnsw-static.h:**
```cpp
searchBaseLayer() {
    // Only traverse intra-cube edges
    for (neighbor in current_node.edges) {
        visit(neighbor);
    }
}
```

**hnsw-cube.h:**
```cpp
searchBaseLayerWithCrossCubes() {
    // Traverse intra-cube edges
    for (neighbor in current_node.edges) {
        visit(neighbor);
    }

    // Traverse cross-cube edges
    for (cross_edge in current_node.cross_edges) {
        neighbor_cube = decode_cube(cross_edge);
        visit(neighbor_cube.node);
    }
}
```

## Edge Count Comparison

### 2D Case (4 neighbors)

| Component | hnsw-static | hnsw-cube | Overhead |
|-----------|-------------|-----------|----------|
| Intra-cube edges | 32 | 32 | 0 |
| Cross-cube edges | 0 | 32 (4×8) | +32 |
| **Total** | **32** | **64** | **2×** |

### 3D Case (6 neighbors)

| Component | hnsw-static | hnsw-cube | Overhead |
|-----------|-------------|-----------|----------|
| Intra-cube edges | 32 | 32 | 0 |
| Cross-cube edges | 0 | 48 (6×8) | +48 |
| **Total** | **32** | **80** | **2.5×** |

## Usage Example

```cpp
// 1. Build index
IndexCube index(2, 128);  // 2D metadata, 128D vectors
index.build_index(
    "sift_base.fvecs",
    "sift_metadata_uniform_2d.bin",
    "sift_cube.index"
);

// 2. Create range filter
CubeBoundingBox bbox(2);
bbox.min_bounds = {20.0f, 30.0f};
bbox.max_bounds = {80.0f, 90.0f};
CubeQuery query(bbox, query_vector);

// 3. Search
auto results = index.search(query_vector, query, 10);

// 4. Process results
while (!results.empty()) {
    auto [dist, id] = results.top();
    std::cout << "ID: " << id << ", dist: " << dist << std::endl;
    results.pop();
}
```

## Performance Characteristics

### Build Time
- **Cube partitioning**: O(n log n)
- **HNSW per cube**: O((n/c) log(n/c)) where c = #cubes
- **Cross-cube edges**: O(n × M_cross × 2d)
- **Total**: O(n log n + n × M_cross × d)

### Search Time
- **Find cubes**: O(log c)
- **Search per cube**: O(log(n/c) + M × ef)
- **Cross-cube**: O(M_cross × 2d)
- **Total**: O(log c + k × (log(n/c) + M × ef + M_cross × d))

### Memory
- **Per node**: (M0 + 2d × M_cross) × 4 bytes
- **2D**: 260 bytes (2× vs regular HNSW)
- **3D**: 324 bytes (2.5× vs regular HNSW)

## Advantages

1. **Efficient Filtered Search**: Only search relevant cubes
2. **Seamless Boundary Traversal**: Cross-cube edges eliminate boundary effects
3. **Scalable**: Hierarchical structure handles large datasets
4. **Flexible**: Supports range and radius queries
5. **Adaptive**: Cube size adapts to data density

## Integration with Existing Code

The cube-based index integrates seamlessly with existing filtered search infrastructure:

```
Existing:
- generate_filters.py → generates query filters
- generate_groundtruth.py → computes groundtruth
- IndexGridND.h → hierarchical grid (no cross-cube edges)
- IndexRTreeND.h → R-tree based (no cross-cube edges)

New:
- hnsw-cube.h → HNSW with cross-cube edges
- IndexCube.h → cube-based index using hnsw-cube
- test_cube_index.cpp → test program

Compatible with:
- All existing filter formats (.json)
- All existing metadata formats (.bin)
- All existing groundtruth formats (.ivecs)
```

## Next Steps

1. **Compile and test**:
   ```bash
   cd build
   cmake ..
   make test_cube_index
   ```

2. **Run with generated data**:
   ```bash
   ./test_cube_index \
       ../DATA/sift_base.fvecs \
       ../DATA/sift_metadata_uniform_2d.bin \
       ../DATA/sift_query.fvecs \
       ../DATA/sift_cube.index \
       2
   ```

3. **Benchmark against other methods**:
   - Compare with IndexGridND
   - Compare with IndexRTreeND
   - Measure recall, latency, memory

4. **Tune parameters**:
   - Adjust CUBE_THRESHOLD
   - Adjust M_cross
   - Test different selectivity levels

## Files Summary

```
CubeGraph/
├── third_party/hnswlib/
│   └── hnsw-cube.h              ← Extended HNSW with cross-cube edges
├── include/
│   └── IndexCube.h              ← Cube-based spatial index
├── src/
│   └── test_cube_index.cpp     ← Test program
└── docs/
    └── CUBE_INDEX.md            ← Documentation
```

All files are ready for compilation and testing!
