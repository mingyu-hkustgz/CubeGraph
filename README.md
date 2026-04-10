# CubeGraph: Hierarchical Cube-Based Filtered Vector Similarity Search

A high-performance C++ library for filtered approximate nearest neighbor (ANN) search using hierarchical cube-based spatial partitioning. Built on HNSWlib with SIMD acceleration (AVX2).

## Key Features

- **Hierarchical Cube Partitioning**: Multi-level spatial indexing that adapts to filter size
- **Cross-Cube Edges**: HNSW extended with edges to adjacent cubes for seamless traversal
- **Multiple Filter Types**: BBox (axis-aligned bounding box), Radius (sphere), and Polygon filters
- **Composite Filters**: AND, OR, NOT compositions for complex filter predicates
- **Two Search Strategies**:
  - `fly_search`: Dynamic cube traversal starting from filter center
  - `predetermined_search`: Precomputed cube list based on BFS from filter region
- **SIMD Acceleration**: AVX2-optimized distance calculations
- **Adaptive Layer Selection**: Automatically selects appropriate cube resolution based on filter size

## Architecture

### Core Components

1. **IndexCube** (`include/IndexCube.h`): Main index with hierarchical cube partitioning
2. **HierarchicalNSWCube** (`third_party/hnsw-cube.h`): Extended HNSW with cross-cube edges
3. **Filter Functors**: BBoxFilter, RadiusFilter, PolygonFilter (implement `BaseFilterFunctor`)
4. **Composite Filters**: AndFilter, OrFilter, NotFilter for filter composition

### Data Formats

- **Vector files**: fvecs format (`[dim: int32][vec[0]: float]...`)
- **Metadata files**: Binary format (`[n: size_t][d: size_t][vector_0: float[d]]...`)

### Hierarchical Cube Structure

Each layer divides attribute space into `(2^(layer_id+1))^attr_dim` cubes:
- Layer 0: 2 cubes per dimension (4 total for 2D)
- Layer 1: 4 cubes per dimension (16 total for 2D)
- Layer 2: 8 cubes per dimension (64 total for 2D)

Layer selection is based on filter size vs cube diagonal:
- Large filters → coarser layers (fewer, larger cubes)
- Small filters → finer layers (more, smaller cubes)

## Building

### Requirements

- C++17 compiler with AVX2 support
- CMake 3.15+
- Boost (program_options, filesystem)
- OpenMP

### Build Instructions

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Example

```cpp
#include "IndexCube.h"

using namespace hnswlib;

// Create index
IndexCube index(/*num_layers=*/4, /*M=*/16, /*ef_construction=*/200, /*cross_edge_count=*/2);

// Build index from data and metadata files
index.build_index("data.fvecs", "metadata.bin");

// Define bounding box filter
BoundingBox bbox(2);
bbox.min_bounds = {0.2f, 0.3f};
bbox.max_bounds = {0.8f, 0.7f};

// Search with fly_search (dynamic traversal)
auto results = index.fly_search(query_vector, k, &bbox);

// Or use predetermined_search (precomputed cube list)
auto results = index.predetermined_search(query_vector, k, &bbox);
```

### Filter Types

#### BBox Filter (Axis-Aligned Bounding Box)

```cpp
BoundingBox bbox(attr_dim);
bbox.min_bounds = {x_min, y_min, ...};
bbox.max_bounds = {x_max, y_max, ...};

auto results = index.fly_search(query_vector, k, &bbox);
```

#### Radius Filter (Sphere)

```cpp
RadiusFilterParams radius;
radius.sphere.center = {0.5f, 0.5f};
radius.sphere.radius = 0.2f;
radius.attr_dim = 2;

auto results = index.fly_search(query_vector, k, &radius);
```

#### Polygon Filter (2D only)

```cpp
PolygonFilterParams polygon;
polygon.vertices = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 1.0f}};
polygon.num_vertices = 3;
polygon.attr_dim = 2;

auto results = index.fly_search(query_vector, k, &polygon);
```

#### Composite Filters (AND/OR/NOT)

```cpp
CompositeFilterParams params;
params.bbox = bbox;
params.spec.use_bbox = true;
params.attr_dim = 2;

RadiusFilterParams radius;
radius.sphere = Sphere({0.5f, 0.5f}, 0.1f);
radius.attr_dim = 2;

params.radius = radius;
params.spec.use_radius = true;

// Default: BBox AND Radius
auto results = index.fly_search(query_vector, k, &params);
```

### Python Scripts

```bash
cd scripts
pip install -r requirements.txt

# Generate test data
python generate_metadata.py --type uniform --count 10000 --attr-dim 2

# Visualize metadata distribution
python visualize_metadata.py <metadata.bin>
```

## Project Structure

```
CubeGraph/
├── CMakeLists.txt
├── README.md
├── include/
│   └── IndexCube.h              # Main index with hierarchical cubes
├── src/
│   ├── bench_hierarchical_cube.cpp    # BBox filter benchmarks
│   ├── bench_hierarchical_ball.cpp      # Radius filter benchmarks
│   ├── bench_hierarchical_polygon.cpp   # Polygon filter benchmarks
│   ├── bench_post_filtering.cpp        # Post-filtering comparison
│   ├── bench_complex_filter.cpp        # Composite filter benchmarks
│   └── ...
├── third_party/
│   ├── hnswlib/                 # HNSWlib base
│   ├── hnsw-cube.h              # Extended HNSW with cross-cube edges
│   ├── matrix.h                 # Matrix utilities
│   └── ...
└── scripts/
    ├── generate_metadata.py     # Generate test metadata
    ├── visualize_metadata.py    # Visualize metadata distribution
    └── ...
```

## Search Algorithms

### Fly Search

Dynamic traversal starting from the cube containing the filter center:
1. Select layer based on filter size vs cube diagonal
2. Start from the cube containing filter center (or centroid for polygon)
3. Traverse HNSW within current cube
4. Follow cross-cube edges to adjacent cubes only if they contain in-filter nodes
5. Continue until all accessible cubes are explored

### Predetermined Search

Precomputed cube list based on BFS from filter region:
1. Select layer based on filter bounding box
2. BFS from central cube, adding all cubes that overlap filter region
3. Search HNSW across all precomputed cubes in parallel

## Build Flags

The project uses aggressive optimization flags:
- `-Ofast -march=core-avx2 -mavx512f -fopenmp -ftree-vectorize`

These require a CPU with AVX2 support.

## References

**CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data** [[paper](https://arxiv.org/abs/2604.06616)] [[bibtex](#bibtex-cite)] - Yang et al., 2026

If you use **CubeGraph** in a research paper, please cite:

```
@article{yang2026cubegraph,
  title={CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data},
  author={Mingyu Yang and Wentao Li and Wei Wang},
  year={2026},
  eprint={2604.06616},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  url={https://arxiv.org/abs/2604.06616}
}
```

<span id="bibtex-cite"></span>
