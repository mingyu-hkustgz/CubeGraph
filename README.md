# Filtered Vector KNN Search with HNSWlib

A high-performance C++ library for filtered vector similarity search with spatial and temporal attributes. Built on top of HNSWlib with SIMD acceleration (AVX2).

## Features

- **Efficient Vector Search**: Uses HNSWlib for fast approximate nearest neighbor search
- **Spatial Filtering**: Filter by geographic location (latitude/longitude) with radius queries
- **Temporal Filtering**: Filter by timestamp ranges
- **Multi-dimensional Range Filtering**: Combine spatial and temporal filters
- **SIMD Acceleration**: AVX2-optimized distance calculations (L2 and inner product)
- **Flexible Filter Composition**: AND, OR, and custom filter predicates
- **High Performance**: Optimized for large-scale datasets

## Architecture

### Core Components

1. **VectorMetadata** (`metadata.h`): Stores spatial (lat/lon) and temporal (timestamp) attributes
2. **SIMD Distance Functions** (`simd_distance.h`): AVX2-accelerated L2 and inner product distance
3. **Filter Predicates** (`filter_predicates.h`): Radius, range, timestamp, AND, OR, and custom filters
4. **FilteredHNSWIndex** (`filtered_hnsw_index.h`): Main index class with filtered search

## Building

### Requirements

- C++17 compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.15+
- AVX2-capable CPU (for SIMD acceleration)

### Build Instructions

```bash
# Clone the repository
cd CubeGraph

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Run example
./example

# Run benchmark
./benchmark
```

## Usage

### Basic Example

```cpp
#include "filtered_hnsw_index.h"

using namespace filtered_knn;

// Create index
FilteredHNSWIndex<L2DistanceSIMD> index(dim, max_elements);

// Add vectors with metadata
VectorMetadata metadata(40.7128f, -74.0060f, 1609459200000LL); // NYC, Jan 1 2021
index.add_vector(vector_data, metadata);

// Search without filter
auto results = index.search(query_vector, k);

// Search with radius filter (50km from NYC)
RadiusFilter filter(40.7128f, -74.0060f, 50.0f);
auto filtered_results = index.search_filtered(query_vector, k, filter);
```

### Filter Examples

#### Radius Filter (Spatial)
```cpp
// Find vectors within 100km of San Francisco
RadiusFilter sf_filter(37.7749f, -122.4194f, 100.0f);
auto results = index.search_filtered(query, k, sf_filter);
```

#### Range Filter (Multi-dimensional)
```cpp
// Find vectors in Europe during 2021
RangeFilter europe_filter(
    35.0f, 71.0f,      // Latitude range
    -10.0f, 40.0f,     // Longitude range
    1609459200000LL,   // Start timestamp
    1640995200000LL    // End timestamp
);
auto results = index.search_filtered(query, k, europe_filter);
```

#### Timestamp Filter
```cpp
// Find vectors from Q1 2021
TimestampFilter q1_filter(1609459200000LL, 1617235200000LL);
auto results = index.search_filtered(query, k, q1_filter);
```

#### Combined Filters (AND)
```cpp
// Find vectors near NYC during Q1 2021
auto radius = std::make_shared<RadiusFilter>(40.7128f, -74.0060f, 100.0f);
auto timestamp = std::make_shared<TimestampFilter>(1609459200000LL, 1617235200000LL);
AndFilter combined({radius, timestamp});
auto results = index.search_filtered(query, k, combined);
```

#### Custom Filter
```cpp
// Custom filter using lambda
CustomFilter custom([](const VectorMetadata& meta) {
    return meta.latitude > 0 && meta.timestamp > 1620000000000LL;
});
auto results = index.search_filtered(query, k, custom);
```

## Performance

### SIMD Acceleration

The library uses AVX2 SIMD instructions for distance calculations:
- **L2 Distance**: ~4x faster than scalar implementation
- **Inner Product**: ~4x faster than scalar implementation
- Automatic fallback to scalar code on non-AVX2 CPUs

### Benchmark Results (Example)

On a typical modern CPU (Intel Core i7, 3.5GHz):
- **Indexing**: ~50,000 vectors/sec (128-dim)
- **Unfiltered Search**: ~10,000 QPS
- **Filtered Search (high selectivity)**: ~5,000 QPS
- **Filtered Search (low selectivity)**: ~8,000 QPS

## API Reference

### FilteredHNSWIndex

```cpp
template<typename DistanceFunc = L2DistanceSIMD>
class FilteredHNSWIndex {
public:
    // Constructor
    FilteredHNSWIndex(size_t dim, size_t max_elements, size_t M = 16, size_t ef_construction = 200);

    // Add single vector
    void add_vector(const float* vector, const VectorMetadata& metadata);

    // Batch add vectors
    void add_vectors(const std::vector<std::vector<float>>& vectors,
                    const std::vector<VectorMetadata>& metadata_list);

    // Search with filter
    std::vector<std::pair<hnswlib::labeltype, float>>
    search_filtered(const float* query, size_t k, const Filter& filter, size_t ef = 50);

    // Search without filter
    std::vector<std::pair<hnswlib::labeltype, float>>
    search(const float* query, size_t k, size_t ef = 50);

    // Get metadata
    const VectorMetadata& get_metadata(hnswlib::labeltype label) const;

    // Save/load index
    void save_index(const std::string& filename);
    void load_index(const std::string& filename, size_t max_elements);
};
```

## Project Structure

```
CubeGraph/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── metadata.h              # Spatial/temporal metadata
│   ├── simd_distance.h         # SIMD distance functions
│   ├── filter_predicates.h     # Filter implementations
│   └── filtered_hnsw_index.h   # Main index class
├── src/
│   ├── example.cpp             # Usage examples
│   └── benchmark.cpp           # Performance benchmarks
└── third_party/
    └── hnswlib/                # HNSWlib (cloned)
```

## License

This project uses HNSWlib, which is licensed under Apache License 2.0.

## References

- [HNSWlib](https://github.com/nmslib/hnswlib) - Fast approximate nearest neighbor search
- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
