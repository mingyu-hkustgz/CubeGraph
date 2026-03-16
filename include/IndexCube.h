#pragma once

#include "hnswlib/hnswlib.h"
#include "hnsw-cube.h"
#include "matrix.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <queue>
#include <omp.h>

#define MAX_DIM 16

// Multi-dimensional bounding box
struct BoundingBox {
    std::vector<float> min_bounds;
    std::vector<float> max_bounds;

    BoundingBox() {}

    BoundingBox(size_t dim) : min_bounds(dim), max_bounds(dim) {}

    bool contains(const std::vector<float>& point) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (point[i] < min_bounds[i] || point[i] > max_bounds[i])
                return false;
        }
        return true;
    }

    bool overlaps(const BoundingBox& other) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (max_bounds[i] < other.min_bounds[i] || min_bounds[i] > other.max_bounds[i])
                return false;
        }
        return true;
    }

    float volume() const {
        float vol = 1.0f;
        for (size_t i = 0; i < min_bounds.size(); i++) {
            vol *= (max_bounds[i] - min_bounds[i]);
        }
        return vol;
    }
};

// Layer selection strategy
enum class LayerSelectionStrategy {
    RANGE_SIZE,      // Match filter range size to cube size
    EXPLICIT,        // User specifies layer_id directly
    SELECTIVITY      // Based on filter selectivity (volume ratio)
};

class IndexCube {
private:
    // Configuration for a single layer
    struct LayerConfig {
        size_t layer_id;                    // 0, 1, 2, ...
        size_t cubes_per_dim;               // 2^(layer_id+1)
        size_t total_cubes;                 // (cubes_per_dim)^attr_dim
        float cube_width[MAX_DIM];          // Cube size per dimension
        hnswlib::HierarchicalNSWCube<float>* hnsw_index;
        std::vector<std::vector<hnswlib::labeltype>> adjacent_cubes;
        BoundingBox global_bbox;
    };

    std::vector<LayerConfig> layers_;
    std::vector<std::vector<float>> metadata_;
    BoundingBox global_bbox_;
    size_t attr_dim_, vec_dim_, num_vectors_, num_layers_;
    size_t M_, ef_construction_, cross_edge_count_;
    hnswlib::SpaceInterface<float>* space_;
    static char* static_base_data_;
    Matrix<float>* base_vectors_;  // Keep base vectors alive

public:
    // Constructor
    IndexCube(hnswlib::SpaceInterface<float>* space,
              size_t vec_dim,
              size_t attr_dim,
              size_t num_layers = 3,
              size_t M = 16,
              size_t ef_construction = 200,
              size_t cross_edge_count = 2)
        : space_(space),
          vec_dim_(vec_dim),
          attr_dim_(attr_dim),
          num_layers_(num_layers),
          M_(M),
          ef_construction_(ef_construction),
          cross_edge_count_(cross_edge_count),
          num_vectors_(0),
          base_vectors_(nullptr) {
        if (attr_dim > MAX_DIM) {
            throw std::runtime_error("attr_dim exceeds MAX_DIM");
        }
        global_bbox_ = BoundingBox(attr_dim);
    }

    // Destructor
    ~IndexCube() {
        for (auto& layer : layers_) {
            if (layer.hnsw_index != nullptr) {
                delete layer.hnsw_index;
            }
        }
        if (base_vectors_ != nullptr) {
            delete base_vectors_;
        }
    }

    // Load metadata from binary file
    void load_metadata(const std::string& metadata_file) {
        std::ifstream in(metadata_file, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open metadata file: " + metadata_file);
        }

        size_t n, d;
        in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&d), sizeof(size_t));

        if (d != attr_dim_) {
            throw std::runtime_error("Metadata dimension mismatch");
        }

        num_vectors_ = n;
        metadata_.resize(n, std::vector<float>(d));

        for (size_t i = 0; i < n; i++) {
            in.read(reinterpret_cast<char*>(metadata_[i].data()), d * sizeof(float));
        }

        in.close();
    }

    // Compute global bounding box from metadata
    void compute_global_bbox() {
        if (metadata_.empty()) {
            throw std::runtime_error("Metadata not loaded");
        }

        for (size_t d = 0; d < attr_dim_; d++) {
            global_bbox_.min_bounds[d] = std::numeric_limits<float>::max();
            global_bbox_.max_bounds[d] = std::numeric_limits<float>::lowest();
        }

        for (const auto& point : metadata_) {
            for (size_t d = 0; d < attr_dim_; d++) {
                global_bbox_.min_bounds[d] = std::min(global_bbox_.min_bounds[d], point[d]);
                global_bbox_.max_bounds[d] = std::max(global_bbox_.max_bounds[d], point[d]);
            }
        }

        // Add small epsilon to avoid boundary issues
        for (size_t d = 0; d < attr_dim_; d++) {
            float range = global_bbox_.max_bounds[d] - global_bbox_.min_bounds[d];
            float epsilon = range * 1e-6f;
            global_bbox_.min_bounds[d] -= epsilon;
            global_bbox_.max_bounds[d] += epsilon;
        }
    }

    // Compute cube ID from metadata coordinates for a specific layer
    size_t compute_cube_id(const std::vector<float>& metadata, const LayerConfig& layer) const {
        size_t cube_id = 0;
        size_t multiplier = 1;

        for (size_t d = 0; d < attr_dim_; d++) {
            float normalized = (metadata[d] - layer.global_bbox.min_bounds[d]) / layer.cube_width[d];
            size_t cube_idx = static_cast<size_t>(normalized);
            // Clamp to valid range
            cube_idx = std::min(cube_idx, layer.cubes_per_dim - 1);
            cube_id += cube_idx * multiplier;
            multiplier *= layer.cubes_per_dim;
        }

        return cube_id;
    }

    // Decode cube ID to multi-dimensional coordinates
    void decode_cube_id(size_t cube_id, const LayerConfig& layer, size_t* coords) const {
        for (size_t d = 0; d < attr_dim_; d++) {
            coords[d] = cube_id % layer.cubes_per_dim;
            cube_id /= layer.cubes_per_dim;
        }
    }

    // Encode multi-dimensional coordinates to cube ID
    size_t encode_cube_id(const size_t* coords, const LayerConfig& layer) const {
        size_t cube_id = 0;
        size_t multiplier = 1;

        for (size_t d = 0; d < attr_dim_; d++) {
            cube_id += coords[d] * multiplier;
            multiplier *= layer.cubes_per_dim;
        }

        return cube_id;
    }

    // Build adjacency list for a layer (2*d face-adjacent neighbors)
    void build_adjacency_list(LayerConfig& layer) {
        layer.adjacent_cubes.resize(layer.total_cubes);

        for (size_t cube_id = 0; cube_id < layer.total_cubes; cube_id++) {
            size_t coords[MAX_DIM];
            decode_cube_id(cube_id, layer, coords);

            // For each dimension, add neighbors at coords[d]±1
            for (size_t d = 0; d < attr_dim_; d++) {
                // Neighbor at coords[d]-1
                if (coords[d] > 0) {
                    size_t neighbor_coords[MAX_DIM];
                    std::copy(coords, coords + attr_dim_, neighbor_coords);
                    neighbor_coords[d]--;
                    size_t neighbor_id = encode_cube_id(neighbor_coords, layer);
                    layer.adjacent_cubes[cube_id].push_back(neighbor_id);
                }

                // Neighbor at coords[d]+1
                if (coords[d] + 1 < layer.cubes_per_dim) {
                    size_t neighbor_coords[MAX_DIM];
                    std::copy(coords, coords + attr_dim_, neighbor_coords);
                    neighbor_coords[d]++;
                    size_t neighbor_id = encode_cube_id(neighbor_coords, layer);
                    layer.adjacent_cubes[cube_id].push_back(neighbor_id);
                }
            }
        }
    }

    // Build a single layer
    void build_layer(size_t layer_id, char* base_data) {
        LayerConfig layer;
        layer.layer_id = layer_id;
        layer.cubes_per_dim = 1 << (layer_id + 1);  // 2^(layer_id+1)
        layer.total_cubes = 1;
        for (size_t d = 0; d < attr_dim_; d++) {
            layer.total_cubes *= layer.cubes_per_dim;
        }

        // Compute cube widths
        layer.global_bbox = global_bbox_;
        for (size_t d = 0; d < attr_dim_; d++) {
            float range = global_bbox_.max_bounds[d] - global_bbox_.min_bounds[d];
            layer.cube_width[d] = range / layer.cubes_per_dim;
        }

        // Create HNSW-CUBE index
        layer.hnsw_index = new hnswlib::HierarchicalNSWCube<float>(
            space_,
            num_vectors_,
            attr_dim_,
            layer.total_cubes,
            cross_edge_count_,
            M_,
            ef_construction_
        );

        // Assign points to cubes
        std::vector<std::vector<size_t>> cube_points(layer.total_cubes);
        for (size_t i = 0; i < num_vectors_; i++) {
            size_t cube_id = compute_cube_id(metadata_[i], layer);
            cube_points[cube_id].push_back(i);
        }

        // Add first point per cube (sequential)
        for (size_t cube_id = 0; cube_id < layer.total_cubes; cube_id++) {
            if (!cube_points[cube_id].empty()) {
                size_t point_id = cube_points[cube_id][0];
                layer.hnsw_index->addCubePoint(
                    base_data + point_id * vec_dim_ * sizeof(float),
                    point_id,
                    cube_id,
                    metadata_[point_id].data()
                );
            }
        }

        // Add remaining points (parallel)
        #pragma omp parallel for schedule(dynamic, 144)
        for (size_t cube_id = 0; cube_id < layer.total_cubes; cube_id++) {
            for (size_t j = 1; j < cube_points[cube_id].size(); j++) {
                size_t point_id = cube_points[cube_id][j];
                layer.hnsw_index->addCubePoint(
                    base_data + point_id * vec_dim_ * sizeof(float),
                    point_id,
                    cube_id,
                    metadata_[point_id].data()
                );
            }
        }

        // Build adjacency list
        build_adjacency_list(layer);

        // Set adjacency in HNSW index
        layer.hnsw_index->setAdjacentCubeIds(layer.adjacent_cubes);

        layers_.push_back(layer);
    }

    // Build all layers and cross-cube edges
    void build_index(const std::string& data_file, const std::string& metadata_file) {
        // Load metadata
        load_metadata(metadata_file);
        compute_global_bbox();

        // Load vector data and keep it alive
        char* data_file_cstr = const_cast<char*>(data_file.c_str());
        base_vectors_ = new Matrix<float>(data_file_cstr);
        if (base_vectors_->n != num_vectors_) {
            throw std::runtime_error("Vector count mismatch");
        }
        if (base_vectors_->d != vec_dim_) {
            throw std::runtime_error("Vector dimension mismatch");
        }

        // Set static base data pointer
        static_base_data_ = reinterpret_cast<char*>(base_vectors_->data);
        hnswlib::HierarchicalNSWCube<float>::static_base_data_ = static_base_data_;

        // Build all layers (can be parallelized, but sequential for now)
        std::cout << "Building " << num_layers_ << " layers..." << std::endl;
        for (size_t layer_id = 0; layer_id < num_layers_; layer_id++) {
            std::cout << "Building layer " << layer_id << "..." << std::endl;
            build_layer(layer_id, static_base_data_);
            std::cout << "Layer " << layer_id << " built: "
                      << layers_[layer_id].cubes_per_dim << "^" << attr_dim_
                      << " = " << layers_[layer_id].total_cubes << " cubes" << std::endl;
        }

        // Build cross-cube edges for all layers (parallel)
        std::cout << "Building cross-cube edges..." << std::endl;
        for (size_t layer_id = 0; layer_id < num_layers_; layer_id++) {
            std::cout << "Building cross-cube edges for layer " << layer_id << "..." << std::endl;
            #pragma omp parallel for schedule(dynamic, 144)
            for (size_t i = 0; i < num_vectors_; i++) {
                layers_[layer_id].hnsw_index->addCrossCubelinks(i);
            }
        }

        std::cout << "Index built successfully!" << std::endl;
    }

    // Layer selection: match filter range size to cube size
    size_t select_layer_by_range_size(const BoundingBox& filter_bbox) const {
        if (layers_.empty()) {
            throw std::runtime_error("No layers built");
        }

        // Compute average span (number of cubes) for each layer
        float best_score = std::numeric_limits<float>::max();
        size_t best_layer = 0;

        for (size_t layer_id = 0; layer_id < layers_.size(); layer_id++) {
            const auto& layer = layers_[layer_id];
            float avg_span = 0.0f;

            for (size_t d = 0; d < attr_dim_; d++) {
                float filter_range = filter_bbox.max_bounds[d] - filter_bbox.min_bounds[d];
                float span = filter_range / layer.cube_width[d];
                avg_span += span;
            }
            avg_span /= attr_dim_;

            // Goal: filter should span 2-3 cubes for optimal performance
            float score = std::abs(avg_span - 2.5f);
            if (score < best_score) {
                best_score = score;
                best_layer = layer_id;
            }
        }

        return best_layer;
    }

    // Layer selection: explicit layer ID
    size_t select_layer_explicit(size_t layer_id) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }
        return layer_id;
    }

    // Layer selection: based on filter selectivity
    size_t select_layer_by_selectivity(const BoundingBox& filter_bbox) const {
        if (layers_.empty()) {
            throw std::runtime_error("No layers built");
        }

        // Compute filter selectivity (volume ratio)
        float filter_volume = filter_bbox.volume();
        float global_volume = global_bbox_.volume();
        float selectivity = filter_volume / global_volume;

        // Small selectivity → fine layer (high layer_id)
        // Large selectivity → coarse layer (low layer_id)
        // Heuristic: layer_id = max(0, log2(selectivity^(-1/d)) - 1)
        float inv_selectivity = 1.0f / selectivity;
        float power = std::pow(inv_selectivity, 1.0f / attr_dim_);
        int layer_id = static_cast<int>(std::log2(power)) - 1;
        layer_id = std::max(0, std::min(layer_id, static_cast<int>(layers_.size()) - 1));

        return static_cast<size_t>(layer_id);
    }

    // Generate list of cube IDs overlapping with filter bbox for a specific layer
    void generate_cube_list(const BoundingBox& filter_bbox, const LayerConfig& layer,
                           std::vector<hnswlib::tableint>& cube_list) const {
        // Compute range of cube indices overlapping with filter bbox
        size_t min_cube_idx[MAX_DIM], max_cube_idx[MAX_DIM];

        for (size_t d = 0; d < attr_dim_; d++) {
            float min_normalized = (filter_bbox.min_bounds[d] - layer.global_bbox.min_bounds[d]) / layer.cube_width[d];
            float max_normalized = (filter_bbox.max_bounds[d] - layer.global_bbox.min_bounds[d]) / layer.cube_width[d];

            // For min: use floor to get the cube containing min_bounds
            min_cube_idx[d] = static_cast<size_t>(std::floor(min_normalized));

            // If min_normalized is exactly an integer (on cube boundary),
            // we need to include the previous cube since overlaps() considers touching as overlapping
            if (min_normalized == std::floor(min_normalized) && min_cube_idx[d] > 0) {
                min_cube_idx[d]--;
            }

            // For max: use floor to get the cube containing max_bounds
            max_cube_idx[d] = static_cast<size_t>(std::floor(max_normalized));

            // If max_normalized is exactly an integer (on cube boundary),
            // the cube at that boundary should be included (since overlaps() considers touching as overlapping)
            // floor already gives us the right cube index, no need to adjust

            // Clamp to valid range
            min_cube_idx[d] = std::max(size_t(0), std::min(min_cube_idx[d], layer.cubes_per_dim - 1));
            max_cube_idx[d] = std::max(size_t(0), std::min(max_cube_idx[d], layer.cubes_per_dim - 1));
        }

        // Enumerate all cubes in the d-dimensional range
        std::vector<hnswlib::tableint> all_cubes;
        enumerate_cubes_recursive(min_cube_idx, max_cube_idx, layer, 0, 0, all_cubes);

        // Filter out empty cubes (cubes with no entry point)
        for (auto cube_id : all_cubes) {
            // Check if cube has a valid entry point (not -1)
            if (cube_id < layer.hnsw_index->cube_entry_points_.size()) {
                auto ep = layer.hnsw_index->cube_entry_points_[cube_id];
                if (static_cast<int>(ep) != -1) {
                    cube_list.push_back(cube_id);
                }
            }
        }
    }

    // Recursively enumerate cubes in d-dimensional range
    void enumerate_cubes_recursive(const size_t* min_idx, const size_t* max_idx,
                                   const LayerConfig& layer, size_t dim,
                                   size_t partial_cube_id, std::vector<hnswlib::tableint>& cube_list) const {
        if (dim == attr_dim_) {
            cube_list.push_back(partial_cube_id);
            return;
        }

        size_t multiplier = 1;
        for (size_t d = 0; d < dim; d++) {
            multiplier *= layer.cubes_per_dim;
        }

        for (size_t idx = min_idx[dim]; idx <= max_idx[dim]; idx++) {
            enumerate_cubes_recursive(min_idx, max_idx, layer, dim + 1,
                                     partial_cube_id + idx * multiplier, cube_list);
        }
    }

    // Filter functor: admits only points whose metadata lies within filter_bbox
    class BBoxFilter : public hnswlib::BaseFilterFunctor {
        const BoundingBox& bbox_;
        size_t attr_dim_;
    public:
        BBoxFilter(const BoundingBox& bbox, size_t attr_dim)
            : bbox_(bbox), attr_dim_(attr_dim) {}

        // Called by searchCubeKnn with a pointer to the point's raw metadata float array
        bool operator()(hnswlib::metatype* meta) override {
            for (size_t d = 0; d < attr_dim_; d++) {
                if (meta[d] < bbox_.min_bounds[d] || meta[d] > bbox_.max_bounds[d])
                    return false;
            }
            return true;
        }
    };

    // Main search method
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search(const float* query_vector, const BoundingBox& filter_bbox, size_t k,
           LayerSelectionStrategy strategy = LayerSelectionStrategy::RANGE_SIZE,
           size_t explicit_layer_id = 0, size_t ef = 100) {
        // Select layer
        size_t layer_id;
        switch (strategy) {
            case LayerSelectionStrategy::RANGE_SIZE:
                layer_id = select_layer_by_range_size(filter_bbox);
                break;
            case LayerSelectionStrategy::EXPLICIT:
                layer_id = select_layer_explicit(explicit_layer_id);
                break;
            case LayerSelectionStrategy::SELECTIVITY:
                layer_id = select_layer_by_selectivity(filter_bbox);
                break;
            default:
                throw std::runtime_error("Unknown layer selection strategy");
        }

        const auto& layer = layers_[layer_id];

        // Generate cube list
        std::vector<hnswlib::tableint> cube_list;
        generate_cube_list(filter_bbox, layer, cube_list);

        if (cube_list.empty()) {
            return std::priority_queue<std::pair<float, hnswlib::labeltype>>();
        }

        // Set ef parameter
        layer.hnsw_index->setEf(ef);

        // Search with in-traversal filter — only admits points inside filter_bbox
        BBoxFilter bbox_filter(filter_bbox, attr_dim_);
        return layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &bbox_filter);
    }

    // Convenience method for radius queries
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    radius_search(const float* query_vector, const std::vector<float>& center, float radius,
                  size_t k, LayerSelectionStrategy strategy = LayerSelectionStrategy::RANGE_SIZE,
                  size_t explicit_layer_id = 0, size_t ef = 100) {
        // Create bounding box from radius
        BoundingBox filter_bbox(attr_dim_);
        for (size_t d = 0; d < attr_dim_; d++) {
            filter_bbox.min_bounds[d] = center[d] - radius;
            filter_bbox.max_bounds[d] = center[d] + radius;
        }

        return search(query_vector, filter_bbox, k, strategy, explicit_layer_id, ef);
    }

    // Get number of layers
    size_t get_num_layers() const {
        return layers_.size();
    }

    // Get layer info
    void get_layer_info(size_t layer_id, size_t& cubes_per_dim, size_t& total_cubes) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }
        cubes_per_dim = layers_[layer_id].cubes_per_dim;
        total_cubes = layers_[layer_id].total_cubes;
    }

    // Get metadata for a point
    const std::vector<float>& get_metadata(size_t point_id) const {
        if (point_id >= metadata_.size()) {
            throw std::runtime_error("Invalid point_id");
        }
        return metadata_[point_id];
    }

    // Get global bounding box
    const BoundingBox& get_global_bbox() const {
        return global_bbox_;
    }

    // Public method to compute cube ID for a point at a specific layer
    size_t compute_cube_id_for_layer(const std::vector<float>& metadata, size_t layer_id) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }
        return compute_cube_id(metadata, layers_[layer_id]);
    }

    // Get cube bounding box for a specific cube and layer
    BoundingBox get_cube_bbox(size_t cube_id, size_t layer_id) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }

        const auto& layer = layers_[layer_id];
        if (cube_id >= layer.total_cubes) {
            throw std::runtime_error("Invalid cube_id");
        }

        // Decode cube_id to coordinates
        size_t coords[MAX_DIM];
        decode_cube_id(cube_id, layer, coords);

        // Compute bounding box
        BoundingBox bbox(attr_dim_);
        for (size_t d = 0; d < attr_dim_; d++) {
            bbox.min_bounds[d] = layer.global_bbox.min_bounds[d] + coords[d] * layer.cube_width[d];
            bbox.max_bounds[d] = bbox.min_bounds[d] + layer.cube_width[d];
        }

        return bbox;
    }

    // Public method to generate cube list for testing
    std::vector<hnswlib::tableint> generate_cube_list_for_layer(
        const BoundingBox& filter_bbox, size_t layer_id) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }

        std::vector<hnswlib::tableint> cube_list;
        generate_cube_list(filter_bbox, layers_[layer_id], cube_list);
        return cube_list;
    }

    // Get layer cube width
    std::vector<float> get_layer_cube_width(size_t layer_id) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }

        std::vector<float> widths(attr_dim_);
        for (size_t d = 0; d < attr_dim_; d++) {
            widths[d] = layers_[layer_id].cube_width[d];
        }
        return widths;
    }
};

// Static member definition
char* IndexCube::static_base_data_ = nullptr;
