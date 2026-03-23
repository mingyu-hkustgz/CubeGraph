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

    bool contains(const std::vector<float> &point) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (point[i] < min_bounds[i] || point[i] > max_bounds[i])
                return false;
        }
        return true;
    }

    bool overlaps(const BoundingBox &other) const {
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
        hnswlib::HierarchicalNSWCube<float> *hnsw_index;
        std::vector<std::vector<hnswlib::labeltype>> adjacent_cubes;
        BoundingBox global_bbox;
    };
    hnswlib::SpaceInterface<float> *space;
    std::vector<LayerConfig> layers_;
    std::vector<std::vector<float>> metadata_;
    BoundingBox global_bbox_;
    size_t attr_dim_, vec_dim_, num_vectors_, num_layers_;
    size_t M_, ef_construction_, cross_edge_count_;
    bool verbose = true;
    float *data_;
public:
    // Constructor
    IndexCube(size_t num_layers = 3,
              size_t M = 16,
              size_t ef_construction = 200,
              size_t cross_edge_count = 2) :
            num_layers_(num_layers),
            M_(M),
            ef_construction_(ef_construction),
            cross_edge_count_(cross_edge_count),
            num_vectors_(0) {};

    // Destructor
    ~IndexCube() {
        for (auto &layer: layers_) {
            if (layer.hnsw_index != nullptr) {
                delete layer.hnsw_index;
            }
        }
    }

    void set_global_ef(size_t ef) {
        for (auto &u: layers_) u.hnsw_index->setEf(ef);
    }


    // Load metadata from binary file
    void load_metadata(char *metadata_file) {
        std::ifstream in(metadata_file, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open metadata file: ");
        }

        size_t n, d;
        in.read(reinterpret_cast<char *>(&n), sizeof(size_t));
        in.read(reinterpret_cast<char *>(&d), sizeof(size_t));

        attr_dim_ = d;

        num_vectors_ = n;
        metadata_.resize(n, std::vector<float>(d));

        for (size_t i = 0; i < n; i++) {
            in.read(reinterpret_cast<char *>(metadata_[i].data()), d * sizeof(float));
        }

        in.close();
    }

    // Compute global bounding box from metadata
    void compute_global_bbox() {
        if (metadata_.empty()) {
            throw std::runtime_error("Metadata not loaded");
        }
        global_bbox_ = BoundingBox(attr_dim_);

        for (size_t d = 0; d < attr_dim_; d++) {
            global_bbox_.min_bounds[d] = std::numeric_limits<float>::max();
            global_bbox_.max_bounds[d] = std::numeric_limits<float>::lowest();
        }

        for (const auto &point: metadata_) {
            for (size_t d = 0; d < attr_dim_; d++) {
                global_bbox_.min_bounds[d] = std::min(global_bbox_.min_bounds[d], point[d]);
                global_bbox_.max_bounds[d] = std::max(global_bbox_.max_bounds[d], point[d]);
            }
        }
        for (size_t d = 0; d < attr_dim_; d++) {
            float range = global_bbox_.max_bounds[d] - global_bbox_.min_bounds[d];
            float epsilon = range * 1e-6f;
            global_bbox_.min_bounds[d] -= epsilon;
            global_bbox_.max_bounds[d] += epsilon;
        }
    }


    // Compute cube ID from metadata coordinates for a specific layer
    size_t compute_cube_id(const std::vector<float> &metadata, const LayerConfig &layer) const {
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
    void decode_cube_id(size_t cube_id, const LayerConfig &layer, size_t *coords) const {
        for (size_t d = 0; d < attr_dim_; d++) {
            coords[d] = cube_id % layer.cubes_per_dim;
            cube_id /= layer.cubes_per_dim;
        }
    }

    // Encode multi-dimensional coordinates to cube ID
    size_t encode_cube_id(const size_t *coords, const LayerConfig &layer) const {
        size_t cube_id = 0;
        size_t multiplier = 1;

        for (size_t d = 0; d < attr_dim_; d++) {
            cube_id += coords[d] * multiplier;
            multiplier *= layer.cubes_per_dim;
        }

        return cube_id;
    }

    // Build adjacency list for a layer (2*d face-adjacent neighbors)
    void build_adjacency_list(LayerConfig &layer) {
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
        if (verbose) {
            for (int i = 0; i < layer.total_cubes; i++) {
                std::cout << "CUBE id:: " << i << " ---- " << layer.cube_width[0] << " " << layer.cube_width[1]
                          << std::endl;
                for (auto u: layer.adjacent_cubes[i]) {
                    std::cout << "ADJ CUBE id:: " << u << " " << std::endl;
                }
            }
        }

    }

    // Build a single layer
    void build_layer(size_t layer_id) {
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
        layer.hnsw_index = new hnswlib::HierarchicalNSWCube<float>(space, num_vectors_, attr_dim_, layer.total_cubes,
                                                                   cross_edge_count_, M_, ef_construction_);
        int check_tag = 0, report = 200000;
#pragma omp parallel for schedule(dynamic, 144)
        for (size_t i = 0; i < num_vectors_; i++) {
            size_t cube_id = compute_cube_id(metadata_[i], layer);
            layer.hnsw_index->addCubePoint(data_ + i * vec_dim_, i, cube_id, metadata_[i].data());
#pragma omp critical
            {
                check_tag++;
                if (check_tag % report == 0) {
                    std::cerr << "Processing Index - " << check_tag << " / " << num_vectors_ << std::endl;
                }
            }
        }

        // Build adjacency list
        build_adjacency_list(layer);

        // Set adjacency in HNSW index
        layer.hnsw_index->setAdjacentCubeIds(layer.adjacent_cubes);

        check_tag = 0;
#pragma omp parallel for schedule(dynamic, 144)
        for (size_t i = 0; i < num_vectors_; i++) {
            layer.hnsw_index->addCrossCubelinks(i);
#pragma omp critical
            {
                check_tag++;
                if (check_tag % report == 0) {
                    std::cerr << "Processing Index - " << check_tag << " / " << num_vectors_ << std::endl;
                }
            }
        }
        layers_.push_back(layer);
    }

    // Build all layers and cross-cube edges
    void build_index(char *data_file, char *metadata_file) {
        // Load metadata
        load_metadata(metadata_file);
        compute_global_bbox();

        auto *X = new Matrix<float>(data_file);
        hnswlib::HierarchicalNSWCube<float>::static_base_data_ = (char *) X->data;
        data_ = X->data;
        num_vectors_ = X->n;
        vec_dim_ = X->d;
        space = new hnswlib::L2Space(X->d);
        std::cout << "Building " << num_layers_ << " layers..." << std::endl;
        for (size_t layer_id = 0; layer_id < num_layers_; layer_id++) {
            std::cout << "Building layer " << layer_id << "..." << std::endl;
            build_layer(layer_id);
            std::cout << "Layer " << layer_id << " built: "
                      << layers_[layer_id].cubes_per_dim << "^" << attr_dim_
                      << " = " << layers_[layer_id].total_cubes << " cubes" << std::endl;

        }

        std::cout << "Index built successfully!" << std::endl;
    }


    // Layer selection: explicit layer ID
    size_t select_layer_explicit(size_t layer_id) const {
        if (layer_id >= layers_.size()) {
            throw std::runtime_error("Invalid layer_id");
        }
        return layer_id;
    }


    // Filter functor: admits only points whose metadata lies within filter_bbox
    class BBoxFilter : public hnswlib::BaseFilterFunctor {
        const BoundingBox &bbox_;
        size_t attr_dim_;
    public:
        BBoxFilter(const BoundingBox &bbox, size_t attr_dim)
                : bbox_(bbox), attr_dim_(attr_dim) {}

        // Called by searchCubeKnn with a pointer to the point's raw metadata float array
        bool operator()(hnswlib::metatype *meta) override {
            for (size_t d = 0; d < attr_dim_; d++) {
                if (meta[d] < bbox_.min_bounds[d] || meta[d] > bbox_.max_bounds[d])
                    return false;
            }
            return true;
        }
    };

    // Main search method
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search(const float *query_vector, size_t k, size_t explicit_layer_id = 0, size_t ef = 100,
           const BoundingBox *filter_bbox = nullptr) {
        auto layer_id = select_layer_explicit(explicit_layer_id);
        const auto &layer = layers_[layer_id];
        // Set ef parameter
//        layer.hnsw_index->setEf(ef);
        std::vector<hnswlib::tableint> cubelist;
        for (int i = 0; i < layer.total_cubes; i++) cubelist.push_back(i);
        // Search with in-traversal filter — only admits points inside filter_bbox
//        BBoxFilter bbox_filter(filter_bbox, attr_dim_);
        return layer.hnsw_index->searchCubeKnn(query_vector, k, cubelist);
    }

    // Get global bounding box
    const BoundingBox &get_global_bbox() const {
        return global_bbox_;
    }

    // Get global bounding box
    const size_t &get_meta_dim() const {
        return attr_dim_;
    }

    // Get metadata vectors
    const std::vector<std::vector<float>> &get_metadata() const { return metadata_; }
};

