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

// N-dimensional sphere (for radius filters)
struct Sphere {
    std::vector<float> center;
    float radius;

    Sphere() : radius(0.0f) {}
    Sphere(size_t dim) : center(dim), radius(0.0f) {}
    Sphere(const std::vector<float>& c, float r) : center(c), radius(r) {}
};

struct RadiusFilterParams {
    Sphere sphere;
    size_t attr_dim;
    RadiusFilterParams() : attr_dim(0) {}
    RadiusFilterParams(const std::vector<float>& center, float radius, size_t attr_dim)
        : sphere(center, radius), attr_dim(attr_dim) {}
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
    bool verbose = false;
    float *data_;

    // Compute squared distance from a point to the closest point in a cube's bounding box
    float dist2_point_to_cube_bbox(const std::vector<float>& point, const BoundingBox& cube_bbox) const {
        float d2 = 0.0f;
        for (size_t d = 0; d < attr_dim_; d++) {
            float coord = point[d];
            float closest = coord;
            if (coord < cube_bbox.min_bounds[d]) {
                closest = cube_bbox.min_bounds[d];
            } else if (coord > cube_bbox.max_bounds[d]) {
                closest = cube_bbox.max_bounds[d];
            }
            float diff = coord - closest;
            d2 += diff * diff;
        }
        return d2;
    }

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
                    std::cerr << "Processing Cross - " << check_tag << " / " << num_vectors_ << std::endl;
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


    // Filter functor: admits only points whose metadata lies within filter_bbox
    class BBoxFilter : public hnswlib::BaseFilterFunctor {
    public:
        const BoundingBox &bbox_;
        size_t attr_dim_;

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

    // Filter functor: admits only points whose metadata lies within radius of sphere center
    class RadiusFilter : public hnswlib::BaseFilterFunctor {
    public:
        const Sphere& sphere_;
        size_t attr_dim_;

        RadiusFilter(const Sphere& sphere, size_t attr_dim)
            : sphere_(sphere), attr_dim_(attr_dim) {}

        bool operator()(hnswlib::metatype *meta) override {
            float d2 = 0.0f;
            for (size_t d = 0; d < attr_dim_; d++) {
                float diff = meta[d] - sphere_.center[d];
                d2 += diff * diff;
            }
            return d2 <= sphere_.radius * sphere_.radius;
        }
    };

    size_t select_layer_with_BBox(const BoundingBox *filter_bbox = nullptr) const {
        if(filter_bbox== nullptr) return 0;
        size_t target_layer = 0;
        while (target_layer < layers_.size()) {
            for (int i = 0; i < attr_dim_; i++) {
                auto edge_length = filter_bbox->max_bounds[i]-filter_bbox->min_bounds[i];
                if(edge_length > layers_[target_layer].cube_width[i]){
                    return target_layer;
                }
            }
            target_layer++;
        }
        target_layer--;
        return target_layer;
    }

    int find_cube_with_BBox(size_t layer_id, const BoundingBox *filter_bbox = nullptr) const {
        if (filter_bbox == nullptr) return 0;
        const auto &layer = layers_[layer_id];

        // Compute center of filter_bbox
        std::vector<float> center(attr_dim_);
        for (size_t d = 0; d < attr_dim_; d++) {
            center[d] = (filter_bbox->min_bounds[d] + filter_bbox->max_bounds[d]) / 2.0f;
        }
        // Compute cube ID for the center point
        return static_cast<int>(compute_cube_id(center, layer));
    }


    std::vector<int> find_cube_list_with_BBox(size_t layer_id, const BoundingBox *filter_bbox = nullptr) const {
        if (filter_bbox == nullptr) return {};

        const auto &layer = layers_[layer_id];
        std::vector<int> result;
        std::vector<char> visited(layer.total_cubes, 0);

        // Start from central cube
        int start_cube = find_cube_with_BBox(layer_id, filter_bbox);
        std::queue<int> bfs_queue;
        bfs_queue.push(start_cube);
        visited[start_cube] = 1;

        while (!bfs_queue.empty()) {
            int current_cube = bfs_queue.front();
            bfs_queue.pop();

            // Get cube bounding box
            BoundingBox cube_bbox(attr_dim_);
            size_t coords[MAX_DIM];
            decode_cube_id(current_cube, layer, coords);
            for (size_t d = 0; d < attr_dim_; d++) {
                cube_bbox.min_bounds[d] = layer.global_bbox.min_bounds[d] + coords[d] * layer.cube_width[d];
                cube_bbox.max_bounds[d] = cube_bbox.min_bounds[d] + layer.cube_width[d];
            }

            // Check overlap with filter_bbox
            if (cube_bbox.overlaps(*filter_bbox)) {
                result.push_back(current_cube);

                // Explore adjacent cubes
                for (int neighbor : layer.adjacent_cubes[current_cube]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = 1;
                        bfs_queue.push(neighbor);
                    }
                }
            }
        }

        return result;
    }

    // Select layer based on sphere radius vs cube diagonal
    size_t select_layer_with_Radius(const RadiusFilterParams* filter_radius = nullptr) const {
        if (filter_radius == nullptr) return 0;
        size_t target_layer = 0;
        float radius = filter_radius->sphere.radius;
        while (target_layer < layers_.size()) {
            // Compute diagonal of cube at this layer
            float diagonal2 = 0.0f;
            for (size_t d = 0; d < attr_dim_; d++) {
                diagonal2 += layers_[target_layer].cube_width[d] * layers_[target_layer].cube_width[d];
            }
            float diagonal = std::sqrt(diagonal2);
            // If sphere fits entirely within cube, we might need finer layer
            // If sphere is larger than cube, we need coarser layer
            if (radius > diagonal) {
                return target_layer;
            }
            target_layer++;
        }
        target_layer--;
        return target_layer;
    }

    int find_cube_with_Radius(size_t layer_id, const RadiusFilterParams* filter_radius = nullptr) const {
        if (filter_radius == nullptr) return 0;
        const auto& layer = layers_[layer_id];
        // Compute cube ID for the sphere center
        return static_cast<int>(compute_cube_id(filter_radius->sphere.center, layer));
    }

    std::vector<int> find_cube_list_with_Radius(size_t layer_id, const RadiusFilterParams* filter_radius = nullptr) const {
        if (filter_radius == nullptr) return {};

        const auto& layer = layers_[layer_id];
        const Sphere& sphere = filter_radius->sphere;
        float radius2 = sphere.radius * sphere.radius;

        std::vector<int> result;
        std::vector<char> visited(layer.total_cubes, 0);

        // Start from cube containing sphere center
        int start_cube = find_cube_with_Radius(layer_id, filter_radius);
        std::queue<int> bfs_queue;
        bfs_queue.push(start_cube);
        visited[start_cube] = 1;

        while (!bfs_queue.empty()) {
            int current_cube = bfs_queue.front();
            bfs_queue.pop();

            // Get cube bounding box
            BoundingBox cube_bbox(attr_dim_);
            size_t coords[MAX_DIM];
            decode_cube_id(current_cube, layer, coords);
            for (size_t d = 0; d < attr_dim_; d++) {
                cube_bbox.min_bounds[d] = layer.global_bbox.min_bounds[d] + coords[d] * layer.cube_width[d];
                cube_bbox.max_bounds[d] = cube_bbox.min_bounds[d] + layer.cube_width[d];
            }

            // Check if sphere intersects cube (distance from sphere center to cube <= radius)
            float d2 = dist2_point_to_cube_bbox(sphere.center, cube_bbox);
            if (d2 <= radius2) {
                result.push_back(current_cube);

                // Explore adjacent cubes
                for (int neighbor : layer.adjacent_cubes[current_cube]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = 1;
                        bfs_queue.push(neighbor);
                    }
                }
            }
        }

        return result;
    }


    // Main search method
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    fly_search(const float *query_vector, size_t k,
           const BoundingBox *filter_bbox = nullptr) {
        auto layer_id = select_layer_with_BBox(filter_bbox);
        const auto &layer = layers_[layer_id];
        BBoxFilter bbox_filter(*filter_bbox, attr_dim_);
        auto cube_id = find_cube_with_BBox(layer_id, filter_bbox);
        return layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &bbox_filter);
    }

    // Radius-filtered search method
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    fly_search(const float *query_vector, size_t k,
           const RadiusFilterParams* filter_radius) {
        auto layer_id = select_layer_with_Radius(filter_radius);
        const auto &layer = layers_[layer_id];
        RadiusFilter radius_filter(filter_radius->sphere, filter_radius->attr_dim);
        auto cube_id = find_cube_with_Radius(layer_id, filter_radius);
        return layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &radius_filter);
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

