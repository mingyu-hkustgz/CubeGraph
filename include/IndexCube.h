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

struct Sphere {
    std::vector<float> center;
    float radius;

    Sphere() : radius(0.0f) {}
    Sphere(size_t dim) : center(dim), radius(0.0f) {}
    Sphere(const std::vector<float>& c, float r) : center(c), radius(r) {}

    // Check if this sphere overlaps with a given BoundingBox
    bool overlaps(const BoundingBox& box) const {
        float dist_squared = 0.0f;
        float radius_squared = radius * radius;

        for (size_t i = 0; i < center.size(); ++i) {
            // Find the closest point on the bounding box to the sphere center along dimension i
            float closest_p = std::max(box.min_bounds[i], std::min(center[i], box.max_bounds[i]));

            // Calculate the 1D squared distance and accumulate
            float diff = closest_p - center[i];
            dist_squared += diff * diff;

            // Early exit: if the accumulated squared distance already exceeds the squared radius,
            // the sphere and the box definitely do not overlap.
            if (dist_squared > radius_squared) {
                return false;
            }
        }

        // If we finish the loop and the distance is within the radius, they overlap
        return true;
    }
};

struct RadiusFilterParams {
    Sphere sphere;
    size_t attr_dim;
    RadiusFilterParams() : attr_dim(0) {}
    RadiusFilterParams(const std::vector<float>& center, float radius, size_t attr_dim)
        : sphere(center, radius), attr_dim(attr_dim) {}
};

struct PolygonFilterParams {
    std::vector<std::vector<float>> vertices;  // polygon vertices in 2D
    size_t num_vertices;                        // 3=triangle, 4=quad, 5=pentagon
    size_t attr_dim;                            // should be 2 for polygon

    PolygonFilterParams() : num_vertices(0), attr_dim(0) {}
    PolygonFilterParams(const std::vector<std::vector<float>>& verts, size_t attr_dim)
        : vertices(verts), num_vertices(verts.size()), attr_dim(attr_dim) {}

    // Compute bounding box of the polygon for cube selection
    BoundingBox get_bbox() const {
        BoundingBox bbox(2);
        if (vertices.empty()) return bbox;
        bbox.min_bounds[0] = bbox.max_bounds[0] = vertices[0][0];
        bbox.min_bounds[1] = bbox.max_bounds[1] = vertices[0][1];
        for (const auto& v : vertices) {
            bbox.min_bounds[0] = std::min(bbox.min_bounds[0], v[0]);
            bbox.max_bounds[0] = std::max(bbox.max_bounds[0], v[0]);
            bbox.min_bounds[1] = std::min(bbox.min_bounds[1], v[1]);
            bbox.max_bounds[1] = std::max(bbox.max_bounds[1], v[1]);
        }
        return bbox;
    }

    // Compute approximate area of polygon using shoelace formula
    float area() const {
        if (num_vertices < 3) return 0.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < num_vertices; i++) {
            const auto& v1 = vertices[i];
            const auto& v2 = vertices[(i + 1) % num_vertices];
            sum += v1[0] * v2[1] - v2[0] * v1[1];
        }
        return std::abs(sum) * 0.5f;
    }
};

// ============================================================================
// Composite Filter Framework: Combine multiple filters with AND/OR/NOT
// ============================================================================

// AND composition: accepts if BOTH filters accept
class AndFilter : public hnswlib::BaseFilterFunctor {
private:
    hnswlib::BaseFilterFunctor* a_;
    hnswlib::BaseFilterFunctor* b_;
public:
    AndFilter(hnswlib::BaseFilterFunctor* a, hnswlib::BaseFilterFunctor* b) : a_(a), b_(b) {}
    bool operator()(hnswlib::metatype *meta) override {
        return (*a_)(meta) && (*b_)(meta);
    }
};

// OR composition: accepts if EITHER filter accepts
class OrFilter : public hnswlib::BaseFilterFunctor {
private:
    hnswlib::BaseFilterFunctor* a_;
    hnswlib::BaseFilterFunctor* b_;
public:
    OrFilter(hnswlib::BaseFilterFunctor* a, hnswlib::BaseFilterFunctor* b) : a_(a), b_(b) {}
    bool operator()(hnswlib::metatype *meta) override {
        return (*a_)(meta) || (*b_)(meta);
    }
};

// NOT composition: accepts if filter REJECTS
class NotFilter : public hnswlib::BaseFilterFunctor {
private:
    hnswlib::BaseFilterFunctor* f_;
public:
    NotFilter(hnswlib::BaseFilterFunctor* f) : f_(f) {}
    bool operator()(hnswlib::metatype *meta) override {
        return !(*f_)(meta);
    }
};

// Composite filter specification for complex filter compositions
enum class CompositeFilterType { BBOX, RADIUS, POLYGON };

struct CompositeFilterSpec {
    bool use_bbox = false;          // use BBox as part of filter
    bool use_radius = false;         // use Radius as part of filter
    bool use_polygon = false;        // use Polygon as part of filter
    bool bbox_negate = false;        // apply NOT to bbox
    bool radius_negate = false;      // apply NOT to radius
    bool polygon_negate = false;     // apply NOT to polygon
    CompositeFilterType primary_op = CompositeFilterType::BBOX;  // primary filter type for cube selection
};

struct CompositeFilterParams {
    BoundingBox bbox;
    RadiusFilterParams radius;
    PolygonFilterParams polygon;
    CompositeFilterSpec spec;
    size_t attr_dim;
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
    size_t min_points_per_cube_;
    bool verbose = false;
    float *data_;

#ifdef COLLECT_LOG
    hnswlib::SearchMetricsAccumulator metrics_accumulator_;
#endif


public:
    // Constructor
    IndexCube(size_t num_layers = 3,
              size_t M = 16,
              size_t ef_construction = 200,
              size_t cross_edge_count = 2,
              size_t min_points_per_cube = 50) :
            num_layers_(num_layers),
            M_(M),
            ef_construction_(ef_construction),
            cross_edge_count_(cross_edge_count),
            num_vectors_(0),
            min_points_per_cube_(min_points_per_cube) {};

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

#ifdef COLLECT_LOG
    const hnswlib::SearchMetricsAccumulator& get_metrics() const { return metrics_accumulator_; }
    void reset_metrics() { metrics_accumulator_.reset(); }
#endif


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
            assert(layer.hnsw_index->getCubeId(i) < layer.total_cubes);
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

    // Compute adaptive number of layers based on minimum points per cube
    // Stops when points_per_cube would fall below min_points_per_cube_
    size_t compute_adaptive_num_layers() const {
        size_t adaptive_layers = 0;
        size_t cubes_per_dim = 2;  // layer 0 has 2^1 = 2 cubes per dim
        while (true) {
            size_t total_cubes = 1;
            for (size_t d = 0; d < attr_dim_; d++) {
                total_cubes *= cubes_per_dim;
            }
            double points_per_cube = static_cast<double>(num_vectors_) / total_cubes;
            if (points_per_cube < min_points_per_cube_ && adaptive_layers > 0) {
                break;
            }
            adaptive_layers++;
            if (adaptive_layers >= num_layers_) {
                break;
            }
            cubes_per_dim *= 2;
        }
        return adaptive_layers;
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

        // Compute adaptive number of layers
        size_t adaptive_layers = compute_adaptive_num_layers();
        std::cout << "Adaptive layers: " << adaptive_layers
                  << " (min points/cube: " << min_points_per_cube_ << ")" << std::endl;
        std::cout << "Building " << adaptive_layers << " layers..." << std::endl;
        num_layers_ = adaptive_layers;
        for (size_t layer_id = 0; layer_id < adaptive_layers; layer_id++) {
            std::cout << "Building layer " << layer_id << "..." << std::endl;
            build_layer(layer_id);
            size_t points_per_cube = num_vectors_ / layers_[layer_id].total_cubes;
            std::cout << "Layer " << layer_id << " built: "
                      << layers_[layer_id].cubes_per_dim << "^" << attr_dim_
                      << " = " << layers_[layer_id].total_cubes << " cubes"
                      << " (" << points_per_cube << " points/cube)" << std::endl;

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

    // Filter functor: admits only points whose metadata lies inside the polygon (ray casting)
    class PolygonFilter : public hnswlib::BaseFilterFunctor {
    public:
        const PolygonFilterParams& polygon_;
        size_t attr_dim_;

        PolygonFilter(const PolygonFilterParams& polygon, size_t attr_dim)
            : polygon_(polygon), attr_dim_(attr_dim) {}

        bool operator()(hnswlib::metatype *meta) override {
            // Ray casting algorithm for point-in-polygon test
            // Only works for 2D polygons (attr_dim must be 2)
            // For higher dimensions, only consider first 2 dimensions
            int crossings = 0;
            size_t n = polygon_.vertices.size();
            for (size_t i = 0; i < n; i++) {
                const auto& v1 = polygon_.vertices[i];
                const auto& v2 = polygon_.vertices[(i + 1) % n];

                // Check if ray from point crosses this edge
                // y-range check ensures we only count each edge once
                if (((v1[1] <= meta[1] && meta[1] < v2[1]) ||
                     (v2[1] <= meta[1] && meta[1] < v1[1])) &&
                    (meta[0] < (v2[0] - v1[0]) * (meta[1] - v1[1]) / (v2[1] - v1[1]) + v1[0])) {
                    crossings++;
                }
            }
            return (crossings % 2) == 1;  // odd crossings = inside
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

    // Get the HNSW index for a specific layer (for custom filter searches)
    hnswlib::HierarchicalNSWCube<float>* get_hnsw_index(size_t layer_id) {
        if (layer_id >= layers_.size()) return nullptr;
        return layers_[layer_id].hnsw_index;
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


    std::vector<hnswlib::tableint> find_cube_list_with_BBox(size_t layer_id, const BoundingBox *filter_bbox = nullptr) const {
        if (filter_bbox == nullptr) return {};

        const auto &layer = layers_[layer_id];
        std::vector<hnswlib::tableint> result;
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
        int target_layer = 0;
        float radius = filter_radius->sphere.radius;
        while (target_layer < layers_.size()) {
            // Compute diagonal of cube at this layer
            float diagonal2 = 0.0f;
            for (size_t d = 0; d < attr_dim_; d++) {
                diagonal2 += layers_[target_layer].cube_width[d] * layers_[target_layer].cube_width[d];
            }
            float diagonal = std::sqrt(diagonal2);
            if (2 * radius > diagonal) {
                return std::max(target_layer, 0);
            }
            target_layer++;
        }
        if(target_layer == num_layers_) target_layer--;
        return target_layer;
    }

    int find_cube_with_Radius(size_t layer_id, const RadiusFilterParams* filter_radius = nullptr) const {
        if (filter_radius == nullptr) return 0;
        const auto& layer = layers_[layer_id];
        // Compute cube ID for the sphere center
        return static_cast<int>(compute_cube_id(filter_radius->sphere.center, layer));
    }

    std::vector<hnswlib::tableint> find_cube_list_with_Radius(size_t layer_id, const RadiusFilterParams* filter_radius = nullptr) const {
        if (filter_radius == nullptr) return {};

        const auto& layer = layers_[layer_id];
        const Sphere& sphere = filter_radius->sphere;
        float radius2 = sphere.radius * sphere.radius;

        std::vector<hnswlib::tableint> result;
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
            if (sphere.overlaps(cube_bbox)) {
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


    // Main search method (fly search - traverses adjacent cubes)
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    fly_search(const float *query_vector, size_t k,
           const BoundingBox *filter_bbox = nullptr) {
        auto layer_id = select_layer_with_BBox(filter_bbox);
        const auto &layer = layers_[layer_id];
        BBoxFilter bbox_filter(*filter_bbox, attr_dim_);
        auto cube_id = find_cube_with_BBox(layer_id, filter_bbox);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        auto result = layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &bbox_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        return layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &bbox_filter);
#endif
    }

    // Predetermined search method (searches fixed cube list)
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    predetermined_search(const float *query_vector, size_t k,
           const BoundingBox *filter_bbox = nullptr) {
        auto layer_id = select_layer_with_BBox(filter_bbox);
        const auto &layer = layers_[layer_id];
        BBoxFilter bbox_filter(*filter_bbox, attr_dim_);
        std::vector<hnswlib::tableint> cube_list = find_cube_list_with_BBox(layer_id, filter_bbox);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        metrics.cubes_visited = cube_list.size();
        auto result = layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &bbox_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        return layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &bbox_filter);
#endif
    }

    // Build composite filter from CompositeFilterParams
    // Returns ownership of the filter - caller must manage memory
    static hnswlib::BaseFilterFunctor* build_composite_filter(const CompositeFilterParams& params) {
        hnswlib::BaseFilterFunctor* result = nullptr;

        // Create individual filters
        BBoxFilter* bbox_filter = nullptr;
        RadiusFilter* radius_filter = nullptr;

        if (params.spec.use_bbox) {
            bbox_filter = new BBoxFilter(params.bbox, params.attr_dim);
            if (params.spec.bbox_negate) {
                result = new NotFilter(bbox_filter);
            } else {
                result = bbox_filter;
            }
        }

        if (params.spec.use_radius) {
            radius_filter = new RadiusFilter(params.radius.sphere, params.attr_dim);
            hnswlib::BaseFilterFunctor* radius_to_use = radius_filter;
            if (params.spec.radius_negate) {
                radius_to_use = new NotFilter(radius_filter);
            }

            if (result == nullptr) {
                result = radius_to_use;
            } else {
                // Combine: result AND radius_to_use
                result = new AndFilter(result, radius_to_use);
            }
        }

        return result;
    }

    // Composite filter search method (fly search - traverses adjacent cubes)
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    fly_search(const float *query_vector, size_t k,
           const CompositeFilterParams* filter_params = nullptr) {
        auto layer_id = select_layer_with_BBox(&filter_params->bbox);
        const auto &layer = layers_[layer_id];
        hnswlib::BaseFilterFunctor* composite_filter = build_composite_filter(*filter_params);
        auto cube_id = find_cube_with_BBox(layer_id, &filter_params->bbox);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        auto result = layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, composite_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        auto result = layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, composite_filter);
#endif
        // Clean up dynamically allocated filters
        delete composite_filter;
        return result;
    }

    // Composite filter search method (predetermined - searches fixed cube list)
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    predetermined_search(const float *query_vector, size_t k,
           const CompositeFilterParams* filter_params = nullptr) {
        auto layer_id = select_layer_with_BBox(&filter_params->bbox);
        const auto &layer = layers_[layer_id];
        hnswlib::BaseFilterFunctor* composite_filter = build_composite_filter(*filter_params);
        std::vector<hnswlib::tableint> cube_list = find_cube_list_with_BBox(layer_id, &filter_params->bbox);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        metrics.cubes_visited = cube_list.size();
        auto result = layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, composite_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        auto result = layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, composite_filter);
#endif
        // Clean up dynamically allocated filters
        delete composite_filter;
        return result;
    }

    // Radius-filtered search method
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    fly_search(const float *query_vector, size_t k,
           const RadiusFilterParams* filter_radius = nullptr) {
        auto layer_id = select_layer_with_Radius(filter_radius);
        const auto &layer = layers_[layer_id];
        RadiusFilter radius_filter(filter_radius->sphere, filter_radius->attr_dim);
        auto cube_id = find_cube_with_Radius(layer_id, filter_radius);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        auto result = layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &radius_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        return layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &radius_filter);
#endif
    }

    // Predetermined radius search method
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    predetermined_search(const float *query_vector, size_t k,
           const RadiusFilterParams* filter_radius = nullptr) {
        auto layer_id = select_layer_with_Radius(filter_radius);
        const auto &layer = layers_[layer_id];
        RadiusFilter radius_filter(filter_radius->sphere, filter_radius->attr_dim);
        std::vector<hnswlib::tableint> cube_list = find_cube_list_with_Radius(layer_id, filter_radius);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        metrics.cubes_visited = cube_list.size();
        auto result = layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &radius_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        return layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &radius_filter);
#endif
    }

    // Select layer based on polygon bounding box vs cube diagonal
    size_t select_layer_with_Polygon(const PolygonFilterParams* filter_polygon = nullptr) const {
        if (filter_polygon == nullptr) return 0;
        BoundingBox bbox = filter_polygon->get_bbox();
        int target_layer = 0;
        while (target_layer < layers_.size()) {
            // Compute diagonal of cube at this layer
            float diagonal2 = 0.0f;
            for (size_t d = 0; d < attr_dim_; d++) {
                diagonal2 += layers_[target_layer].cube_width[d] * layers_[target_layer].cube_width[d];
            }
            float diagonal = std::sqrt(diagonal2);

            // Check if polygon bbox edge length exceeds diagonal
            for (size_t d = 0; d < 2 && d < attr_dim_; d++) {
                float edge_length = bbox.max_bounds[d] - bbox.min_bounds[d];
                if (2 * edge_length > diagonal) {
                    return std::max(target_layer, 0);
                }
            }
            target_layer++;
        }
        if (target_layer == num_layers_) target_layer--;
        return target_layer;
    }

    int find_cube_with_Polygon(size_t layer_id, const PolygonFilterParams* filter_polygon = nullptr) const {
        if (filter_polygon == nullptr) return 0;
        const auto& layer = layers_[layer_id];
        // Use centroid of polygon as the reference point
        BoundingBox bbox = filter_polygon->get_bbox();
        std::vector<float> center(2);
        center[0] = (bbox.min_bounds[0] + bbox.max_bounds[0]) / 2.0f;
        center[1] = (bbox.min_bounds[1] + bbox.max_bounds[1]) / 2.0f;
        return static_cast<int>(compute_cube_id(center, layer));
    }

    std::vector<hnswlib::tableint> find_cube_list_with_Polygon(size_t layer_id, const PolygonFilterParams* filter_polygon = nullptr) const {
        if (filter_polygon == nullptr) return {};

        const auto& layer = layers_[layer_id];
        BoundingBox polygon_bbox = filter_polygon->get_bbox();

        std::vector<hnswlib::tableint> result;
        std::vector<char> visited(layer.total_cubes, 0);

        // Start from cube containing polygon centroid
        int start_cube = find_cube_with_Polygon(layer_id, filter_polygon);
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

            // Check overlap with polygon bounding box (use bbox for quick rejection)
            if (cube_bbox.overlaps(polygon_bbox)) {
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

    // Polygon-filtered search method (fly search)
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    fly_search(const float *query_vector, size_t k,
           const PolygonFilterParams* filter_polygon = nullptr) {
        auto layer_id = select_layer_with_Polygon(filter_polygon);
        const auto &layer = layers_[layer_id];
        PolygonFilter polygon_filter(*filter_polygon, filter_polygon->attr_dim);
        auto cube_id = find_cube_with_Polygon(layer_id, filter_polygon);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        auto result = layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &polygon_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        return layer.hnsw_index->searchFlyKnn(query_vector, k, cube_id, &polygon_filter);
#endif
    }

    // Polygon-filtered search method (predetermined)
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    predetermined_search(const float *query_vector, size_t k,
           const PolygonFilterParams* filter_polygon = nullptr) {
        auto layer_id = select_layer_with_Polygon(filter_polygon);
        const auto &layer = layers_[layer_id];
        PolygonFilter polygon_filter(*filter_polygon, filter_polygon->attr_dim);
        std::vector<hnswlib::tableint> cube_list = find_cube_list_with_Polygon(layer_id, filter_polygon);
#ifdef COLLECT_LOG
        hnswlib::SearchMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.total_cubes_in_layer = layer.total_cubes;
        metrics.total_nodes_in_layer = layer.hnsw_index->getCurrentElementCount();
        metrics.cubes_visited = cube_list.size();
        auto result = layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &polygon_filter, &metrics);
        metrics_accumulator_.add(metrics);
        return result;
#else
        return layer.hnsw_index->searchCubeKnn(query_vector, k, cube_list, &polygon_filter);
#endif
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

    // Save index to file
    void save_index(const std::string &path) {
        std::ofstream output(path, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Cannot open index file for writing: " + path);
        }

        // Save header metadata
        size_t actual_layers = layers_.size();
        hnswlib::writeBinaryPOD(output, actual_layers);  // actual number of layers built
        hnswlib::writeBinaryPOD(output, M_);
        hnswlib::writeBinaryPOD(output, ef_construction_);
        hnswlib::writeBinaryPOD(output, cross_edge_count_);
        hnswlib::writeBinaryPOD(output, attr_dim_);
        hnswlib::writeBinaryPOD(output, vec_dim_);
        hnswlib::writeBinaryPOD(output, num_vectors_);
        hnswlib::writeBinaryPOD(output, min_points_per_cube_);

        // Save global_bbox_
        hnswlib::writeBinaryPOD(output, attr_dim_);
        for (size_t d = 0; d < attr_dim_; d++) {
            hnswlib::writeBinaryPOD(output, global_bbox_.min_bounds[d]);
        }
        for (size_t d = 0; d < attr_dim_; d++) {
            hnswlib::writeBinaryPOD(output, global_bbox_.max_bounds[d]);
        }

        // Save metadata_
        hnswlib::writeBinaryPOD(output, num_vectors_);
        hnswlib::writeBinaryPOD(output, attr_dim_);
        for (size_t i = 0; i < num_vectors_; i++) {
            output.write(reinterpret_cast<const char*>(metadata_[i].data()), attr_dim_ * sizeof(float));
        }

        // Save per-layer data
        for (size_t layer_idx = 0; layer_idx < layers_.size(); layer_idx++) {
            const auto &layer = layers_[layer_idx];

            // Save layer metadata
            hnswlib::writeBinaryPOD(output, layer.layer_id);
            hnswlib::writeBinaryPOD(output, layer.cubes_per_dim);
            hnswlib::writeBinaryPOD(output, layer.total_cubes);
            for (size_t d = 0; d < attr_dim_; d++) {
                hnswlib::writeBinaryPOD(output, layer.cube_width[d]);
            }

            // Save adjacent_cubes
            hnswlib::writeBinaryPOD(output, layer.total_cubes);
            for (size_t c = 0; c < layer.total_cubes; c++) {
                hnswlib::writeBinaryPOD(output, layer.adjacent_cubes[c].size());
                for (const auto &adj : layer.adjacent_cubes[c]) {
                    hnswlib::writeBinaryPOD(output, adj);
                }
            }

            // Save global_bbox for this layer
            hnswlib::writeBinaryPOD(output, attr_dim_);
            for (size_t d = 0; d < attr_dim_; d++) {
                hnswlib::writeBinaryPOD(output, layer.global_bbox.min_bounds[d]);
            }
            for (size_t d = 0; d < attr_dim_; d++) {
                hnswlib::writeBinaryPOD(output, layer.global_bbox.max_bounds[d]);
            }

            // Save HNSW index for this layer
            std::string layer_path = path + ".layer_" + std::to_string(layer_idx);
            layer.hnsw_index->saveIndex(layer_path);
        }

        output.close();
        std::cout << "Index saved to " << path << std::endl;
    }

    // Load index from file
    bool load_index(const std::string &path, char *data_file) {
        std::ifstream input(path, std::ios::binary);
        if (!input.is_open()) {
            return false;
        }

        // Load header metadata
        hnswlib::readBinaryPOD(input, num_layers_);  // actual layers built
        hnswlib::readBinaryPOD(input, M_);
        hnswlib::readBinaryPOD(input, ef_construction_);
        hnswlib::readBinaryPOD(input, cross_edge_count_);
        hnswlib::readBinaryPOD(input, attr_dim_);
        hnswlib::readBinaryPOD(input, vec_dim_);
        hnswlib::readBinaryPOD(input, num_vectors_);
        hnswlib::readBinaryPOD(input, min_points_per_cube_);

        // Load global_bbox_
        size_t bbox_dim;
        hnswlib::readBinaryPOD(input, bbox_dim);
        global_bbox_.min_bounds.resize(bbox_dim);
        global_bbox_.max_bounds.resize(bbox_dim);
        for (size_t d = 0; d < bbox_dim; d++) {
            hnswlib::readBinaryPOD(input, global_bbox_.min_bounds[d]);
        }
        for (size_t d = 0; d < bbox_dim; d++) {
            hnswlib::readBinaryPOD(input, global_bbox_.max_bounds[d]);
        }

        // Load metadata_
        size_t meta_n, meta_d;
        hnswlib::readBinaryPOD(input, meta_n);
        hnswlib::readBinaryPOD(input, meta_d);
        if (meta_d != attr_dim_) {
            std::cerr << "ERROR: metadata dimension mismatch - loaded " << meta_d
                      << " but expected " << attr_dim_ << std::endl;
            return false;
        }
        metadata_.resize(meta_n, std::vector<float>(meta_d));
        for (size_t i = 0; i < meta_n; i++) {
            input.read(reinterpret_cast<char*>(metadata_[i].data()), meta_d * sizeof(float));
        }

        // Load vector data (for static_base_data_)
        auto *X = new Matrix<float>(data_file);
        hnswlib::HierarchicalNSWCube<float>::static_base_data_ = (char *) X->data;
        data_ = X->data;
        vec_dim_ = X->d;
        space = new hnswlib::L2Space(X->d);

        // Load per-layer data
        layers_.resize(num_layers_);
        for (size_t layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
            LayerConfig layer;

            // Load layer metadata
            hnswlib::readBinaryPOD(input, layer.layer_id);
            hnswlib::readBinaryPOD(input, layer.cubes_per_dim);
            hnswlib::readBinaryPOD(input, layer.total_cubes);
            for (size_t d = 0; d < attr_dim_; d++) {
                hnswlib::readBinaryPOD(input, layer.cube_width[d]);
            }

            // Load adjacent_cubes
            size_t total_cubes;
            hnswlib::readBinaryPOD(input, total_cubes);
            layer.adjacent_cubes.resize(total_cubes);
            for (size_t c = 0; c < total_cubes; c++) {
                size_t adj_size;
                hnswlib::readBinaryPOD(input, adj_size);
                layer.adjacent_cubes[c].resize(adj_size);
                for (size_t a = 0; a < adj_size; a++) {
                    hnswlib::readBinaryPOD(input, layer.adjacent_cubes[c][a]);
                }
            }

            // Load global_bbox for this layer
            size_t layer_bbox_dim;
            hnswlib::readBinaryPOD(input, layer_bbox_dim);
            layer.global_bbox.min_bounds.resize(layer_bbox_dim);
            layer.global_bbox.max_bounds.resize(layer_bbox_dim);
            for (size_t d = 0; d < layer_bbox_dim; d++) {
                hnswlib::readBinaryPOD(input, layer.global_bbox.min_bounds[d]);
            }
            for (size_t d = 0; d < layer_bbox_dim; d++) {
                hnswlib::readBinaryPOD(input, layer.global_bbox.max_bounds[d]);
            }

            // Create HNSW index and load from file
            layer.hnsw_index = new hnswlib::HierarchicalNSWCube<float>(space, num_vectors_, attr_dim_, layer.total_cubes,
                                                                       cross_edge_count_, M_, ef_construction_);

            // Load HNSW index for this layer
            std::string layer_path = path + ".layer_" + std::to_string(layer_idx);
            layer.hnsw_index->loadIndex(layer_path, space, num_vectors_);

            // Set adjacency (needed for search)
            layer.hnsw_index->setAdjacentCubeIds(layer.adjacent_cubes);

            layers_[layer_idx] = std::move(layer);
        }

        input.close();
        std::cout << "Index loaded from " << path << std::endl;
        return true;
    }
};

