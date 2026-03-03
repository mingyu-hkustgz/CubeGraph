#pragma once

#include <vector>
#include <queue>
#include <utility>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <memory>
#include <omp.h>
#include <unordered_map>

// Forward declarations needed by utils.h (before matrix.h includes utils.h)
struct SegQuery {
    SegQuery() : L(0), R(0), data_(nullptr) {}
    SegQuery(unsigned left_range, unsigned right_range, float *data)
        : L(left_range), R(right_range), data_(data) {}
    unsigned L, R;
    float *data_;
};

// Stub for bruteforce_range_search to satisfy utils.h
inline auto bruteforce_range_search(const SegQuery&, float*, unsigned, unsigned) {
    return std::priority_queue<std::pair<float, unsigned>>();
}

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnsw-cube.h"
#include "matrix.h"

#define HNSW_CUBE_M 32  // Internal edges per node
#define HNSW_CUBE_efConstruction 200  // Reduced for faster build
#define CUBE_THRESHOLD 5000  // Minimum vectors per cube (larger for faster build)

// Multi-dimensional bounding box
struct CubeBoundingBox {
    std::vector<float> min_bounds;
    std::vector<float> max_bounds;

    CubeBoundingBox() {}

    CubeBoundingBox(size_t dim) : min_bounds(dim), max_bounds(dim) {}

    bool contains(const std::vector<float>& point) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (point[i] < min_bounds[i] || point[i] > max_bounds[i])
                return false;
        }
        return true;
    }

    bool overlaps(const CubeBoundingBox& other) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (max_bounds[i] < other.min_bounds[i] || min_bounds[i] > other.max_bounds[i])
                return false;
        }
        return true;
    }

    // Check if this cube is adjacent to another cube
    bool is_adjacent(const CubeBoundingBox& other, float epsilon = 1e-6) const {
        int touching_dims = 0;
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (std::abs(max_bounds[i] - other.min_bounds[i]) < epsilon ||
                std::abs(min_bounds[i] - other.max_bounds[i]) < epsilon) {
                touching_dims++;
            }
        }
        // Adjacent if touching in exactly one dimension and overlapping in others
        return touching_dims >= 1;
    }

    // Get direction to another cube (for 2D: 0=left, 1=right, 2=bottom, 3=top)
    int get_direction_to(const CubeBoundingBox& other, float epsilon = 1e-6) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (std::abs(max_bounds[i] - other.min_bounds[i]) < epsilon) {
                return 2 * i + 1;  // Right/top/front direction
            }
            if (std::abs(min_bounds[i] - other.max_bounds[i]) < epsilon) {
                return 2 * i;  // Left/bottom/back direction
            }
        }
        return -1;
    }
};

// Cube query structure
struct CubeQuery {
    CubeBoundingBox bbox;
    std::vector<float> center;
    float radius;
    bool is_radius_query;
    float* data_;

    CubeQuery() : radius(0.0f), is_radius_query(false), data_(nullptr) {}

    CubeQuery(const CubeBoundingBox& box, float* data)
        : bbox(box), radius(0.0f), is_radius_query(false), data_(data) {}

    CubeQuery(const std::vector<float>& c, float r, float* data)
        : center(c), radius(r), is_radius_query(true), data_(data) {
        size_t dim = c.size();
        bbox = CubeBoundingBox(dim);
        for (size_t i = 0; i < dim; i++) {
            bbox.min_bounds[i] = c[i] - r;
            bbox.max_bounds[i] = c[i] + r;
        }
    }
};

// Cube filter for filtering points based on metadata
class CubeFilter : public hnswlib::BaseFilterFunctor {
public:
    const std::vector<std::vector<float>>* metadata_;
    CubeQuery query_;

    CubeFilter(const std::vector<std::vector<float>>* metadata, const CubeQuery& query)
        : metadata_(metadata), query_(query) {}

    bool operator()(hnswlib::labeltype id) override {
        if (id >= metadata_->size()) return false;

        const auto& point = (*metadata_)[id];

        if (query_.is_radius_query) {
            float dist_sq = 0.0f;
            for (size_t i = 0; i < point.size(); i++) {
                float diff = point[i] - query_.center[i];
                dist_sq += diff * diff;
            }
            return dist_sq <= (query_.radius * query_.radius);
        } else {
            return query_.bbox.contains(point);
        }
    }
};

// Cube node in hierarchical structure
class CubeNode {
public:
    CubeBoundingBox bbox;
    std::vector<hnswlib::labeltype> point_ids;
    hnswlib::HierarchicalNSWCube<float>* index;
    std::vector<CubeNode*> children;
    std::vector<CubeNode*> neighbors;  // Adjacent cubes at same level
    int level;
    int cube_id;  // Unique ID for this cube (linear index in grid)
    std::vector<int> grid_position;  // Position in the grid (for 2D: [i,j], for 3D: [i,j,k])
    std::vector<int> grid_size;  // Size of the grid at this level (for each dimension)

    CubeNode() : index(nullptr), level(0), cube_id(-1) {}

    ~CubeNode() {
        if (index) delete index;
        for (auto child : children) {
            if (child) delete child;
        }
    }

    bool is_leaf() const {
        return children.empty();
    }

    // Compute linear index from grid position
    int compute_linear_index() const {
        if (grid_position.empty() || grid_size.empty()) return -1;
        int idx = 0;
        int multiplier = 1;
        for (size_t d = 0; d < grid_position.size(); d++) {
            idx += grid_position[d] * multiplier;
            multiplier *= grid_size[d];
        }
        return idx;
    }

    // Get neighbor cube ID in a specific direction
    // direction: 0=negative, 1=positive for each dimension
    // For 2D: 0=left, 1=right, 2=bottom, 3=top
    // For 3D: 0=left, 1=right, 2=bottom, 3=top, 4=back, 5=front
    // Returns linear index based on grid position
    int get_neighbor_id_in_direction(int dir) const {
        if (grid_position.empty() || grid_size.empty()) return -1;

        int dim = dir / 2;  // Which dimension (0=dim0_negative, 1=dim0_positive, 2=dim1_negative, ...)
        int sign = dir % 2;  // 0 = negative direction, 1 = positive direction

        if (dim >= static_cast<int>(grid_position.size())) return -1;

        int new_pos = grid_position[dim] + (sign == 0 ? -1 : 1);

        // Check bounds
        if (new_pos < 0 || new_pos >= grid_size[dim]) return -1;

        // Compute neighbor's linear index using row-major order
        int neighbor_id = 0;
        int multiplier = 1;
        for (size_t d = 0; d < grid_position.size(); d++) {
            int pos = grid_position[d];
            if (d == static_cast<size_t>(dim)) pos = new_pos;
            neighbor_id += pos * multiplier;
            multiplier *= grid_size[d];
        }

        return neighbor_id;
    }
};

// Main cube-based index class
class IndexCube {
public:
    IndexCube() : attr_dim_(0), vec_dim_(0), num_vectors_(0), root_(nullptr) {}

    IndexCube(size_t attr_dim, size_t vec_dim)
        : attr_dim_(attr_dim), vec_dim_(vec_dim), num_vectors_(0), root_(nullptr) {}

    ~IndexCube() {
        if (root_) delete root_;
    }

    // Build hierarchical cube index
    void build_index(const char* data_path, const char* metadata_path, const char* output_path) {
        // Load vector data
        Matrix<float>* X = new Matrix<float>(const_cast<char*>(data_path));
        vec_dim_ = X->d;
        num_vectors_ = X->n;
        hnswlib::HierarchicalNSWCube<float>::static_base_data_ = (char*)X->data;

        // Load metadata
        load_metadata(metadata_path);

        std::cout << "Building cube index..." << std::endl;
        std::cout << "  Vectors: " << num_vectors_ << std::endl;
        std::cout << "  Vector dim: " << vec_dim_ << std::endl;
        std::cout << "  Attribute dim: " << attr_dim_ << std::endl;

        // Compute global bounding box
        CubeBoundingBox global_bbox(attr_dim_);
        for (size_t i = 0; i < attr_dim_; i++) {
            global_bbox.min_bounds[i] = std::numeric_limits<float>::max();
            global_bbox.max_bounds[i] = std::numeric_limits<float>::lowest();
        }

        for (const auto& point : metadata_) {
            for (size_t i = 0; i < attr_dim_; i++) {
                global_bbox.min_bounds[i] = std::min(global_bbox.min_bounds[i], point[i]);
                global_bbox.max_bounds[i] = std::max(global_bbox.max_bounds[i], point[i]);
            }
        }

        // Build hierarchical cube structure
        std::vector<hnswlib::labeltype> all_ids(num_vectors_);
        for (size_t i = 0; i < num_vectors_; i++) {
            all_ids[i] = i;
        }

        root_ = build_cube_recursive(global_bbox, all_ids, 0, {}, {});

        // Build cross-cube edges
        std::cout << "Building cross-cube edges..." << std::endl;
        build_cross_cube_connections(root_);

        // Save index
        if (output_path) {
            save_index(output_path);
        }

        std::cout << "Cube index built successfully!" << std::endl;
    }

    // Search with cube-based filtering
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search(const float* query_vec, const CubeQuery& query_filter, size_t k) {
        if (!root_) {
            return std::priority_queue<std::pair<float, hnswlib::labeltype>>();
        }

        // Find cubes that overlap with the query filter
        std::vector<CubeNode*> overlapping_cubes;
        find_overlapping_cubes(root_, query_filter.bbox, overlapping_cubes);

        std::cout << "Found " << overlapping_cubes.size() << " overlapping cubes" << std::endl;

        if (overlapping_cubes.empty()) {
            return std::priority_queue<std::pair<float, hnswlib::labeltype>>();
        }

        // Create filter functor
        CubeFilter filter(&metadata_, query_filter);

        // Merge results from all overlapping cubes
        std::priority_queue<std::pair<float, hnswlib::labeltype>> merged_results;

        for (auto cube : overlapping_cubes) {
            if (cube->index == nullptr || cube->index->cur_element_count == 0) {
                continue;
            }

            // Search in this cube with cross-cube edges
            auto results = cube->index->searchKnn(query_vec, k, &filter);

            // Merge results
            while (!results.empty()) {
                merged_results.push(results.top());
                results.pop();
            }
        }

        // Keep only top k
        std::priority_queue<std::pair<float, hnswlib::labeltype>> final_results;
        std::vector<std::pair<float, hnswlib::labeltype>> temp;

        while (!merged_results.empty()) {
            temp.push_back(merged_results.top());
            merged_results.pop();
        }

        std::sort(temp.begin(), temp.end());
        for (size_t i = 0; i < std::min(k, temp.size()); i++) {
            final_results.push(temp[i]);
        }

        return final_results;
    }

private:
    size_t attr_dim_;
    size_t vec_dim_;
    size_t num_vectors_;
    CubeNode* root_;
    std::vector<std::vector<float>> metadata_;

    // Load metadata from binary file
    void load_metadata(const char* metadata_path) {
        std::ifstream fin(metadata_path, std::ios::binary);
        if (!fin.is_open()) {
            throw std::runtime_error("Cannot open metadata file");
        }

        size_t n, d;
        fin.read((char*)&n, sizeof(size_t));
        fin.read((char*)&d, sizeof(size_t));

        if (n != num_vectors_) {
            throw std::runtime_error("Metadata size mismatch");
        }

        attr_dim_ = d;
        metadata_.resize(n);

        for (size_t i = 0; i < n; i++) {
            metadata_[i].resize(d);
            fin.read((char*)metadata_[i].data(), d * sizeof(float));
        }

        fin.close();
    }

    // Recursively build cube structure
    CubeNode* build_cube_recursive(
            const CubeBoundingBox& bbox,
            const std::vector<hnswlib::labeltype>& point_ids,
            int level,
            const std::vector<int>& grid_pos = {},
            const std::vector<int>& grid_sz = {}) {

        CubeNode* node = new CubeNode();
        node->bbox = bbox;
        node->point_ids = point_ids;
        node->level = level;
        node->grid_position = grid_pos;
        node->grid_size = grid_sz;
        node->cube_id = node->compute_linear_index();  // Use linear index as cube_id

        // If few enough points, create leaf cube with HNSW index
        if (point_ids.size() <= CUBE_THRESHOLD) {
            build_leaf_cube(node);
            return node;
        }

        // Otherwise, subdivide into 2^attr_dim_ children
        subdivide_cube(node, grid_pos, grid_sz);

        return node;
    }

    // Build HNSW index for a leaf cube
    void build_leaf_cube(CubeNode* node) {
        if (node->point_ids.empty()) {
            return;
        }

        auto l2space = new hnswlib::L2Space(vec_dim_);
        // M = 32 internal edges
        // M_cross = 16 total (will be distributed as M/(2*d) per direction)
        node->index = new hnswlib::HierarchicalNSWCube<float>(
            l2space,
            node->point_ids.size(),
            HNSW_CUBE_M,  // M = 32 internal edges
            HNSW_CUBE_efConstruction,
            100,
            attr_dim_,
            16  // M_cross = 16 total (distributed across 2*d directions)
        );

        // Add points to index
        for (size_t i = 0; i < node->point_ids.size(); i++) {
            hnswlib::labeltype label = node->point_ids[i];
            node->index->addPoint(
                hnswlib::HierarchicalNSWCube<float>::static_base_data_ + label * vec_dim_ * sizeof(float),
                label
            );
        }
    }

    // Subdivide cube into 2^attr_dim_ children
    void subdivide_cube(CubeNode* node, const std::vector<int>& parent_grid_pos, const std::vector<int>& parent_grid_sz) {
        size_t num_children = 1 << attr_dim_;  // 2^attr_dim_
        node->children.resize(num_children, nullptr);

        // Compute midpoints
        std::vector<float> midpoints(attr_dim_);
        for (size_t i = 0; i < attr_dim_; i++) {
            midpoints[i] = (node->bbox.min_bounds[i] + node->bbox.max_bounds[i]) / 2.0f;
        }

        // Calculate child grid size (double the parent in each dimension)
        std::vector<int> child_grid_sz(attr_dim_, 2);
        if (!parent_grid_sz.empty()) {
            for (size_t i = 0; i < attr_dim_; i++) {
                child_grid_sz[i] = parent_grid_sz[i] * 2;
            }
        }

        // Create child bounding boxes and assign points
        std::vector<std::vector<hnswlib::labeltype>> child_points(num_children);

        for (auto id : node->point_ids) {
            const auto& point = metadata_[id];

            // Determine which child this point belongs to
            size_t child_idx = 0;
            for (size_t i = 0; i < attr_dim_; i++) {
                if (point[i] >= midpoints[i]) {
                    child_idx |= (1 << i);
                }
            }

            child_points[child_idx].push_back(id);
        }

        // Create children with grid positions
        for (size_t i = 0; i < num_children; i++) {
            if (child_points[i].empty()) {
                continue;
            }

            // Compute child's grid position
            std::vector<int> child_grid_pos = parent_grid_pos;
            for (size_t d = 0; d < attr_dim_; d++) {
                int bit = (i >> d) & 1;  // 0 or 1
                // Position doubles in each dimension and adds the bit
                int base = parent_grid_pos.empty() ? 0 : parent_grid_pos[d] * 2;
                child_grid_pos.push_back(base + bit);
            }

            CubeBoundingBox child_bbox(attr_dim_);
            for (size_t d = 0; d < attr_dim_; d++) {
                if (i & (1 << d)) {
                    child_bbox.min_bounds[d] = midpoints[d];
                    child_bbox.max_bounds[d] = node->bbox.max_bounds[d];
                } else {
                    child_bbox.min_bounds[d] = node->bbox.min_bounds[d];
                    child_bbox.max_bounds[d] = midpoints[d];
                }
            }

            node->children[i] = build_cube_recursive(child_bbox, child_points[i], node->level + 1, child_grid_pos, child_grid_sz);
        }
    }

    // Build cross-cube connections using grid-based neighbor lookup (O(n) instead of O(n²))
    void build_cross_cube_connections(CubeNode* root) {
        // Collect all leaf cubes
        std::vector<CubeNode*> leaf_cubes;
        collect_leaf_cubes(root, leaf_cubes);

        std::cout << "  Total leaf cubes: " << leaf_cubes.size() << std::endl;

        // Build a map from cube_id to cube pointer for O(1) lookup
        std::unordered_map<int, CubeNode*> cube_by_id;
        for (auto cube : leaf_cubes) {
            cube_by_id[cube->cube_id] = cube;
        }

        // Find neighbors using grid positions (O(n * d) instead of O(n²))
        // For each cube, check 2*attr_dim_ directions
        int num_directions = 2 * attr_dim_;
        for (auto cube : leaf_cubes) {
            for (int dir = 0; dir < num_directions; dir++) {
                int neighbor_id = cube->get_neighbor_id_in_direction(dir);

                if (neighbor_id >= 0) {
                    auto it = cube_by_id.find(neighbor_id);
                    if (it != cube_by_id.end() && it->second != cube) {
                        cube->neighbors.push_back(it->second);
                    }
                }
            }
        }

        // Count total neighbor connections
        size_t total_neighbors = 0;
        for (auto cube : leaf_cubes) {
            total_neighbors += cube->neighbors.size();
        }
        std::cout << "  Total neighbor connections: " << (total_neighbors / 2) << " (bidirectional)" << std::endl;

        std::cout << "  Building cross-cube edges..." << std::endl;

        // Build cross-cube edges - sequential to avoid race conditions
        for (size_t i = 0; i < leaf_cubes.size(); i++) {
            auto cube = leaf_cubes[i];
            if (cube->index == nullptr) continue;

            // Add neighbor connections
            for (auto neighbor : cube->neighbors) {
                if (neighbor->index == nullptr) continue;

                int direction = cube->bbox.get_direction_to(neighbor->bbox);
                cube->index->add_neighbor_cube(neighbor->cube_id, direction, neighbor->index);
            }

            // Build the actual cross-cube edges
            cube->index->build_cross_cube_edges();

            if (i % 10 == 0) {
                std::cout << "    Processed " << i << "/" << leaf_cubes.size() << " cubes" << std::endl;
            }
        }

        std::cout << "  Cross-cube edges built" << std::endl;
    }

    // Collect all leaf cubes
    void collect_leaf_cubes(CubeNode* node, std::vector<CubeNode*>& leaves) {
        if (node == nullptr) return;

        if (node->is_leaf()) {
            leaves.push_back(node);
        } else {
            for (auto child : node->children) {
                collect_leaf_cubes(child, leaves);
            }
        }
    }

    // Find cubes that overlap with query bounding box
    void find_overlapping_cubes(
            CubeNode* node,
            const CubeBoundingBox& query_bbox,
            std::vector<CubeNode*>& result) {

        if (node == nullptr) return;

        if (!node->bbox.overlaps(query_bbox)) {
            return;
        }

        if (node->is_leaf()) {
            result.push_back(node);
        } else {
            for (auto child : node->children) {
                find_overlapping_cubes(child, query_bbox, result);
            }
        }
    }

    // Save index to file
    void save_index(const char* output_path) {
        std::ofstream fout(output_path, std::ios::binary);
        if (!fout.is_open()) {
            throw std::runtime_error("Cannot open output file");
        }

        // Write header
        fout.write((char*)&attr_dim_, sizeof(size_t));
        fout.write((char*)&vec_dim_, sizeof(size_t));
        fout.write((char*)&num_vectors_, sizeof(size_t));

        // TODO: Implement full serialization
        // For now, just write basic info

        fout.close();
        std::cout << "Index saved to: " << output_path << std::endl;
    }
};
