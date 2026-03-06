#pragma once

#include "hnswlib/hnswlib.h"
#include "hnsw-static.h"
#include "matrix.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <queue>

#define HNSW_GRID_M 16
#define HNSW_GRID_efConstruction 200
#define GRID_THRESHOLD 4096  // Minimum vectors per grid cell

// Multi-dimensional bounding box
struct BoundingBox {
    std::vector<float> min_bounds;  // Minimum values for each dimension
    std::vector<float> max_bounds;  // Maximum values for each dimension

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
};

// Grid query structure
struct GridQuery {
    BoundingBox bbox;           // Query bounding box
    std::vector<float> center;  // Query center (for radius queries)
    float radius;               // Query radius (for radius queries)
    bool is_radius_query;       // True if radius query, false if range query
    float* data_;               // Query vector

    GridQuery() : radius(0.0f), is_radius_query(false), data_(nullptr) {}

    GridQuery(const BoundingBox& box, float* data)
        : bbox(box), radius(0.0f), is_radius_query(false), data_(data) {}

    GridQuery(const std::vector<float>& c, float r, float* data)
        : center(c), radius(r), is_radius_query(true), data_(data) {
        // Create bounding box from radius
        size_t dim = c.size();
        bbox = BoundingBox(dim);
        for (size_t i = 0; i < dim; i++) {
            bbox.min_bounds[i] = c[i] - r;
            bbox.max_bounds[i] = c[i] + r;
        }
    }
};

// Multi-dimensional grid filter
class GridFilter : public hnswlib::BaseFilterFunctor {
public:
    const std::vector<std::vector<float>>* metadata_;  // Pointer to metadata
    GridQuery query_;

    GridFilter(const std::vector<std::vector<float>>* metadata, const GridQuery& query)
        : metadata_(metadata), query_(query) {}

    bool operator()(hnswlib::labeltype id) override {
        if (id >= metadata_->size()) return false;

        const auto& point = (*metadata_)[id];

        if (query_.is_radius_query) {
            // Radius query: check Euclidean distance
            float dist_sq = 0.0f;
            for (size_t i = 0; i < point.size(); i++) {
                float diff = point[i] - query_.center[i];
                dist_sq += diff * diff;
            }
            return dist_sq <= (query_.radius * query_.radius);
        } else {
            // Range query: check if point is in bounding box
            return query_.bbox.contains(point);
        }
    }
};

// Grid cell node in hierarchical structure
class GridNode {
public:
    BoundingBox bbox;                           // Bounding box of this grid cell
    std::vector<hnswlib::labeltype> point_ids;  // Vector IDs in this cell
    hnswlib::HierarchicalNSWStatic<float>* index; // HNSW index for this cell
    std::vector<GridNode*> children;            // Child grid cells (2^dim children)
    int level;                                  // Level in hierarchy (0 = root)

    GridNode() : index(nullptr), level(0) {}

    ~GridNode() {
        if (index) delete index;
        for (auto child : children) {
            if (child) delete child;
        }
    }
};

// Main hierarchical grid index class
class IndexGridND {
public:
    IndexGridND() : attr_dim_(0), vec_dim_(0), num_vectors_(0), root_(nullptr) {}

    IndexGridND(size_t attr_dim, size_t vec_dim)
        : attr_dim_(attr_dim), vec_dim_(vec_dim), num_vectors_(0), root_(nullptr) {}

    ~IndexGridND() {
        if (root_) delete root_;
    }

    // Build hierarchical grid index
    void build_index(const char* data_path, const char* metadata_path, const char* output_path) {
        // Load vector data
        Matrix<float>* X = new Matrix<float>(const_cast<char*>(data_path));
        vec_dim_ = X->d;
        num_vectors_ = X->n;
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char*)X->data;

        // Load metadata (attribute values)
        load_metadata(metadata_path);

        // Compute global bounding box
        BoundingBox global_bbox = compute_global_bbox();

        // Build hierarchical grid
        root_ = new GridNode();
        root_->bbox = global_bbox;
        root_->level = 0;

        // Add all points to root
        for (size_t i = 0; i < num_vectors_; i++) {
            root_->point_ids.push_back(i);
        }

        // Recursively subdivide
        subdivide_node(root_);

        // Save index
        save_index(output_path);
    }

    // Load index from file
    void load_index(const char* index_path, const char* metadata_path) {
        load_metadata(metadata_path);
        std::ifstream fin(index_path, std::ios::binary);
        root_ = load_node(fin);
        fin.close();
    }

    // Range query: find k nearest neighbors within bounding box
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    range_search(float* query, const BoundingBox& bbox, size_t k, size_t ef = 100) {
        GridQuery grid_query(bbox, query);
        return search_grid(query, grid_query, k, ef);
    }

    // Radius query: find k nearest neighbors within radius
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    radius_search(float* query, const std::vector<float>& center, float radius, size_t k, size_t ef = 100) {
        GridQuery grid_query(center, radius, query);
        return search_grid(query, grid_query, k, ef);
    }

private:
    size_t attr_dim_;      // Dimension of attribute space
    size_t vec_dim_;       // Dimension of vector space
    size_t num_vectors_;   // Number of vectors
    GridNode* root_;       // Root of hierarchical grid
    std::vector<std::vector<float>> metadata_;  // Attribute values for each vector

    // Load metadata from file
    void load_metadata(const char* metadata_path) {
        std::ifstream fin(metadata_path, std::ios::binary);
        size_t n, d;
        fin.read((char*)&n, sizeof(size_t));
        fin.read((char*)&d, sizeof(size_t));

        metadata_.resize(n);
        for (size_t i = 0; i < n; i++) {
            metadata_[i].resize(d);
            fin.read((char*)metadata_[i].data(), d * sizeof(float));
        }
        attr_dim_ = d;
        num_vectors_ = n;
        fin.close();
    }

    // Compute global bounding box
    BoundingBox compute_global_bbox() {
        BoundingBox bbox(attr_dim_);
        for (size_t i = 0; i < attr_dim_; i++) {
            bbox.min_bounds[i] = std::numeric_limits<float>::max();
            bbox.max_bounds[i] = std::numeric_limits<float>::lowest();
        }

        for (const auto& point : metadata_) {
            for (size_t i = 0; i < attr_dim_; i++) {
                bbox.min_bounds[i] = std::min(bbox.min_bounds[i], point[i]);
                bbox.max_bounds[i] = std::max(bbox.max_bounds[i], point[i]);
            }
        }
        return bbox;
    }

    // Subdivide node into 2^dim children
    void subdivide_node(GridNode* node) {
        // Stop if below threshold
        if (node->point_ids.size() < GRID_THRESHOLD) {
            build_hnsw_for_node(node);
            return;
        }

        // Create 2^attr_dim_ children
        size_t num_children = 1 << attr_dim_;  // 2^attr_dim_
        node->children.resize(num_children, nullptr);

        // Compute midpoints for each dimension
        std::vector<float> midpoints(attr_dim_);
        for (size_t i = 0; i < attr_dim_; i++) {
            midpoints[i] = (node->bbox.min_bounds[i] + node->bbox.max_bounds[i]) / 2.0f;
        }

        // Create children and distribute points
        for (size_t child_idx = 0; child_idx < num_children; child_idx++) {
            GridNode* child = new GridNode();
            child->level = node->level + 1;
            child->bbox = BoundingBox(attr_dim_);

            // Determine child's bounding box based on binary representation of child_idx
            for (size_t dim = 0; dim < attr_dim_; dim++) {
                bool upper_half = (child_idx >> dim) & 1;
                if (upper_half) {
                    child->bbox.min_bounds[dim] = midpoints[dim];
                    child->bbox.max_bounds[dim] = node->bbox.max_bounds[dim];
                } else {
                    child->bbox.min_bounds[dim] = node->bbox.min_bounds[dim];
                    child->bbox.max_bounds[dim] = midpoints[dim];
                }
            }

            // Assign points to child
            for (auto id : node->point_ids) {
                if (child->bbox.contains(metadata_[id])) {
                    child->point_ids.push_back(id);
                }
            }

            node->children[child_idx] = child;

            // Recursively subdivide child
            if (!child->point_ids.empty()) {
                subdivide_node(child);
            }
        }

        // Build HNSW index for current node (for queries that span multiple children)
        build_hnsw_for_node(node);
    }

    // Build HNSW index for a node
    void build_hnsw_for_node(GridNode* node) {
        if (node->point_ids.empty()) return;

        auto l2space = new hnswlib::L2Space(vec_dim_);
        node->index = new hnswlib::HierarchicalNSWStatic<float>(
            l2space, node->point_ids.size(), HNSW_GRID_M, HNSW_GRID_efConstruction);

        // Add points to index
        for (size_t i = 0; i < node->point_ids.size(); i++) {
            hnswlib::labeltype id = node->point_ids[i];
            node->index->addPoint(
                hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + id * node->index->data_size_,
                id);
        }
    }

    // Search hierarchical grid
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search_grid(float* query, const GridQuery& grid_query, size_t k, size_t ef) {
        if (!root_) {
            return std::priority_queue<std::pair<float, hnswlib::labeltype>>();
        }

        // Find appropriate level and nodes
        std::vector<GridNode*> target_nodes = find_target_nodes(root_, grid_query);

        // Merge results from all target nodes
        std::priority_queue<std::pair<float, hnswlib::labeltype>> merged_results;

        for (auto node : target_nodes) {
            if (!node->index) continue;

            GridFilter filter(&metadata_, grid_query);
            node->index->setEf(ef);
            auto results = node->index->searchKnn(query, k, &filter);

            // Merge results
            while (!results.empty()) {
                merged_results.push(results.top());
                results.pop();
            }
        }

        // Keep only top k
        std::priority_queue<std::pair<float, hnswlib::labeltype>> final_results;
        while (!merged_results.empty() && final_results.size() < k) {
            final_results.push(merged_results.top());
            merged_results.pop();
        }

        return final_results;
    }

    // Find target nodes that overlap with query
    std::vector<GridNode*> find_target_nodes(GridNode* node, const GridQuery& query) {
        std::vector<GridNode*> result;

        if (!node) return result;

        // Check if query overlaps with this node
        if (!node->bbox.overlaps(query.bbox)) {
            return result;
        }

        // If node has no children, use this node
        if (node->children.empty()) {
            result.push_back(node);
            return result;
        }

        // Check if query bbox is fully contained in this node's bbox
        bool query_contained = true;
        for (size_t i = 0; i < query.bbox.min_bounds.size(); i++) {
            if (query.bbox.min_bounds[i] < node->bbox.min_bounds[i] ||
                query.bbox.max_bounds[i] > node->bbox.max_bounds[i]) {
                query_contained = false;
                break;
            }
        }

        // If query is contained and node has no children, use this node
        if (query_contained && node->children.empty()) {
            result.push_back(node);
            return result;
        }

        // Otherwise, recursively search children
        for (auto child : node->children) {
            if (child) {
                auto child_results = find_target_nodes(child, query);
                result.insert(result.end(), child_results.begin(), child_results.end());
            }
        }

        // If no children matched, use current node
        if (result.empty()) {
            result.push_back(node);
        }

        return result;
    }

    // Save index to file
    void save_index(const char* output_path) {
        std::ofstream fout(output_path, std::ios::binary);
        save_node(root_, fout);
        fout.close();
    }

    void save_node(GridNode* node, std::ofstream& fout) {
        if (!node) {
            bool is_null = true;
            fout.write((char*)&is_null, sizeof(bool));
            return;
        }

        bool is_null = false;
        fout.write((char*)&is_null, sizeof(bool));

        // Save node metadata
        fout.write((char*)&node->level, sizeof(int));
        fout.write((char*)node->bbox.min_bounds.data(), attr_dim_ * sizeof(float));
        fout.write((char*)node->bbox.max_bounds.data(), attr_dim_ * sizeof(float));

        size_t num_points = node->point_ids.size();
        fout.write((char*)&num_points, sizeof(size_t));
        fout.write((char*)node->point_ids.data(), num_points * sizeof(hnswlib::labeltype));

        // Save HNSW index if exists
        bool has_index = (node->index != nullptr);
        fout.write((char*)&has_index, sizeof(bool));
        if (has_index) {
            save_hnsw_index(node->index, fout);
        }

        // Save children
        size_t num_children = node->children.size();
        fout.write((char*)&num_children, sizeof(size_t));
        for (auto child : node->children) {
            save_node(child, fout);
        }
    }

    void save_hnsw_index(hnswlib::HierarchicalNSWStatic<float>* index, std::ofstream& fout) {
        fout.write((char*)&index->enterpoint_node_, sizeof(unsigned int));
        fout.write((char*)&index->maxlevel_, sizeof(unsigned int));

        for (size_t j = 0; j < index->cur_element_count; j++) {
            unsigned int linkListSize = index->element_levels_[j] > 0 ?
                index->size_links_per_element_ * index->element_levels_[j] : 0;
            fout.write((char*)&linkListSize, sizeof(unsigned));
            if (linkListSize)
                fout.write((char*)index->linkLists_[j], linkListSize);
        }
        fout.write((char*)index->data_level0_memory_,
                  index->cur_element_count * index->size_data_per_element_);
    }

    // Load node from file
    GridNode* load_node(std::ifstream& fin) {
        bool is_null;
        fin.read((char*)&is_null, sizeof(bool));
        if (is_null) return nullptr;

        GridNode* node = new GridNode();

        // Load node metadata
        fin.read((char*)&node->level, sizeof(int));
        node->bbox = BoundingBox(attr_dim_);
        fin.read((char*)node->bbox.min_bounds.data(), attr_dim_ * sizeof(float));
        fin.read((char*)node->bbox.max_bounds.data(), attr_dim_ * sizeof(float));

        size_t num_points;
        fin.read((char*)&num_points, sizeof(size_t));
        node->point_ids.resize(num_points);
        fin.read((char*)node->point_ids.data(), num_points * sizeof(hnswlib::labeltype));

        // Load HNSW index if exists
        bool has_index;
        fin.read((char*)&has_index, sizeof(bool));
        if (has_index) {
            node->index = load_hnsw_index(fin, num_points);
        }

        // Load children
        size_t num_children;
        fin.read((char*)&num_children, sizeof(size_t));
        node->children.resize(num_children);
        for (size_t i = 0; i < num_children; i++) {
            node->children[i] = load_node(fin);
        }

        return node;
    }

    hnswlib::HierarchicalNSWStatic<float>* load_hnsw_index(std::ifstream& fin, size_t num_elements) {
        auto l2space = new hnswlib::L2Space(vec_dim_);
        auto index = new hnswlib::HierarchicalNSWStatic<float>(
            l2space, num_elements, HNSW_GRID_M, HNSW_GRID_efConstruction);

        fin.read((char*)&index->enterpoint_node_, sizeof(unsigned int));
        fin.read((char*)&index->maxlevel_, sizeof(unsigned int));
        index->cur_element_count = num_elements;

        for (size_t j = 0; j < num_elements; j++) {
            unsigned int linkListSize;
            fin.read((char*)&linkListSize, sizeof(unsigned));
            if (linkListSize == 0) {
                index->element_levels_[j] = 0;
                index->linkLists_[j] = nullptr;
            } else {
                index->element_levels_[j] = linkListSize / index->size_links_per_element_;
                index->linkLists_[j] = (char*)malloc(linkListSize);
                fin.read(index->linkLists_[j], linkListSize);
            }
        }
        fin.read((char*)index->data_level0_memory_,
                index->cur_element_count * index->size_data_per_element_);

        return index;
    }
};

