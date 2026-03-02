#pragma once

// Define SegQuery to satisfy utils.h (included by matrix.h)
// This is a minimal definition from Index2D.h
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
#include "hnswlib/hnsw-static.h"
#include "matrix.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <queue>

#define HNSW_RTREE_M 16
#define HNSW_RTREE_efConstruction 200
#define RTREE_MAX_ENTRIES 16    // Maximum entries per R-tree node
#define RTREE_MIN_ENTRIES 4     // Minimum entries per R-tree node
#define RTREE_LEAF_THRESHOLD 4096  // Maximum vectors in leaf node

// Multi-dimensional bounding box (MBR - Minimum Bounding Rectangle)
struct MBR {
    std::vector<float> min_bounds;
    std::vector<float> max_bounds;

    MBR() {}

    MBR(size_t dim) : min_bounds(dim, std::numeric_limits<float>::max()),
                      max_bounds(dim, std::numeric_limits<float>::lowest()) {}

    MBR(const std::vector<float>& mins, const std::vector<float>& maxs)
        : min_bounds(mins), max_bounds(maxs) {}

    size_t dim() const { return min_bounds.size(); }

    // Check if point is contained in MBR
    bool contains(const std::vector<float>& point) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (point[i] < min_bounds[i] || point[i] > max_bounds[i])
                return false;
        }
        return true;
    }

    // Check if this MBR overlaps with another MBR
    bool overlaps(const MBR& other) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (max_bounds[i] < other.min_bounds[i] || min_bounds[i] > other.max_bounds[i])
                return false;
        }
        return true;
    }

    // Check if this MBR is fully contained in another MBR
    bool contained_in(const MBR& other) const {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (min_bounds[i] < other.min_bounds[i] || max_bounds[i] > other.max_bounds[i])
                return false;
        }
        return true;
    }

    // Compute area (volume in higher dimensions)
    float area() const {
        float result = 1.0f;
        for (size_t i = 0; i < min_bounds.size(); i++) {
            result *= (max_bounds[i] - min_bounds[i]);
        }
        return result;
    }

    // Compute enlargement needed to include a point
    float enlargement(const std::vector<float>& point) const {
        MBR enlarged = *this;
        for (size_t i = 0; i < min_bounds.size(); i++) {
            enlarged.min_bounds[i] = std::min(enlarged.min_bounds[i], point[i]);
            enlarged.max_bounds[i] = std::max(enlarged.max_bounds[i], point[i]);
        }
        return enlarged.area() - area();
    }

    // Expand MBR to include a point
    void expand(const std::vector<float>& point) {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            min_bounds[i] = std::min(min_bounds[i], point[i]);
            max_bounds[i] = std::max(max_bounds[i], point[i]);
        }
    }

    // Expand MBR to include another MBR
    void expand(const MBR& other) {
        for (size_t i = 0; i < min_bounds.size(); i++) {
            min_bounds[i] = std::min(min_bounds[i], other.min_bounds[i]);
            max_bounds[i] = std::max(max_bounds[i], other.max_bounds[i]);
        }
    }

    // Check if point is within radius of center
    bool within_radius(const std::vector<float>& center, float radius) const {
        // Check if MBR intersects with sphere
        float dist_sq = 0.0f;
        for (size_t i = 0; i < min_bounds.size(); i++) {
            if (center[i] < min_bounds[i]) {
                float d = min_bounds[i] - center[i];
                dist_sq += d * d;
            } else if (center[i] > max_bounds[i]) {
                float d = center[i] - max_bounds[i];
                dist_sq += d * d;
            }
        }
        return dist_sq <= (radius * radius);
    }
};

// R-tree query structure
struct RTreeQuery {
    MBR mbr;                    // Query MBR (for range queries)
    std::vector<float> center;  // Query center (for radius queries)
    float radius;               // Query radius
    bool is_radius_query;       // True if radius query
    float* data_;               // Query vector

    RTreeQuery() : radius(0.0f), is_radius_query(false), data_(nullptr) {}

    RTreeQuery(const MBR& m, float* data)
        : mbr(m), radius(0.0f), is_radius_query(false), data_(data) {}

    RTreeQuery(const std::vector<float>& c, float r, float* data)
        : center(c), radius(r), is_radius_query(true), data_(data) {
        // Create MBR from radius
        size_t dim = c.size();
        mbr = MBR(dim);
        for (size_t i = 0; i < dim; i++) {
            mbr.min_bounds[i] = c[i] - r;
            mbr.max_bounds[i] = c[i] + r;
        }
    }

    bool overlaps_with(const MBR& node_mbr) const {
        if (is_radius_query) {
            return node_mbr.within_radius(center, radius);
        } else {
            return mbr.overlaps(node_mbr);
        }
    }
};

// R-tree filter for HNSW search
class RTreeFilter : public hnswlib::BaseFilterFunctor {
public:
    const std::vector<std::vector<float>>* metadata_;
    RTreeQuery query_;

    RTreeFilter(const std::vector<std::vector<float>>* metadata, const RTreeQuery& query)
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
            // Range query: check if point is in MBR
            return query_.mbr.contains(point);
        }
    }
};

// R-tree node
class RTreeNode {
public:
    MBR mbr;                                    // Bounding box of this node
    bool is_leaf;                               // True if leaf node
    std::vector<RTreeNode*> children;           // Child nodes (for internal nodes)
    std::vector<hnswlib::labeltype> point_ids;  // Vector IDs (for leaf nodes)
    hnswlib::HierarchicalNSWStatic<float>* index; // HNSW index (for leaf nodes)

    RTreeNode(size_t dim, bool leaf = false)
        : mbr(dim), is_leaf(leaf), index(nullptr) {}

    ~RTreeNode() {
        if (index) delete index;
        for (auto child : children) {
            if (child) delete child;
        }
    }
};

// Main R-tree index class
class IndexRTreeND {
public:
    IndexRTreeND() : attr_dim_(0), vec_dim_(0), num_vectors_(0), root_(nullptr) {}

    IndexRTreeND(size_t attr_dim, size_t vec_dim)
        : attr_dim_(attr_dim), vec_dim_(vec_dim), num_vectors_(0), root_(nullptr) {}

    ~IndexRTreeND() {
        if (root_) delete root_;
    }

    // Build R-tree index
    void build_index(const char* data_path, const char* metadata_path, const char* output_path) {
        // Load vector data
        Matrix<float>* X = new Matrix<float>(const_cast<char*>(data_path));
        vec_dim_ = X->d;
        num_vectors_ = X->n;
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char*)X->data;

        // Load metadata
        load_metadata(metadata_path);

        // Build R-tree using bulk loading
        std::vector<hnswlib::labeltype> all_ids(num_vectors_);
        for (size_t i = 0; i < num_vectors_; i++) {
            all_ids[i] = i;
        }

        root_ = bulk_load(all_ids, 0);

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

    // Range query
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    range_search(float* query, const MBR& mbr, size_t k, size_t ef = 100) {
        RTreeQuery rtree_query(mbr, query);
        return search_rtree(query, rtree_query, k, ef);
    }

    // Radius query
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    radius_search(float* query, const std::vector<float>& center, float radius, size_t k, size_t ef = 100) {
        RTreeQuery rtree_query(center, radius, query);
        return search_rtree(query, rtree_query, k, ef);
    }

private:
    size_t attr_dim_;
    size_t vec_dim_;
    size_t num_vectors_;
    RTreeNode* root_;
    std::vector<std::vector<float>> metadata_;

    // Load metadata
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

    // Bulk load R-tree using Sort-Tile-Recursive (STR) algorithm
    RTreeNode* bulk_load(std::vector<hnswlib::labeltype>& ids, int depth) {
        if (ids.empty()) return nullptr;

        // Create leaf node if below threshold
        if (ids.size() <= RTREE_LEAF_THRESHOLD) {
            return create_leaf_node(ids);
        }

        // Sort points along alternating dimensions
        size_t sort_dim = depth % attr_dim_;
        std::sort(ids.begin(), ids.end(), [this, sort_dim](hnswlib::labeltype a, hnswlib::labeltype b) {
            return metadata_[a][sort_dim] < metadata_[b][sort_dim];
        });

        // Split into slices
        size_t slice_size = std::ceil(std::sqrt(ids.size() / (float)RTREE_MAX_ENTRIES));
        size_t num_slices = std::ceil(ids.size() / (float)slice_size);

        RTreeNode* node = new RTreeNode(attr_dim_, false);

        for (size_t i = 0; i < num_slices; i++) {
            size_t start = i * slice_size;
            size_t end = std::min(start + slice_size, ids.size());

            std::vector<hnswlib::labeltype> slice_ids(ids.begin() + start, ids.begin() + end);
            RTreeNode* child = bulk_load(slice_ids, depth + 1);

            if (child) {
                node->children.push_back(child);
                node->mbr.expand(child->mbr);
            }
        }

        return node;
    }

    // Create leaf node with HNSW index
    RTreeNode* create_leaf_node(const std::vector<hnswlib::labeltype>& ids) {
        RTreeNode* node = new RTreeNode(attr_dim_, true);
        node->point_ids = ids;

        // Compute MBR
        for (auto id : ids) {
            node->mbr.expand(metadata_[id]);
        }

        // Build HNSW index
        auto l2space = new hnswlib::L2Space(vec_dim_);
        node->index = new hnswlib::HierarchicalNSWStatic<float>(
            l2space, ids.size(), HNSW_RTREE_M, HNSW_RTREE_efConstruction);

        for (size_t i = 0; i < ids.size(); i++) {
            hnswlib::labeltype id = ids[i];
            node->index->addPoint(
                hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + id * node->index->data_size_,
                id);
        }

        return node;
    }

    // Search R-tree
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search_rtree(float* query, const RTreeQuery& rtree_query, size_t k, size_t ef) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> merged_results;

        if (!root_) return merged_results;

        // Recursively search R-tree
        search_node(root_, query, rtree_query, k, ef, merged_results);

        // Keep only top k
        std::priority_queue<std::pair<float, hnswlib::labeltype>> final_results;
        while (!merged_results.empty() && final_results.size() < k) {
            final_results.push(merged_results.top());
            merged_results.pop();
        }

        return final_results;
    }

    // Recursively search R-tree node
    void search_node(RTreeNode* node, float* query, const RTreeQuery& rtree_query, size_t k, size_t ef,
                    std::priority_queue<std::pair<float, hnswlib::labeltype>>& results) {
        if (!node) return;

        // Check if query overlaps with node's MBR
        if (!rtree_query.overlaps_with(node->mbr)) {
            return;
        }

        if (node->is_leaf) {
            // Leaf node: search HNSW index with filter
            if (node->index) {
                RTreeFilter filter(&metadata_, rtree_query);
                node->index->setEf(ef);
                auto node_results = node->index->searchKnn(query, k, &filter);

                // Merge results
                while (!node_results.empty()) {
                    results.push(node_results.top());
                    node_results.pop();
                }
            }
        } else {
            // Internal node: recursively search children
            for (auto child : node->children) {
                search_node(child, query, rtree_query, k, ef, results);
            }
        }
    }

    // Save index
    void save_index(const char* output_path) {
        std::ofstream fout(output_path, std::ios::binary);
        save_node(root_, fout);
        fout.close();
    }

    void save_node(RTreeNode* node, std::ofstream& fout) {
        if (!node) {
            bool is_null = true;
            fout.write((char*)&is_null, sizeof(bool));
            return;
        }

        bool is_null = false;
        fout.write((char*)&is_null, sizeof(bool));

        // Save node metadata
        fout.write((char*)&node->is_leaf, sizeof(bool));
        fout.write((char*)node->mbr.min_bounds.data(), attr_dim_ * sizeof(float));
        fout.write((char*)node->mbr.max_bounds.data(), attr_dim_ * sizeof(float));

        if (node->is_leaf) {
            // Save leaf node data
            size_t num_points = node->point_ids.size();
            fout.write((char*)&num_points, sizeof(size_t));
            fout.write((char*)node->point_ids.data(), num_points * sizeof(hnswlib::labeltype));

            // Save HNSW index
            bool has_index = (node->index != nullptr);
            fout.write((char*)&has_index, sizeof(bool));
            if (has_index) {
                save_hnsw_index(node->index, fout);
            }
        } else {
            // Save internal node children
            size_t num_children = node->children.size();
            fout.write((char*)&num_children, sizeof(size_t));
            for (auto child : node->children) {
                save_node(child, fout);
            }
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

    // Load node
    RTreeNode* load_node(std::ifstream& fin) {
        bool is_null;
        fin.read((char*)&is_null, sizeof(bool));
        if (is_null) return nullptr;

        bool is_leaf;
        fin.read((char*)&is_leaf, sizeof(bool));

        RTreeNode* node = new RTreeNode(attr_dim_, is_leaf);
        fin.read((char*)node->mbr.min_bounds.data(), attr_dim_ * sizeof(float));
        fin.read((char*)node->mbr.max_bounds.data(), attr_dim_ * sizeof(float));

        if (is_leaf) {
            // Load leaf node
            size_t num_points;
            fin.read((char*)&num_points, sizeof(size_t));
            node->point_ids.resize(num_points);
            fin.read((char*)node->point_ids.data(), num_points * sizeof(hnswlib::labeltype));

            bool has_index;
            fin.read((char*)&has_index, sizeof(bool));
            if (has_index) {
                node->index = load_hnsw_index(fin, num_points);
            }
        } else {
            // Load internal node
            size_t num_children;
            fin.read((char*)&num_children, sizeof(size_t));
            node->children.resize(num_children);
            for (size_t i = 0; i < num_children; i++) {
                node->children[i] = load_node(fin);
            }
        }

        return node;
    }

    hnswlib::HierarchicalNSWStatic<float>* load_hnsw_index(std::ifstream& fin, size_t num_elements) {
        auto l2space = new hnswlib::L2Space(vec_dim_);
        auto index = new hnswlib::HierarchicalNSWStatic<float>(
            l2space, num_elements, HNSW_RTREE_M, HNSW_RTREE_efConstruction);

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
