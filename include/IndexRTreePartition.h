#pragma once

#include "hnswlib/hnswlib.h"
#include "hnsw-static.h"
#include "IndexCube.h"
#include "matrix.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <queue>
#include <numeric>
#include <omp.h>

// R-Tree partitioned HNSW baseline index.
// Builds an R-Tree (STR bulk-loaded) over 2D metadata space.
// Internal nodes store MBRs; leaf nodes own subsets of vectors.
// Leaf nodes with > leaf_scan_threshold vectors get an HNSW index;
// smaller leaves use brute-force scan.
// At query time, traverse the R-Tree to find leaves overlapping the filter
// bounding box, search each leaf, and merge results.

struct RTreeBoundingBox {
    float min_bounds[2];
    float max_bounds[2];

    RTreeBoundingBox() {
        min_bounds[0] = min_bounds[1] = std::numeric_limits<float>::max();
        max_bounds[0] = max_bounds[1] = std::numeric_limits<float>::lowest();
    }

    void expand(float x, float y) {
        min_bounds[0] = std::min(min_bounds[0], x);
        min_bounds[1] = std::min(min_bounds[1], y);
        max_bounds[0] = std::max(max_bounds[0], x);
        max_bounds[1] = std::max(max_bounds[1], y);
    }

    void expand(const RTreeBoundingBox &other) {
        min_bounds[0] = std::min(min_bounds[0], other.min_bounds[0]);
        min_bounds[1] = std::min(min_bounds[1], other.min_bounds[1]);
        max_bounds[0] = std::max(max_bounds[0], other.max_bounds[0]);
        max_bounds[1] = std::max(max_bounds[1], other.max_bounds[1]);
    }

    bool overlaps(const BoundingBox &bbox) const {
        return !(max_bounds[0] < bbox.min_bounds[0] || min_bounds[0] > bbox.max_bounds[0] ||
                 max_bounds[1] < bbox.min_bounds[1] || min_bounds[1] > bbox.max_bounds[1]);
    }

    bool contains_point(float x, float y) const {
        return x >= min_bounds[0] && x <= max_bounds[0] &&
               y >= min_bounds[1] && y <= max_bounds[1];
    }
};

struct RTreeNode {
    RTreeBoundingBox mbr;
    bool is_leaf;

    // For internal nodes: children indices into nodes_ array
    std::vector<size_t> children;

    // For leaf nodes: vector IDs (global labels)
    std::vector<hnswlib::labeltype> vector_ids;

    // For leaf nodes with enough points: HNSW index
    hnswlib::HierarchicalNSWStatic<float> *hnsw_index = nullptr;

    ~RTreeNode() {
        if (hnsw_index != nullptr) {
            delete hnsw_index;
            hnsw_index = nullptr;
        }
    }

    // Non-copyable due to ownership of hnsw_index
    RTreeNode() : is_leaf(false) {}
    RTreeNode(RTreeNode &&other) noexcept
        : mbr(other.mbr), is_leaf(other.is_leaf),
          children(std::move(other.children)),
          vector_ids(std::move(other.vector_ids)),
          hnsw_index(other.hnsw_index) {
        other.hnsw_index = nullptr;
    }
    RTreeNode &operator=(RTreeNode &&other) noexcept {
        if (this != &other) {
            if (hnsw_index) delete hnsw_index;
            mbr = other.mbr;
            is_leaf = other.is_leaf;
            children = std::move(other.children);
            vector_ids = std::move(other.vector_ids);
            hnsw_index = other.hnsw_index;
            other.hnsw_index = nullptr;
        }
        return *this;
    }
    RTreeNode(const RTreeNode &) = delete;
    RTreeNode &operator=(const RTreeNode &) = delete;
};


class IndexRTreePartition {
private:
    std::vector<RTreeNode> nodes_;
    size_t root_idx_;

    std::vector<std::vector<float>> metadata_;  // [n][2]
    float *data_;           // raw vector data pointer
    size_t num_vectors_;
    size_t vec_dim_;
    size_t attr_dim_;       // always 2 for now

    size_t leaf_capacity_;          // max vectors per leaf (STR parameter)
    size_t leaf_scan_threshold_;    // leaves <= this size use brute-force
    size_t leaf_k_expand_factor_;   // retrieve more candidates per leaf before global merge
    size_t leaf_ef_min_;            // minimum ef used for per-leaf HNSW search
    size_t M_;
    size_t ef_construction_;
    size_t ef_;
    bool force_scan_;
    bool leaf_use_meta_filter_;    // true: filter during HNSW traversal, false: post-filter results

    hnswlib::SpaceInterface<float> *space_;

    size_t num_leaves_;
    size_t num_hnsw_leaves_;
    size_t num_scan_leaves_;

public:
    IndexRTreePartition(size_t leaf_capacity = 1000,
                        size_t leaf_scan_threshold = 100,
                                                size_t leaf_k_expand_factor = 4,
                                                size_t leaf_ef_min = 0,
                        bool leaf_use_meta_filter = false,
                        size_t M = 16,
                        size_t ef_construction = 200)
        : root_idx_(0), data_(nullptr), num_vectors_(0), vec_dim_(0), attr_dim_(2),
          leaf_capacity_(leaf_capacity), leaf_scan_threshold_(leaf_scan_threshold),
                    leaf_k_expand_factor_(leaf_k_expand_factor),
                    leaf_ef_min_(leaf_ef_min),
                      M_(M), ef_construction_(ef_construction), ef_(10), force_scan_(false),
                      leaf_use_meta_filter_(leaf_use_meta_filter), space_(nullptr),
          num_leaves_(0), num_hnsw_leaves_(0), num_scan_leaves_(0) {}

    ~IndexRTreePartition() {
        // RTreeNode destructors handle hnsw_index cleanup
        nodes_.clear();
    }

    void set_ef(size_t ef) {
        ef_ = ef;
        // Update ef on all leaf HNSW indexes
        for (auto &node : nodes_) {
            if (node.is_leaf && node.hnsw_index != nullptr) {
                node.hnsw_index->setEf(ef);
            }
        }
    }

    size_t get_num_leaves() const { return num_leaves_; }
    size_t get_num_hnsw_leaves() const { return num_hnsw_leaves_; }
    size_t get_num_scan_leaves() const { return num_scan_leaves_; }
    void set_force_scan(bool v) { force_scan_ = v; }
    void set_leaf_k_expand_factor(size_t v) { leaf_k_expand_factor_ = std::max<size_t>(1, v); }
    void set_leaf_ef_min(size_t v) { leaf_ef_min_ = v; }
    void set_leaf_use_meta_filter(bool v) { leaf_use_meta_filter_ = v; }


    // ========================================================================
    // Metadata loading
    // ========================================================================
    void load_metadata(const char *metadata_file) {
        std::ifstream in(metadata_file, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error(std::string("Cannot open metadata file: ") + metadata_file);
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

    // ========================================================================
    // STR bulk-loading: Sort-Tile-Recursive R-Tree construction
    // ========================================================================

    // Build R-Tree using STR on the given set of vector IDs
    // Returns index into nodes_ for the root of the subtree
    size_t str_build(std::vector<hnswlib::labeltype> &ids) {
        if (ids.size() <= leaf_capacity_) {
            // Create leaf node
            size_t idx = nodes_.size();
            nodes_.emplace_back();
            RTreeNode &leaf = nodes_.back();
            leaf.is_leaf = true;
            leaf.vector_ids = ids;

            // Compute MBR
            for (auto id : ids) {
                leaf.mbr.expand(metadata_[id][0], metadata_[id][1]);
            }

            num_leaves_++;
            return idx;
        }

        // STR: sort by dimension 0, split into S slices, then sort each slice
        // by dimension 1 and split into groups of leaf_capacity_.
        size_t n = ids.size();
        size_t num_leaves_target = (n + leaf_capacity_ - 1) / leaf_capacity_;
        size_t S = (size_t)std::ceil(std::sqrt((double)num_leaves_target));

        // Sort by dim 0
        std::sort(ids.begin(), ids.end(), [this](hnswlib::labeltype a, hnswlib::labeltype b) {
            return metadata_[a][0] < metadata_[b][0];
        });

        size_t slice_size = S * leaf_capacity_;
        std::vector<size_t> child_indices;

        for (size_t i = 0; i < n; i += slice_size) {
            size_t slice_end = std::min(i + slice_size, n);

            // Sort this slice by dim 1
            std::sort(ids.begin() + i, ids.begin() + slice_end,
                      [this](hnswlib::labeltype a, hnswlib::labeltype b) {
                          return metadata_[a][1] < metadata_[b][1];
                      });

            // Split into leaf-sized groups
            for (size_t j = i; j < slice_end; j += leaf_capacity_) {
                size_t group_end = std::min(j + leaf_capacity_, slice_end);
                std::vector<hnswlib::labeltype> group(ids.begin() + j, ids.begin() + group_end);
                size_t child_idx = str_build(group);
                child_indices.push_back(child_idx);
            }
        }

        if (child_indices.size() == 1) {
            return child_indices[0];
        }

        // Create internal node
        size_t idx = nodes_.size();
        nodes_.emplace_back();
        RTreeNode &internal = nodes_.back();
        internal.is_leaf = false;
        internal.children = child_indices;

        // Compute MBR from children
        for (auto ci : child_indices) {
            internal.mbr.expand(nodes_[ci].mbr);
        }

        return idx;
    }


    // ========================================================================
    // Build per-leaf HNSW indexes (or mark for brute-force scan)
    // ========================================================================
    void build_leaf_indexes() {
        // Collect leaf node indices
        std::vector<size_t> leaf_indices;
        for (size_t i = 0; i < nodes_.size(); i++) {
            if (nodes_[i].is_leaf) {
                leaf_indices.push_back(i);
            }
        }

        long long hnsw_leaves = 0;
        long long scan_leaves = 0;

#pragma omp parallel for schedule(dynamic, 144) reduction(+:hnsw_leaves, scan_leaves)
        for (size_t li = 0; li < leaf_indices.size(); li++) {
            size_t ni = leaf_indices[li];
            RTreeNode &leaf = nodes_[ni];
            size_t count = leaf.vector_ids.size();

            if (count <= leaf_scan_threshold_) {
                // Small leaf: brute-force scan, no HNSW
                leaf.hnsw_index = nullptr;
                scan_leaves++;
            } else {
                // Build HNSW for this leaf
                leaf.hnsw_index = new hnswlib::HierarchicalNSWStatic<float>(
                    space_, count, M_, ef_construction_);

                // Share global metadata pointer (no copy) so searchKnn can filter by external label
                leaf.hnsw_index->set_meta_dim(attr_dim_);
                leaf.hnsw_index->set_shared_metadata(&metadata_);

                for (auto id : leaf.vector_ids) {
                    leaf.hnsw_index->addPoint(data_ + (size_t)id * vec_dim_, id);
                }

                hnsw_leaves++;
            }
        }

        num_hnsw_leaves_ = static_cast<size_t>(hnsw_leaves);
        num_scan_leaves_ = static_cast<size_t>(scan_leaves);
    }


    // ========================================================================
    // Build the full index: load data, build R-Tree, build per-leaf HNSW
    // ========================================================================
    void build_index(char *data_file, char *metadata_file) {
        load_metadata(metadata_file);

        auto *X = new Matrix<float>(data_file);
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *)X->data;
        data_ = X->data;
        num_vectors_ = X->n;
        vec_dim_ = X->d;
        space_ = new hnswlib::L2Space(X->d);

        std::cout << "Building R-Tree (STR) with leaf_capacity=" << leaf_capacity_
                  << ", scan_threshold=" << leaf_scan_threshold_ << "..." << std::endl;

        // Prepare vector IDs
        std::vector<hnswlib::labeltype> all_ids(num_vectors_);
        std::iota(all_ids.begin(), all_ids.end(), 0);

        // Reserve space to avoid reallocation invalidating references
        size_t est_nodes = (num_vectors_ / leaf_capacity_) * 2 + 100;
        nodes_.reserve(est_nodes);

        root_idx_ = str_build(all_ids);

        std::cout << "R-Tree built: " << nodes_.size() << " nodes, "
                  << num_leaves_ << " leaves" << std::endl;

        std::cout << "Building per-leaf HNSW indexes..." << std::endl;
        build_leaf_indexes();

        std::cout << "Done. HNSW leaves: " << num_hnsw_leaves_
                  << ", scan leaves: " << num_scan_leaves_ << std::endl;
    }


    // ========================================================================
    // Search: traverse R-Tree, search matching leaves, merge results
    // ========================================================================

    // Brute-force scan a leaf node with filter
    void scan_leaf(const RTreeNode &leaf, const float *query, size_t k,
                   const BoundingBox &filter,
                   std::priority_queue<std::pair<float, hnswlib::labeltype>> &results) const {
        for (auto id : leaf.vector_ids) {
            // Check metadata filter
            const float *meta = metadata_[id].data();
            bool pass = true;
            for (size_t d = 0; d < attr_dim_; d++) {
                if (meta[d] < filter.min_bounds[d] || meta[d] > filter.max_bounds[d]) {
                    pass = false;
                    break;
                }
            }
            if (!pass) continue;

            float dist = sqr_dist(query, data_ + (size_t)id * vec_dim_, vec_dim_);

            if (results.size() < k) {
                results.emplace(dist, id);
            } else if (dist < results.top().first) {
                results.pop();
                results.emplace(dist, id);
            }
        }
    }

    // Search the index with a bounding box filter
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search(const float *query, size_t k, const BoundingBox &filter) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> results;  // max-heap

        // BFS/DFS traversal of R-Tree to find overlapping leaves
        std::vector<size_t> stack;
        stack.push_back(root_idx_);

        while (!stack.empty()) {
            size_t ni = stack.back();
            stack.pop_back();
            RTreeNode &node = nodes_[ni];

            if (!node.mbr.overlaps(filter)) {
                continue;
            }

            if (node.is_leaf) {
                if (node.hnsw_index != nullptr && !force_scan_) {
                    // Search HNSW with metadata filter
                    size_t local_k = std::max<size_t>(k, k * leaf_k_expand_factor_);
                    local_k = std::min(local_k, node.vector_ids.size());
                    size_t local_ef = std::max(ef_, leaf_ef_min_);
                    local_ef = std::max(local_ef, local_k);
                    local_ef = std::min(local_ef, node.vector_ids.size());
                    node.hnsw_index->setEf(local_ef);
                    std::priority_queue<std::pair<float, hnswlib::labeltype>> leaf_results;
                    if (leaf_use_meta_filter_) {
                        BBoxFilterMeta bboxFilter(filter, attr_dim_);
                        leaf_results = node.hnsw_index->searchKnn(query, local_k, &bboxFilter);
                    } else {
                        leaf_results = node.hnsw_index->searchKnn(query, local_k, (hnswlib::BaseFilterFunctor*)nullptr);
                    }

                    // Merge into global results
                    while (!leaf_results.empty()) {
                        auto &top = leaf_results.top();
                        if (!leaf_use_meta_filter_) {
                            const float *meta = metadata_[top.second].data();
                            bool pass = true;
                            for (size_t d = 0; d < attr_dim_; d++) {
                                if (meta[d] < filter.min_bounds[d] || meta[d] > filter.max_bounds[d]) {
                                    pass = false;
                                    break;
                                }
                            }
                            if (!pass) {
                                leaf_results.pop();
                                continue;
                            }
                        }
                        if (results.size() < k) {
                            results.push(top);
                        } else if (top.first < results.top().first) {
                            results.pop();
                            results.push(top);
                        }
                        leaf_results.pop();
                    }
                } else {
                    // Brute-force scan
                    scan_leaf(node, query, k, filter, results);
                }
            } else {
                // Internal node: push children
                for (auto ci : node.children) {
                    stack.push_back(ci);
                }
            }
        }

        return results;
    }

    // Get metadata for external use (groundtruth computation etc.)
    const std::vector<std::vector<float>> &get_metadata() const { return metadata_; }
    size_t get_attr_dim() const { return attr_dim_; }
    size_t get_num_vectors() const { return num_vectors_; }

private:
    // Metadata-based BBox filter for HierarchicalNSWStatic::searchKnn
    class BBoxFilterMeta : public hnswlib::MetaFilterFunctor {
    public:
        const BoundingBox &bbox_;
        size_t attr_dim_;

        BBoxFilterMeta(const BoundingBox &bbox, size_t attr_dim)
            : bbox_(bbox), attr_dim_(attr_dim) {}

        bool operator()(hnswlib::metatype *meta) override {
            for (size_t d = 0; d < attr_dim_; d++) {
                if (meta[d] < bbox_.min_bounds[d] || meta[d] > bbox_.max_bounds[d])
                    return false;
            }
            return true;
        }
    };
};
