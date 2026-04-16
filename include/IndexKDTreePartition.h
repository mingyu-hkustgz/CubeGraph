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
#include <atomic>
#include <chrono>
#include <iomanip>
#include <omp.h>

// KD-Tree partitioned HNSW index (baseline).
// Builds a standard KD-tree over N-dimensional metadata space.
// Every node (internal and leaf) stores all vector IDs in its subtree.
// Nodes with >= scan_threshold vectors get an HNSW-static index;
// smaller nodes use brute-force scan.
// At query time, traverse the KD-tree:
//   - If a node's bbox is fully contained by the filter, search its HNSW
//   - If partially overlapping, recurse to children
//   - At single-vector leaves, brute-force filter check
// Memory: O(N log N) — every vector appears in O(log N) nodes.

struct KDTreeBoundingBox {
    std::vector<float> min_bounds;
    std::vector<float> max_bounds;

    KDTreeBoundingBox() {}

    KDTreeBoundingBox(size_t dim) : min_bounds(dim, std::numeric_limits<float>::max()),
                                     max_bounds(dim, std::numeric_limits<float>::lowest()) {}

    void expand(const std::vector<float> &point) {
        for (size_t d = 0; d < min_bounds.size(); d++) {
            min_bounds[d] = std::min(min_bounds[d], point[d]);
            max_bounds[d] = std::max(max_bounds[d], point[d]);
        }
    }

    void expand(const KDTreeBoundingBox &other) {
        for (size_t d = 0; d < min_bounds.size(); d++) {
            min_bounds[d] = std::min(min_bounds[d], other.min_bounds[d]);
            max_bounds[d] = std::max(max_bounds[d], other.max_bounds[d]);
        }
    }

    bool overlaps(const BoundingBox &bbox) const {
        for (size_t d = 0; d < min_bounds.size(); d++) {
            if (max_bounds[d] < bbox.min_bounds[d] || min_bounds[d] > bbox.max_bounds[d])
                return false;
        }
        return true;
    }

    bool fully_contained_by(const BoundingBox &bbox) const {
        for (size_t d = 0; d < min_bounds.size(); d++) {
            if (min_bounds[d] < bbox.min_bounds[d] || max_bounds[d] > bbox.max_bounds[d])
                return false;
        }
        return true;
    }
};

struct KDTreeNode {
    KDTreeBoundingBox mbr;
    bool is_leaf;

    // For internal nodes
    size_t split_dim;
    float split_val;
    size_t left_child;   // index into nodes_ array
    size_t right_child;

    // ALL nodes store vector IDs for their entire subtree
    std::vector<hnswlib::labeltype> vector_ids;

    // Nodes with enough points get an HNSW index
    hnswlib::HierarchicalNSWStatic<float> *hnsw_index = nullptr;

    ~KDTreeNode() {
        if (hnsw_index != nullptr) {
            delete hnsw_index;
            hnsw_index = nullptr;
        }
    }

    KDTreeNode() : is_leaf(false), split_dim(0), split_val(0), left_child(0), right_child(0) {}
    KDTreeNode(KDTreeNode &&other) noexcept
        : mbr(std::move(other.mbr)), is_leaf(other.is_leaf),
          split_dim(other.split_dim), split_val(other.split_val),
          left_child(other.left_child), right_child(other.right_child),
          vector_ids(std::move(other.vector_ids)),
          hnsw_index(other.hnsw_index) {
        other.hnsw_index = nullptr;
    }
    KDTreeNode &operator=(KDTreeNode &&other) noexcept {
        if (this != &other) {
            if (hnsw_index) delete hnsw_index;
            mbr = std::move(other.mbr);
            is_leaf = other.is_leaf;
            split_dim = other.split_dim;
            split_val = other.split_val;
            left_child = other.left_child;
            right_child = other.right_child;
            vector_ids = std::move(other.vector_ids);
            hnsw_index = other.hnsw_index;
            other.hnsw_index = nullptr;
        }
        return *this;
    }
    KDTreeNode(const KDTreeNode &) = delete;
    KDTreeNode &operator=(const KDTreeNode &) = delete;
};


class IndexKDTreePartition {
private:
    std::vector<KDTreeNode> nodes_;
    size_t root_idx_;

    std::vector<std::vector<float>> metadata_;  // [n][attr_dim]
    float *data_;           // raw vector data pointer
    size_t num_vectors_;
    size_t vec_dim_;
    size_t attr_dim_;

    size_t scan_threshold_;         // nodes < this use brute-force (no HNSW)
    size_t M_;
    size_t ef_construction_;
    size_t ef_;
    bool force_scan_;

    hnswlib::SpaceInterface<float> *space_;

    size_t num_hnsw_nodes_;
    size_t num_scan_nodes_;

public:
        IndexKDTreePartition(size_t scan_threshold = 1000,
                                                 size_t M = 16,
                                                 size_t ef_construction = 200)
                : root_idx_(0), data_(nullptr), num_vectors_(0), vec_dim_(0), attr_dim_(2),
                    scan_threshold_(scan_threshold),
                    M_(M), ef_construction_(ef_construction), ef_(10), force_scan_(false),
                    space_(nullptr),
                    num_hnsw_nodes_(0), num_scan_nodes_(0) {}

    ~IndexKDTreePartition() {
        nodes_.clear();
    }

    void set_ef(size_t ef) {
        ef_ = ef;
        for (auto &node : nodes_) {
            if (node.hnsw_index != nullptr) {
                node.hnsw_index->setEf(ef);
            }
        }
    }

    size_t get_num_nodes() const { return nodes_.size(); }
    size_t get_num_hnsw_nodes() const { return num_hnsw_nodes_; }
    size_t get_num_scan_nodes() const { return num_scan_nodes_; }
    void set_force_scan(bool v) { force_scan_ = v; }
    // removed set_leaf_use_meta_filter

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
    // KD-Tree construction: split by dimension with largest spread at median.
    // Every node stores ALL vector IDs in its subtree.
    // Recursion stops when a single vector remains (true leaf).
    // ========================================================================
    size_t kd_build(std::vector<hnswlib::labeltype> &ids, size_t depth) {
        size_t n = ids.size();

        // Create node first — every node stores its subtree's vector_ids
        size_t idx = nodes_.size();
        nodes_.emplace_back();
        // Use index-based access (not references) since recursive calls may reallocate nodes_
        nodes_[idx].vector_ids = ids;
        nodes_[idx].mbr = KDTreeBoundingBox(attr_dim_);
        for (auto id : ids) {
            nodes_[idx].mbr.expand(metadata_[id]);
        }

        if (n <= 1) {
            // Single-vector leaf: no children
            nodes_[idx].is_leaf = true;
            return idx;
        }

        // Find dimension with largest spread
        size_t best_dim = 0;
        float best_spread = -1;
        for (size_t d = 0; d < attr_dim_; d++) {
            float lo = std::numeric_limits<float>::max();
            float hi = std::numeric_limits<float>::lowest();
            for (auto id : ids) {
                float v = metadata_[id][d];
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            float spread = hi - lo;
            if (spread > best_spread) {
                best_spread = spread;
                best_dim = d;
            }
        }

        // Sort by best dimension, split at median
        std::sort(ids.begin(), ids.end(), [&](hnswlib::labeltype a, hnswlib::labeltype b) {
            return metadata_[a][best_dim] < metadata_[b][best_dim];
        });

        size_t mid = n / 2;
        float split_val = metadata_[ids[mid]][best_dim];

        std::vector<hnswlib::labeltype> left_ids(ids.begin(), ids.begin() + mid);
        std::vector<hnswlib::labeltype> right_ids(ids.begin() + mid, ids.end());

        // Recurse (nodes_[idx] stays valid via index-based access)
        size_t left_child = kd_build(left_ids, depth + 1);
        size_t right_child = kd_build(right_ids, depth + 1);

        // Fill in internal node fields
        nodes_[idx].is_leaf = false;
        nodes_[idx].split_dim = best_dim;
        nodes_[idx].split_val = split_val;
        nodes_[idx].left_child = left_child;
        nodes_[idx].right_child = right_child;

        return idx;
    }

    // ========================================================================
    // Build HNSW indexes for all nodes with >= scan_threshold_ vectors
    // ========================================================================
    void build_node_indexes() {
        long long hnsw_nodes = 0;
        long long scan_nodes = 0;
        const size_t total_nodes = nodes_.size();
        const size_t report_step = std::max<size_t>(size_t(1), total_nodes / 200);
        std::atomic<size_t> processed_nodes{0};
        std::atomic<size_t> built_hnsw_nodes{0};
        std::atomic<size_t> scanned_nodes{0};
        std::atomic<size_t> next_report{report_step};
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nodes_.size(); i++) {
            KDTreeNode &node = nodes_[i];
            size_t count = node.vector_ids.size();

            if (count < scan_threshold_) {
                node.hnsw_index = nullptr;
                scan_nodes++;
                scanned_nodes.fetch_add(1, std::memory_order_relaxed);
            } else {
                node.hnsw_index = new hnswlib::HierarchicalNSWStatic<float>(
                    space_, count, M_, ef_construction_);

                node.hnsw_index->set_meta_dim(attr_dim_);
                node.hnsw_index->set_shared_metadata(&metadata_);

#pragma omp parallel for schedule(dynamic, 144)
                for (auto id : node.vector_ids) {
                    node.hnsw_index->addPoint(data_ + (size_t)id * vec_dim_, id);
                }

                hnsw_nodes++;
                built_hnsw_nodes.fetch_add(1, std::memory_order_relaxed);
            }

            size_t done = processed_nodes.fetch_add(1, std::memory_order_relaxed) + 1;
            size_t target = next_report.load(std::memory_order_relaxed);
            if (done >= target || done == total_nodes) {
#pragma omp critical(kdtree_progress)
                {
                    target = next_report.load(std::memory_order_relaxed);
                    if (done >= target || done == total_nodes) {
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - start).count();
                        double pct = total_nodes == 0 ? 100.0
                                                      : 100.0 * static_cast<double>(done) / static_cast<double>(total_nodes);
                        std::cout << "\r  Progress: "
                                  << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%"
                                  << " (" << done << "/" << total_nodes << " nodes"
                                  << ", HNSW: " << built_hnsw_nodes.load(std::memory_order_relaxed)
                                  << ", scan: " << scanned_nodes.load(std::memory_order_relaxed)
                                  << ", elapsed: " << elapsed << "s)" << std::flush;
                        if (done == total_nodes) {
                            std::cout << std::endl;
                        }
                        while (next_report.load(std::memory_order_relaxed) <= done) {
                            next_report.fetch_add(report_step, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }

        num_hnsw_nodes_ = static_cast<size_t>(hnsw_nodes);
        num_scan_nodes_ = static_cast<size_t>(scan_nodes);
    }

    // ========================================================================
    // Build the full index
    // ========================================================================
    void build_index(char *data_file, char *metadata_file) {
        load_metadata(metadata_file);

        auto *X = new Matrix<float>(data_file);
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *)X->data;
        data_ = X->data;
        num_vectors_ = X->n;
        vec_dim_ = X->d;
        space_ = new hnswlib::L2Space(X->d);

        std::cout << "Building KD-Tree (scan_threshold=" << scan_threshold_ << ")..." << std::endl;

        std::vector<hnswlib::labeltype> all_ids(num_vectors_);
        std::iota(all_ids.begin(), all_ids.end(), 0);

        // Complete binary tree has 2N-1 nodes
        size_t est_nodes = num_vectors_ * 2;
        nodes_.reserve(est_nodes);

        root_idx_ = kd_build(all_ids, 0);

        std::cout << "KD-Tree built: " << nodes_.size() << " total nodes" << std::endl;

        std::cout << "Building per-node HNSW indexes..." << std::endl;
        build_node_indexes();

        std::cout << "Done. HNSW nodes: " << num_hnsw_nodes_
                  << ", scan nodes: " << num_scan_nodes_ << std::endl;
    }

    // ========================================================================
    // Search
    // ========================================================================

    // Brute-force scan a node's vectors with filter check
    void scan_node_filtered(const KDTreeNode &node, const float *query, size_t k,
                            const BoundingBox &filter,
                            std::priority_queue<std::pair<float, hnswlib::labeltype>> &results) const {
        for (auto id : node.vector_ids) {
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

    // Brute-force scan a node's vectors WITHOUT filter (node fully contained in filter)
    void scan_node_unfiltered(const KDTreeNode &node, const float *query, size_t k,
                              std::priority_queue<std::pair<float, hnswlib::labeltype>> &results) const {
        for (auto id : node.vector_ids) {
            float dist = sqr_dist(query, data_ + (size_t)id * vec_dim_, vec_dim_);

            if (results.size() < k) {
                results.emplace(dist, id);
            } else if (dist < results.top().first) {
                results.pop();
                results.emplace(dist, id);
            }
        }
    }

    // Search a node's HNSW without filter (node fully contained in filter)
    void search_node_hnsw_unfiltered(KDTreeNode &node, const float *query, size_t k,
                                     std::priority_queue<std::pair<float, hnswlib::labeltype>> &results) {
        node.hnsw_index->setEf(ef_);

        auto node_results = node.hnsw_index->searchKnn(query, k, (hnswlib::BaseFilterFunctor*)nullptr);

        while (!node_results.empty()) {
            auto &top = node_results.top();
            if (results.size() < k) {
                results.push(top);
            } else if (top.first < results.top().first) {
                results.pop();
                results.push(top);
            }
            node_results.pop();
        }
    }

    // Search a node's HNSW with filter and merge results
    void search_node_hnsw_filtered(KDTreeNode &node, const float *query, size_t k,
                                   const BoundingBox &filter,
                                   std::priority_queue<std::pair<float, hnswlib::labeltype>> &results) {
        node.hnsw_index->setEf(ef_);

        std::priority_queue<std::pair<float, hnswlib::labeltype>> node_results;
        node_results = node.hnsw_index->searchKnn(query, k, (hnswlib::BaseFilterFunctor*)nullptr);

        while (!node_results.empty()) {
            auto &top = node_results.top();
            const float *meta = metadata_[top.second].data();
            bool pass = true;
            for (size_t d = 0; d < attr_dim_; d++) {
                if (meta[d] < filter.min_bounds[d] || meta[d] > filter.max_bounds[d]) {
                    pass = false;
                    break;
                }
            }
            if (!pass) {
                node_results.pop();
                continue;
            }
            if (results.size() < k) {
                results.push(top);
            } else if (top.first < results.top().first) {
                results.pop();
                results.push(top);
            }
            node_results.pop();
        }
    }

    // Search the index with a bounding box filter
    std::priority_queue<std::pair<float, hnswlib::labeltype>>
    search(const float *query, size_t k, const BoundingBox &filter) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> results;

        // DFS traversal of KD-tree
        std::vector<size_t> stack;
        stack.push_back(root_idx_);

        while (!stack.empty()) {
            size_t ni = stack.back();
            stack.pop_back();
            KDTreeNode &node = nodes_[ni];

            // Skip if no overlap
            if (!node.mbr.overlaps(filter)) {
                continue;
            }

            // If fully contained in filter: search this node directly (no filter needed)
            if (node.mbr.fully_contained_by(filter)) {
                if (node.hnsw_index != nullptr && !force_scan_) {
                    search_node_hnsw_unfiltered(node, query, k, results);
                } else {
                    scan_node_unfiltered(node, query, k, results);
                }
                continue;  // Don't recurse deeper — children are subset
            }

            // Partially overlapping
            if (node.is_leaf) {
                // Single-vector leaf: brute-force filter check
                scan_node_filtered(node, query, k, filter, results);
            } else {
                // Internal node: recurse to children
                stack.push_back(node.left_child);
                stack.push_back(node.right_child);
            }
        }

        return results;
    }

    const std::vector<std::vector<float>> &get_metadata() const { return metadata_; }
    size_t get_attr_dim() const { return attr_dim_; }
    size_t get_num_vectors() const { return num_vectors_; }

private:
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
