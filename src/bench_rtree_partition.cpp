#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include "hnsw-static.h"
#include "matrix.h"
#include "utils.h"
#include <getopt.h>
#include "config.h"
#include "IndexCube.h"
#include "IndexRTreePartition.h"

using namespace std;
using namespace hnswlib;

static void log_index_time(const char* dataset, const char* program, long long build_time_ms) {
    char log_path[512];
    sprintf(log_path, "./results/time-log/%s/%s-%s.log", dataset, dataset, program);
    ofstream log_file(log_path, ios::app);
    log_file << "index_time_sec: " << (build_time_ms / 1000.0) << "\n";
    log_file.close();
}

const int MAXK = 100;

int efSearch = 20;
double outer_recall = 0;

// Generate filter bounding box centered at a random point
BoundingBox generate_filter_bbox(const BoundingBox &global_bbox, float filter_ratio, size_t meta_dim, mt19937 &rng) {
    BoundingBox filter_bbox(meta_dim);

    for (size_t d = 0; d < meta_dim; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        float filter_size = range * sqrt(filter_ratio);

        uniform_real_distribution<float> dist(
                global_bbox.min_bounds[d] + filter_size / 2,
                global_bbox.max_bounds[d] - filter_size / 2
        );
        float center = dist(rng);

        filter_bbox.min_bounds[d] = center - filter_size / 2;
        filter_bbox.max_bounds[d] = center + filter_size / 2;
    }

    return filter_bbox;
}


template<typename GT>
static void
get_gt(Matrix<float> &Q, Matrix<float> &X, GT G, vector<std::priority_queue<std::pair<float, labeltype >>> &answers,
       size_t subk) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(Q.n)).swap(answers);
    for (int i = 0; i < Q.n; i++) {
        for (int j = 0; j < subk; j++) {
            auto gt = G.data[G.d * i + j];
            if (gt < 0) break;
            answers[i].emplace(sqr_dist(Q.data + i * Q.d, X.data + gt * X.d, X.d), gt);
        }
    }
}

static void test_approx(float *massQ, size_t vecsize, size_t qsize, IndexRTreePartition &appr_alg,
                        size_t vecdim,
                        vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                        vector<BoundingBox> &filters) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;
    long double total_ratio = 0;

    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);
#endif

        std::priority_queue<std::pair<float, labeltype >> result =
            appr_alg.search(massQ + vecdim * i, k, filters[i]);

#ifndef WIN32
        GetCurTime(&run_end);
        GetTime(&run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        total_ratio += Ratio(result, gt);
        correct += tmp;
    }
    long double time_us_per_query = total_time / qsize;
    long double recall_val = 1.0f * correct / total;
    long double dist_ratio = total_ratio / qsize;

    cout << recall_val * 100.0 << " " << 1e6 / (time_us_per_query) << " " << dist_ratio << endl;
    cerr << recall_val * 100.0 << " " << 1e6 / (time_us_per_query) << " " << dist_ratio << endl;
    outer_recall = recall_val * 100;
    return;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, IndexRTreePartition &appr_alg,
                           size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                           vector<BoundingBox> &filters) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        if(efBase >= k) efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.set_ef(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, filters);
        if (outer_recall > 99.5) break;
    }
}


// Compute filtered groundtruth
static Matrix<int> compute_filtered_gt(
        Matrix<float> &Q,
        Matrix<float> &X_base,
        const std::vector<BoundingBox> &filters,
        const std::vector<std::vector<float>> &metadata,
        size_t meta_dim,
        size_t k) {
    Matrix<int> G(Q.n, k);
    double total_selectivity = 0.0;

#pragma omp parallel for schedule(dynamic, 144) reduction(+:total_selectivity)
    for (size_t i = 0; i < Q.n; i++) {
        const float *query = Q.data + i * Q.d;
        const BoundingBox &fb = filters[i];

        std::vector<size_t> passed;
        passed.reserve(X_base.n);
        for (size_t j = 0; j < X_base.n; j++) {
            const float *meta = metadata[j].data();
            bool ok = true;
            for (size_t d = 0; d < meta_dim; d++) {
                float v = meta[d];
                if (v < fb.min_bounds[d] || v > fb.max_bounds[d]) {
                    ok = false;
                    break;
                }
            }
            if (ok) passed.push_back(j);
        }

        double selectivity = (double)passed.size() / X_base.n;
        total_selectivity += selectivity;

        if (passed.size() <= k) {
            std::vector<std::pair<float, size_t>> dists;
            dists.reserve(passed.size());
            for (size_t idx: passed) {
                float d = sqr_dist(query, X_base.data + idx * X_base.d, Q.d);
                dists.emplace_back(d, idx);
            }
            std::sort(dists.begin(), dists.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            size_t r = 0;
            for (auto &p: dists) G.data[G.d * i + r++] = (int) p.second;
            for (; r < k; r++) G.data[G.d * i + r] = -1;
        } else {
            std::vector<std::pair<float, size_t>> dists;
            dists.reserve(passed.size());
            for (size_t idx: passed) {
                float d = sqr_dist(query, X_base.data + idx * X_base.d, Q.d);
                dists.emplace_back(d, idx);
            }
            std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
                             [](const auto &a, const auto &b) { return a.first < b.first; });
            std::sort(dists.begin(), dists.begin() + k,
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (size_t r = 0; r < k; r++)
                G.data[G.d * i + r] = (int) dists[r].second;
        }
    }
    double avg_selectivity = total_selectivity / Q.n;
    cout << "Average filter selectivity: " << avg_selectivity << endl;
    cerr << "Average filter selectivity: " << avg_selectivity << endl;
    return G;
}

// Load metadata from binary file
static std::vector<std::vector<float>> load_metadata(const char *path, size_t &n, size_t &dim) {
    std::vector<std::vector<float>> metadata;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        cerr << "Cannot open metadata file: " << path << endl;
        return metadata;
    }
    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
    cerr << "  Metadata file header: n=" << n << ", d=" << dim << endl;

    metadata.resize(n, std::vector<float>(dim));
    for (size_t i = 0; i < n; i++) {
        in.read(reinterpret_cast<char*>(metadata[i].data()), dim * sizeof(float));
    }
    in.close();
    return metadata;
}

// Compute global bounding box from metadata
static BoundingBox compute_global_bbox(const std::vector<std::vector<float>> &metadata, size_t meta_dim) {
    BoundingBox bbox(meta_dim);
    for (size_t d = 0; d < meta_dim; d++) {
        bbox.min_bounds[d] = std::numeric_limits<float>::max();
        bbox.max_bounds[d] = std::numeric_limits<float>::lowest();
    }
    for (size_t i = 0; i < metadata.size(); i++) {
        for (size_t d = 0; d < meta_dim; d++) {
            float v = metadata[i][d];
            bbox.min_bounds[d] = std::min(bbox.min_bounds[d], v);
            bbox.max_bounds[d] = std::max(bbox.max_bounds[d], v);
        }
    }
    return bbox;
}


int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            {"help",                no_argument,       0, 'h'},
            {"randomized",          required_argument, 0, 'd'},
            {"k",                   required_argument, 0, 'k'},
            {"epsilon0",            required_argument, 0, 'e'},
            {"gap",                 required_argument, 0, 'p'},
            {"dataset",             required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"query_path",          required_argument, 0, 'q'},
            {"groundtruth_path",    required_argument, 0, 'g'},
            {"result_path",         required_argument, 0, 'r'},
            {"transformation_path", required_argument, 0, 't'},
            {"meta",                required_argument, 0, 'm'},
            {"filter-ratio",        required_argument, 0, 'f'},
            {"leaf-capacity",       required_argument, 0, 'l'},
                {"leaf-k-expand",       required_argument, 0, 'x'},
                {"leaf-ef-min",         required_argument, 0, 'E'},
                    {"leaf-meta-filter",    no_argument,       0, 'P'},
                {"max-queries",         required_argument, 0, 'u'},
    };

    int ind;
    int iarg = 0, K = 10;
    opterr = 1;
    char source[256] = "";
    char dataset[256] = "";
    char index_path[256] = "";
    char query_path[256] = "";
    char data_path[256] = "";
    char meta_path[256] = "";
    char result_path[256] = "";
    char file_type[256] = "fvecs";
    char meta[256] = "uniform_2d";
    float filter_ratio = FILTER_RATIO;
    size_t leaf_capacity = 1000;
    size_t leaf_k_expand = 4;
    size_t leaf_ef_min = 0;
    bool leaf_meta_filter = false;
    size_t max_queries = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:r:f:m:l:x:E:Pu:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) strcpy(dataset, optarg);
                break;
            case 's':
                if (optarg) strcpy(source, optarg);
                break;
            case 'f':
                if (optarg) filter_ratio = atof(optarg);
                break;
            case 'm':
                if (optarg) strcpy(meta, optarg);
                break;
            case 'l':
                if (optarg) leaf_capacity = atoi(optarg);
                break;
            case 'x':
                if (optarg) leaf_k_expand = atoi(optarg);
                break;
            case 'E':
                if (optarg) leaf_ef_min = atoi(optarg);
                break;
            case 'P':
                leaf_meta_filter = true;
                break;
            case 'u':
                if (optarg) max_queries = atoi(optarg);
                break;
        }
    }

    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
    sprintf(query_path, "%s%s_query.%s", source, dataset, file_type);
    sprintf(meta_path, "%s%s_metadata_%s.bin", source, dataset, meta);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    size_t query_count = (max_queries > 0) ? std::min<size_t>(Q.n, max_queries) : Q.n;
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;

    size_t M = HNSW_M;
    size_t ef_construction = HNSW_efConstruction;
    size_t leaf_scan_threshold = 100;

    // Load metadata for global bbox and groundtruth
    cout << "Loading metadata from " << meta_path << "..." << endl;
    size_t n_vectors, meta_dim;
    std::vector<std::vector<float>> metadata = load_metadata(meta_path, n_vectors, meta_dim);
    cout << "  Loaded metadata for " << metadata.size() << " vectors, dim=" << meta_dim << endl;

    BoundingBox global_bbox = compute_global_bbox(metadata, meta_dim);
    cout << "  Global bbox: ";
    for (size_t d = 0; d < meta_dim; d++) {
        cout << "[" << global_bbox.min_bounds[d] << ", " << global_bbox.max_bounds[d] << "] ";
    }
    cout << endl;

    // Build R-Tree partitioned index
    IndexRTreePartition index(leaf_capacity, leaf_scan_threshold, leaf_k_expand, leaf_ef_min,
                              leaf_meta_filter, M, ef_construction);

    cout << "Building R-Tree partitioned index (leaf_capacity=" << leaf_capacity
         << ", scan_threshold=" << leaf_scan_threshold << ")..." << endl;
    auto start = chrono::high_resolution_clock::now();
    index.build_index(data_path, meta_path);
    auto end = chrono::high_resolution_clock::now();
    auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Index built in " << build_time << " ms" << endl;
    log_index_time(dataset, "bench_rtree_partition", build_time);

    cout << "  Leaves: " << index.get_num_leaves()
         << " (HNSW: " << index.get_num_hnsw_leaves()
         << ", scan: " << index.get_num_scan_leaves() << ")" << endl;

    // Generate random filters
    vector<BoundingBox> filters;
    cout << "Generating random filters (ratio=" << filter_ratio << ", seed=42)..." << endl;
    mt19937 rng(42);
    for (size_t i = 0; i < query_count; i++) {
        filters.push_back(generate_filter_bbox(global_bbox, filter_ratio, meta_dim, rng));
    }
    cout << "  Generated " << filters.size() << " filters" << endl;

    // Compute filtered groundtruth
    cout << "Computing filtered groundtruth (k=" << 100 << ")..." << endl;
    Matrix<float> Q_eval(query_count, Q.d);
    memcpy(Q_eval.data, Q.data, query_count * Q.d * sizeof(float));
    Matrix<int> G = compute_filtered_gt(Q_eval, X, filters, metadata, meta_dim, 100);
    cout << "  Done." << endl;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    // Evaluate recall@20
    K = 20;
    sprintf(result_path, "./results/recall@%d/%s/%s-rtree-partition-%s-%.2f.log", K, dataset, dataset, meta, filter_ratio);
    freopen(result_path, "a", stdout);
    get_gt(Q_eval, X, G, answers, K);
    test_vs_recall(Q_eval.data, X.n, query_count, index, Q_eval.d, answers, K, filters);
    answers.clear();

    // Evaluate recall@100
    K = 100;
    sprintf(result_path, "./results/recall@%d/%s/%s-rtree-partition-%s-%.2f.log", K, dataset, dataset, meta, filter_ratio);
    freopen(result_path, "a", stdout);
    get_gt(Q_eval, X, G, answers, K);
    test_vs_recall(Q_eval.data, X.n, query_count, index, Q_eval.d, answers, K, filters);

    return 0;
}
