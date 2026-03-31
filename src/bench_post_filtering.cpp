#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include "hnswlib/hnswlib.h"
#include "matrix.h"
#include "utils.h"
#include <getopt.h>
#include "../third_party/config.h"

using namespace std;
using namespace hnswlib;

static void log_index_time(const char* dataset, const char* program, long long build_time_ms) {
    char log_path[512];
    sprintf(log_path, "./results/time-log/%s-%s.log", dataset, program);
    ofstream log_file(log_path, ios::app);
    log_file << "index_time_sec: " << (build_time_ms / 1000.0) << "\n";
    log_file.close();
}

const int MAXK = 100;
int efSearch = 20;
double outer_recall = 0;

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
};

BoundingBox generate_filter_bbox(const BoundingBox &global_bbox, float filter_ratio, size_t attr_dim, mt19937 &rng) {
    BoundingBox filter_bbox(attr_dim);
    for (size_t d = 0; d < attr_dim; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        float filter_size = range * sqrt(filter_ratio);
        uniform_real_distribution<float> dist(
                global_bbox.min_bounds[d] + filter_size / 2,
                global_bbox.max_bounds[d] - filter_size / 2);
        float center = dist(rng);
        filter_bbox.min_bounds[d] = center - filter_size / 2;
        filter_bbox.max_bounds[d] = center + filter_size / 2;
    }
    return filter_bbox;
}

vector<vector<float>> load_metadata(const char *metadata_file, size_t &n, size_t &d) {
    ifstream in(metadata_file, ios::binary);
    if (!in.is_open()) throw runtime_error("Cannot open metadata file");
    in.read(reinterpret_cast<char *>(&n), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&d), sizeof(size_t));
    vector<vector<float>> metadata(n, vector<float>(d));
    for (size_t i = 0; i < n; i++)
        in.read(reinterpret_cast<char *>(metadata[i].data()), d * sizeof(float));
    in.close();
    return metadata;
}

// POST filtering: search full HNSW with oversample, then filter results
static void test_approx_post(float *massQ, size_t vecsize, size_t qsize,
                              HierarchicalNSW<float> &appr_alg, size_t vecdim,
                              vector<priority_queue<pair<float, labeltype>>> &answers, size_t k,
                              vector<BoundingBox> &filters,
                              const vector<vector<float>> &metadata, size_t attr_dim) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;
    long double total_ratio = 0;

    for (int i = 0; i < (int)qsize; i++) {
        float sys_t, usr_t;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);

        // Oversample: retrieve k * oversample_factor candidates
        size_t oversample = max((size_t)k * 10, (size_t)1000);
        auto raw_result = appr_alg.searchKnn(massQ + vecdim * i, oversample);

        // Post-filter: keep only those passing the bounding box filter
        priority_queue<pair<float, labeltype>> result;
        while (!raw_result.empty() && result.size() < k) {
            auto [dist, label] = raw_result.top();
            raw_result.pop();
            const auto &meta = metadata[label];
            bool pass = true;
            for (size_t d = 0; d < attr_dim; d++) {
                if (meta[d] < filters[i].min_bounds[d] || meta[d] > filters[i].max_bounds[d]) {
                    pass = false;
                    break;
                }
            }
            if (pass) result.emplace(dist, label);
        }

        GetCurTime(&run_end);
        GetTime(&run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;

        priority_queue<pair<float, labeltype>> gt(answers[i]);
        total += gt.size();
        correct += recall(result, gt);
        total_ratio += Ratio(result, gt);
    }

    long double time_us_per_query = total_time / qsize;
    long double rec = 1.0f * correct / total;
    long double dist_ratio = total_ratio / qsize;

    cout << rec * 100.0 << " " << 1e6 / time_us_per_query << " " << dist_ratio << endl;
    cerr << rec * 100.0 << " " << 1e6 / time_us_per_query << " " << dist_ratio << endl;
    outer_recall = rec * 100;
}

static void test_vs_recall_post(float *massQ, size_t vecsize, size_t qsize,
                                 HierarchicalNSW<float> &appr_alg, size_t vecdim,
                                 vector<priority_queue<pair<float, labeltype>>> &answers, size_t k,
                                 vector<BoundingBox> &filters,
                                 const vector<vector<float>> &metadata, size_t attr_dim) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        if (efBase >= k) efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        test_approx_post(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, filters, metadata, attr_dim);
        if (outer_recall > 99.5) break;
    }
}

template<typename GT>
static void get_gt(Matrix<float> &Q, Matrix<float> &X, GT G,
                   vector<priority_queue<pair<float, labeltype>>> &answers, size_t subk) {
    (vector<priority_queue<pair<float, labeltype>>>(Q.n)).swap(answers);
    for (int i = 0; i < (int)Q.n; i++) {
        for (int j = 0; j < (int)subk; j++) {
            auto gt = G.data[G.d * i + j];
            if (gt < 0) break;
            answers[i].emplace(sqr_dist(Q.data + i * Q.d, X.data + gt * X.d, X.d), gt);
        }
    }
}

static Matrix<int> compute_filtered_gt(
        Matrix<float> &Q, Matrix<float> &X_base,
        const vector<BoundingBox> &filters,
        const vector<vector<float>> &metadata,
        size_t attr_dim, size_t k) {
    Matrix<int> G(Q.n, k);
#pragma omp parallel for schedule(dynamic, 144)
    for (size_t i = 0; i < Q.n; i++) {
        const float *query = Q.data + i * Q.d;
        const BoundingBox &fb = filters[i];
        vector<size_t> passed;
        for (size_t j = 0; j < X_base.n; j++) {
            const float *meta = metadata[j].data();
            bool ok = true;
            for (size_t d = 0; d < attr_dim; d++) {
                if (meta[d] < fb.min_bounds[d] || meta[d] > fb.max_bounds[d]) { ok = false; break; }
            }
            if (ok) passed.push_back(j);
        }
        vector<pair<float, size_t>> dists;
        dists.reserve(passed.size());
        for (size_t idx : passed)
            dists.emplace_back(sqr_dist(query, X_base.data + idx * X_base.d, Q.d), idx);
        if (dists.size() > k)
            nth_element(dists.begin(), dists.begin() + k, dists.end(),
                        [](const auto &a, const auto &b) { return a.first < b.first; });
        sort(dists.begin(), dists.begin() + min(k, dists.size()),
             [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t r = 0; r < k; r++)
            G.data[G.d * i + r] = (r < dists.size()) ? (int)dists[r].second : -1;
    }
    return G;
}

int main(int argc, char *argv[]) {
    int iarg = 0;
    char source[256] = "";
    char dataset[256] = "";
    char file_type[256] = "fvecs";

    const struct option longopts[] = {
        {"dataset", required_argument, 0, 'd'},
        {"source",  required_argument, 0, 's'},
        {0, 0, 0, 0}
    };
    int ind;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
        switch (iarg) {
            case 'd': if (optarg) strcpy(dataset, optarg); break;
            case 's': if (optarg) strcpy(source, optarg); break;
        }
    }

    char data_path[256], query_path[256], meta_path[256], result_path[256];
    sprintf(data_path,  "%s%s_base.%s",                source, dataset, file_type);
    sprintf(query_path, "%s%s_query.%s",               source, dataset, file_type);
    sprintf(meta_path,  "%s%s_metadata_uniform_2d.bin", source, dataset);

    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);

    size_t meta_n, attr_dim;
    auto metadata = load_metadata(meta_path, meta_n, attr_dim);

    // Compute global bounding box
    BoundingBox global_bbox(attr_dim);
    for (size_t d = 0; d < attr_dim; d++) {
        global_bbox.min_bounds[d] = numeric_limits<float>::max();
        global_bbox.max_bounds[d] = numeric_limits<float>::lowest();
    }
    for (const auto &m : metadata) {
        for (size_t d = 0; d < attr_dim; d++) {
            global_bbox.min_bounds[d] = min(global_bbox.min_bounds[d], m[d]);
            global_bbox.max_bounds[d] = max(global_bbox.max_bounds[d], m[d]);
        }
    }

    // Build standard HNSW index
    cout << "Building HNSW index (M=" << HNSW_M << ", ef=" << HNSW_efConstruction << ")..." << endl;
    auto start = chrono::high_resolution_clock::now();
    L2Space space(X.d);
    HierarchicalNSW<float> hnsw(&space, X.n, HNSW_M, HNSW_efConstruction);
#pragma omp parallel for schedule(dynamic, 144)
    for (size_t i = 0; i < X.n; i++)
        hnsw.addPoint(X.data + i * X.d, i);
    auto end = chrono::high_resolution_clock::now();
    auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Index built in " << build_time << " ms" << endl;
    log_index_time(dataset, "bench_post_filtering", build_time);

    // Generate random filters
    mt19937 rng(42);
    vector<BoundingBox> filters;
    for (size_t i = 0; i < Q.n; i++)
        filters.push_back(generate_filter_bbox(global_bbox, FILTER_RATIO, attr_dim, rng));

    // Compute filtered groundtruth
    cout << "Computing filtered groundtruth..." << endl;
    Matrix<int> G = compute_filtered_gt(Q, X, filters, metadata, attr_dim, 100);

    vector<priority_queue<pair<float, labeltype>>> answers;
    int K = 20;
    sprintf(result_path, "./results/recall@%d/%s/%s-post-bbox-%.2f.log", K, dataset, dataset, FILTER_RATIO);
    freopen(result_path, "a", stdout);
    get_gt(Q, X, G, answers, K);
    test_vs_recall_post(Q.data, X.n, Q.n, hnsw, Q.d, answers, K, filters, metadata, attr_dim);

    answers.clear();
    K = 100;
    sprintf(result_path, "./results/recall@%d/%s/%s-post-bbox-%.2f.log", K, dataset, dataset, FILTER_RATIO);
    freopen(result_path, "a", stdout);
    get_gt(Q, X, G, answers, K);
    test_vs_recall_post(Q.data, X.n, Q.n, hnsw, Q.d, answers, K, filters, metadata, attr_dim);

    return 0;
}
