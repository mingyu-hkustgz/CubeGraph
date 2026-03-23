#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include "hnsw-cube.h"
#include "matrix.h"
#include "utils.h"
#include <getopt.h>
#include "config.h"
#include "IndexCube.h"
using namespace std;
using namespace hnswlib;

const int MAXK = 100;


int efSearch = 100;
double outer_recall = 0;

// Generate filter bounding box centered at a random point
BoundingBox generate_filter_bbox(const BoundingBox& global_bbox, float filter_ratio, size_t attr_dim, mt19937& rng) {
    BoundingBox filter_bbox(attr_dim);

    for (size_t d = 0; d < attr_dim; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        float filter_size = range * sqrt(filter_ratio);  // sqrt for 2D to get linear dimension

        // Random center within valid range
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
static void get_gt(Matrix<float> &Q, Matrix<float> &X, GT G, vector<std::priority_queue<std::pair<float, labeltype >>> &answers,
                   size_t subk) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(Q.n)).swap(answers);
    for (int i = 0; i < Q.n; i++) {
        for (int j = 0; j < subk; j++) {
            auto gt = G.data[G.d * i + j];
            if (gt < 0) break;  // no more valid ground-truth entries
            answers[i].emplace(sqr_dist(Q.data + i * Q.d ,X.data + gt * X.d, X.d), gt);
        }
    }
}

static void test_approx(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg, size_t vecdim,
                        vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;
    long double total_ratio = 0;
    size_t dist_count = 0;
    std::vector<tableint> cubelist;
    for(int i=0;i<CUBE;i++) cubelist.push_back(i);
    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);
#endif

        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.search(massQ + vecdim * i, k, 0);
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
    long double recall = 1.0f * correct / total;
    long double dist_ratio =  total_ratio/ qsize;

    cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << " "<<dist_ratio<< endl;
    cerr << recall * 100.0 << " " << 1e6 / (time_us_per_query) << " "<<dist_ratio<< endl;
    outer_recall = recall * 100;
    return;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.set_global_ef(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        if(outer_recall > 99.5) break;
    }
}

// Compute filtered groundtruth: for each query, find top-k L2 nearest base vectors
// among those whose metadata lies within the query's random bounding-box filter.
// Returns Matrix<int> G where G.data[G.d * q + r] = ID of r-th nearest neighbor
// for query q (same layout as ivecs-loaded Matrix<unsigned>).
static Matrix<int> compute_filtered_gt(
    Matrix<float> &Q,
    Matrix<float> &X_base,
    const std::vector<BoundingBox> &filters,
    const std::vector<std::vector<float>> &metadata,
    size_t attr_dim,
    size_t k)
{
    Matrix<int> G(Q.n, k);
#pragma omp parallel for schedule(dynamic, 144)
    for (size_t i = 0; i < Q.n; i++) {
        const float *query = Q.data + i * Q.d;
        const BoundingBox &fb = filters[i];

        // Pass 1: collect indices of base vectors passing the filter
        std::vector<size_t> passed;
        passed.reserve(X_base.n);
        for (size_t j = 0; j < X_base.n; j++) {
            const float *meta = metadata[j].data();
            bool ok = true;
            for (size_t d = 0; d < attr_dim; d++) {
                float v = meta[d];
                if (v < fb.min_bounds[d] || v > fb.max_bounds[d]) {
                    ok = false;
                    break;
                }
            }
            if (ok) passed.push_back(j);
        }

        // Pass 2: compute L2 distances and find top-k
        if (passed.size() <= k) {
            std::vector<std::pair<float, size_t>> dists;
            dists.reserve(passed.size());
            for (size_t idx : passed) {
                float d = sqr_dist(query, X_base.data + idx * X_base.d, Q.d);
                dists.emplace_back(d, idx);
            }
            std::sort(dists.begin(), dists.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            size_t r = 0;
            for (auto &p : dists) G.data[G.d * i + r++] = (int)p.second;
            for (; r < k; r++) G.data[G.d * i + r] = -1;
        } else {
            std::vector<std::pair<float, size_t>> dists;
            dists.reserve(passed.size());
            for (size_t idx : passed) {
                float d = sqr_dist(query, X_base.data + idx * X_base.d, Q.d);
                dists.emplace_back(d, idx);
            }
            // O(n) partial sort: nth_element to find top-k, then sort top-k
            std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
                             [](const auto &a, const auto &b) { return a.first < b.first; });
            std::sort(dists.begin(), dists.begin() + k,
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (size_t r = 0; r < k; r++)
                G.data[G.d * i + r] = (int)dists[r].second;
        }
    }
    return G;
}

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",                no_argument,       0, 'h'},

            // Query Parameter
            {"randomized",          required_argument, 0, 'd'},
            {"k",                   required_argument, 0, 'k'},
            {"epsilon0",            required_argument, 0, 'e'},
            {"gap",                 required_argument, 0, 'p'},

            // Indexing Path
            {"dataset",             required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"query_path",          required_argument, 0, 'q'},
            {"groundtruth_path",    required_argument, 0, 'g'},
            {"result_path",         required_argument, 0, 'r'},
            {"transformation_path", required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0, K = 10, num_thread = 1;
    opterr = 1; //getopt error message (off: 0)
    char source[256] = "";
    char dataset[256] = "";
    char index_path[256] = "";
    char query_path[256] = "";
    char data_path[256] = "";
    char meta_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char file_type[256] = "fvecs";
    int subk = 100;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:k:r", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if (optarg) {
                    strcpy(source, optarg);
                }
                break;
            case 'k':
                if (optarg) K = atoi(optarg);
                break;
        }
    }
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
    sprintf(query_path, "%s%s_query.%s", source, dataset, file_type);
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    sprintf(meta_path, "%s%s_metadata_uniform_2d.bin", source, dataset);
    sprintf(result_path, "./results/recall@%d/%s/%s-hnsw-cube-merge-layer-%d.log", K, dataset, dataset, 0);
    sprintf(index_path, "%s%s.cube", source, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    hnswlib::HierarchicalNSWCube<float>::static_base_data_ = (char *) X.data;

    size_t num_layers = 3;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t cross_edge_count = 2;

    auto start = chrono::high_resolution_clock::now();
    IndexCube index(num_layers, M, ef_construction, cross_edge_count);
    index.build_index(data_path, meta_path);
    auto end = chrono::high_resolution_clock::now();
    auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Index built in " << build_time << " ms" << endl;

//    vector<BoundingBox> filters;
//    cout << "Generating random filters (ratio=" << FILTER_RATIO << ", seed=42)..." << endl;
//    mt19937 rng(42);
//    BoundingBox global_bbox = index.get_global_bbox();
//    for (size_t i = 0; i < Q.n; i++) {
//        filters.push_back(generate_filter_bbox(global_bbox, FILTER_RATIO, index.get_meta_dim(), rng));
//    }
//    cout << "  Generated " << filters.size() << " filters" << endl;
//
//
//    cout << "Computing filtered groundtruth (k=" << subk << ")..." << endl;
//    Matrix<int> G = compute_filtered_gt(Q, X, filters,
//                                        index.get_metadata(),
//                                        index.get_meta_dim(), subk);
//    cout << "  Done." << endl;

    Matrix<unsigned> G(groundtruth_path);
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    freopen(result_path, "a", stdout);
    get_gt(Q, X, G, answers, subk);
    test_vs_recall(Q.data, X.n, Q.n, index, Q.d, answers, subk);
    return 0;
}