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

// Generate filter radius (sphere) centered at a random point
RadiusFilterParams generate_filter_radius(const BoundingBox &global_bbox, float filter_ratio, size_t attr_dim, mt19937 &rng) {
    std::vector<float> center(attr_dim);
    float max_range = 0.0f;

    for (size_t d = 0; d < attr_dim; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        max_range = std::max(max_range, range);
        // Random center within valid range
        uniform_real_distribution<float> dist(
                global_bbox.min_bounds[d] + range * 0.1f,
                global_bbox.max_bounds[d] - range * 0.1f
        );
        center[d] = dist(rng);
    }

    // Radius based on filter_ratio (fraction of volume in N-dimensional space)
    // Volume of N-ball scales as r^N, so we need pow(filter_ratio, 1/dim) for correct radius
    float radius = max_range * pow(filter_ratio, 1.0f / attr_dim) / 2.0f;

    return RadiusFilterParams(center, radius, attr_dim);
}


template<typename GT>
static void
get_gt(Matrix<float> &Q, Matrix<float> &X, GT G, vector<std::priority_queue<std::pair<float, labeltype >>> &answers,
       size_t subk) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(Q.n)).swap(answers);
    for (int i = 0; i < Q.n; i++) {
        for (int j = 0; j < subk; j++) {
            auto gt = G.data[G.d * i + j];
            if (gt < 0) break;  // no more valid ground-truth entries
            answers[i].emplace(sqr_dist(Q.data + i * Q.d, X.data + gt * X.d, X.d), gt);
        }
    }
}

static void test_approx_radius(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg, size_t vecdim,
                        vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                        vector<RadiusFilterParams> &filters) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;
    long double total_ratio = 0;
#ifdef COLLECT_LOG
    appr_alg.reset_metrics();
#endif
    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);
#endif

        std::priority_queue<std::pair<float, labeltype >> result =
#if USE_FLY_SEARCH
            appr_alg.fly_search(massQ + vecdim * i, k, &filters[i]);
#else
            appr_alg.predetermined_search(massQ + vecdim * i, k, &filters[i]);
#endif
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
    long double dist_ratio = total_ratio / qsize;

    cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << " " << dist_ratio << endl;
    cerr << recall * 100.0 << " " << 1e6 / (time_us_per_query) << " " << dist_ratio << endl;
#ifdef COLLECT_LOG
    const auto& m = appr_alg.get_metrics();
    cerr << "Metrics: avg_layer=" << m.avg_layer()
         << " avg_cubes_visited=" << m.avg_cubes_visited()
         << " avg_selectivity=" << m.avg_selectivity()
         << " avg_cross_cube_edges=" << m.avg_cross_cube_edges()
         << " avg_distance_computations=" << m.avg_distance_computations()
         << " avg_hops=" << m.avg_hops() << endl;
#endif
    outer_recall = recall * 100;
    return;
}

static void test_vs_recall_radius(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                           vector<RadiusFilterParams> &filters) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        if(efBase >= k) efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.set_global_ef(ef);
        test_approx_radius(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, filters);
        if (outer_recall > 99.5) break;
    }
}

// Compute filtered groundtruth with radius (sphere) filter
static Matrix<int> compute_filtered_gt_radius(
        Matrix<float> &Q,
        Matrix<float> &X_base,
        const std::vector<RadiusFilterParams> &filters,
        const std::vector<std::vector<float>> &metadata,
        size_t attr_dim,
        size_t k) {
    Matrix<int> G(Q.n, k);
    double total_selectivity = 0.0;

#pragma omp parallel for schedule(dynamic, 144) reduction(+:total_selectivity)
    for (size_t i = 0; i < Q.n; i++) {
        const float *query = Q.data + i * Q.d;
        const RadiusFilterParams &fr = filters[i];
        const Sphere &sphere = fr.sphere;
        float radius2 = sphere.radius * sphere.radius;

        // Pass 1: collect indices of base vectors passing the filter
        std::vector<size_t> passed;
        passed.reserve(X_base.n);
        for (size_t j = 0; j < X_base.n; j++) {
            const float *meta = metadata[j].data();
            float d2 = 0.0f;
            for (size_t d = 0; d < attr_dim; d++) {
                float diff = meta[d] - sphere.center[d];
                d2 += diff * diff;
            }
            if (d2 <= radius2) {
                passed.push_back(j);
            }
        }

        // Compute selectivity for this query
        double selectivity = (double)passed.size() / X_base.n;
        total_selectivity += selectivity;

        // Pass 2: compute L2 distances and find top-k
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
            {"meta",                required_argument, 0, 'm'},

            // Filter Parameter
            {"filter-ratio",        required_argument, 0, 'f'},
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
    char meta[256] = "uniform_2d";  // default metadata type
    float filter_ratio = FILTER_RATIO;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:r:f:m:", longopts, &ind);
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
            case 'f':
                if (optarg) {
                    filter_ratio = atof(optarg);
                }
                break;
            case 'm':
                if (optarg) {
                    strcpy(meta, optarg);
                }
                break;
        }
    }
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
    sprintf(query_path, "%s%s_query.%s", source, dataset, file_type);
    sprintf(meta_path, "%s%s_metadata_%s.bin", source, dataset, meta);
    sprintf(index_path, "%s%s_%s.cube", source, meta, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    hnswlib::HierarchicalNSWCube<float>::static_base_data_ = (char *) X.data;

    size_t num_layers = 6;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t cross_edge_count = 2;

    IndexCube index(num_layers, M, ef_construction, cross_edge_count);

    if (isFileExists_ifstream(index_path)) {
        cout << "Loading existing index from " << index_path << "..." << endl;
        auto start = chrono::high_resolution_clock::now();
        index.load_index(index_path, data_path);
        auto end = chrono::high_resolution_clock::now();
        auto load_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Index loaded in " << load_time << " ms" << endl;
        log_index_time(dataset, "bench_hierarchical_ball", load_time);
    } else {
        cout << "Building new index..." << endl;
        auto start = chrono::high_resolution_clock::now();
        index.build_index(data_path, meta_path);
        auto end = chrono::high_resolution_clock::now();
        auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Index built in " << build_time << " ms" << endl;
        log_index_time(dataset, "bench_hierarchical_ball", build_time);

        cout << "Saving index to " << index_path << "..." << endl;
        index.save_index(index_path);
    }

    // Test with radius filters
    vector<RadiusFilterParams> radius_filters;
    cout << "Generating random radius filters (ratio=" << filter_ratio << ", seed=42)..." << endl;
    mt19937 rng(42);
    BoundingBox global_bbox = index.get_global_bbox();
    for (size_t i = 0; i < Q.n; i++) {
        radius_filters.push_back(generate_filter_radius(global_bbox, filter_ratio, index.get_meta_dim(), rng));
    }
    cout << "  Generated " << radius_filters.size() << " radius filters" << endl;

    cout << "Computing radius-filtered groundtruth (k=" << 100 << ")..." << endl;
    Matrix<int> G_radius = compute_filtered_gt_radius(Q, X, radius_filters,
                                        index.get_metadata(),
                                        index.get_meta_dim(), 100);
    cout << "  Done." << endl;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    K = 20;
    sprintf(result_path, "./results/recall@%d/%s/%s-hnsw-cube-merge-layer-radius-%s-%.2f.log", K, dataset, dataset, meta, filter_ratio);
    freopen(result_path, "w", stdout);
    get_gt(Q, X, G_radius, answers, K);
    test_vs_recall_radius(Q.data, X.n, Q.n, index, Q.d, answers, K, radius_filters);
    answers.clear();
    K = 100;
    sprintf(result_path, "./results/recall@%d/%s/%s-hnsw-cube-merge-layer-radius-%s-%.2f.log", K, dataset, dataset, meta, filter_ratio);
    freopen(result_path, "w", stdout);
    get_gt(Q, X, G_radius, answers, K);
    test_vs_recall_radius(Q.data, X.n, Q.n, index, Q.d, answers, K, radius_filters);

    return 0;
}
