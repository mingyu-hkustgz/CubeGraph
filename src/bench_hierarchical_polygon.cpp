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

// Generate a random polygon with specified number of vertices
// The polygon is centered at a random point and has area determined by filter_ratio
PolygonFilterParams generate_filter_polygon(const BoundingBox &global_bbox, float filter_ratio, size_t num_vertices, size_t attr_dim, mt19937 &rng) {
    // For polygon, we use 2D
    if (attr_dim < 2) attr_dim = 2;

    // Generate center within valid range
    vector<float> center(2);
    for (size_t d = 0; d < 2; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        uniform_real_distribution<float> dist(
                global_bbox.min_bounds[d] + range * 0.2f,
                global_bbox.max_bounds[d] - range * 0.2f
        );
        center[d] = dist(rng);
    }

    // Target area based on filter_ratio
    // We compute the equivalent area and then determine the polygon radius
    float global_area = 1.0f;
    for (size_t d = 0; d < 2 && d < attr_dim; d++) {
        global_area *= (global_bbox.max_bounds[d] - global_bbox.min_bounds[d]);
    }
    float target_area = global_area * filter_ratio;

    // Generate random polygon vertices
    // Use a simple approach: generate random angles and radii
    vector<vector<float>> vertices(num_vertices);

    // First, generate a reference radius based on target area
    // For a rough approximation, assume polygon approximates a circle
    float ref_radius = sqrt(target_area / (num_vertices * sin(2 * M_PI / num_vertices)));

    // Generate vertices with some randomness
    for (size_t i = 0; i < num_vertices; i++) {
        float angle = 2 * M_PI * i / num_vertices;
        // Add some variation to make it look more like a random polygon
        uniform_real_distribution<float> radius_dist(ref_radius * 0.5f, ref_radius * 1.5f);
        float r = radius_dist(rng);
        uniform_real_distribution<float> angle_var(-M_PI / num_vertices * 0.5f, M_PI / num_vertices * 0.5f);
        angle += angle_var(rng);

        vertices[i].resize(2);
        vertices[i][0] = center[0] + r * cos(angle);
        vertices[i][1] = center[1] + r * sin(angle);

        // Clamp to global bounds
        vertices[i][0] = max(global_bbox.min_bounds[0], min(global_bbox.max_bounds[0], vertices[i][0]));
        vertices[i][1] = max(global_bbox.min_bounds[1], min(global_bbox.max_bounds[1], vertices[i][1]));
    }

    return PolygonFilterParams(vertices, 2);
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

static void test_approx_polygon(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg, size_t vecdim,
                        vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                        vector<PolygonFilterParams> &filters) {
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
#ifdef USE_FLY_SEARCH
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

static void test_vs_recall_polygon(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                           vector<PolygonFilterParams> &filters) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        if(efBase >= k) efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.set_global_ef(ef);
        test_approx_polygon(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, filters);
        if (outer_recall > 99.5) break;
    }
}

// Ray casting point-in-polygon test
static bool point_in_polygon(const float* meta, const PolygonFilterParams& polygon) {
    int crossings = 0;
    size_t n = polygon.vertices.size();
    for (size_t i = 0; i < n; i++) {
        const auto& v1 = polygon.vertices[i];
        const auto& v2 = polygon.vertices[(i + 1) % n];

        if (((v1[1] <= meta[1] && meta[1] < v2[1]) ||
             (v2[1] <= meta[1] && meta[1] < v1[1])) &&
            (meta[0] < (v2[0] - v1[0]) * (meta[1] - v1[1]) / (v2[1] - v1[1]) + v1[0])) {
            crossings++;
        }
    }
    return (crossings % 2) == 1;
}

// Compute filtered groundtruth with polygon filter
static Matrix<int> compute_filtered_gt_polygon(
        Matrix<float> &Q,
        Matrix<float> &X_base,
        const std::vector<PolygonFilterParams> &filters,
        const std::vector<std::vector<float>> &metadata,
        size_t attr_dim,
        size_t k) {
    Matrix<int> G(Q.n, k);
    double total_selectivity = 0.0;

#pragma omp parallel for schedule(dynamic, 144) reduction(+:total_selectivity)
    for (size_t i = 0; i < Q.n; i++) {
        const float *query = Q.data + i * Q.d;
        const PolygonFilterParams &fr = filters[i];

        // Pass 1: collect indices of base vectors passing the filter
        std::vector<size_t> passed;
        passed.reserve(X_base.n);
        for (size_t j = 0; j < X_base.n; j++) {
            const float *meta = metadata[j].data();
            if (point_in_polygon(meta, fr)) {
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
            {"num-vertices",        required_argument, 0, 'v'},
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
    size_t num_vertices = 4;  // default to quadrilateral

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:r:f:m:v:", longopts, &ind);
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
            case 'v':
                if (optarg) {
                    num_vertices = atoi(optarg);
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
        log_index_time(dataset, "bench_polygon", load_time);
    } else {
        cout << "Building new index..." << endl;
        auto start = chrono::high_resolution_clock::now();
        index.build_index(data_path, meta_path);
        auto end = chrono::high_resolution_clock::now();
        auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Index built in " << build_time << " ms" << endl;
        log_index_time(dataset, "bench_polygon", build_time);

        cout << "Saving index to " << index_path << "..." << endl;
        index.save_index(index_path);
    }

    // Test with polygon filters
    vector<PolygonFilterParams> polygon_filters;
    cout << "Generating random polygon filters (vertices=" << num_vertices << ", ratio=" << filter_ratio << ", seed=42)..." << endl;
    mt19937 rng(42);
    BoundingBox global_bbox = index.get_global_bbox();
    for (size_t i = 0; i < Q.n; i++) {
        polygon_filters.push_back(generate_filter_polygon(global_bbox, filter_ratio, num_vertices, index.get_meta_dim(), rng));
    }
    cout << "  Generated " << polygon_filters.size() << " polygon filters" << endl;

    cout << "Computing polygon-filtered groundtruth (k=" << 100 << ")..." << endl;
    Matrix<int> G_polygon = compute_filtered_gt_polygon(Q, X, polygon_filters,
                                        index.get_metadata(),
                                        index.get_meta_dim(), 100);
    cout << "  Done." << endl;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    K = 20;
    sprintf(result_path, "./results/recall@%d/%s/%s-hnsw-cube-merge-layer-polygon-%s-%zu-%.2f.log", K, dataset, dataset, meta, num_vertices, filter_ratio);
    freopen(result_path, "a", stdout);
    get_gt(Q, X, G_polygon, answers, K);
    test_vs_recall_polygon(Q.data, X.n, Q.n, index, Q.d, answers, K, polygon_filters);
    answers.clear();
    K = 100;
    sprintf(result_path, "./results/recall@%d/%s/%s-hnsw-cube-merge-layer-polygon-%s-%zu-%.2f.log", K, dataset, dataset, meta, num_vertices, filter_ratio);
    freopen(result_path, "a", stdout);
    get_gt(Q, X, G_polygon, answers, K);
    test_vs_recall_polygon(Q.data, X.n, Q.n, index, Q.d, answers, K, polygon_filters);

    return 0;
}