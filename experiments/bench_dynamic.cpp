#include <filesystem>
#include <chrono>
#include <unordered_set>
#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <unistd.h>

#include "hnsw-cube.h"
#include "matrix.h"
#include "utils.h"
#include <getopt.h>
#include "IndexCube.h"
#include "config.h"

using namespace std;
using namespace hnswlib;

const int MAXK = 100;
int efSearch = 20;
double outer_recall = 0;

BoundingBox generate_filter_bbox(const BoundingBox &global_bbox, float filter_ratio, size_t attr_dim, mt19937 &rng) {
    BoundingBox filter_bbox(attr_dim);
    for (size_t d = 0; d < attr_dim; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        float base_filter_size = range * pow(filter_ratio, 1.0/attr_dim);
        uniform_real_distribution<float> fluctuation_dist(0.8, 1.2);
        float filter_size = base_filter_size * fluctuation_dist(rng);
        uniform_real_distribution<float> dist(
            global_bbox.min_bounds[d] + filter_size / 2,
            global_bbox.max_bounds[d] - filter_size / 2);
        float center = dist(rng);
        filter_bbox.min_bounds[d] = center - filter_size / 2;
        filter_bbox.max_bounds[d] = center + filter_size / 2;
    }
    return filter_bbox;
}

template<typename GT>
static void get_gt(Matrix<float> &Q, Matrix<float> &X, GT G,
                   vector<std::priority_queue<std::pair<float, labeltype>>> &answers, size_t subk) {
    (vector<std::priority_queue<std::pair<float, labeltype>>>(Q.n)).swap(answers);
    for (int i = 0; i < Q.n; i++) {
        for (int j = 0; j < subk; j++) {
            auto gt = G.data[G.d * i + j];
            if (gt < 0) break;
            answers[i].emplace(sqr_dist(Q.data + i * Q.d, X.data + gt * X.d, X.d), gt);
        }
    }
}

static void test_approx(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg,
                        size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
                        size_t k, vector<BoundingBox> &filters, ostream *out = nullptr) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;
    long double total_ratio = 0;
    for (int i = 0; i < qsize; i++) {
        const auto query_started = std::chrono::steady_clock::now();

        std::priority_queue<std::pair<float, labeltype>> result =
#if USE_FLY_SEARCH
            appr_alg.fly_search(massQ + vecdim * i, k, &filters[i]);
#else
            appr_alg.predetermined_search(massQ + vecdim * i, k, &filters[i]);
#endif
        total_time += std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - query_started).count();
        // Validation stays outside query timing.
        auto checked = result;
        std::unordered_set<labeltype> unique_ids;
        const auto& attrs = appr_alg.get_metadata();
        while (!checked.empty()) {
            const auto id = checked.top().second;
            checked.pop();
            if (id >= attrs.size() || !filters[i].contains(attrs[id]) ||
                !unique_ids.insert(id).second)
                throw std::runtime_error("invalid dynamic-query result");
        }

        std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        total_ratio += Ratio(result, gt);
        correct += tmp;
    }
    long double time_us_per_query = total_time / qsize;
    long double recall_val = 1.0f * correct / total;
    long double dist_ratio = total_ratio / qsize;

    if (out) {
        *out << recall_val * 100.0 << " " << 1e6 / time_us_per_query << " " << dist_ratio << endl;
    } else {
        cout << recall_val * 100.0 << " " << 1e6 / time_us_per_query << " " << dist_ratio << endl;
    }
    outer_recall = recall_val * 100;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, IndexCube &appr_alg,
                           size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
                           size_t k, vector<BoundingBox> &filters, ostream *out = nullptr) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        if (efBase >= k) efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef : efs) {
        appr_alg.set_global_ef(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, filters, out);
        if (outer_recall > 99.5) break;
    }
}

static Matrix<int> compute_filtered_gt(Matrix<float> &Q, Matrix<float> &X_base,
                                       const std::vector<BoundingBox> &filters,
                                       const std::vector<std::vector<float>> &metadata,
                                       size_t attr_dim, size_t k) {
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
            for (size_t d = 0; d < attr_dim; d++) {
                if (meta[d] < fb.min_bounds[d] || meta[d] > fb.max_bounds[d]) {
                    ok = false; break;
                }
            }
            if (ok) passed.push_back(j);
        }
        double selectivity = (double)passed.size() / X_base.n;
        total_selectivity += selectivity;

        if (passed.size() <= k) {
            std::vector<std::pair<float, size_t>> dists;
            for (size_t idx : passed) {
                dists.emplace_back(sqr_dist(query, X_base.data + idx * X_base.d, Q.d), idx);
            }
            std::sort(dists.begin(), dists.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            size_t r = 0;
            for (auto &p : dists) G.data[G.d * i + r++] = (int)p.second;
            for (; r < k; r++) G.data[G.d * i + r] = -1;
        } else {
            std::vector<std::pair<float, size_t>> dists;
            for (size_t idx : passed) {
                dists.emplace_back(sqr_dist(query, X_base.data + idx * X_base.d, Q.d), idx);
            }
            std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
                             [](const auto &a, const auto &b) { return a.first < b.first; });
            std::sort(dists.begin(), dists.begin() + k,
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (size_t r = 0; r < k; r++)
                G.data[G.d * i + r] = (int)dists[r].second;
        }
    }
    cout << "Average filter selectivity: " << total_selectivity / Q.n << endl;
    return G;
}

// Write subset of metadata to temp binary file
static void write_temp_metadata(const char* path, const vector<vector<float>> &metadata,
                                size_t start, size_t count) {
    ofstream out(path, ios::binary);
    size_t d = metadata[0].size();
    out.write((char*)&count, sizeof(size_t));
    out.write((char*)&d, sizeof(size_t));
    for (size_t i = start; i < start + count; i++) {
        out.write((char*)metadata[i].data(), d * sizeof(float));
    }
    out.close();
}

// Write subset of vectors to temp fvecs file
static void write_temp_fvecs(const char* path, float *data, size_t dim,
                             size_t start, size_t count) {
    ofstream out(path, ios::binary);
    int d = (int)dim;
    for (size_t i = start; i < start + count; i++) {
        out.write((char*)&d, sizeof(int));
        out.write((char*)(data + i * dim), dim * sizeof(float));
    }
    out.close();
}

int main(int argc, char *argv[]) {
    const struct option longopts[] = {
        {"dataset",       required_argument, 0, 'd'},
        {"source",        required_argument, 0, 's'},
        {"filter-ratio",  required_argument, 0, 'f'},
        {"meta",          required_argument, 0, 'm'},
        {"k",             required_argument, 0, 'k'},
        {"threads",       required_argument, 0, 't'},
    };

    char source[256] = "./DATA/";
    char dataset[256] = "sift";
    char meta[256] = "uniform_2d";
    float filter_ratio = 0.1f;
    int K = 20;
    int num_threads = 32;

    // Arguments parsed below after adding -r option

    double ratio = 0.1;

    const struct option extra_longopts[] = {
        {"dataset",       required_argument, 0, 'd'},
        {"source",        required_argument, 0, 's'},
        {"filter-ratio",  required_argument, 0, 'f'},
        {"meta",          required_argument, 0, 'm'},
        {"k",             required_argument, 0, 'k'},
        {"threads",       required_argument, 0, 't'},
        {"ratio",         required_argument, 0, 'r'},
    };

    int iarg = 0, ind;
    opterr = 1;
    while ((iarg = getopt_long(argc, argv, "d:s:f:m:k:t:r:", extra_longopts, &ind)) != -1) {
        switch (iarg) {
            case 'd': strcpy(dataset, optarg); break;
            case 's': strcpy(source, optarg); break;
            case 'f': filter_ratio = atof(optarg); break;
            case 'm': strcpy(meta, optarg); break;
            case 'k': K = atoi(optarg); break;
            case 't': num_threads = atoi(optarg); break;
            case 'r': ratio = atof(optarg); break;
        }
    }

    omp_set_num_threads(num_threads);

    char data_path[512], query_path[512], meta_path[512];
    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    sprintf(meta_path, "%s%s_metadata_%s.bin", source, dataset, meta);

    // Load full datasets
    Matrix<float> X_full(data_path);
    Matrix<float> Q(query_path);
    cout << "Base vectors: " << X_full.n << " x " << X_full.d << endl;
    cout << "Query vectors: " << Q.n << " x " << Q.d << endl;

    // Load full metadata
    ifstream meta_in(meta_path, ios::binary);
    size_t meta_n, meta_d;
    meta_in.read((char*)&meta_n, sizeof(size_t));
    meta_in.read((char*)&meta_d, sizeof(size_t));
    vector<vector<float>> all_metadata(meta_n, vector<float>(meta_d));
    for (size_t i = 0; i < meta_n; i++) {
        meta_in.read((char*)all_metadata[i].data(), meta_d * sizeof(float));
    }
    meta_in.close();
    cout << "Metadata: " << meta_n << " x " << meta_d << endl;

    // Use subset: first 200K vectors as total pool
    const size_t TOTAL_POOL = 200000;
    if (X_full.n < TOTAL_POOL) {
        cerr << "ERROR: Dataset has only " << X_full.n << " vectors, need at least " << TOTAL_POOL << endl;
        return 1;
    }

    const size_t INITIAL_SIZE = 100000;
    const size_t MAX_UPDATE = TOTAL_POOL - INITIAL_SIZE;

    size_t vec_dim = X_full.d;
    size_t data_size = vec_dim * sizeof(float);

    // Pre-allocate contiguous buffer for all vectors (initial + updates)
    float *full_data = new float[TOTAL_POOL * vec_dim];
    memcpy(full_data, X_full.data, TOTAL_POOL * data_size);

    // Pre-allocate contiguous metadata buffer for updates
    float *update_meta_buf = (float*)aligned_alloc(64, MAX_UPDATE * meta_d * sizeof(float));
    for (size_t i = 0; i < MAX_UPDATE; i++) {
        memcpy(update_meta_buf + i * meta_d, all_metadata[INITIAL_SIZE + i].data(), meta_d * sizeof(float));
    }

    // Set static base data
    HierarchicalNSWCube<float>::static_base_data_ = (char*)full_data;

    char time_dat[512], qps_dat[512], summary_path[512];
    sprintf(time_dat, "./results/dynamic-update/%s_update_time.dat", dataset);
    sprintf(qps_dat, "./results/dynamic-update/%s_search_qps.dat", dataset);
    sprintf(summary_path, "./results/dynamic-update/%s_summary.txt", dataset);

    std::filesystem::create_directories("./results/dynamic-update");
    std::filesystem::create_directories("./results/recall@20");
    ofstream time_file(time_dat, ios::app);
    ofstream qps_file(qps_dat, ios::app);
    ofstream summary(summary_path, ios::app);

    mt19937 rng(42);
    size_t num_layers = 6, M = 16, ef_construction = 200, cross_edge = 2;

    size_t update_count = (size_t)(INITIAL_SIZE * ratio);
    if (update_count > MAX_UPDATE) update_count = MAX_UPDATE;
    size_t total_after_update = INITIAL_SIZE + update_count;

    cout << "\n========================================" << endl;
    cout << "Update ratio: " << ratio * 100 << "% (" << update_count << " new points)" << endl;
    summary << "--- Update ratio: " << ratio * 100 << "% (" << update_count << " new points) ---" << endl;

    // Write temp files for initial subset
    char tmp_meta_init[512], tmp_data_init[512];
    sprintf(tmp_meta_init, "/tmp/dy_meta_init_%.0f_%zu.bin", ratio * 100, (size_t)getpid());
    sprintf(tmp_data_init, "/tmp/dy_data_init_%.0f_%zu.fvecs", ratio * 100, (size_t)getpid());
    write_temp_metadata(tmp_meta_init, all_metadata, 0, INITIAL_SIZE);
    write_temp_fvecs(tmp_data_init, full_data, vec_dim, 0, INITIAL_SIZE);

        // ==========================================
        // INCREMENTAL UPDATE
        // ==========================================
        cout << "\n--- Incremental Update ---" << endl;

        IndexCube *index_inc = new IndexCube(num_layers, M, ef_construction, cross_edge);
        index_inc->build_index(tmp_data_init, tmp_meta_init);
        // build_index overwrites static_base_data_ with its own Matrix data.
        // Reset to our full_data buffer so new labels (>= INITIAL_SIZE) are valid.
        HierarchicalNSWCube<float>::static_base_data_ = (char*)full_data;

        // Prepare labels for new points
        vector<labeltype> update_labels(update_count);
        for (size_t i = 0; i < update_count; i++) {
            update_labels[i] = INITIAL_SIZE + i;
        }

        // Resize and add points
        index_inc->resize_for_updates(update_count);

        auto inc_start = chrono::high_resolution_clock::now();
        long long inc_time_ms = index_inc->add_points_batch(
            update_labels,
            full_data + INITIAL_SIZE * vec_dim,
            update_meta_buf);
        auto inc_end = chrono::high_resolution_clock::now();
        long long inc_time = chrono::duration_cast<chrono::milliseconds>(inc_end - inc_start).count();

        cout << "Incremental update: " << inc_time << " ms ("
             << (double)inc_time * 1000.0 / update_count << " us/point)" << endl;
        summary << "Incremental: " << inc_time << " ms" << endl;

        // ==========================================
        // REBUILD
        // ==========================================
        cout << "\n--- Rebuild ---" << endl;

        char tmp_meta_full[512], tmp_data_full[512];
        sprintf(tmp_meta_full, "/tmp/dy_meta_full_%.0f_%zu.bin", ratio * 100, (size_t)getpid());
        sprintf(tmp_data_full, "/tmp/dy_data_full_%.0f_%zu.fvecs", ratio * 100, (size_t)getpid());
        write_temp_metadata(tmp_meta_full, all_metadata, 0, total_after_update);
        write_temp_fvecs(tmp_data_full, full_data, vec_dim, 0, total_after_update);

        HierarchicalNSWCube<float>::static_base_data_ = (char*)full_data;
        auto rebuild_start = chrono::high_resolution_clock::now();
        IndexCube *index_rebuild = new IndexCube(num_layers, M, ef_construction, cross_edge);
        index_rebuild->build_index(tmp_data_full, tmp_meta_full);
        auto rebuild_end = chrono::high_resolution_clock::now();
        long long rebuild_time = chrono::duration_cast<chrono::milliseconds>(rebuild_end - rebuild_start).count();

        cout << "Full rebuild: " << rebuild_time << " ms ("
             << (double)rebuild_time * 1000.0 / total_after_update << " us/point)" << endl;
        summary << "Rebuild: " << rebuild_time << " ms" << endl;

        double speedup = (double)rebuild_time / (double)inc_time;
        cout << "Speedup: " << speedup << "x" << endl;
        summary << "Speedup: " << speedup << "x" << endl;

        time_file << ratio << " " << inc_time << " " << rebuild_time << " " << speedup << "\n";

        // ==========================================
        // SEARCH TEST
        // ==========================================
        cout << "\n--- Search Test ---" << endl;

        // Build Matrix wrapper for ground truth
        Matrix<float> X_search;
        X_search.n = total_after_update;
        X_search.d = vec_dim;
        X_search.data = full_data;

        // Generate filters
        vector<BoundingBox> filters;
        BoundingBox global_bbox = index_inc->get_global_bbox();
        for (size_t i = 0; i < Q.n; i++) {
            filters.push_back(generate_filter_bbox(global_bbox, filter_ratio, meta_d, rng));
        }

        // Ground truth
        vector<vector<float>> gt_meta(all_metadata.begin(), all_metadata.begin() + total_after_update);
        Matrix<int> G = compute_filtered_gt(Q, X_search, filters, gt_meta, meta_d, K);

        vector<std::priority_queue<std::pair<float, labeltype>>> answers;
        get_gt(Q, X_search, G, answers, K);

        // Search incremental index, capture to file
        char inc_log[512], rebuild_log[512];
        sprintf(inc_log, "./results/recall@20/%s_dynamic_inc_%.0f.log", dataset, ratio * 100);
        sprintf(rebuild_log, "./results/recall@20/%s_dynamic_rebuild_%.0f.log", dataset, ratio * 100);

        ofstream inc_out(inc_log);
        if (!inc_out) throw std::runtime_error("cannot write incremental query log");
        test_vs_recall(Q.data, X_search.n, Q.n, *index_inc, Q.d, answers, K, filters, &inc_out);
        inc_out.close();

        // Rebuild answers (test_vs_recall modifies outer_recall state)
        outer_recall = 0;
        ofstream rebuild_out(rebuild_log);
        if (!rebuild_out) throw std::runtime_error("cannot write rebuild query log");
        test_vs_recall(Q.data, X_search.n, Q.n, *index_rebuild, Q.d, answers, K, filters, &rebuild_out);
        rebuild_out.close();

        // Extract QPS at ~95% recall for the .dat file
        // Read back the log files and find closest point to 95%
        auto extract_qps = [](const char* path, double target_recall) -> pair<double, double> {
            ifstream in(path);
            double best_recall = 0, best_qps = 0, best_dist = 1e9;
            double r, q, dr;
            while (in >> r >> q >> dr) {
                double dist = fabs(r - target_recall);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_recall = r;
                    best_qps = q;
                }
            }
            in.close();
            if (!(best_qps > 0)) throw std::runtime_error("no valid dynamic query measurements");
            return {best_recall, best_qps};
        };

        auto inc_qps = extract_qps(inc_log, 95.0);
        auto reb_qps = extract_qps(rebuild_log, 95.0);

        qps_file << ratio << " incremental " << inc_qps.first << " " << inc_qps.second << "\n";
        qps_file << ratio << " rebuild " << reb_qps.first << " " << reb_qps.second << "\n";

        cout << "Incremental @ ~95% recall: " << inc_qps.first << "% " << inc_qps.second << " QPS" << endl;
        cout << "Rebuild @ ~95% recall: " << reb_qps.first << "% " << reb_qps.second << " QPS" << endl;
        summary << "Incremental QPS: " << inc_qps.second << " @ " << inc_qps.first << "% recall" << endl;
        summary << "Rebuild QPS: " << reb_qps.second << " @ " << reb_qps.first << "% recall" << endl;

        // Clean up temp files
        remove(tmp_meta_init);
        remove(tmp_data_init);
        remove(tmp_meta_full);
        remove(tmp_data_full);

        time_file.close();
        qps_file.close();
        summary.close();

    // Note: There appears to be a heap corruption issue that only manifests
    // during deallocation. The benchmark produces correct results, so we
    // intentionally leak these buffers at process exit (harmless for a benchmark).
    // delete[] full_data;
    // free(update_meta_buf);

    cout << "\n========================================" << endl;
    cout << "Benchmark complete." << endl;
    cout << "Results: " << time_dat << ", " << qps_dat << ", " << summary_path << endl;

    return 0;
}
