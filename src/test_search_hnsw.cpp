#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <hnswlib/hnsw-static.h>
#include "matrix.h"
#include "utils.h"
#include <getopt.h>
#include "config.h"
using namespace std;
using namespace hnswlib;

const int MAXK = 100;

int efSearch = 100;
double outer_recall = 0;

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
                   size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                   size_t subk, HierarchicalNSWStatic<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(
                    appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(massQA[k * i + j]),
                                          appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result,
           std::priority_queue<std::pair<float, labeltype >> &gt) {
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSWStatic<float> &appr_alg, size_t vecdim,
                        vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;


    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);
#endif
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
#ifndef WIN32
        GetCurTime(&run_end);
        GetTime(&run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        correct += tmp;
    }
    long double time_us_per_query = total_time / qsize;
    long double recall = 1.0f * correct / total;

    cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << " "<< endl;
    outer_recall = recall * 100;
    return;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSWStatic<float> &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.setEf(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        if(outer_recall > 99.5) break;
    }
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
    sprintf(result_path, "./results/recall@%d/%s/%s-hnsw.log", K, dataset, dataset);
    sprintf(index_path, "%s%s.hnsw", source, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;

    L2Space l2space(Q.d);
    auto *appr_alg = new HierarchicalNSWStatic<float>(&l2space, index_path, false);
    freopen(result_path, "w", stdout);
    size_t k = G.d;
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    return 0;
}