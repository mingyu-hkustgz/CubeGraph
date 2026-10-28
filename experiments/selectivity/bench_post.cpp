// 0.5% box driver for the frozen PostFiltering index.
// Search is unchanged; raw IDs are written after the timed pass.
#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "IndexCube.h"
#include "config.h"
#include "hnsw-cube.h"
#include "hnsw-static.h"
#include "matrix.h"

namespace fs = std::filesystem;
using hnswlib::labeltype;

static const std::vector<unsigned> EF_VALUES{20, 32, 64, 128, 256, 512, 1024, 2048};

struct SharedWorkload {
    uint64_t nq = 0, dims = 0, k = 0;
    std::vector<BoundingBox> boxes;
    std::vector<int64_t> groundtruth;
};

static SharedWorkload load_workload(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open workload " + path.string());
    SharedWorkload w;
    in.read(reinterpret_cast<char *>(&w.nq), sizeof(w.nq));
    in.read(reinterpret_cast<char *>(&w.dims), sizeof(w.dims));
    in.read(reinterpret_cast<char *>(&w.k), sizeof(w.k));
    if (!in || w.nq == 0 || w.dims == 0 || w.k == 0) throw std::runtime_error("bad workload header");
    if (w.dims > MAX_DIM || w.nq > 1000000 || w.k > 10000 ||
        fs::file_size(path) != 24 + w.nq * w.dims * 2 * sizeof(float) + w.nq * w.k * sizeof(int64_t))
        throw std::runtime_error("invalid workload dimensions or byte length");
    for (uint64_t qi = 0; qi < w.nq; ++qi) {
        BoundingBox box(w.dims);
        in.read(reinterpret_cast<char *>(box.min_bounds.data()), w.dims * sizeof(float));
        in.read(reinterpret_cast<char *>(box.max_bounds.data()), w.dims * sizeof(float));
        w.boxes.push_back(std::move(box));
    }
    w.groundtruth.resize(w.nq * w.k);
    in.read(reinterpret_cast<char *>(w.groundtruth.data()), w.groundtruth.size() * sizeof(int64_t));
    if (!in) throw std::runtime_error("truncated workload");
    return w;
}

static double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min<size_t>(static_cast<size_t>(q * (v.size() - 1) + 0.5), v.size() - 1)];
}

static uint64_t file_bytes(const fs::path &p) { return fs::exists(p) ? fs::file_size(p) : 0; }

int main(int argc, char **argv) {
    try {
        if (argc != 8) {
            std::cerr << "usage: bench_baselines_shared ALGO DATA_ROOT DATASET METADATA INDEX_PATH "
                         "WORKLOAD_DIR OUT_CSV\n";
            return 2;
        }
        const std::string algo = argv[1];
        const fs::path data_root = argv[2];
        const std::string dataset = argv[3], meta = argv[4];
        const fs::path index_path = argv[5];
        const fs::path workload_dir = argv[6];
        const fs::path out_csv = argv[7];
        if (algo != "post") throw std::runtime_error("unknown ALGO " + algo);
        if (fs::exists(out_csv)) throw std::runtime_error("refusing to overwrite " + out_csv.string());

        const fs::path source = data_root / dataset;
        std::string data_path = (source / (dataset + "_base.fvecs")).string();
        std::string query_path = (source / (dataset + "_query.fvecs")).string();
        std::string meta_path = (source / (dataset + "_metadata_" + meta + ".bin")).string();

        Matrix<float> X(data_path.data());
        Matrix<float> Q(query_path.data());

        // metadata is needed to decide whether a retrieved candidate satisfies
        // the box; the vector index itself stores none
        std::vector<std::vector<float>> meta_vec;
        {
            std::ifstream mi(meta_path, std::ios::binary);
            if (!mi) throw std::runtime_error("cannot open metadata " + meta_path);
            size_t mn = 0, md = 0;
            mi.read(reinterpret_cast<char *>(&mn), sizeof(size_t));
            mi.read(reinterpret_cast<char *>(&md), sizeof(size_t));
            if (!mi || mn == 0 || md == 0) throw std::runtime_error("bad metadata header");
            meta_vec.assign(mn, std::vector<float>(md));
            for (auto &row : meta_vec) mi.read(reinterpret_cast<char *>(row.data()), md * sizeof(float));
        }
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
        auto *space = new hnswlib::L2Space(X.d);

        double build_seconds = 0.0;
        hnswlib::HierarchicalNSWStatic<float> *post = nullptr;
        uint64_t index_bytes = 0;
        if (!fs::exists(index_path)) throw std::runtime_error("missing frozen index");
        const auto load_start = std::chrono::steady_clock::now();

        if (algo == "post") {
            const size_t M = 16, efc = 200;
            if (fs::exists(index_path)) {
                post = new hnswlib::HierarchicalNSWStatic<float>(space, index_path.string());
            } else {
                const auto t0 = std::chrono::steady_clock::now();
                post = new hnswlib::HierarchicalNSWStatic<float>(space, X.n, M, efc);
                for (size_t i = 0; i < X.n; ++i)
                    post->addPoint(X.data + i * X.d, static_cast<labeltype>(i));
                post->saveIndex(index_path.string());
                build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
            index_bytes = file_bytes(index_path);
        }
        const double load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();

        std::ofstream csv(out_csv);
        if (!csv) throw std::runtime_error("cannot open output");
        csv << "algorithm,dataset,metadata,target_ratio,queries,k,budget,recall,qps,p50_us,"
               "p95_us,repeat,index_bytes,load_seconds,build_seconds\n";
        csv << std::setprecision(10);

        const std::vector<std::pair<std::string, double>> workloads{
            {dataset + "_" + meta + "_r0050.bin", 0.005},
        };

        for (const auto &[name, ratio] : workloads) {
            const auto w = load_workload(workload_dir / name);
            if (w.k != 20 || static_cast<size_t>(Q.n) < w.nq)
                throw std::runtime_error("workload does not match queries");
            for (unsigned rep = 1; rep <= 3; ++rep) {
                for (const unsigned budget : EF_VALUES) {
                    const size_t warm = std::min<uint64_t>(20, w.nq);
                    for (size_t qi = 0; qi < warm; ++qi) {
                        if (algo == "post") post->searchKnn(Q.data + qi * Q.d, budget);
                    }
                    std::vector<double> lat(w.nq);
                    uint64_t hits = 0, available = 0;
                    std::vector<int64_t> raw_ids(w.nq * 20, -1);
                    const auto started = std::chrono::steady_clock::now();
                    for (size_t qi = 0; qi < w.nq; ++qi) {
                        const auto q0 = std::chrono::steady_clock::now();
                        std::vector<std::pair<float, labeltype>> got;
                        if (algo == "post") {
                            auto cand = post->searchKnn(Q.data + qi * Q.d, budget);
                            std::vector<std::pair<float, labeltype>> keep;
                            keep.reserve(cand.size());
                            while (!cand.empty()) {
                                const auto &top = cand.top();
                                if (static_cast<size_t>(top.second) < meta_vec.size() &&
                                    w.boxes[qi].contains(meta_vec[top.second]))
                                    keep.emplace_back(top.first, top.second);
                                cand.pop();
                            }
                            const size_t take = std::min<size_t>(20, keep.size());
                            if (take) {
                                std::partial_sort(keep.begin(), keep.begin() + take, keep.end());
                                got.assign(keep.begin(), keep.begin() + take);
                            }
                        }
                        lat[qi] = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - q0).count();
                        std::unordered_set<labeltype> truth;
                        for (size_t r = 0; r < 20; ++r)
                            if (w.groundtruth[qi * 20 + r] >= 0)
                                truth.insert(static_cast<labeltype>(w.groundtruth[qi * 20 + r]));
                        available += truth.size();
                        for (size_t j=0; j<got.size(); ++j) { hits += truth.count(got[j].second); raw_ids[qi*20+j]=int64_t(got[j].second); }
                    }
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                    std::ofstream ids(out_csv.string()+".r50.ef"+std::to_string(budget)+".rep"+std::to_string(rep)+".ids", std::ios::binary);
                    ids.write(reinterpret_cast<const char*>(raw_ids.data()), raw_ids.size()*sizeof(int64_t));
                    if (!ids) throw std::runtime_error("failed writing returned IDs");
                    csv << algo << ',' << dataset << ',' << meta << ',' << ratio << ',' << w.nq
                        << ",20," << budget << ','
                        << (available ? static_cast<double>(hits) / available : 0.0) << ','
                        << static_cast<double>(w.nq) / seconds << ',' << percentile(lat, 0.50) << ','
                        << percentile(lat, 0.95) << ',' << rep << ',' << index_bytes << ','
                        << load_seconds << ',' << build_seconds << '\n';
                    csv.flush();
                    std::cout << "ratio=" << ratio << " budget=" << budget << " rep=" << rep
                              << " recall=" << (available ? static_cast<double>(hits) / available : 0.0)
                              << " qps=" << static_cast<double>(w.nq) / seconds << std::endl;
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
