// P1 physical-structure ablation driver (new-host-20260919/revision-ablation).
//
// Builds or queries one physical index variant:
//   multi-cross2            full hierarchy, 2 cross edges per node   (production)
//   multi-cross0            full hierarchy, cross edges disabled
//   singlefinest-cross2     only the finest adaptive layer, cross edges on
//   singlefinest-cross0     only the finest adaptive layer, no cross edges
//   single<K>-cross<N>      only layer K at its native 2^(K+1) resolution
//
// Build mode times the build, reports the layer geometry and saves the index.
// Query mode loads an index and runs the shared fixed-cardinality grid.
//
// Compiled twice: plain for timing, with -DABLATION_METRICS for the
// traversal counters, so the counters never perturb the timing runs.
#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// COLLECT_LOG is defined in config.h, which IndexCube.h does not include.
// Including config.h first therefore switches IndexCube's traversal counters on;
// leaving it after IndexCube.h keeps the timed binary free of their overhead.
#ifdef ABLATION_METRICS
#include "config.h"
#endif

#include "IndexCube.h"
#include "config.h"
#include "hnsw-cube.h"
#include "matrix.h"

namespace fs = std::filesystem;
using hnswlib::labeltype;

struct SharedWorkload {
    uint64_t nq = 0;
    uint64_t dims = 0;
    uint64_t k = 0;
    std::vector<BoundingBox> boxes;
    std::vector<int64_t> groundtruth;
};

static SharedWorkload load_workload(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open workload " + path.string());
    SharedWorkload work;
    in.read(reinterpret_cast<char*>(&work.nq), sizeof(work.nq));
    in.read(reinterpret_cast<char*>(&work.dims), sizeof(work.dims));
    in.read(reinterpret_cast<char*>(&work.k), sizeof(work.k));
    if (!in || work.nq == 0 || work.dims == 0 || work.k == 0) {
        throw std::runtime_error("bad workload header");
    }
    if (work.dims > MAX_DIM || work.nq > 1000000 || work.k > 10000 ||
        fs::file_size(path) != 24 + work.nq * work.dims * 2 * sizeof(float) +
                                    work.nq * work.k * sizeof(int64_t))
        throw std::runtime_error("invalid workload dimensions or byte length");
    work.boxes.reserve(work.nq);
    for (uint64_t qi = 0; qi < work.nq; ++qi) {
        BoundingBox box(work.dims);
        in.read(reinterpret_cast<char*>(box.min_bounds.data()), work.dims * sizeof(float));
        in.read(reinterpret_cast<char*>(box.max_bounds.data()), work.dims * sizeof(float));
        work.boxes.push_back(std::move(box));
    }
    work.groundtruth.resize(work.nq * work.k);
    in.read(reinterpret_cast<char*>(work.groundtruth.data()),
            work.groundtruth.size() * sizeof(int64_t));
    if (!in) throw std::runtime_error("truncated workload " + path.string());
    return work;
}

static double recall_at_k(const std::vector<std::vector<labeltype>>& actual,
                          const std::vector<int64_t>& expected, size_t k) {
    uint64_t hits = 0;
    for (size_t qi = 0; qi < actual.size(); ++qi) {
        std::unordered_set<labeltype> gt;
        for (size_t rank = 0; rank < k; ++rank) {
            const int64_t id = expected[qi * k + rank];
            if (id >= 0) gt.insert(static_cast<labeltype>(id));
        }
        for (const auto id : actual[qi]) hits += gt.count(id);
    }
    return static_cast<double>(hits) / static_cast<double>(actual.size() * k);
}

struct Variant {
    long long single_layer;  // -1 multi, -2 finest adaptive layer, >=0 that id
    size_t cross;
    std::string name;
};

static Variant parse_variant(const std::string& s) {
    const size_t dash = s.rfind("-cross");
    if (dash == std::string::npos)
        throw std::runtime_error("variant must look like <layers>-cross<N>: " + s);
    const std::string layers = s.substr(0, dash);
    Variant v;
    v.cross = std::stoul(s.substr(dash + 6));
    v.name = s;
    if (layers == "multi") {
        v.single_layer = -1;
    } else if (layers == "singlefinest") {
        v.single_layer = -2;
    } else if (layers.rfind("single", 0) == 0 && layers.size() > 6) {
        v.single_layer = std::stoll(layers.substr(6));
    } else {
        throw std::runtime_error("unknown layer spec: " + layers);
    }
    return v;
}

static void usage() {
    std::cerr <<
        "usage:\n"
        "  bench_ablation build DATA_ROOT DATASET METADATA VARIANT INDEX_PATH\n"
        "  bench_ablation query DATA_ROOT DATASET METADATA VARIANT INDEX_PATH "
        "WORKLOAD_DIR OUTPUT_CSV\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 2; }
        const std::string mode = argv[1];
        if (mode == "build") {
            if (argc != 7) { usage(); return 2; }
            const fs::path source = fs::path(argv[2]) / argv[3];
            const std::string dataset = argv[3], meta = argv[4];
            const Variant variant = parse_variant(argv[5]);
            const fs::path index_path = argv[6];
            if (fs::exists(index_path))
                throw std::runtime_error("refusing to overwrite " + index_path.string());

            std::string data_path = (source / (dataset + "_base.fvecs")).string();
            std::string meta_path = (source / (dataset + "_metadata_" + meta + ".bin")).string();
            if (!fs::exists(data_path) || !fs::exists(meta_path))
                throw std::runtime_error("missing base vectors or metadata");

            IndexCube index(6, 16, 200, variant.cross, 50);
            index.set_ablation_single_layer(variant.single_layer);
            const auto started = std::chrono::steady_clock::now();
            index.build_index(const_cast<char*>(data_path.c_str()),
                              const_cast<char*>(meta_path.c_str()));
            const double build_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            index.save_index(index_path.string());

            uint64_t index_bytes = fs::file_size(index_path);
            size_t layers = 0;
            while (index.get_hnsw_index(layers)) {
                const fs::path layer_path = index_path.string() + ".layer_" + std::to_string(layers);
                if (fs::exists(layer_path)) index_bytes += fs::file_size(layer_path);
                ++layers;
            }
            std::cout << "ABLATION_BUILD variant=" << variant.name
                      << " dataset=" << dataset << " metadata=" << meta
                      << " cross_edge_count=" << variant.cross
                      << " layers=" << layers
                      << " build_seconds=" << std::setprecision(6) << build_seconds
                      << " index_bytes=" << index_bytes << std::endl;
            return 0;
        }

        if (mode == "query") {
            if (argc != 9) { usage(); return 2; }
            const fs::path source = fs::path(argv[2]) / argv[3];
            const std::string dataset = argv[3], meta = argv[4];
            const Variant variant = parse_variant(argv[5]);
            const fs::path index_path = argv[6];
            const fs::path workload_dir = argv[7];
            const fs::path output_path = argv[8];
            if (fs::exists(output_path))
                throw std::runtime_error("refusing to overwrite " + output_path.string());

            std::string data_path = (source / (dataset + "_base.fvecs")).string();
            std::string query_path = (source / (dataset + "_query.fvecs")).string();
            if (!fs::exists(index_path) || !fs::exists(data_path) || !fs::exists(query_path))
                throw std::runtime_error("missing index, base vectors or query vectors");

            IndexCube index(6, 16, 200, variant.cross, 50);
            const auto load_start = std::chrono::steady_clock::now();
            if (!index.load_index(index_path.string(), const_cast<char*>(data_path.c_str())))
                throw std::runtime_error("failed to load index");
            const double load_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();

            Matrix<float> queries(query_path.data());
            std::ifstream base_header(data_path, std::ios::binary);
            int32_t base_dim = 0;
            base_header.read(reinterpret_cast<char*>(&base_dim), sizeof(base_dim));
            if (!base_header || base_dim <= 0) throw std::runtime_error("bad vector header");
            if (queries.d != size_t(base_dim))
                throw std::runtime_error("query/vector dimensionality mismatch");

            uint64_t index_bytes = fs::file_size(index_path);
            size_t layers = 0;
            while (index.get_hnsw_index(layers)) {
                const fs::path layer_path = index_path.string() + ".layer_" + std::to_string(layers);
                if (fs::exists(layer_path)) index_bytes += fs::file_size(layer_path);
                ++layers;
            }

            fs::create_directories(output_path.parent_path());
            std::ofstream csv(output_path);
            if (!csv) throw std::runtime_error("cannot open output " + output_path.string());
            csv << "variant,dataset,metadata,target_ratio,queries,k,ef_search,recall,qps,repeat,"
                   "layers,index_bytes,load_seconds";
#ifdef ABLATION_METRICS
            csv << ",avg_layer,cubes_visited_per_query,nodes_visited_per_query,"
                   "cross_edges_per_query,distance_computations_per_query,hops_per_query";
#endif
            csv << '\n';
            csv << std::setprecision(10);

            const bool predetermined = std::getenv("CG_SEARCH_MODE") &&
                std::string(std::getenv("CG_SEARCH_MODE")) == "pred";
            auto search = [&](const float* q, size_t k, const BoundingBox* box) {
                return predetermined ? index.predetermined_search(q, k, box) : index.fly_search(q, k, box);
            };
            const std::vector<unsigned> ef_values{20, 32, 64, 128, 256, 512, 1024, 2048};
            const std::vector<std::pair<std::string, double>> workloads{
                {dataset + "_" + meta + "_r0100.bin", 0.01},
                {dataset + "_" + meta + "_r0200.bin", 0.02},
                {dataset + "_" + meta + "_r0500.bin", 0.05},
                {dataset + "_" + meta + "_r1000.bin", 0.10},
            };

            for (const auto& [name, ratio] : workloads) {
                const auto work = load_workload(workload_dir / name);
                if (work.dims != index.get_meta_dim() || work.k != 20 || work.nq > queries.n)
                    throw std::runtime_error("workload does not match index");
                for (unsigned rep = 1; rep <= 3; ++rep) {
                    for (const auto ef : ef_values) {
                        index.set_global_ef(ef);
#ifdef ABLATION_METRICS
                        index.reset_metrics();
#endif
                        const size_t warmups = std::min<uint64_t>(20, work.nq);
                        for (size_t qi = 0; qi < warmups; ++qi)
                            search(queries.data + qi * queries.d, work.k, &work.boxes[qi]);

                        std::vector<std::vector<labeltype>> actual(work.nq);
                        const auto started = std::chrono::steady_clock::now();
                        for (size_t qi = 0; qi < work.nq; ++qi) {
                            auto result =
                                search(queries.data + qi * queries.d, work.k, &work.boxes[qi]);
                            actual[qi].reserve(result.size());
                            while (!result.empty()) {
                                actual[qi].push_back(result.top().second);
                                result.pop();
                            }
                        }
                        const double seconds =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                        const double recall = recall_at_k(actual, work.groundtruth, work.k);
                        const double qps = static_cast<double>(work.nq) / seconds;
                        for (size_t qi = 0; qi < work.nq; ++qi) {
                            std::unordered_set<labeltype> seen;
                            for (auto id : actual[qi]) {
                                if (id >= index.get_metadata().size() ||
                                    !work.boxes[qi].contains(index.get_metadata()[id]))
                                    throw std::runtime_error("out-of-filter or invalid ID");
                                if (!seen.insert(id).second)
                                    throw std::runtime_error("duplicate result");
                            }
                        }
                        const std::string dump_name = output_path.string() + ".r" +
                            std::to_string(int(std::llround(ratio*10000))) + ".ef" +
                            std::to_string(ef) + ".rep" + std::to_string(rep) + ".ids";
                        std::ofstream dump(dump_name, std::ios::binary);
                        for (const auto& row : actual) {
                            if (row.size() > work.k) throw std::runtime_error("too many IDs");
                            for (size_t rank=0; rank<work.k; ++rank) {
                                int64_t id = rank<row.size() ? int64_t(row[rank]) : -1;
                                dump.write(reinterpret_cast<const char*>(&id),8);
                            }
                        }
                        if (!dump) throw std::runtime_error("ID dump failed");
                        csv << variant.name << ',' << dataset << ',' << meta << ',' << ratio << ','
                            << work.nq << ',' << work.k << ',' << ef << ',' << recall << ',' << qps
                            << ',' << rep << ',' << layers << ',' << index_bytes << ','
                            << load_seconds;
#ifdef ABLATION_METRICS
                        const auto& m = index.get_metrics();
                        const double n = static_cast<double>(m.query_count);
                        csv << ',' << m.avg_layer() << ',' << m.total_cubes_visited / n << ','
                            << m.total_nodes_visited / n << ',' << m.total_cross_cube_edges / n
                            << ',' << m.total_distance_computations / n << ','
                            << m.total_hops / n;
#endif
                        csv << '\n';
                        csv.flush();
                        std::cout << "variant=" << variant.name << " ratio=" << ratio
                                  << " ef=" << ef << " rep=" << rep << " recall=" << recall
                                  << " qps=" << qps << '\n';
                    }
                }
            }
            return 0;
        }

        usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
