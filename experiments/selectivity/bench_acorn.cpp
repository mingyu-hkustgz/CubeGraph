#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>
#include <sys/stat.h>

namespace fs = std::filesystem;

struct Args {
    std::string dataset = "sift";
    std::string cube_data_root = "/home/bld/Research/CubeGraph/DATA";
    std::string metadata_name = "uniform_2d";
    std::string index_dir = "/home/bld/Research/acorn-codex-results/indexes";
    std::string output_dir = "/home/bld/Research/acorn-codex-results/logs";
    std::string workload_dir;
    int m = 32;
    int gamma = 12;
    int m_beta = 32;
    int k = 20;
    int query_count = 1000;
    int gt_threads = 72;
    uint32_t seed = 42;
    std::vector<double> ratios{0.01, 0.02, 0.05, 0.10};
    std::vector<unsigned> ef_values{20, 32, 64, 128, 256, 512, 1024, 2048};
};

static std::vector<std::string> split(const std::string& value, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--dataset") args.dataset = next();
        else if (key == "--cube-data-root") args.cube_data_root = next();
        else if (key == "--metadata") args.metadata_name = next();
        else if (key == "--index-dir") args.index_dir = next();
        else if (key == "--output-dir") args.output_dir = next();
        else if (key == "--workload-dir") args.workload_dir = next();
        else if (key == "--M") args.m = std::stoi(next());
        else if (key == "--gamma") args.gamma = std::stoi(next());
        else if (key == "--M-beta") args.m_beta = std::stoi(next());
        else if (key == "--k") args.k = std::stoi(next());
        else if (key == "--queries") args.query_count = std::stoi(next());
        else if (key == "--gt-threads") args.gt_threads = std::stoi(next());
        else if (key == "--seed") args.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (key == "--ratios") {
            args.ratios.clear();
            for (const auto& x : split(next(), ',')) args.ratios.push_back(std::stod(x));
        } else if (key == "--ef-values") {
            args.ef_values.clear();
            for (const auto& x : split(next(), ',')) args.ef_values.push_back(static_cast<unsigned>(std::stoul(x)));
        } else if (key == "--help") {
            std::cout
                << "ACORN sweep using CubeGraph fvecs and metadata\n"
                << "  --dataset sift --metadata uniform_2d\n"
                << "  --M 32 --gamma 12 --M-beta 32\n"
                << "  --ratios 0.01,0.02,0.05,0.10\n"
                << "  --ef-values 20,32,64,128,256,512,1024,2048\n"
                << "  --queries 1000 --gt-threads 72\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.m <= 0 || args.gamma <= 0 || args.m_beta <= 0 || args.k <= 0 ||
        args.query_count <= 0 || args.gt_threads <= 0) {
        throw std::runtime_error("integer parameters must be positive");
    }
    return args;
}

struct Fvecs {
    size_t n = 0;
    size_t d = 0;
    std::vector<float> values;
};

static Fvecs read_fvecs(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!input || d <= 0 || d > 1000000) throw std::runtime_error("bad fvecs dimension");
    input.seekg(0, std::ios::end);
    const auto bytes = static_cast<uint64_t>(input.tellg());
    const uint64_t row_bytes = static_cast<uint64_t>(d + 1) * sizeof(float);
    if (bytes % row_bytes != 0) throw std::runtime_error("bad fvecs file size");
    const size_t n = static_cast<size_t>(bytes / row_bytes);
    input.seekg(0, std::ios::beg);
    Fvecs out;
    out.n = n;
    out.d = static_cast<size_t>(d);
    out.values.resize(n * out.d);
    std::vector<float> row(out.d);
    for (size_t i = 0; i < n; ++i) {
        int32_t row_d = 0;
        input.read(reinterpret_cast<char*>(&row_d), sizeof(row_d));
        if (row_d != d) throw std::runtime_error("inconsistent fvecs row dimension");
        input.read(reinterpret_cast<char*>(row.data()), d * sizeof(float));
        if (!input) throw std::runtime_error("truncated fvecs file");
        std::copy(row.begin(), row.end(), out.values.begin() + i * out.d);
    }
    return out;
}

struct Metadata {
    size_t n = 0;
    size_t d = 0;
    std::vector<float> values;
};

static Metadata read_metadata(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    Metadata out;
    input.read(reinterpret_cast<char*>(&out.n), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&out.d), sizeof(size_t));
    if (!input || out.n == 0 || out.d == 0 || out.d > 64) {
        throw std::runtime_error("bad metadata header");
    }
    out.values.resize(out.n * out.d);
    input.read(reinterpret_cast<char*>(out.values.data()), out.values.size() * sizeof(float));
    if (!input) throw std::runtime_error("truncated metadata file");
    return out;
}

struct Box {
    std::vector<float> lo;
    std::vector<float> hi;
};

static std::pair<std::vector<float>, std::vector<float>> metadata_bounds(const Metadata& meta) {
    std::vector<float> lo(meta.d, std::numeric_limits<float>::max());
    std::vector<float> hi(meta.d, std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < meta.n; ++i) {
        for (size_t j = 0; j < meta.d; ++j) {
            const float v = meta.values[i * meta.d + j];
            lo[j] = std::min(lo[j], v);
            hi[j] = std::max(hi[j], v);
        }
    }
    return {lo, hi};
}

static std::vector<Box> generate_boxes(
        int count,
        double target_ratio,
        const std::vector<float>& global_lo,
        const std::vector<float>& global_hi,
        uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t dim = global_lo.size();
    const double side_fraction = std::pow(target_ratio, 1.0 / static_cast<double>(dim));
    std::vector<Box> boxes;
    boxes.reserve(static_cast<size_t>(count));
    for (int qi = 0; qi < count; ++qi) {
        Box box{std::vector<float>(dim), std::vector<float>(dim)};
        for (size_t j = 0; j < dim; ++j) {
            const double width = (global_hi[j] - global_lo[j]) * side_fraction;
            const double half = width / 2.0;
            std::uniform_real_distribution<double> center(global_lo[j] + half, global_hi[j] - half);
            const double c = center(rng);
            box.lo[j] = static_cast<float>(c - half);
            box.hi[j] = static_cast<float>(c + half);
        }
        boxes.push_back(std::move(box));
    }
    return boxes;
}

static bool passes(const Metadata& meta, size_t id, const Box& box) {
    const float* row = meta.values.data() + id * meta.d;
    for (size_t j = 0; j < meta.d; ++j) {
        if (row[j] < box.lo[j] || row[j] > box.hi[j]) return false;
    }
    return true;
}

static float l2_squared(const float* a, const float* b, size_t d) {
    float result = 0.0f;
    #pragma omp simd reduction(+:result)
    for (size_t j = 0; j < d; ++j) {
        const float diff = a[j] - b[j];
        result += diff * diff;
    }
    return result;
}

struct Workload {
    std::vector<char> masks;
    std::vector<faiss::idx_t> groundtruth;
    double measured_selectivity = 0.0;
    double prep_seconds = 0.0;
};

static Workload read_shared_workload(const fs::path& path, const Fvecs& base,
                                    const Metadata& meta, int queries, int k, int threads) {
    std::ifstream in(path,std::ios::binary);
    uint64_t nq=0,dims=0,topk=0;
    in.read(reinterpret_cast<char*>(&nq),8);
    in.read(reinterpret_cast<char*>(&dims),8);
    in.read(reinterpret_cast<char*>(&topk),8);
    if(!in || nq!=uint64_t(queries) || dims!=meta.d || topk!=uint64_t(k) ||
       fs::file_size(path)!=24+nq*dims*8+nq*topk*8)
        throw std::runtime_error("invalid shared workload " + path.string());
    std::vector<Box> boxes(nq);
    for(auto& box:boxes) {
        box.lo.resize(dims); box.hi.resize(dims);
        in.read(reinterpret_cast<char*>(box.lo.data()),dims*4);
        in.read(reinterpret_cast<char*>(box.hi.data()),dims*4);
    }
    Workload work;
    work.groundtruth.resize(nq*topk);
    static_assert(sizeof(faiss::idx_t)==8);
    in.read(reinterpret_cast<char*>(work.groundtruth.data()),nq*topk*8);
    if(!in) throw std::runtime_error("truncated shared workload");
    for(size_t qi=0;qi<nq;++qi)
        for(size_t rank=0;rank<topk;++rank) {
            const auto id=work.groundtruth[qi*topk+rank];
            if(id>=0 && (size_t(id)>=base.n || !passes(meta,id,boxes[qi])))
                throw std::runtime_error("invalid ground-truth ID");
        }
    const auto started=std::chrono::steady_clock::now();
    work.masks.resize(nq*base.n);
    uint64_t selected=0;
    #pragma omp parallel for reduction(+:selected) num_threads(threads)
    for(int64_t qi=0;qi<int64_t(nq);++qi) {
        char* mask=work.masks.data()+size_t(qi)*base.n;
        for(size_t id=0;id<base.n;++id) {
            const bool ok=passes(meta,id,boxes[qi]);
            mask[id]=ok; selected+=ok;
        }
    }
    work.measured_selectivity=double(selected)/(nq*double(base.n));
    work.prep_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return work;
}

static double recall_at_k(
        const std::vector<faiss::idx_t>& actual,
        const std::vector<faiss::idx_t>& expected,
        int k) {
    uint64_t hits = 0;
    const size_t nq = actual.size() / static_cast<size_t>(k);
    for (size_t qi = 0; qi < nq; ++qi) {
        std::unordered_set<faiss::idx_t> gt;
        for (int r = 0; r < k; ++r) {
            const auto id = expected[qi * static_cast<size_t>(k) + static_cast<size_t>(r)];
            if (id >= 0) gt.insert(id);
        }
        for (int r = 0; r < k; ++r) {
            if (gt.count(actual[qi * static_cast<size_t>(k) + static_cast<size_t>(r)])) ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * static_cast<size_t>(k));
}

static void dump_workload(
        const fs::path& output_path,
        const std::vector<Box>& boxes,
        const Workload& workload,
        int k) {
    std::ofstream out(output_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open workload output " + output_path.string());
    const uint64_t nq = boxes.size();
    const uint64_t dims = boxes.empty() ? 0 : boxes.front().lo.size();
    const uint64_t topk = static_cast<uint64_t>(k);
    out.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    out.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(&topk), sizeof(topk));
    for (const auto& box : boxes) {
        out.write(reinterpret_cast<const char*>(box.lo.data()), dims * sizeof(float));
        out.write(reinterpret_cast<const char*>(box.hi.data()), dims * sizeof(float));
    }
    out.write(
        reinterpret_cast<const char*>(workload.groundtruth.data()),
        workload.groundtruth.size() * sizeof(faiss::idx_t));
    if (!out) throw std::runtime_error("failed writing workload output " + output_path.string());
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path dataset_dir = fs::path(args.cube_data_root) / args.dataset;
        const fs::path base_path = dataset_dir / (args.dataset + "_base.fvecs");
        const fs::path query_path = dataset_dir / (args.dataset + "_query.fvecs");
        const fs::path metadata_path = dataset_dir / (args.dataset + "_metadata_" + args.metadata_name + ".bin");
        fs::create_directories(args.index_dir);
        fs::create_directories(args.output_dir);
        if (!args.workload_dir.empty()) fs::create_directories(args.workload_dir);

        std::cerr << "Loading vectors and metadata...\n";
        const Fvecs base = read_fvecs(base_path);
        Fvecs queries = read_fvecs(query_path);
        const Metadata meta = read_metadata(metadata_path);
        if (base.d != queries.d || base.n != meta.n) throw std::runtime_error("dataset dimensions/counts do not match");
        if (args.query_count > static_cast<int>(queries.n)) throw std::runtime_error("query count exceeds query file");
        queries.n = static_cast<size_t>(args.query_count);
        queries.values.resize(queries.n * queries.d);
        const auto [global_lo, global_hi] = metadata_bounds(meta);

        std::ostringstream stem_builder;
        stem_builder << args.dataset << "_M" << args.m << "_g" << args.gamma << "_mb" << args.m_beta;
        const std::string stem = stem_builder.str();
        const fs::path index_path = fs::path(args.index_dir) / (stem + ".index");
        const fs::path csv_path = fs::path(args.output_dir) / (stem + "_" + args.metadata_name + ".csv");

        faiss::IndexACORNFlat* index = nullptr;
        double build_seconds = 0.0;
        if (!fs::exists(index_path)) throw std::runtime_error("required cached index missing");
        if (fs::exists(index_path)) {
            std::cerr << "Loading index " << index_path << "\n";
            index = dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(index_path.c_str()));
            if (!index) throw std::runtime_error("saved index is not ACORN");
        } else {
            std::cerr << "Building ACORN M=" << args.m << " gamma=" << args.gamma
                      << " M_beta=" << args.m_beta << "...\n";
            index = new faiss::IndexACORNFlat(base.d, args.m, args.gamma, args.m_beta);
            const auto started = std::chrono::steady_clock::now();
            index->add(base.n, base.values.data());
            build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            faiss::write_index(index, index_path.c_str());
        }
        if(index->d != base.d || index->ntotal != base.n) throw std::runtime_error("index/vector shape mismatch");
        const uint64_t index_bytes = fs::file_size(index_path);

        if(args.workload_dir.empty()) throw std::runtime_error("shared workload directory required");
        if(fs::exists(csv_path)) throw std::runtime_error("refusing to append to existing CSV");
        const bool new_csv = true;
        std::ofstream csv(csv_path, std::ios::app);
        if (!csv) throw std::runtime_error("cannot open output " + csv_path.string());
        if (new_csv) {
            csv << "dataset,metadata,target_ratio,measured_selectivity,queries,k,M,gamma,M_beta,ef_search,recall,qps,build_seconds,index_bytes,prep_seconds,seed,valid,predicate_violations\n";
        }
        csv << std::setprecision(10);

        for (const double ratio : args.ratios) {
            std::cerr << "Preparing ratio=" << ratio << "...\n";
            std::ostringstream name;
            name << args.dataset << '_' << args.metadata_name << "_r"
                 << std::setw(4) << std::setfill('0') << int(std::llround(ratio*10000)) << ".bin";
            auto workload=read_shared_workload(fs::path(args.workload_dir)/name.str(),
                                              base,meta,args.query_count,args.k,args.gt_threads);
            std::cerr << "Read shared workload; measured_selectivity=" << workload.measured_selectivity
                      << " mask_preparation_seconds=" << workload.prep_seconds << "\n";
            omp_set_num_threads(1);
            std::vector<faiss::idx_t> ids(static_cast<size_t>(args.query_count) * args.k);
            std::vector<float> distances(ids.size());

            const int warmup_count = std::min(20, args.query_count);
            for (const unsigned ef : args.ef_values) {
                index->acorn.efSearch = std::max<unsigned>(ef, static_cast<unsigned>(args.k));
                for (int qi = 0; qi < warmup_count; ++qi) {
                    index->search(
                        1,
                        queries.values.data() + static_cast<size_t>(qi) * queries.d,
                        args.k,
                        distances.data() + static_cast<size_t>(qi) * args.k,
                        ids.data() + static_cast<size_t>(qi) * args.k,
                        workload.masks.data() + static_cast<size_t>(qi) * base.n);
                }
                const auto started = std::chrono::steady_clock::now();
                for (int qi = 0; qi < args.query_count; ++qi) {
                    index->search(
                        1,
                        queries.values.data() + static_cast<size_t>(qi) * queries.d,
                        args.k,
                        distances.data() + static_cast<size_t>(qi) * args.k,
                        ids.data() + static_cast<size_t>(qi) * args.k,
                        workload.masks.data() + static_cast<size_t>(qi) * base.n);
                }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                const double qps = static_cast<double>(args.query_count) / seconds;
                const double recall = recall_at_k(ids, workload.groundtruth, args.k);
                uint64_t predicate_violations=0;
                for (int qi=0; qi<args.query_count; ++qi) {
                    std::unordered_set<faiss::idx_t> seen;
                    for (int rank=0; rank<args.k; ++rank) {
                        auto id=ids[size_t(qi)*args.k+rank];
                        if (id==-1) continue;
                        if (id<0 || size_t(id)>=base.n)
                            throw std::runtime_error("invalid result ID");
                        if (!workload.masks[size_t(qi)*base.n+id]) ++predicate_violations;
                        if (!seen.insert(id).second) throw std::runtime_error("duplicate result");
                    }
                }
                const std::string dump_name=csv_path.string()+".r"+
                    std::to_string(int(std::llround(ratio*10000)))+".ef"+std::to_string(ef)+".ids";
                std::ofstream dump(dump_name,std::ios::binary);
                dump.write(reinterpret_cast<const char*>(ids.data()),ids.size()*sizeof(faiss::idx_t));
                if (!dump) throw std::runtime_error("ID dump failed");

                std::cout << "dataset=" << args.dataset
                          << " metadata=" << args.metadata_name
                          << " target=" << ratio
                          << " actual=" << workload.measured_selectivity
                          << " M=" << args.m
                          << " gamma=" << args.gamma
                          << " M_beta=" << args.m_beta
                          << " ef=" << index->acorn.efSearch
                          << " recall=" << recall
                          << " qps=" << qps << "\n";
                csv << args.dataset << ',' << args.metadata_name << ',' << ratio << ','
                    << workload.measured_selectivity << ',' << args.query_count << ',' << args.k << ','
                    << args.m << ',' << args.gamma << ',' << args.m_beta << ',' << index->acorn.efSearch << ','
                    << recall << ',' << qps << ',' << build_seconds << ',' << index_bytes << ','
                    << workload.prep_seconds << ',' << args.seed << ',' << (predicate_violations==0) << ',' << predicate_violations << '\n';
                if(predicate_violations) std::cerr << "INVALID_PREDICATE ef=" << ef << " violations=" << predicate_violations << "; raw IDs retained, cell excluded from valid performance\n";
                csv.flush();
            }
        }
        delete index;
        std::cerr << "Results written to " << csv_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
