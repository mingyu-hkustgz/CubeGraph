// 0.5% extension of the frozen fixed-cardinality box generator.
// Same seed-42 object anchors and stored float32 boxes; exhaustive float64 GT.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Args {
    fs::path data_root = "/home/bld/Research/CubeGraph/DATA";
    fs::path output_dir = "/home/bld/Research/codex-workloads";
    std::string dataset = "sift";
    std::string metadata = "uniform_2d";
    int queries = 1000;
    int k = 20;
    int threads = 72;
    uint32_t seed = 42;
    std::string mode = "geometric";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--data-root") a.data_root = next();
        else if (key == "--output-dir") a.output_dir = next();
        else if (key == "--dataset") a.dataset = next();
        else if (key == "--metadata") a.metadata = next();
        else if (key == "--queries") a.queries = std::stoi(next());
        else if (key == "--k") a.k = std::stoi(next());
        else if (key == "--threads") a.threads = std::stoi(next());
        else if (key == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (key == "--mode") a.mode = next();
        else if (key == "--help") {
            std::cout << "Generate deterministic shared range-filter workloads with exact ground truth\n"
                      << "  --dataset sift --metadata uniform_2d --queries 1000 --k 20\n"
                      << "  --data-root /path/to/DATA --output-dir /path/to/workloads\n"
                      << "  --mode geometric|paper-perturbed|fixed-cardinality\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + key);
    }
    if (a.queries <= 0 || a.k <= 0 || a.threads <= 0)
        throw std::runtime_error("queries, k, and threads must be positive");
    if (a.mode != "geometric" && a.mode != "paper-perturbed" && a.mode != "fixed-cardinality")
        throw std::runtime_error("mode must be geometric, paper-perturbed, or fixed-cardinality");
    return a;
}

class FvecsView {
public:
    explicit FvecsView(const fs::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) throw std::runtime_error("bad fvecs stat");
        bytes_ = static_cast<size_t>(st.st_size);
        data_ = static_cast<const uint8_t*>(::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            throw std::runtime_error("mmap failed for " + path.string());
        }
        int32_t d = 0;
        std::memcpy(&d, data_, sizeof(d));
        if (d <= 0) throw std::runtime_error("bad fvecs dimension");
        d_ = static_cast<size_t>(d);
        row_bytes_ = (d_ + 1) * sizeof(float);
        if (bytes_ % row_bytes_ != 0) throw std::runtime_error("bad fvecs size");
        n_ = bytes_ / row_bytes_;
    }
    ~FvecsView() {
        if (data_) ::munmap(const_cast<uint8_t*>(data_), bytes_);
        if (fd_ >= 0) ::close(fd_);
    }
    FvecsView(const FvecsView&) = delete;
    FvecsView& operator=(const FvecsView&) = delete;
    size_t n() const { return n_; }
    size_t d() const { return d_; }
    const float* row(size_t i) const {
        return reinterpret_cast<const float*>(data_ + i * row_bytes_ + sizeof(int32_t));
    }
private:
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t bytes_ = 0;
    size_t row_bytes_ = 0;
    size_t n_ = 0;
    size_t d_ = 0;
};

struct Metadata {
    size_t n = 0;
    size_t d = 0;
    std::vector<float> values;
};

static Metadata read_metadata(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    Metadata m;
    in.read(reinterpret_cast<char*>(&m.n), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&m.d), sizeof(size_t));
    if (!in || m.n == 0 || m.d == 0 || m.d > 64) throw std::runtime_error("bad metadata header");
    m.values.resize(m.n * m.d);
    in.read(reinterpret_cast<char*>(m.values.data()), m.values.size() * sizeof(float));
    if (!in) throw std::runtime_error("truncated metadata");
    return m;
}

struct Box {
    std::vector<float> lo;
    std::vector<float> hi;
};

static std::vector<Box> make_boxes(int count, double ratio, const Metadata& m, uint32_t seed) {
    std::vector<float> glo(m.d, std::numeric_limits<float>::infinity());
    std::vector<float> ghi(m.d, -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < m.n; ++i) {
        for (size_t j = 0; j < m.d; ++j) {
            const float v = m.values[i * m.d + j];
            glo[j] = std::min(glo[j], v);
            ghi[j] = std::max(ghi[j], v);
        }
    }
    std::mt19937 rng(seed);
    const double side_fraction = std::pow(ratio, 1.0 / static_cast<double>(m.d));
    std::vector<Box> boxes;
    boxes.reserve(static_cast<size_t>(count));
    for (int qi = 0; qi < count; ++qi) {
        Box b{std::vector<float>(m.d), std::vector<float>(m.d)};
        for (size_t j = 0; j < m.d; ++j) {
            const double width = (ghi[j] - glo[j]) * side_fraction;
            const double half = width / 2.0;
            std::uniform_real_distribution<double> center(glo[j] + half, ghi[j] - half);
            const double c = center(rng);
            b.lo[j] = static_cast<float>(c - half);
            b.hi[j] = static_cast<float>(c + half);
        }
        boxes.push_back(std::move(b));
    }
    return boxes;
}

// Reproduce src/bench_hierarchical_cube.cpp's per-axis +/-20% side variation.
static std::vector<Box> make_paper_perturbed_boxes(
        int count, double ratio, const Metadata& m, uint32_t seed) {
    std::vector<float> glo(m.d, std::numeric_limits<float>::infinity());
    std::vector<float> ghi(m.d, -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < m.n; ++i) {
        for (size_t j = 0; j < m.d; ++j) {
            const float v = m.values[i * m.d + j];
            glo[j] = std::min(glo[j], v);
            ghi[j] = std::max(ghi[j], v);
        }
    }
    std::mt19937 rng(seed);
    std::vector<Box> boxes;
    boxes.reserve(static_cast<size_t>(count));
    for (int qi = 0; qi < count; ++qi) {
        Box b{std::vector<float>(m.d), std::vector<float>(m.d)};
        for (size_t j = 0; j < m.d; ++j) {
            const float range = ghi[j] - glo[j];
            const float base_size = range * std::pow(static_cast<float>(ratio), 1.0 / m.d);
            std::uniform_real_distribution<float> fluctuation(0.8f, 1.2f);
            const float size = base_size * fluctuation(rng);
            std::uniform_real_distribution<float> center(glo[j] + size / 2.0f,
                                                          ghi[j] - size / 2.0f);
            const float c = center(rng);
            b.lo[j] = c - size / 2.0f;
            b.hi[j] = c + size / 2.0f;
        }
        boxes.push_back(std::move(b));
    }
    return boxes;
}

template <size_t R>
static std::array<std::vector<Box>, R> make_fixed_cardinality_boxes(
        size_t count, const std::array<double, R>& ratios, const Metadata& m,
        int k, int threads, uint32_t seed) {
    std::vector<float> glo(m.d, std::numeric_limits<float>::infinity());
    std::vector<float> ghi(m.d, -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < m.n; ++i) {
        for (size_t j = 0; j < m.d; ++j) {
            const float v = m.values[i * m.d + j];
            glo[j] = std::min(glo[j], v);
            ghi[j] = std::max(ghi[j], v);
        }
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> choose(0, m.n - 1);
    std::vector<size_t> center_ids(count);
    for (size_t qi = 0; qi < count; ++qi) center_ids[qi] = choose(rng);

    std::array<std::vector<Box>, R> boxes;
    for (auto& by_ratio : boxes) by_ratio.resize(count);
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t qii = 0; qii < static_cast<int64_t>(count); ++qii) {
        const size_t qi = static_cast<size_t>(qii);
        const float* center = m.values.data() + center_ids[qi] * m.d;
        std::vector<float> radius(m.n);
        for (size_t id = 0; id < m.n; ++id) {
            const float* row = m.values.data() + id * m.d;
            float normalized = 0.0f;
            for (size_t j = 0; j < m.d; ++j) {
                const float span = ghi[j] - glo[j];
                const float delta = span > 0 ? std::abs(row[j] - center[j]) / span : 0.0f;
                normalized = std::max(normalized, delta);
            }
            radius[id] = normalized;
        }
        size_t lower = 0;
        for (size_t r = 0; r < R; ++r) {
            const size_t target = std::min(m.n, std::max<size_t>(
                static_cast<size_t>(k), static_cast<size_t>(std::llround(ratios[r] * m.n))));
            const size_t nth = target - 1;
            std::nth_element(radius.begin() + static_cast<std::ptrdiff_t>(lower),
                             radius.begin() + static_cast<std::ptrdiff_t>(nth), radius.end());
            const float scale = radius[nth];
            Box b{std::vector<float>(m.d), std::vector<float>(m.d)};
            for (size_t j = 0; j < m.d; ++j) {
                const float half = scale * (ghi[j] - glo[j]);
                b.lo[j] = std::max(glo[j], center[j] - half);
                b.hi[j] = std::min(ghi[j], center[j] + half);
            }
            boxes[r][qi] = std::move(b);
            lower = target;
        }
    }
    return boxes;
}

static bool passes(const Metadata& m, size_t id, const Box& b) {
    const float* row = m.values.data() + id * m.d;
    for (size_t j = 0; j < m.d; ++j)
        if (row[j] < b.lo[j] || row[j] > b.hi[j]) return false;
    return true;
}

static double l2(const float* a, const float* b, size_t d) {
    double result = 0.0;
    #pragma omp simd reduction(+:result)
    for (size_t j = 0; j < d; ++j) {
        const double diff = double(a[j]) - double(b[j]);
        result += diff * diff;
    }
    return result;
}

static void write_workload(const fs::path& path, const std::vector<Box>& boxes,
                           const std::vector<int64_t>& gt, int k) {
    if (fs::exists(path)) throw std::runtime_error("refusing to overwrite workload");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create " + path.string());
    const uint64_t nq = boxes.size();
    const uint64_t dims = boxes.front().lo.size();
    const uint64_t topk = static_cast<uint64_t>(k);
    out.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    out.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(&topk), sizeof(topk));
    for (const auto& b : boxes) {
        out.write(reinterpret_cast<const char*>(b.lo.data()), dims * sizeof(float));
        out.write(reinterpret_cast<const char*>(b.hi.data()), dims * sizeof(float));
    }
    out.write(reinterpret_cast<const char*>(gt.data()), gt.size() * sizeof(int64_t));
    if (!out) throw std::runtime_error("failed writing " + path.string());
}

int main(int argc, char** argv) {
    try {
        const Args a = parse_args(argc, argv);
        const fs::path dir = a.data_root / a.dataset;
        const fs::path base_path = dir / (a.dataset + "_base.fvecs");
        const fs::path query_path = dir / (a.dataset + "_query.fvecs");
        const fs::path meta_path = dir / (a.dataset + "_metadata_" + a.metadata + ".bin");
        FvecsView base(base_path), queries(query_path);
        const Metadata meta = read_metadata(meta_path);
        const size_t nq = std::min<size_t>(static_cast<size_t>(a.queries), queries.n());
        if (base.n() != meta.n || base.d() != queries.d()) throw std::runtime_error("data shape mismatch");

        constexpr size_t R = 1;
        const std::array<double, R> ratios{0.005};
        const std::array<std::string, R> tags{"r0050"};
        std::array<std::vector<Box>, R> boxes;
        std::array<std::vector<int64_t>, R> gt;
        std::array<std::vector<uint64_t>, R> selected;
        if (a.mode == "fixed-cardinality") {
            boxes = make_fixed_cardinality_boxes(nq, ratios, meta, a.k, a.threads, a.seed);
        } else if (a.mode == "paper-perturbed") {
            for (size_t r = 0; r < R; ++r)
                boxes[r] = make_paper_perturbed_boxes(static_cast<int>(nq), ratios[r], meta, a.seed);
        } else {
            for (size_t r = 0; r < R; ++r)
                boxes[r] = make_boxes(static_cast<int>(nq), ratios[r], meta, a.seed);
        }
        for (size_t r = 0; r < R; ++r) {
            gt[r].assign(nq * static_cast<size_t>(a.k), -1);
            selected[r].assign(nq, 0);
        }

        const auto started = std::chrono::steady_clock::now();
        omp_set_num_threads(a.threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t qii = 0; qii < static_cast<int64_t>(nq); ++qii) {
            const size_t qi = static_cast<size_t>(qii);
            using Candidate = std::pair<double, int64_t>;
            std::array<std::priority_queue<Candidate>, R> best;
            const float* q = queries.row(qi);
            for (size_t id = 0; id < base.n(); ++id) {
                std::array<bool, R> match{};
                bool any = false;
                for (size_t r = 0; r < R; ++r) {
                    match[r] = passes(meta, id, boxes[r][qi]);
                    if (match[r]) { ++selected[r][qi]; any = true; }
                }
                if (!any) continue;
                const double distance = l2(q, base.row(id), base.d());
                for (size_t r = 0; r < R; ++r) {
                    if (!match[r]) continue;
                    auto& heap = best[r];
                    if (heap.size() < static_cast<size_t>(a.k)) heap.emplace(distance, static_cast<int64_t>(id));
                    else if (Candidate{distance, static_cast<int64_t>(id)} < heap.top()) {
                        heap.pop();
                        heap.emplace(distance, static_cast<int64_t>(id));
                    }
                }
            }
            for (size_t r = 0; r < R; ++r) {
                std::vector<Candidate> ordered;
                while (!best[r].empty()) { ordered.push_back(best[r].top()); best[r].pop(); }
                std::sort(ordered.begin(), ordered.end());
                for (size_t rank = 0; rank < ordered.size(); ++rank)
                    gt[r][qi * static_cast<size_t>(a.k) + rank] = ordered[rank].second;
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        fs::create_directories(a.output_dir);
        const fs::path stats_path = a.output_dir / "workload_stats.csv";
        const bool new_stats = !fs::exists(stats_path);
        std::ofstream stats(stats_path, std::ios::app);
        if (new_stats) stats << "dataset,metadata,mode,target_ratio,queries,k,measured_selectivity,min_candidates,max_candidates,prep_seconds,seed\n";
        stats << std::setprecision(10);
        for (size_t r = 0; r < R; ++r) {
            const fs::path out = a.output_dir / (a.dataset + "_" + a.metadata + "_" + tags[r] + ".bin");
            write_workload(out, boxes[r], gt[r], a.k);
            const uint64_t total = std::accumulate(selected[r].begin(), selected[r].end(), uint64_t{0});
            const auto [mn, mx] = std::minmax_element(selected[r].begin(), selected[r].end());
            const double measured = static_cast<double>(total) / static_cast<double>(nq * base.n());
            stats << a.dataset << ',' << a.metadata << ',' << a.mode << ',' << ratios[r] << ',' << nq << ',' << a.k << ','
                  << measured << ',' << *mn << ',' << *mx << ',' << seconds << ',' << a.seed << '\n';
            std::cout << out.filename().string() << " selectivity=" << measured
                      << " candidates=[" << *mn << ',' << *mx << "]\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
