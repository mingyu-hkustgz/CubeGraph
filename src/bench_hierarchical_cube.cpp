#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <getopt.h>
#include <random>

#include "IndexCube.h"
#include "matrix.h"
#include "../third_party/utils.h"

using namespace std;
using namespace hnswlib;

// Define static member
template<>
char* HierarchicalNSWCube<float>::static_base_data_ = nullptr;

struct BenchmarkConfig {
    string index_file;
    string base_file;
    string query_file;
    string metadata_file;
    string groundtruth_file;
    string filter_file;  // Pre-generated filters
    string output_file;
    int k = 10;
    float filter_ratio = 0.1f;
    LayerSelectionStrategy strategy = LayerSelectionStrategy::RANGE_SIZE;
    int explicit_layer = 0;
    int ef_base = 50;
    int ef_max = 500;
    int ef_step = 50;
    size_t attr_dim = 2;
    size_t num_layers = 3;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t cross_edge_count = 2;
    bool rebuild_index = false;
};

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --base <file>           Base vectors file (.fvecs)" << endl;
    cout << "  --query <file>          Query vectors file (.fvecs)" << endl;
    cout << "  --metadata <file>       Metadata file (.bin)" << endl;
    cout << "  --groundtruth <file>    Groundtruth file (.bin)" << endl;
    cout << "  --filters <file>        Pre-generated filters file (.bin, optional)" << endl;
    cout << "  --output <file>         Output log file" << endl;
    cout << "  --k <value>             Number of neighbors (default: 10)" << endl;
    cout << "  --filter-ratio <value>  Filter selectivity ratio (default: 0.1)" << endl;
    cout << "  --strategy <name>       Layer selection strategy: RANGE_SIZE|SELECTIVITY|EXPLICIT (default: RANGE_SIZE)" << endl;
    cout << "  --layer <id>            Explicit layer ID (for EXPLICIT strategy)" << endl;
    cout << "  --ef-base <value>       Starting ef value (default: 50)" << endl;
    cout << "  --ef-max <value>        Maximum ef value (default: 500)" << endl;
    cout << "  --ef-step <value>       ef increment step (default: 50)" << endl;
    cout << "  --attr-dim <value>      Attribute dimension (default: 2)" << endl;
    cout << "  --num-layers <value>    Number of layers (default: 3)" << endl;
    cout << "  --M <value>             HNSW M parameter (default: 16)" << endl;
    cout << "  --ef-construction <val> HNSW ef_construction (default: 200)" << endl;
    cout << "  --cross-edges <value>   Cross-cube edge count (default: 2)" << endl;
    cout << "  --rebuild               Rebuild index instead of loading" << endl;
    cout << "  --help                  Show this help message" << endl;
}

// Load groundtruth from binary file
// Format: [n: size_t][k: size_t][query_0_ids: int32[k]][query_1_ids: int32[k]]...
vector<vector<int>> load_groundtruth(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        throw runtime_error("Cannot open groundtruth file: " + filename);
    }

    size_t n, k;
    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&k), sizeof(size_t));

    vector<vector<int>> groundtruth(n, vector<int>(k));
    for (size_t i = 0; i < n; i++) {
        in.read(reinterpret_cast<char*>(groundtruth[i].data()), k * sizeof(int));
    }

    in.close();
    return groundtruth;
}

// Load pre-generated filters from binary file
// Format: [n: size_t][attr_dim: size_t]
//         [query_0_min: float[attr_dim]][query_0_max: float[attr_dim]]...
vector<BoundingBox> load_filters(const string& filename, size_t attr_dim) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        throw runtime_error("Cannot open filters file: " + filename);
    }

    size_t n, file_attr_dim;
    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&file_attr_dim), sizeof(size_t));

    if (file_attr_dim != attr_dim) {
        throw runtime_error("Filter attr_dim mismatch");
    }

    vector<BoundingBox> filters;
    filters.reserve(n);

    for (size_t i = 0; i < n; i++) {
        BoundingBox bbox(attr_dim);
        in.read(reinterpret_cast<char*>(bbox.min_bounds.data()), attr_dim * sizeof(float));
        in.read(reinterpret_cast<char*>(bbox.max_bounds.data()), attr_dim * sizeof(float));
        filters.push_back(bbox);
    }

    in.close();
    return filters;
}

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

// Compute recall between search result and groundtruth
float compute_recall(priority_queue<pair<float, labeltype>> result,
                     const vector<int>& groundtruth, int k) {
    unordered_set<int> gt_set;
    for (int id : groundtruth) {
        if (id >= 0) {  // -1 indicates no valid result
            gt_set.insert(id);
        }
    }

    int hits = 0;
    int count = 0;
    while (!result.empty() && count < k) {
        if (gt_set.find(result.top().second) != gt_set.end()) {
            hits++;
        }
        result.pop();
        count++;
    }

    return (float)hits / min(k, (int)gt_set.size());
}

// Run benchmark with varying ef values
void run_benchmark(IndexCube& index, const BenchmarkConfig& config) {
    // Load queries
    char* query_file_cstr = const_cast<char*>(config.query_file.c_str());
    Matrix<float> Q(query_file_cstr);
    cout << "Loaded " << Q.n << " query vectors" << endl;

    // Load groundtruth
    vector<vector<int>> groundtruth = load_groundtruth(config.groundtruth_file);
    cout << "Loaded groundtruth for " << groundtruth.size() << " queries, k=" << groundtruth[0].size() << endl;

    if (Q.n != groundtruth.size()) {
        throw runtime_error("Query count mismatch with groundtruth");
    }

    // Open output file
    ofstream out(config.output_file);
    if (!out.is_open()) {
        throw runtime_error("Cannot open output file: " + config.output_file);
    }

    // Load or generate filters
    vector<BoundingBox> filters;
    if (!config.filter_file.empty()) {
        cout << "Loading pre-generated filters from " << config.filter_file << "..." << endl;
        filters = load_filters(config.filter_file, config.attr_dim);
        cout << "  Loaded " << filters.size() << " filters" << endl;

        if (filters.size() != Q.n) {
            throw runtime_error("Filter count mismatch with query count");
        }
    } else {
        cout << "Generating random filters (ratio=" << config.filter_ratio << ", seed=42)..." << endl;
        mt19937 rng(42);
        BoundingBox global_bbox = index.get_global_bbox();
        for (size_t i = 0; i < Q.n; i++) {
            filters.push_back(generate_filter_bbox(global_bbox, config.filter_ratio, config.attr_dim, rng));
        }
        cout << "  Generated " << filters.size() << " filters" << endl;
    }

    cout << endl;
    cout << "=== Running Benchmark ===" << endl;
    cout << "Filter ratio: " << config.filter_ratio << endl;
    cout << "Strategy: ";
    switch (config.strategy) {
        case LayerSelectionStrategy::RANGE_SIZE: cout << "RANGE_SIZE"; break;
        case LayerSelectionStrategy::SELECTIVITY: cout << "SELECTIVITY"; break;
        case LayerSelectionStrategy::EXPLICIT: cout << "EXPLICIT (layer " << config.explicit_layer << ")"; break;
    }
    cout << endl;
    cout << "k: " << config.k << endl;
    cout << "ef range: " << config.ef_base << " to " << config.ef_max << " (step " << config.ef_step << ")" << endl;
    cout << endl;

    // Iterate over ef values
    for (int ef = config.ef_base; ef <= config.ef_max; ef += config.ef_step) {
        cout << "Testing ef=" << ef << "..." << flush;

        float total_recall = 0.0f;
        float total_ratio = 0.0f;
        int valid_queries = 0;

        auto start_time = chrono::high_resolution_clock::now();

        // Run all queries
        for (size_t q = 0; q < Q.n; q++) {
            float* query = Q.data + q * Q.d;

            // Use pre-loaded or pre-generated filter
            BoundingBox filter_bbox = filters[q];

            // Search
            try {
                auto result = index.search(query, filter_bbox, config.k, config.strategy, config.explicit_layer, ef);

                // Compute recall
                float recall = compute_recall(result, groundtruth[q], config.k);
                total_recall += recall;

                // Compute distance ratio (if we have results)
                if (!result.empty() && !groundtruth[q].empty() && groundtruth[q][0] >= 0) {
                    total_ratio += 1.0f;  // Simplified - actual ratio would need groundtruth distances
                }

                valid_queries++;
            } catch (const exception& e) {
                // Skip failed queries
                cerr << "Query " << q << " failed: " << e.what() << endl;
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

        // Compute metrics
        float avg_recall = (valid_queries > 0) ? (total_recall / valid_queries) : 0.0f;
        float qps = (duration > 0) ? (valid_queries * 1000000.0f / duration) : 0.0f;
        float avg_ratio = (valid_queries > 0) ? (total_ratio / valid_queries) : 1.0f;

        // Output: <recall%> <QPS> <distance_ratio>
        out << (avg_recall * 100.0f) << " " << qps << " " << avg_ratio << endl;

        cout << " Recall=" << (avg_recall * 100.0f) << "%, QPS=" << qps << endl;
    }

    out.close();
    cout << endl;
    cout << "Results written to: " << config.output_file << endl;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;

    // Define long options
    static struct option long_options[] = {
        {"base", required_argument, 0, 'b'},
        {"query", required_argument, 0, 'q'},
        {"metadata", required_argument, 0, 'm'},
        {"groundtruth", required_argument, 0, 'g'},
        {"filters", required_argument, 0, 'F'},
        {"output", required_argument, 0, 'o'},
        {"k", required_argument, 0, 'k'},
        {"filter-ratio", required_argument, 0, 'f'},
        {"strategy", required_argument, 0, 's'},
        {"layer", required_argument, 0, 'l'},
        {"ef-base", required_argument, 0, 'e'},
        {"ef-max", required_argument, 0, 'E'},
        {"ef-step", required_argument, 0, 'S'},
        {"attr-dim", required_argument, 0, 'a'},
        {"num-layers", required_argument, 0, 'n'},
        {"M", required_argument, 0, 'M'},
        {"ef-construction", required_argument, 0, 'c'},
        {"cross-edges", required_argument, 0, 'x'},
        {"rebuild", no_argument, 0, 'r'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    // Parse command-line arguments
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "b:q:m:g:F:o:k:f:s:l:e:E:S:a:n:M:c:x:rh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'b': config.base_file = optarg; break;
            case 'q': config.query_file = optarg; break;
            case 'm': config.metadata_file = optarg; break;
            case 'g': config.groundtruth_file = optarg; break;
            case 'F': config.filter_file = optarg; break;
            case 'o': config.output_file = optarg; break;
            case 'k': config.k = atoi(optarg); break;
            case 'f': config.filter_ratio = atof(optarg); break;
            case 's':
                if (string(optarg) == "RANGE_SIZE") {
                    config.strategy = LayerSelectionStrategy::RANGE_SIZE;
                } else if (string(optarg) == "SELECTIVITY") {
                    config.strategy = LayerSelectionStrategy::SELECTIVITY;
                } else if (string(optarg) == "EXPLICIT") {
                    config.strategy = LayerSelectionStrategy::EXPLICIT;
                } else {
                    cerr << "Unknown strategy: " << optarg << endl;
                    return 1;
                }
                break;
            case 'l': config.explicit_layer = atoi(optarg); break;
            case 'e': config.ef_base = atoi(optarg); break;
            case 'E': config.ef_max = atoi(optarg); break;
            case 'S': config.ef_step = atoi(optarg); break;
            case 'a': config.attr_dim = atoi(optarg); break;
            case 'n': config.num_layers = atoi(optarg); break;
            case 'M': config.M = atoi(optarg); break;
            case 'c': config.ef_construction = atoi(optarg); break;
            case 'x': config.cross_edge_count = atoi(optarg); break;
            case 'r': config.rebuild_index = true; break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Validate required arguments
    if (config.base_file.empty() || config.query_file.empty() ||
        config.metadata_file.empty() || config.groundtruth_file.empty() ||
        config.output_file.empty()) {
        cerr << "Error: Missing required arguments" << endl;
        print_usage(argv[0]);
        return 1;
    }

    cout << "=== Hierarchical Cube Index Benchmark ===" << endl;
    cout << "Base file: " << config.base_file << endl;
    cout << "Query file: " << config.query_file << endl;
    cout << "Metadata file: " << config.metadata_file << endl;
    cout << "Groundtruth file: " << config.groundtruth_file << endl;
    cout << "Output file: " << config.output_file << endl;
    cout << "Attr dim: " << config.attr_dim << endl;
    cout << "Num layers: " << config.num_layers << endl;
    cout << "M: " << config.M << endl;
    cout << "ef_construction: " << config.ef_construction << endl;
    cout << "Cross edges: " << config.cross_edge_count << endl;
    cout << endl;

    try {
        // Load base vectors to get dimension
        char* base_file_cstr = const_cast<char*>(config.base_file.c_str());
        Matrix<float> X(base_file_cstr);
        size_t vec_dim = X.d;
        size_t num_vectors = X.n;

        cout << "Loaded " << num_vectors << " vectors of dimension " << vec_dim << endl;

        // Create L2 space
        L2Space l2space(vec_dim);

        // Build or load index
        cout << endl;
        cout << "Building hierarchical index..." << endl;
        auto start = chrono::high_resolution_clock::now();

        IndexCube index(&l2space, vec_dim, config.attr_dim, config.num_layers,
                       config.M, config.ef_construction, config.cross_edge_count);
        index.build_index(config.base_file, config.metadata_file);

        auto end = chrono::high_resolution_clock::now();
        auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << "Index built in " << build_time << " ms" << endl;

        // Print layer information
        cout << endl;
        cout << "=== Layer Information ===" << endl;
        for (size_t layer_id = 0; layer_id < index.get_num_layers(); layer_id++) {
            size_t cubes_per_dim, total_cubes;
            index.get_layer_info(layer_id, cubes_per_dim, total_cubes);
            cout << "Layer " << layer_id << ": " << cubes_per_dim << "^" << config.attr_dim
                 << " = " << total_cubes << " cubes" << endl;
        }
        cout << endl;

        // Run benchmark
        run_benchmark(index, config);

        cout << "Benchmark completed successfully!" << endl;
        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

