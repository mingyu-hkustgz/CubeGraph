// Benchmark program for IndexCube
// Measures recall and QPS with different ef values

#include "IndexCube.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <unordered_set>
#include <getopt.h>

// Define static member - using HierarchicalNSWStatic
template<typename dist_t>
char* hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = nullptr;

struct Config {
    const char* base_path = nullptr;
    const char* metadata_path = nullptr;
    const char* query_path = nullptr;
    const char* groundtruth_path = nullptr;
    const char* output_path = nullptr;
    size_t attr_dim = 2;
    size_t k = 20;
    size_t ef = 100;
    bool build_only = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -d <base_path>       Base vectors path" << std::endl;
    std::cout << "  -m <metadata_path>  Metadata path" << std::endl;
    std::cout << "  -q <query_path>     Query vectors path" << std::endl;
    std::cout << "  -g <gt_path>        Groundtruth path" << std::endl;
    std::cout << "  -o <output_path>    Output index path" << std::endl;
    std::cout << "  -a <attr_dim>       Attribute dimension (default: 2)" << std::endl;
    std::cout << "  -k <k>              K neighbors (default: 20)" << std::endl;
    std::cout << "  -e <ef>             EF search parameter (default: 100)" << std::endl;
    std::cout << "  -b                  Build index only" << std::endl;
}

// Load groundtruth
std::vector<std::vector<int>> load_groundtruth(const char* gt_path, size_t k) {
    std::vector<std::vector<int>> gt;

    std::ifstream fin(gt_path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open groundtruth file: " << gt_path << std::endl;
        return gt;
    }

    size_t n, d;
    fin.read((char*)&n, sizeof(size_t));  // num queries
    fin.read((char*)&d, sizeof(size_t));  // k

    gt.resize(n);
    for (size_t i = 0; i < n; i++) {
        gt[i].resize(d);
        fin.read((char*)gt[i].data(), d * sizeof(int));
    }

    fin.close();
    return gt;
}

// Calculate recall
float calculate_recall(const std::vector<int>& result, const std::vector<int>& groundtruth) {
    std::unordered_set<int> gt_set;
    for (int id : groundtruth) {
        gt_set.insert(id);
    }

    int correct = 0;
    for (int id : result) {
        if (gt_set.find(id) != gt_set.end()) {
            correct++;
        }
    }

    return (float)correct / groundtruth.size() * 100.0f;
}

// Load query vectors
std::vector<std::vector<float>> load_queries(const char* query_path, size_t& dim) {
    std::ifstream fin(query_path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open query file");
    }

    std::vector<std::vector<float>> queries;

    while (true) {
        int d;
        fin.read((char*)&d, sizeof(int));
        if (fin.eof()) break;

        dim = d;
        std::vector<float> query(d);
        fin.read((char*)query.data(), d * sizeof(float));
        queries.push_back(query);
    }

    fin.close();
    return queries;
}

int main(int argc, char** argv) {
    Config config;

    int opt;
    while ((opt = getopt(argc, argv, "d:m:q:g:o:a:k:e:b")) != -1) {
        switch (opt) {
            case 'd': config.base_path = optarg; break;
            case 'm': config.metadata_path = optarg; break;
            case 'q': config.query_path = optarg; break;
            case 'g': config.groundtruth_path = optarg; break;
            case 'o': config.output_path = optarg; break;
            case 'a': config.attr_dim = atoi(optarg); break;
            case 'k': config.k = atoi(optarg); break;
            case 'e': config.ef = atoi(optarg); break;
            case 'b': config.build_only = true; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (!config.base_path || !config.metadata_path || !config.query_path || !config.output_path) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "CubeGraph Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "EF: " << config.ef << std::endl;
    std::cout << "K: " << config.k << std::endl;
    std::cout << std::endl;

    try {
        // Build index
        std::cout << "Building cube index..." << std::endl;
        auto start_build = std::chrono::high_resolution_clock::now();

        IndexCube index(config.attr_dim, 0);
        index.build_index(config.base_path, config.metadata_path, config.output_path);

        auto end_build = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
        std::cout << "Build time: " << build_time << " ms" << std::endl;
        std::cout << std::endl;

        if (config.build_only) {
            return 0;
        }

        // Load queries
        std::cout << "Loading queries..." << std::endl;
        size_t query_dim;
        auto queries = load_queries(config.query_path, query_dim);
        std::cout << "Loaded " << queries.size() << " queries (dim=" << query_dim << ")" << std::endl;
        std::cout << std::endl;

        // Load groundtruth if available
        float avg_recall = 0.0f;
        if (config.groundtruth_path) {
            std::cout << "Loading groundtruth..." << std::endl;
            auto gt = load_groundtruth(config.groundtruth_path, config.k);
            if (!gt.empty()) {
                std::cout << "Loaded groundtruth for " << gt.size() << " queries" << std::endl;
                std::cout << std::endl;

                // Test queries with recall
                std::cout << "Testing queries with recall..." << std::endl;
                float total_recall = 0.0f;
                long long total_time_us = 0;
                size_t num_queries = queries.size();

                for (size_t i = 0; i < num_queries; i++) {
                    // Create range filter [20, 80] in each dimension
                    CubeBoundingBox filter_bbox(config.attr_dim);
                    for (size_t d = 0; d < config.attr_dim; d++) {
                        filter_bbox.min_bounds[d] = 20.0f;
                        filter_bbox.max_bounds[d] = 80.0f;
                    }
                    CubeQuery query_filter(filter_bbox, queries[i].data());

                    auto start_search = std::chrono::high_resolution_clock::now();
                    auto results = index.search(queries[i].data(), query_filter, config.k);
                    auto end_search = std::chrono::high_resolution_clock::now();

                    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
                    total_time_us += search_time;

                    // Get result IDs
                    std::vector<int> result_ids;
                    while (!results.empty()) {
                        result_ids.push_back(results.top().second);
                        results.pop();
                    }
                    std::reverse(result_ids.begin(), result_ids.end());

                    // Calculate recall
                    if (i < gt.size()) {
                        float recall = calculate_recall(result_ids, gt[i]);
                        total_recall += recall;
                    }
                }

                avg_recall = total_recall / num_queries;
                double qps = num_queries * 1000000.0 / total_time_us;

                std::cout << std::endl;
                std::cout << "Results:" << std::endl;
                std::cout << "  Recall: " << avg_recall << "%" << std::endl;
                std::cout << "  QPS: " << qps << std::endl;
                std::cout << "  Total time: " << (total_time_us / 1000.0) << " ms" << std::endl;
                std::cout << std::endl;
                std::cout << avg_recall << " " << qps << std::endl;
            } else {
                std::cerr << "Failed to load groundtruth, skipping recall calculation" << std::endl;
            }
        } else {
            // No groundtruth - just measure QPS
            std::cout << "Testing queries (QPS only)..." << std::endl;
            long long total_time_us = 0;
            size_t num_queries = queries.size();

            for (size_t i = 0; i < num_queries; i++) {
                CubeBoundingBox filter_bbox(config.attr_dim);
                for (size_t d = 0; d < config.attr_dim; d++) {
                    filter_bbox.min_bounds[d] = 20.0f;
                    filter_bbox.max_bounds[d] = 80.0f;
                }
                CubeQuery query_filter(filter_bbox, queries[i].data());

                auto start_search = std::chrono::high_resolution_clock::now();
                auto results = index.search(queries[i].data(), query_filter, config.k);
                auto end_search = std::chrono::high_resolution_clock::now();

                auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
                total_time_us += search_time;
            }

            double qps = num_queries * 1000000.0 / total_time_us;

            std::cout << std::endl;
            std::cout << "Results:" << std::endl;
            std::cout << "  QPS: " << qps << std::endl;
            std::cout << "  Total time: " << (total_time_us / 1000.0) << " ms" << std::endl;
            std::cout << std::endl;
            // Output format for comparison: recall=0 QPS
            std::cout << "0 " << qps << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
