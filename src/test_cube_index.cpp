// Test program for IndexCube
// Demonstrates cube-based filtered vector search with cross-cube edges

#include "IndexCube.h"
#include <iostream>
#include <chrono>
#include <fstream>

// Define static member
template<typename dist_t>
char* hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = nullptr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <base_vectors> <metadata> <query_vectors> <output_index> <attr_dim>" << std::endl;
    std::cout << "  base_vectors: Path to base vectors (.fvecs)" << std::endl;
    std::cout << "  metadata: Path to metadata (.bin)" << std::endl;
    std::cout << "  query_vectors: Path to query vectors (.fvecs)" << std::endl;
    std::cout << "  output_index: Path to save index (.bin)" << std::endl;
    std::cout << "  attr_dim: Attribute dimension (e.g., 2 for 2D)" << std::endl;
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
    if (argc != 6) {
        print_usage(argv[0]);
        return 1;
    }

    const char* base_path = argv[1];
    const char* metadata_path = argv[2];
    const char* query_path = argv[3];
    const char* output_path = argv[4];
    size_t attr_dim = std::atoi(argv[5]);

    std::cout << "========================================" << std::endl;
    std::cout << "Cube-Based Filtered Vector Search Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Base vectors: " << base_path << std::endl;
    std::cout << "Metadata: " << metadata_path << std::endl;
    std::cout << "Query vectors: " << query_path << std::endl;
    std::cout << "Output index: " << output_path << std::endl;
    std::cout << "Attribute dimension: " << attr_dim << std::endl;
    std::cout << std::endl;

    try {
        // Build index
        std::cout << "Building cube index..." << std::endl;
        auto start_build = std::chrono::high_resolution_clock::now();

        IndexCube index(attr_dim, 0);  // vec_dim will be determined from data
        index.build_index(base_path, metadata_path, output_path);

        auto end_build = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
        std::cout << "Build time: " << build_time << " ms" << std::endl;
        std::cout << std::endl;

        // Load queries
        std::cout << "Loading queries..." << std::endl;
        size_t query_dim;
        auto queries = load_queries(query_path, query_dim);
        std::cout << "Loaded " << queries.size() << " queries (dim=" << query_dim << ")" << std::endl;
        std::cout << std::endl;

        // Test queries with different filters
        std::cout << "Testing queries..." << std::endl;
        size_t k = 10;
        size_t num_test_queries = std::min(size_t(5), queries.size());

        for (size_t i = 0; i < num_test_queries; i++) {
            std::cout << "Query " << i << ":" << std::endl;

            // Create a range filter (example: [20, 80] in each dimension)
            CubeBoundingBox filter_bbox(attr_dim);
            for (size_t d = 0; d < attr_dim; d++) {
                filter_bbox.min_bounds[d] = 20.0f;
                filter_bbox.max_bounds[d] = 80.0f;
            }
            CubeQuery query_filter(filter_bbox, queries[i].data());

            // Search
            auto start_search = std::chrono::high_resolution_clock::now();
            auto results = index.search(queries[i].data(), query_filter, k);
            auto end_search = std::chrono::high_resolution_clock::now();
            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();

            std::cout << "  Search time: " << search_time << " μs" << std::endl;
            std::cout << "  Results: " << results.size() << " neighbors" << std::endl;

            // Print top 5 results
            std::vector<std::pair<float, hnswlib::labeltype>> result_vec;
            while (!results.empty()) {
                result_vec.push_back(results.top());
                results.pop();
            }
            std::reverse(result_vec.begin(), result_vec.end());

            for (size_t j = 0; j < std::min(size_t(5), result_vec.size()); j++) {
                std::cout << "    " << j << ": ID=" << result_vec[j].second
                         << ", dist=" << result_vec[j].first << std::endl;
            }
            std::cout << std::endl;
        }

        // Test with radius filter
        std::cout << "Testing with radius filter..." << std::endl;
        if (!queries.empty()) {
            std::vector<float> center(attr_dim, 50.0f);  // Center at (50, 50, ...)
            float radius = 30.0f;
            CubeQuery radius_query(center, radius, queries[0].data());

            auto start_search = std::chrono::high_resolution_clock::now();
            auto results = index.search(queries[0].data(), radius_query, k);
            auto end_search = std::chrono::high_resolution_clock::now();
            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();

            std::cout << "  Radius query: center=(50, 50, ...), radius=" << radius << std::endl;
            std::cout << "  Search time: " << search_time << " μs" << std::endl;
            std::cout << "  Results: " << results.size() << " neighbors" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
