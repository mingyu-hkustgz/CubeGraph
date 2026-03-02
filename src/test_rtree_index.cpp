#include "IndexRTreeND.h"
#include <iostream>
#include <chrono>
#include <random>

using namespace std;

// Generate random metadata (attribute values)
void generate_metadata(const char* output_path, size_t n, size_t attr_dim) {
    ofstream fout(output_path, ios::binary);
    fout.write((char*)&n, sizeof(size_t));
    fout.write((char*)&attr_dim, sizeof(size_t));

    mt19937 gen(42);
    uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < attr_dim; j++) {
            float val = dist(gen);
            fout.write((char*)&val, sizeof(float));
        }
    }
    fout.close();
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <data_path> <metadata_path> <index_path> <attr_dim>" << endl;
        cerr << "Example: " << argv[0] << " data.fvecs metadata.bin index.rtree 2" << endl;
        return 1;
    }

    char* data_path = argv[1];
    char* metadata_path = argv[2];
    char* index_path = argv[3];
    size_t attr_dim = atoi(argv[4]);

    cout << "=== R-tree Index Test ===" << endl;
    cout << "Data path: " << data_path << endl;
    cout << "Metadata path: " << metadata_path << endl;
    cout << "Index path: " << index_path << endl;
    cout << "Attribute dimension: " << attr_dim << endl;

    // Load data to get dimensions
    Matrix<float> X(const_cast<char*>(data_path));
    size_t vec_dim = X.d;
    size_t num_vectors = X.n;

    cout << "\nDataset info:" << endl;
    cout << "  Vectors: " << num_vectors << endl;
    cout << "  Vector dimension: " << vec_dim << endl;

    // Generate metadata if it doesn't exist
    ifstream test_meta(metadata_path);
    if (!test_meta.good()) {
        cout << "\nGenerating random metadata..." << endl;
        generate_metadata(metadata_path, num_vectors, attr_dim);
    }
    test_meta.close();

    // Build index
    cout << "\nBuilding R-tree index..." << endl;
    auto start = chrono::high_resolution_clock::now();

    IndexRTreeND index(attr_dim, vec_dim);
    index.build_index(data_path, metadata_path, index_path);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Index built in " << duration.count() << " seconds" << endl;

    // Load index
    cout << "\nLoading index..." << endl;
    IndexRTreeND loaded_index(attr_dim, vec_dim);
    loaded_index.load_index(index_path, metadata_path);
    cout << "Index loaded successfully" << endl;

    // Test range query
    cout << "\n=== Testing Range Query ===" << endl;
    MBR query_mbr(attr_dim);
    for (size_t i = 0; i < attr_dim; i++) {
        query_mbr.min_bounds[i] = 20.0f;
        query_mbr.max_bounds[i] = 40.0f;
    }

    vector<float> query_vec(vec_dim, 0.5f);
    size_t k = 10;

    start = chrono::high_resolution_clock::now();
    auto results = loaded_index.range_search(query_vec.data(), query_mbr, k, 100);
    end = chrono::high_resolution_clock::now();
    auto query_duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Range query completed in " << query_duration.count() << " μs" << endl;
    cout << "Found " << results.size() << " results" << endl;
    cout << "Top 5 results:" << endl;
    int count = 0;
    while (!results.empty() && count < 5) {
        auto [dist, label] = results.top();
        results.pop();
        cout << "  " << count + 1 << ". Label: " << label << ", Distance: " << dist << endl;
        count++;
    }

    // Test radius query
    cout << "\n=== Testing Radius Query ===" << endl;
    vector<float> center(attr_dim, 50.0f);
    float radius = 20.0f;

    start = chrono::high_resolution_clock::now();
    results = loaded_index.radius_search(query_vec.data(), center, radius, k, 100);
    end = chrono::high_resolution_clock::now();
    query_duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Radius query completed in " << query_duration.count() << " μs" << endl;
    cout << "Found " << results.size() << " results within radius " << radius << endl;
    cout << "Top 5 results:" << endl;
    count = 0;
    while (!results.empty() && count < 5) {
        auto [dist, label] = results.top();
        results.pop();
        cout << "  " << count + 1 << ". Label: " << label << ", Distance: " << dist << endl;
        count++;
    }

    cout << "\n=== Test completed ===" << endl;

    return 0;
}
