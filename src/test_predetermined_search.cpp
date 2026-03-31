#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>
#include <vector>
#include "hnswlib/hnswlib.h"
#include "hnsw-cube.h"
#include "IndexCube.h"
#include "utils.h"
#include "matrix.h"
#include "config.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <data_file> <metadata_file> <attr_dim>" << endl;
        return 1;
    }

    char *data_file = argv[1];
    char *meta_file = argv[2];
    size_t attr_dim = atoi(argv[3]);

    cout << "Testing fly_search vs predetermined_search with attr_dim=" << attr_dim << endl;
    cout << "Data: " << data_file << endl;
    cout << "Metadata: " << meta_file << endl;

    // Build index
    size_t num_layers = 3;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t cross_edge_count = 2;

    IndexCube index(num_layers, M, ef_construction, cross_edge_count);

    cout << "\nBuilding index..." << endl;
    auto start = chrono::high_resolution_clock::now();
    index.build_index(data_file, meta_file);
    auto end = chrono::high_resolution_clock::now();
    auto build_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Index built in " << build_time << " ms" << endl;

    // Get global bbox
    BoundingBox global_bbox = index.get_global_bbox();

    // Generate a sphere filter
    std::vector<float> center(attr_dim);
    float max_range = 0.0f;
    for (size_t d = 0; d < attr_dim; d++) {
        float range = global_bbox.max_bounds[d] - global_bbox.min_bounds[d];
        max_range = std::max(max_range, range);
        center[d] = (global_bbox.min_bounds[d] + global_bbox.max_bounds[d]) / 2.0f;
    }
    float radius = max_range * 0.2f;  // 20% of max range
    RadiusFilterParams sphere_filter(center, radius, attr_dim);

    cout << "\nSphere filter: radius=" << radius << ", center=[";
    for (size_t d = 0; d < attr_dim; d++) {
        cout << center[d] << (d < attr_dim - 1 ? ", " : "");
    }
    cout << "]" << endl;

    // Create a query vector (use first vector from data)
    Matrix<float> X(data_file);
    float *query_vector = X.data;

    cout << "\n=== Testing select_layer_with_Radius ===" << endl;
    size_t selected_layer = index.select_layer_with_Radius(&sphere_filter);
    cout << "select_layer_with_Radius returned: " << selected_layer << endl;

    cout << "\n=== Testing fly_search (USE_FLY_SEARCH=true) ===" << endl;
    cout << "Testing fly_search with query vector..." << endl;
    try {
        auto result = index.fly_search(query_vector, 10, &sphere_filter);
        cout << "fly_search returned " << result.size() << " results" << endl;
    } catch (const exception& e) {
        cout << "fly_search EXCEPTION: " << e.what() << endl;
    }

    cout << "\n=== Testing predetermined_search (USE_FLY_SEARCH=false) ===" << endl;
    cout << "Testing predetermined_search with query vector..." << endl;
    try {
        auto result = index.predetermined_search(query_vector, 10, &sphere_filter);
        cout << "predetermined_search returned " << result.size() << " results" << endl;
    } catch (const exception& e) {
        cout << "predetermined_search EXCEPTION: " << e.what() << endl;
    }

    cout << "\n=== Tests complete! ===" << endl;
    return 0;
}
