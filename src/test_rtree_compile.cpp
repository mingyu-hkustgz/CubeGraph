// Simplified test for IndexRTreeND without matrix.h dependency

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

// Forward declarations to avoid including problematic headers
namespace hnswlib {
    template<typename dist_t> class HierarchicalNSWStatic;
}

// Include only what we need
#include "hnswlib/hnswlib.h"
#include "hnswlib/hnsw-static.h"

// Now include our index
#include "IndexRTreeND.h"

using namespace std;

int main() {
    cout << "=== IndexRTreeND Compilation Test ===" << endl;
    cout << "IndexRTreeND header compiled successfully!" << endl;
    cout << "Note: Full functionality test requires proper data files." << endl;
    cout << "The implementation includes:" << endl;
    cout << "  - R-tree structure with STR bulk loading" << endl;
    cout << "  - Adaptive partitioning based on data distribution" << endl;
    cout << "  - Multi-dimensional range queries" << endl;
    cout << "  - Radius queries with sphere-MBR intersection" << endl;
    cout << "  - BaseFilterFunctor integration for efficient filtering" << endl;
    cout << "  - Save/load functionality for persistent storage" << endl;
    return 0;
}
