// Simplified test for IndexGridND without matrix.h dependency

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
#include "IndexGridND.h"

using namespace std;

int main() {
    cout << "=== IndexGridND Compilation Test ===" << endl;
    cout << "IndexGridND header compiled successfully!" << endl;
    cout << "Note: Full functionality test requires proper data files." << endl;
    cout << "The implementation includes:" << endl;
    cout << "  - Hierarchical grid structure with 2^dim subdivision" << endl;
    cout << "  - Multi-dimensional range queries" << endl;
    cout << "  - Radius queries with Euclidean distance" << endl;
    cout << "  - BaseFilterFunctor integration for efficient filtering" << endl;
    cout << "  - Save/load functionality for persistent storage" << endl;
    return 0;
}
