#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "IndexCube.h"
#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswalg.h"
using namespace std;


int main(){
    // Create L2 space
    hnswlib::L2Space l2space(100);

    IndexCube index(&l2space, 100, 2, 10, 32, 200, 2);

    return 0;
}
