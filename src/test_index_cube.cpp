#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "hnsw-cube.h"
#include "config.h"
using namespace std;
using namespace hnswlib;

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
            // General Parameter
            {"help",                        no_argument,       0, 'h'},

            // Index Parameter
            {"efConstruction",              required_argument, 0, 'e'},
            {"M",                           required_argument, 0, 'm'},

            // Indexing Path
            {"data_path",                   required_argument, 0, 'd'},
            {"index_path",                  required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char source[256] = "";
    char dataset[256] = "";
    char data_path[256] = "";
    char index_path[256] = "";
    char file_type[256] = "fvecs";
    size_t efConstruction = 0;
    size_t M = 0;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
        switch (iarg){
            case 'd':
                if (optarg) {
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if (optarg) {
                    strcpy(source, optarg);
                }
                break;
        }
    }
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
    sprintf(index_path, "%s%s.cube", source, dataset);

    auto *X = new Matrix<float>(data_path);
    hnswlib::HierarchicalNSWCube<float>::static_base_data_ = (char *) X->data;
    size_t D = X->d;
    size_t N = X->n;
    size_t report = 50000;
    L2Space l2space(D);
    auto* appr_alg = new HierarchicalNSWCube<float> (&l2space, N,2,2, HNSW_M, HNSW_efConstruction);
    appr_alg->addPoint(X->data , 0);
    unsigned check_tag = 1;
#pragma omp parallel for schedule(dynamic, 144)
    for(int i=1;i<N;i++){
        appr_alg->addPoint(X->data + i * D, i);
#pragma omp critical
        {
            check_tag++;
            if(check_tag % report == 0){
                cerr << "Processing - " << check_tag << " / " << N << endl;
            }
        }
    }

    appr_alg->saveIndex(index_path);
    return 0;
}