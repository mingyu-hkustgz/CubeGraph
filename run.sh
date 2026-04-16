source set.sh

bash mkdir.sh
#
#
#bash tests/run_cube.sh

#for data in "${datasets[@]}"; do
#  echo "Indexing - ${data}"
#
#  ./build/src/index_hnsw -d ${data} -s "./DATA/${data}/"
#
#done
#
#for K in {20,100}; do
#  for data in "${datasets[@]}"; do
#    echo "Searching - ${data}"
#
#    ./build/src/search_hnsw -d ${data} -s "./DATA/${data}/" -k ${K}
#
#  done
#done


#for data in "${datasets[@]}"; do
#  echo "Indexing - ${data}"
#
#  ./build/src/index_cube -d ${data} -s "./DATA/${data}/"
#
#done
#
#for K in {20,100}; do
#  for data in "${datasets[@]}"; do
#    echo "Searching - ${data}"
#
#    ./build/src/search_cube -d ${data} -s "./DATA/${data}/" -k ${K}
#
#  done
#done


#for data in "${datasets[@]}"; do
#  echo "Indexing - ${data}"
#
#  ./build/src/link_adjacent_cube -d ${data} -s "./DATA/${data}/"
#
#done
#
#for K in {20,100}; do
#  for data in "${datasets[@]}"; do
#    echo "Searching - ${data}"
#
#    ./build/src/cross_cube_search -d ${data} -s "./DATA/${data}/" -k ${K}
#
#  done
#done

#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} Cube"
#
##  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d"
##
##  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_2d"
#
##  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_2d"
#
#done
#
#
#
#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} (ball)"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.2 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_3d"
#
#done
#
#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} Cube"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_4d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_4d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.2 -m "uniform_4d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_4d"
#
#done

#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} (ball)"
#
#  ./build/src/bench_hierarchical_polygon -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d" -v 3
#
#  ./build/src/bench_hierarchical_polygon -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d" -v 3
#
#  ./build/src/bench_hierarchical_polygon -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d" -v 4
#
#  ./build/src/bench_hierarchical_polygon -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d" -v 4
#
#  ./build/src/bench_hierarchical_polygon -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d" -v 5
#
#  ./build/src/bench_hierarchical_polygon -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d" -v 5
#
#done

#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} (ball)"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.2 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_3d"
#
#done
#
#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} (ball)"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.2 -m "uniform_3d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_3d"
#
#done


# YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.05 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.05 -m "real_3d"
#
#
## YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.1 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.1 -m "real_3d"
#
#
# YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.02 -m "real_2d"
#
#
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_post_filtering -d yfcc -s "./DATA/yfcc/" -f 0.02 -m "real_2d"

# YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.2 -m "real_3d"
#
# YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.1 -m "real_3d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.01 -m "real_3d"
#
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_post_filtering -d yfcc -s "./DATA/yfcc/" -f 0.1 -m "real_3d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_post_filtering -d yfcc -s "./DATA/yfcc/" -f 0.01 -m "real_3d"
#
#
## YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.05 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.05 -m "real_3d"
#
#
## YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.1 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.1 -m "real_3d"
#
#
## YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.2 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.2 -m "real_3d"
#
## YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.01 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_ball -d yfcc -s "./DATA/yfcc/" -f 0.01 -m "real_3d"

#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} (Cube)"
#
#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_3d"
#
#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_3d"
#
#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_4d"
#
#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_4d"
#
#done


# for data in "${datasets[@]}"; do
#   echo "Searching - ${data} (Cube)"
#
##   ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_2d"
##
##   ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_2d"
#
#   OMP_NUM_THREADS=16 ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_2d"
#
# done

#for data in "${datasets[@]}"; do
#  echo "Searching - ${data} (Cube)"
#
#  ./build/src/bench_complex_filter -d ${data} -s "./DATA/${data}/" -m "uniform_2d" -f 0.1 -r 0.3
#
#  ./build/src/bench_complex_filter -d ${data} -s "./DATA/${data}/" -m "uniform_2d" -f 0.05 -r 0.3
#
#
#done

 KD-Tree partitioned HNSW benchmarks

for data in yfcc msmarc10m sift; do
  case $data in
    sift)
      ratios=(0.01 0.02 0.05 0.10)
      metas=(uniform_2d )
      ;;
    yfcc)
      ratios=(0.01 0.02 0.05 0.10)
      metas=(real_2d real_3d)
      ;;
    msmarc10m)
      ratios=(0.01 0.02 0.05 0.10)
      metas=(uniform_2d )
      ;;
  esac
  for meta in "${metas[@]}"; do
    for ratio in "${ratios[@]}"; do
      echo "[KDTree] $data $meta ratio=$ratio"
      OMP_NUM_THREADS=16 ./build/src/bench_kdtree_partition -d $data -s ./DATA/$data/ -f $ratio -m $meta
    done
  done
done

# ============================================================================
# Various Distributions Benchmark (SIFT 2D only)
# ============================================================================
# Distributions: uniform, normal, clustered, skewed, hollow
# Filter ratios: 0.05, 0.10

#echo "========== SIFT Various Distributions Benchmark (2D) =========="
#
#for dist in normal clustered skewed hollow; do
#    echo "Searching - sift 2D ${dist}"
#    ./build/src/bench_hierarchical_cube -d sift -s "./DATA/sift/" -f 0.05 -m "${dist}_2d"
#    ./build/src/bench_hierarchical_cube -d sift -s "./DATA/sift/" -f 0.10 -m "${dist}_2d"
#done