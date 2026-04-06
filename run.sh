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

for data in "${datasets[@]}"; do
  echo "Searching - ${data} Cube"

#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d"
#
#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d"

  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_2d"

#  ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_2d"

done
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
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.2 -m "uniform_2d"
#
#  ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_2d"
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
echo "Searching - yfcc (real_2d)"
./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.02 -m "real_2d"


echo "Searching - yfcc (real_2d)"
./build/src/bench_post_filtering -d yfcc -s "./DATA/yfcc/" -f 0.02 -m "real_2d"

# YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.2 -m "real_3d"
#
## YFCC real_2d benchmark
#echo "Searching - yfcc (real_2d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.01 -m "real_2d"
#
## YFCC real_3d benchmark
#echo "Searching - yfcc (real_3d)"
#./build/src/bench_hierarchical_cube -d yfcc -s "./DATA/yfcc/" -f 0.01 -m "real_3d"
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

for data in "${datasets[@]}"; do
  echo "Searching - ${data} (Cube)"

#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.05 -m "uniform_2d"
#
#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.1 -m "uniform_2d"

  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.02 -m "uniform_2d"

#  ./build/src/bench_post_filtering -d ${data} -s "./DATA/${data}/" -f 0.01 -m "uniform_2d"

done

for data in "${datasets[@]}"; do
  echo "Searching - ${data} (Cube)"

  ./build/src/bench_complex_filter -d ${data} -s "./DATA/${data}/" -m "uniform_2d" -f 0.1 -r 0.3

  ./build/src/bench_complex_filter -d ${data} -s "./DATA/${data}/" -m "uniform_2d" -f 0.05 -r 0.3


done