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

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    ./build/src/bench_hierarchical_cube -d ${data} -s "./DATA/${data}/" -k ${K}

  done
done

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data} (ball)"

    ./build/src/bench_hierarchical_ball -d ${data} -s "./DATA/${data}/"

  done
done
