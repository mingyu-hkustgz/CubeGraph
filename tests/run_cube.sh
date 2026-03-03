#!/bin/bash
# Test script for CubeGraph index

source ../set.sh

echo "========================================="
echo "CubeGraph Index Test"
echo "========================================="

# Create results directory
mkdir -p ${result_path}/recall@20/${data}
mkdir -p ${result_path}/recall@100/${data}

# Generate groundtruth if not exists
if [ ! -f "${store_path}/${data}/${data}_metadata_uniform_2d.bin" ]; then
    echo "Generating metadata..."
    python3 ../scripts/generate_metadata.py \
        --data ${store_path}/${data}/${data}_base.fvecs \
        --output-dir ${store_path}/${data} \
        --attr-dim 2 \
        --distributions uniform
fi

# Generate groundtruth
echo "Generating groundtruth..."
python3 ../scripts/generate_groundtruth.py \
    ${store_path}/${data}/${data}_base.fvecs \
    ${store_path}/${data}/${data}_query.fvecs \
    ${store_path}/${data}/${data}_metadata_uniform_2d.bin \
    ${result_path}/${data}_groundtruth.bin 2>/dev/null

# Test different ef values
for ef in 50 100 200 500; do
    echo ""
    echo "========================================="
    echo "Testing with ef=${ef}..."
    echo "========================================="

    # Run benchmark
    ./build/src/bench_cube_index \
        -d ${store_path}/${data}/${data}_base.fvecs \
        -m ${store_path}/${data}/${data}_metadata_uniform_2d.bin \
        -q ${store_path}/${data}/${data}_query.fvecs \
        -g ${result_path}/${data}_groundtruth.bin \
        -o ${result_path}/${data}_cube_ef${ef}.index \
        -a 2 \
        -k 20 \
        -e ${ef} 2>&1 | tee ${result_path}/recall@20/${data}/${data}-cube-ef${ef}.log

    # Extract recall and QPS and save in required format
    recall=$(grep "^Results:" -A 3 ${result_path}/recall@20/${data}/${data}-cube-ef${ef}.log | grep "Recall:" | awk '{print $2}' | tr -d '%')
    qps=$(grep "^Results:" -A 3 ${result_path}/recall@20/${data}/${data}-cube-ef${ef}.log | grep "QPS:" | awk '{print $2}')

    # Write in format: "recall Qps"
    echo "${recall} ${qps}" > ${result_path}/recall@20/${data}/${data}-cube.log
done

echo ""
echo "========================================="
echo "Done! Results:"
echo "========================================="
cat ${result_path}/recall@20/${data}/${data}-cube.log
