#!/bin/bash
# Test script for CubeGraph index

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

source set.sh

# Use first dataset from list
data=${datasets[0]}

echo "========================================="
echo "CubeGraph Index Test"
echo "========================================="
echo "Dataset: $data"
echo "Store path: $store_path"
echo "Result path: $result_path"

# Create results directory
mkdir -p ${result_path}/recall@20/${data}
mkdir -p ${result_path}/recall@100/${data}

# Generate metadata if not exists
if [ ! -f "${store_path}/${data}/${data}_metadata_uniform_2d.bin" ]; then
    echo "Generating metadata..."
    python3 scripts/generate_metadata.py \
        --data ${store_path}/${data}/${data}_base.fvecs \
        --output-dir ${store_path}/${data} \
        --attr-dim 2 \
        --distributions uniform
fi

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
        -o ${result_path}/${data}_cube_ef${ef}.index \
        -a 2 \
        -k 20 \
        -e ${ef} 2>&1 | tee ${result_path}/recall@20/${data}/${data}-cube-ef${ef}.log

    # Extract recall and QPS and save in required format
    recall=$(grep "^Results:" -A 3 ${result_path}/recall@20/${data}/${data}-cube-ef${ef}.log 2>/dev/null | grep "Recall:" | awk '{print $2}' | tr -d '%')
    qps=$(grep "^Results:" -A 3 ${result_path}/recall@20/${data}/${data}-cube-ef${ef}.log 2>/dev/null | grep "QPS:" | awk '{print $2}')

    if [ -z "$recall" ]; then
        recall="0"
    fi
    if [ -z "$qps" ]; then
        qps="0"
    fi

    # Write in format: "recall Qps"
    echo "${recall} ${qps}" > ${result_path}/recall@20/${data}/${data}-cube.log

    echo "  Result: recall=${recall}%, qps=${qps}"
done

echo ""
echo "========================================="
echo "Done! Results:"
echo "========================================="
cat ${result_path}/recall@20/${data}/${data}-cube.log
