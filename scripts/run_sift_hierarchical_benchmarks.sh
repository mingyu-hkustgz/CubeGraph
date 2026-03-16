#!/bin/bash
# Comprehensive benchmark script for hierarchical cube index on SIFT dataset
# This script automates the entire testing pipeline from the implementation plan

set -e  # Exit on error

# Configuration
DATASET="sift"
DATA_DIR="./DATA/${DATASET}"
RESULTS_DIR="./results/recall@10/${DATASET}"
FIGURE_DIR="./figure/${DATASET}"
BUILD_DIR="./build/src"

# SIFT dataset files (must exist)
BASE_FILE="${DATA_DIR}/sift_base.fvecs"
QUERY_FILE="${DATA_DIR}/sift_query.fvecs"

# Metadata distributions
DISTRIBUTIONS=("uniform" "normal" "skewed" "clustered")

# Filter ratios to test
FILTER_RATIOS=(0.01 0.05 0.1 0.25 0.5 1.0)

# Layer selection strategies
STRATEGIES=("RANGE_SIZE" "SELECTIVITY")

# Index parameters
ATTR_DIM=2
NUM_LAYERS=3
M=16
EF_CONSTRUCTION=200
CROSS_EDGES=2

# Benchmark parameters
K=10
EF_BASE=50
EF_MAX=500
EF_STEP=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Hierarchical Cube Index SIFT Benchmark"
echo "========================================="
echo ""

# Check if SIFT data exists
if [ ! -f "${BASE_FILE}" ] || [ ! -f "${QUERY_FILE}" ]; then
    echo -e "${RED}ERROR: SIFT dataset not found!${NC}"
    echo ""
    echo "Expected files:"
    echo "  - ${BASE_FILE}"
    echo "  - ${QUERY_FILE}"
    echo ""
    echo "Please download or generate the SIFT dataset with the following metadata files:"
    for dist in "${DISTRIBUTIONS[@]}"; do
        echo "  - ${DATA_DIR}/sift_metadata_${dist}_2d.bin"
    done
    echo ""
    echo "You can use the generate_test_data.py script to create test data."
    exit 1
fi

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${FIGURE_DIR}"
mkdir -p "${DATA_DIR}/indexes"
mkdir -p "${DATA_DIR}/groundtruth"

echo "Step 1: Building Hierarchical Indexes"
echo "======================================="
echo ""

for dist in "${DISTRIBUTIONS[@]}"; do
    METADATA_FILE="${DATA_DIR}/sift_metadata_${dist}_2d.bin"
    INDEX_FILE="${DATA_DIR}/indexes/sift_hierarchical_${dist}.index"

    if [ ! -f "${METADATA_FILE}" ]; then
        echo -e "${YELLOW}WARNING: Metadata file not found: ${METADATA_FILE}${NC}"
        echo "Skipping ${dist} distribution..."
        continue
    fi

    echo -e "${GREEN}Building index for ${dist} distribution...${NC}"
    ${BUILD_DIR}/test_hierarchical_cube \
        "${BASE_FILE}" \
        "${METADATA_FILE}" \
        "${QUERY_FILE}" \
        "${INDEX_FILE}" \
        ${ATTR_DIM} \
        ${NUM_LAYERS} \
        ${M} \
        ${EF_CONSTRUCTION} \
        ${CROSS_EDGES}

    echo ""
done

echo ""
echo "Step 2: Generating Filtered Groundtruth"
echo "========================================"
echo ""

for dist in "${DISTRIBUTIONS[@]}"; do
    METADATA_FILE="${DATA_DIR}/sift_metadata_${dist}_2d.bin"

    if [ ! -f "${METADATA_FILE}" ]; then
        continue
    fi

    for ratio in "${FILTER_RATIOS[@]}"; do
        GT_FILE="${DATA_DIR}/groundtruth/sift_filtered_groundtruth_${dist}_${ratio}.bin"

        if [ -f "${GT_FILE}" ]; then
            echo "Groundtruth already exists: ${GT_FILE}"
            continue
        fi

        echo -e "${GREEN}Generating groundtruth for ${dist}, filter ratio ${ratio}...${NC}"
        python3 scripts/generate_filtered_groundtruth.py \
            --base "${BASE_FILE}" \
            --query "${QUERY_FILE}" \
            --metadata "${METADATA_FILE}" \
            --output "${GT_FILE}" \
            --filter-ratio ${ratio} \
            --k ${K}

        echo ""
    done
done

echo ""
echo "Step 3: Running Benchmarks"
echo "==========================="
echo ""

TOTAL_BENCHMARKS=$((${#DISTRIBUTIONS[@]} * ${#FILTER_RATIOS[@]} * ${#STRATEGIES[@]}))
CURRENT=0

for dist in "${DISTRIBUTIONS[@]}"; do
    METADATA_FILE="${DATA_DIR}/sift_metadata_${dist}_2d.bin"

    if [ ! -f "${METADATA_FILE}" ]; then
        continue
    fi

    for ratio in "${FILTER_RATIOS[@]}"; do
        GT_FILE="${DATA_DIR}/groundtruth/sift_filtered_groundtruth_${dist}_${ratio}.bin"

        if [ ! -f "${GT_FILE}" ]; then
            echo -e "${YELLOW}WARNING: Groundtruth not found: ${GT_FILE}${NC}"
            continue
        fi

        for strategy in "${STRATEGIES[@]}"; do
            CURRENT=$((CURRENT + 1))

            # Format ratio for filename (replace . with _)
            RATIO_STR=$(echo ${ratio} | sed 's/\./_/g')
            OUTPUT_FILE="${RESULTS_DIR}/sift-hierarchical-${dist}-${RATIO_STR}-${strategy,,}.log"

            echo -e "${GREEN}[${CURRENT}/${TOTAL_BENCHMARKS}] Running: ${dist}, ratio=${ratio}, strategy=${strategy}${NC}"

            ${BUILD_DIR}/bench_hierarchical_cube \
                --base "${BASE_FILE}" \
                --query "${QUERY_FILE}" \
                --metadata "${METADATA_FILE}" \
                --groundtruth "${GT_FILE}" \
                --output "${OUTPUT_FILE}" \
                --k ${K} \
                --filter-ratio ${ratio} \
                --strategy ${strategy} \
                --ef-base ${EF_BASE} \
                --ef-max ${EF_MAX} \
                --ef-step ${EF_STEP} \
                --attr-dim ${ATTR_DIM} \
                --num-layers ${NUM_LAYERS} \
                --M ${M} \
                --ef-construction ${EF_CONSTRUCTION} \
                --cross-edges ${CROSS_EDGES}

            echo ""
        done
    done
done

echo ""
echo "Step 4: Generating Visualizations"
echo "=================================="
echo ""

if [ -f "scripts/visualize_qps_recall.py" ]; then
    echo -e "${GREEN}Generating QPS-Recall plots...${NC}"
    python3 scripts/visualize_qps_recall.py \
        --result-path ./results \
        --figure-path ./figure \
        --datasets ${DATASET} \
        --k-values ${K}
    echo ""
else
    echo -e "${YELLOW}WARNING: visualize_qps_recall.py not found${NC}"
fi

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Figures saved to: ${FIGURE_DIR}"
echo ""
echo "Summary:"
echo "  - Tested ${#DISTRIBUTIONS[@]} metadata distributions"
echo "  - Tested ${#FILTER_RATIOS[@]} filter ratios"
echo "  - Tested ${#STRATEGIES[@]} layer selection strategies"
echo "  - Total benchmark runs: ${TOTAL_BENCHMARKS}"
echo ""
