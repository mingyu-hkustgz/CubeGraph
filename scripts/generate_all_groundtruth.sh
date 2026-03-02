#!/bin/bash
# Generate groundtruth for all filter files

# Configuration
DATA_DIR="./DATA"
BASE_FILE="${DATA_DIR}/sift_base.fvecs"
QUERY_FILE="${DATA_DIR}/sift_query.fvecs"
FILTER_DIR="${DATA_DIR}/filters"
OUTPUT_DIR="${DATA_DIR}/groundtruth"
K=100
ATTR_DIM=2

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Generating Groundtruth for Filtered Queries"
echo "=========================================="
echo "Base file: ${BASE_FILE}"
echo "Query file: ${QUERY_FILE}"
echo "Filter directory: ${FILTER_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "K: ${K}"
echo ""

# Check if required files exist
if [ ! -f "${BASE_FILE}" ]; then
    echo "Error: Base file not found: ${BASE_FILE}"
    exit 1
fi

if [ ! -f "${QUERY_FILE}" ]; then
    echo "Error: Query file not found: ${QUERY_FILE}"
    exit 1
fi

if [ ! -d "${FILTER_DIR}" ]; then
    echo "Error: Filter directory not found: ${FILTER_DIR}"
    exit 1
fi

# Count filter files
num_filters=$(ls "${FILTER_DIR}"/*.json 2>/dev/null | wc -l)
if [ ${num_filters} -eq 0 ]; then
    echo "Error: No filter files found in ${FILTER_DIR}"
    exit 1
fi

echo "Found ${num_filters} filter files"
echo ""

# Process each filter file
count=0
success=0
failed=0

for filter_file in "${FILTER_DIR}"/*.json; do
    count=$((count + 1))
    filename=$(basename "${filter_file}" .json)

    # Extract distribution from filename (e.g., sift_uniform_range_1pct.json -> uniform)
    dist=$(echo "${filename}" | cut -d'_' -f2)

    # Metadata file
    metadata_file="${DATA_DIR}/sift_metadata_${dist}_${ATTR_DIM}d.bin"

    if [ ! -f "${metadata_file}" ]; then
        echo "[${count}/${num_filters}] ✗ Skipping ${filename}: metadata not found"
        failed=$((failed + 1))
        continue
    fi

    # Output groundtruth file
    output_file="${OUTPUT_DIR}/${filename}_gt.ivecs"

    echo "[${count}/${num_filters}] Processing: ${filename}"
    echo "  Metadata: ${metadata_file}"
    echo "  Output: ${output_file}"

    python3 scripts/generate_groundtruth.py \
        --base "${BASE_FILE}" \
        --query "${QUERY_FILE}" \
        --metadata "${metadata_file}" \
        --filters "${filter_file}" \
        --output "${output_file}" \
        --k ${K}

    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
        success=$((success + 1))
    else
        echo "  ✗ Failed"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "=========================================="
echo "Groundtruth generation complete!"
echo "=========================================="
echo "Total: ${count}"
echo "Success: ${success}"
echo "Failed: ${failed}"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_DIR}"/*.ivecs 2>/dev/null | wc -l
echo " groundtruth files"
