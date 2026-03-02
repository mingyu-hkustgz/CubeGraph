#!/bin/bash
# Generate metadata for all datasets with all distributions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../DATA"

echo "=========================================="
echo "Generating Metadata for All Datasets"
echo "=========================================="

# SIFT dataset
if [ -f "${DATA_DIR}/sift/sift_base.fvecs" ]; then
    echo ""
    echo "Processing SIFT dataset..."
    python3 "${SCRIPT_DIR}/generate_metadata.py" \
        --data "${DATA_DIR}/sift/sift_base.fvecs" \
        --output-dir "${DATA_DIR}/sift" \
        --attr-dim 2 \
        --distributions all \
        --seed 42

    echo ""
    echo "Generating 3D metadata for SIFT..."
    python3 "${SCRIPT_DIR}/generate_metadata.py" \
        --data "${DATA_DIR}/sift/sift_base.fvecs" \
        --output-dir "${DATA_DIR}/sift" \
        --attr-dim 3 \
        --distributions all \
        --seed 42
else
    echo "SIFT dataset not found, skipping..."
fi

# GIST dataset
if [ -f "${DATA_DIR}/gist/gist_base.fvecs" ]; then
    echo ""
    echo "Processing GIST dataset..."
    python3 "${SCRIPT_DIR}/generate_metadata.py" \
        --data "${DATA_DIR}/gist/gist_base.fvecs" \
        --output-dir "${DATA_DIR}/gist" \
        --attr-dim 2 \
        --distributions all \
        --seed 42

    echo ""
    echo "Generating 3D metadata for GIST..."
    python3 "${SCRIPT_DIR}/generate_metadata.py" \
        --data "${DATA_DIR}/gist/gist_base.fvecs" \
        --output-dir "${DATA_DIR}/gist" \
        --attr-dim 3 \
        --distributions all \
        --seed 42
else
    echo "GIST dataset not found, skipping..."
fi

echo ""
echo "=========================================="
echo "Metadata generation completed!"
echo "=========================================="
echo ""
echo "Generated files:"
find "${DATA_DIR}" -name "*_metadata_*.bin" -type f
