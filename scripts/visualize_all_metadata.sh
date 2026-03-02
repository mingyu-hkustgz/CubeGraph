#!/bin/bash
# Visualize all generated metadata

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../DATA"
VIS_DIR="${SCRIPT_DIR}/../visualizations"

echo "=========================================="
echo "Visualizing All Metadata"
echo "=========================================="

mkdir -p "${VIS_DIR}"

# SIFT 2D metadata
if ls "${DATA_DIR}"/sift/sift_metadata_*_2d.bin 1> /dev/null 2>&1; then
    echo ""
    echo "Visualizing SIFT 2D metadata..."
    python3 "${SCRIPT_DIR}/visualize_metadata.py" \
        --metadata "${DATA_DIR}"/sift/sift_metadata_*_2d.bin \
        --output-dir "${VIS_DIR}/sift_2d" \
        --compare
fi

# SIFT 3D metadata
if ls "${DATA_DIR}"/sift/sift_metadata_*_3d.bin 1> /dev/null 2>&1; then
    echo ""
    echo "Visualizing SIFT 3D metadata..."
    python3 "${SCRIPT_DIR}/visualize_metadata.py" \
        --metadata "${DATA_DIR}"/sift/sift_metadata_*_3d.bin \
        --output-dir "${VIS_DIR}/sift_3d" \
        --compare
fi

# GIST 2D metadata
if ls "${DATA_DIR}"/gist/gist_metadata_*_2d.bin 1> /dev/null 2>&1; then
    echo ""
    echo "Visualizing GIST 2D metadata..."
    python3 "${SCRIPT_DIR}/visualize_metadata.py" \
        --metadata "${DATA_DIR}"/gist/gist_metadata_*_2d.bin \
        --output-dir "${VIS_DIR}/gist_2d" \
        --compare
fi

# GIST 3D metadata
if ls "${DATA_DIR}"/gist/gist_metadata_*_3d.bin 1> /dev/null 2>&1; then
    echo ""
    echo "Visualizing GIST 3D metadata..."
    python3 "${SCRIPT_DIR}/visualize_metadata.py" \
        --metadata "${DATA_DIR}"/gist/gist_metadata_*_3d.bin \
        --output-dir "${VIS_DIR}/gist_3d" \
        --compare
fi

echo ""
echo "=========================================="
echo "Visualization completed!"
echo "=========================================="
echo ""
echo "Visualizations saved to: ${VIS_DIR}"
echo ""
echo "Generated visualizations:"
find "${VIS_DIR}" -name "*.png" -type f | head -20
