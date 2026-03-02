#!/bin/bash
# Generate query filters for all selectivity levels and distributions

# Configuration
DATA_DIR="./DATA"
QUERY_FILE="${DATA_DIR}/sift_query.fvecs"
OUTPUT_DIR="${DATA_DIR}/filters"
ATTR_DIM=2

# Selectivity levels
SELECTIVITIES=(0.01 0.05 0.10 0.30)
SELECTIVITY_NAMES=("1pct" "5pct" "10pct" "30pct")

# Filter types
FILTER_TYPES=("range" "radius")

# Distributions
DISTRIBUTIONS=("uniform" "normal" "clustered" "skewed")

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Generating Query Filters"
echo "=========================================="
echo "Query file: ${QUERY_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Attribute dimension: ${ATTR_DIM}"
echo ""

# Check if query file exists
if [ ! -f "${QUERY_FILE}" ]; then
    echo "Error: Query file not found: ${QUERY_FILE}"
    exit 1
fi

# Generate filters for each combination
for dist in "${DISTRIBUTIONS[@]}"; do
    METADATA_FILE="${DATA_DIR}/sift_metadata_${dist}_${ATTR_DIM}d.bin"

    if [ ! -f "${METADATA_FILE}" ]; then
        echo "Warning: Metadata file not found: ${METADATA_FILE}"
        echo "Skipping distribution: ${dist}"
        continue
    fi

    echo "=========================================="
    echo "Distribution: ${dist}"
    echo "=========================================="

    for filter_type in "${FILTER_TYPES[@]}"; do
        echo ""
        echo "Filter type: ${filter_type}"
        echo "------------------------------------------"

        for i in "${!SELECTIVITIES[@]}"; do
            selectivity="${SELECTIVITIES[$i]}"
            sel_name="${SELECTIVITY_NAMES[$i]}"

            output_file="${OUTPUT_DIR}/sift_${dist}_${filter_type}_${sel_name}.json"

            echo "  Generating ${sel_name} (${selectivity})..."
            python3 scripts/generate_filters.py \
                --metadata "${METADATA_FILE}" \
                --query "${QUERY_FILE}" \
                --output "${output_file}" \
                --selectivity "${selectivity}" \
                --filter-type "${filter_type}" \
                --seed 42

            if [ $? -eq 0 ]; then
                echo "  ✓ Success: ${output_file}"
            else
                echo "  ✗ Failed: ${output_file}"
            fi
        done
    done
    echo ""
done

echo "=========================================="
echo "Filter generation complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_DIR}"/*.json 2>/dev/null | wc -l
echo " filter files"
