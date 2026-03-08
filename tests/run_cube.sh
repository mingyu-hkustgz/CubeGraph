#!/bin/bash
# Test script for CubeGraph with cube-aware indexing and search

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source set.sh

data=${datasets[0]}
echo "Dataset: $data"

DATA_DIR="${store_path}/${data}"
RESULT_DIR="${result_path}/recall@20/${data}"

mkdir -p "$RESULT_DIR"
mkdir -p "$DATA_DIR"

# Generate metadata if not exists
if [ ! -f "${DATA_DIR}/${data}_metadata_uniform_2d.bin" ]; then
    echo "Generating metadata..."
    python3 scripts/generate_metadata.py \
        --data ${DATA_DIR}/${data}_base.fvecs \
        --output-dir ${DATA_DIR} \
        --attr-dim 2 \
        --distributions uniform
fi

# Generate regular groundtruth (for recall comparison)
if [ ! -f "${DATA_DIR}/${data}_groundtruth.ivecs" ]; then
    echo "Generating groundtruth..."
    python3 -c "
import numpy as np
import struct

def read_fvecs(path):
    with open(path, 'rb') as f:
        d = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        vecs = []
        while True:
            b = f.read(4)
            if not b: break
            dim = struct.unpack('i', b)[0]
            vecs.append(struct.unpack('f' * dim, f.read(4 * dim)))
    return np.array(vecs, dtype=np.float32)

base = read_fvecs('${DATA_DIR}/${data}_base.fvecs')
query = read_fvecs('${DATA_DIR}/${data}_query.fvecs')

# Compute brute-force KNN
k = 100
dist = np.sum((base - query[:, None])**2, axis=2)
nn = np.argsort(dist, axis=1)[:, :k]

# Save as ivecs
with open('${DATA_DIR}/${data}_groundtruth.ivecs', 'wb') as f:
    for i in range(len(query)):
        f.write(struct.pack('i', k))
        for j in range(k):
            f.write(struct.pack('i', int(nn[i, j])))
print('Groundtruth saved')
"
fi

# Generate cube assignments
if [ ! -f "${DATA_DIR}/${data}_cube_id_base.bin" ]; then
    echo "Generating cube assignments..."
    python3 scripts/generate_cube_assignment.py \
        --base ${DATA_DIR}/${data}_base.fvecs \
        --query ${DATA_DIR}/${data}_query.fvecs \
        --metadata ${DATA_DIR}/${data}_metadata_uniform_2d.bin \
        --output-base ${DATA_DIR}/${data}_cube_id_base.bin \
        --output-query ${DATA_DIR}/${data}_cube_id_query.bin \
        --num-cubes 16
fi

# Generate cube-filtered groundtruth
if [ ! -f "${DATA_DIR}/${data}_cube_groundtruth.ivecs" ]; then
    echo "Generating cube-filtered groundtruth..."
    python3 scripts/generate_cube_groundtruth.py \
        --base ${DATA_DIR}/${data}_base.fvecs \
        --query ${DATA_DIR}/${data}_query.fvecs \
        --cube-base ${DATA_DIR}/${data}_cube_id_base.bin \
        --cube-query ${DATA_DIR}/${data}_cube_id_query.bin \
        --output ${DATA_DIR}/${data}_cube_groundtruth.ivecs \
        --k 100
fi

# Build index with cube_id
echo "Building index with cube_id..."
./build/src/index_cube -d "$data" -s "${DATA_DIR}/" -t "${DATA_DIR}/${data}_metadata_uniform_2d.bin" -c 16

# Search with cube-aware search
echo "Searching with cube-aware search..."
./build/src/search_cube -d "$data" -s "${DATA_DIR}/" -k 20 -m "${DATA_DIR}/${data}_metadata_uniform_2d.bin" -c 16

echo "Done!"
