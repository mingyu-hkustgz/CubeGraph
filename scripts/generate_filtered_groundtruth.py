#!/usr/bin/env python3
"""
Generate filtered groundtruth for CubeGraph benchmark.

Computes KNN only among base vectors that satisfy a bounding box filter,
matching the actual filtered search behavior in bench_cube_index.cpp.
"""

import argparse
import struct
import sys
import numpy as np


def read_fvecs(filename):
    """Read .fvecs file format."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def read_metadata(filename):
    """Read metadata .bin file (format: [n: uint64][d: uint64][vectors...])."""
    with open(filename, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        d = struct.unpack('Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32)
    return data.reshape(n, d)


def write_groundtruth(filename, groundtruth):
    """Write groundtruth in binary format matching load_groundtruth() in bench_cube_index.cpp.

    Format: [n: size_t][k: size_t][query_0_ids: int32[k]][query_1_ids: int32[k]]...
    """
    n = len(groundtruth)
    k = len(groundtruth[0]) if n > 0 else 0
    with open(filename, 'wb') as f:
        f.write(struct.pack('QQ', n, k))
        for gt in groundtruth:
            f.write(struct.pack(f'{k}i', *gt))


def apply_filter(metadata, filter_min, filter_max):
    """Return indices of metadata rows satisfying the bounding box filter."""
    mask = np.ones(len(metadata), dtype=bool)
    for d in range(metadata.shape[1]):
        mask &= (metadata[:, d] >= filter_min[d])
        mask &= (metadata[:, d] <= filter_max[d])
    return np.where(mask)[0]


def compute_filtered_knn(queries, base_vectors, filtered_indices, k):
    """Compute KNN on filtered base vectors, returning original IDs."""
    filtered_vectors = base_vectors[filtered_indices]
    n_filtered = len(filtered_indices)
    groundtruth = []

    for i, query in enumerate(queries):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing query {i+1}/{len(queries)}...", end='\r', flush=True)

        if n_filtered == 0:
            top_k = [-1] * k
        elif n_filtered <= k:
            dists = np.sum((filtered_vectors - query) ** 2, axis=1)
            sorted_idx = np.argsort(dists)
            top_k = filtered_indices[sorted_idx].tolist()
            top_k += [-1] * (k - len(top_k))
        else:
            dists = np.sum((filtered_vectors - query) ** 2, axis=1)
            part = np.argpartition(dists, k - 1)[:k]
            part_sorted = part[np.argsort(dists[part])]
            top_k = filtered_indices[part_sorted].tolist()

        groundtruth.append(top_k)

    print()  # newline after progress
    return groundtruth


def main():
    parser = argparse.ArgumentParser(
        description='Generate filtered groundtruth for CubeGraph benchmark')
    parser.add_argument('--base', required=True, help='Base vectors (.fvecs)')
    parser.add_argument('--query', required=True, help='Query vectors (.fvecs)')
    parser.add_argument('--metadata', required=True, help='Metadata file (.bin)')
    parser.add_argument('--output', required=True, help='Output groundtruth file (.bin)')
    parser.add_argument('--filter-min', nargs='+', type=float, required=True,
                        help='Min bounds per dimension')
    parser.add_argument('--filter-max', nargs='+', type=float, required=True,
                        help='Max bounds per dimension')
    parser.add_argument('--k', type=int, default=100, help='Number of neighbors (default: 100)')
    args = parser.parse_args()

    if len(args.filter_min) != len(args.filter_max):
        print("Error: --filter-min and --filter-max must have the same number of values")
        sys.exit(1)

    print(f"Loading base vectors from {args.base}...")
    base = read_fvecs(args.base)
    print(f"  {len(base)} vectors, dim={base.shape[1]}")

    print(f"Loading query vectors from {args.query}...")
    queries = read_fvecs(args.query)
    print(f"  {len(queries)} queries, dim={queries.shape[1]}")

    print(f"Loading metadata from {args.metadata}...")
    metadata = read_metadata(args.metadata)
    print(f"  {metadata.shape[0]} entries, attr_dim={metadata.shape[1]}")

    filter_min = np.array(args.filter_min, dtype=np.float32)
    filter_max = np.array(args.filter_max, dtype=np.float32)
    print(f"Applying filter: min={filter_min.tolist()}, max={filter_max.tolist()}")

    filtered_indices = apply_filter(metadata, filter_min, filter_max)
    pct = 100.0 * len(filtered_indices) / len(base)
    print(f"  {len(filtered_indices)} / {len(base)} vectors pass filter ({pct:.1f}%)")

    print(f"Computing filtered KNN (k={args.k})...")
    groundtruth = compute_filtered_knn(queries, base, filtered_indices, args.k)

    counts = [sum(1 for x in gt if x >= 0) for gt in groundtruth]
    print(f"  Avg neighbors found: {np.mean(counts):.1f}, "
          f"min={min(counts)}, max={max(counts)}")

    print(f"Writing groundtruth to {args.output}...")
    write_groundtruth(args.output, groundtruth)
    print("Done.")


if __name__ == '__main__':
    main()
