#!/usr/bin/env python3
"""
Generate filtered groundtruth with random filters per query.

This matches the benchmark behavior in bench_hierarchical_cube.cpp where
each query gets a random filter centered at a random location.
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
    """Write groundtruth in binary format.

    Format: [n: size_t][k: size_t][query_0_ids: int32[k]][query_1_ids: int32[k]]...
    """
    n = len(groundtruth)
    k = len(groundtruth[0]) if n > 0 else 0
    with open(filename, 'wb') as f:
        f.write(struct.pack('QQ', n, k))
        for gt in groundtruth:
            f.write(struct.pack(f'{k}i', *gt))


def write_filters(filename, filters):
    """Write filter bounding boxes to binary file.

    Format: [n: size_t][attr_dim: size_t]
            [query_0_min: float[attr_dim]][query_0_max: float[attr_dim]]
            [query_1_min: float[attr_dim]][query_1_max: float[attr_dim]]...
    """
    n = len(filters)
    attr_dim = len(filters[0][0]) if n > 0 else 0
    with open(filename, 'wb') as f:
        f.write(struct.pack('QQ', n, attr_dim))
        for filter_min, filter_max in filters:
            f.write(struct.pack(f'{attr_dim}f', *filter_min))
            f.write(struct.pack(f'{attr_dim}f', *filter_max))


def generate_random_filter(global_min, global_max, filter_ratio, rng):
    """Generate a random filter bounding box.

    Matches the logic in bench_hierarchical_cube.cpp:
    - filter_size = range * sqrt(filter_ratio) for each dimension
    - center is randomly placed within valid range
    """
    attr_dim = len(global_min)
    filter_min = np.zeros(attr_dim, dtype=np.float32)
    filter_max = np.zeros(attr_dim, dtype=np.float32)

    for d in range(attr_dim):
        range_d = global_max[d] - global_min[d]
        filter_size = range_d * np.sqrt(filter_ratio)

        # Random center within valid range
        center_min = global_min[d] + filter_size / 2
        center_max = global_max[d] - filter_size / 2
        center = rng.uniform(center_min, center_max)

        filter_min[d] = center - filter_size / 2
        filter_max[d] = center + filter_size / 2

    return filter_min, filter_max


def apply_filter(metadata, filter_min, filter_max):
    """Return indices of metadata rows satisfying the bounding box filter."""
    mask = np.ones(len(metadata), dtype=bool)
    for d in range(metadata.shape[1]):
        mask &= (metadata[:, d] >= filter_min[d])
        mask &= (metadata[:, d] <= filter_max[d])
    return np.where(mask)[0]


def compute_filtered_knn_per_query(queries, base_vectors, metadata, filters, k):
    """Compute KNN with a different filter per query."""
    groundtruth = []

    for i, (query, (filter_min, filter_max)) in enumerate(zip(queries, filters)):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing query {i+1}/{len(queries)}...", end='\r', flush=True)

        # Apply filter for this query
        filtered_indices = apply_filter(metadata, filter_min, filter_max)

        if len(filtered_indices) == 0:
            top_k = [-1] * k
        elif len(filtered_indices) <= k:
            filtered_vectors = base_vectors[filtered_indices]
            dists = np.sum((filtered_vectors - query) ** 2, axis=1)
            sorted_idx = np.argsort(dists)
            top_k = filtered_indices[sorted_idx].tolist()
            top_k += [-1] * (k - len(top_k))
        else:
            filtered_vectors = base_vectors[filtered_indices]
            dists = np.sum((filtered_vectors - query) ** 2, axis=1)
            part = np.argpartition(dists, k - 1)[:k]
            part_sorted = part[np.argsort(dists[part])]
            top_k = filtered_indices[part_sorted].tolist()

        groundtruth.append(top_k)

    print()  # newline after progress
    return groundtruth


def main():
    parser = argparse.ArgumentParser(
        description='Generate filtered groundtruth with random filters per query')
    parser.add_argument('--base', required=True, help='Base vectors (.fvecs)')
    parser.add_argument('--query', required=True, help='Query vectors (.fvecs)')
    parser.add_argument('--metadata', required=True, help='Metadata file (.bin)')
    parser.add_argument('--output', required=True, help='Output groundtruth file (.bin)')
    parser.add_argument('--filter-output', required=True, help='Output filter file (.bin)')
    parser.add_argument('--filter-ratio', type=float, required=True,
                        help='Filter selectivity ratio (e.g., 0.1 for 10%%)')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    print(f"Loading base vectors from {args.base}...")
    base = read_fvecs(args.base)
    print(f"  {len(base)} vectors, dim={base.shape[1]}")

    print(f"Loading query vectors from {args.query}...")
    queries = read_fvecs(args.query)
    print(f"  {len(queries)} queries, dim={queries.shape[1]}")

    print(f"Loading metadata from {args.metadata}...")
    metadata = read_metadata(args.metadata)
    print(f"  {metadata.shape[0]} entries, attr_dim={metadata.shape[1]}")

    # Compute global bounding box
    global_min = np.min(metadata, axis=0)
    global_max = np.max(metadata, axis=0)
    print(f"Global bbox: min={global_min.tolist()}, max={global_max.tolist()}")

    # Generate random filters (one per query)
    print(f"\nGenerating random filters (ratio={args.filter_ratio}, seed={args.seed})...")
    rng = np.random.RandomState(args.seed)
    filters = []
    filter_sizes = []

    for i in range(len(queries)):
        filter_min, filter_max = generate_random_filter(global_min, global_max,
                                                         args.filter_ratio, rng)
        filters.append((filter_min, filter_max))

        # Count how many points pass this filter
        filtered_indices = apply_filter(metadata, filter_min, filter_max)
        filter_sizes.append(len(filtered_indices))

    print(f"  Generated {len(filters)} filters")
    print(f"  Avg filter size: {np.mean(filter_sizes):.1f} vectors "
          f"({100.0 * np.mean(filter_sizes) / len(base):.1f}%)")
    print(f"  Min/Max filter size: {min(filter_sizes)} / {max(filter_sizes)}")

    print(f"\nComputing filtered KNN (k={args.k})...")
    groundtruth = compute_filtered_knn_per_query(queries, base, metadata, filters, args.k)

    counts = [sum(1 for x in gt if x >= 0) for gt in groundtruth]
    print(f"  Avg neighbors found: {np.mean(counts):.1f}, "
          f"min={min(counts)}, max={max(counts)}")

    print(f"\nWriting groundtruth to {args.output}...")
    write_groundtruth(args.output, groundtruth)

    print(f"Writing filters to {args.filter_output}...")
    write_filters(args.filter_output, filters)

    print("Done.")


if __name__ == '__main__':
    main()
