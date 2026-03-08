#!/usr/bin/env python3
"""
Cube-Aware Groundtruth Generator

Generates groundtruth (K nearest neighbors) for queries where only base vectors
with the same cube_id as the query are considered. This simulates the filtered
search where we only look for neighbors within the same spatial cube.

Output format (same as regular groundtruth .ivecs):
    For each query: [k: int (4 bytes)] [id_0: int] [id_1: int] ... [id_k-1: int]
"""

import numpy as np
import struct
import argparse
from pathlib import Path


def read_fvecs(filename):
    """Read .fvecs file format"""
    with open(filename, 'rb') as f:
        # Read first vector to get dimension
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)

        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            d = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * d, f.read(4 * d))
            vectors.append(vec)

        return np.array(vectors), dim


def load_cube_ids(cube_id_path):
    """
    Load cube_ids from binary file

    Format: [n: uint64] [cube_id_0: uint32] [cube_id_1: uint32] ...
    """
    with open(cube_id_path, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        cube_ids = np.frombuffer(f.read(4 * n), dtype=np.uint32)

    return cube_ids


def compute_l2_distances(query, base_vectors):
    """
    Compute L2 distances between query and base vectors

    Args:
        query: numpy array of shape (d,)
        base_vectors: numpy array of shape (n, d)

    Returns:
        numpy array of distances of shape (n,)
    """
    diff = base_vectors - query
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return distances


def cube_filtered_knn(query, base_vectors, base_cube_ids, query_cube_id, k):
    """
    Find K nearest neighbors from base vectors with same cube_id as query

    Args:
        query: numpy array of shape (d,)
        base_vectors: numpy array of shape (n, d)
        base_cube_ids: numpy array of shape (n,) with cube_id for each base vector
        query_cube_id: int cube_id of the query
        k: number of nearest neighbors

    Returns:
        numpy array of k nearest neighbor IDs (may be fewer if not enough points in cube)
    """
    # Create mask for same cube
    mask = (base_cube_ids == query_cube_id)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        # No points in same cube, return empty result
        return np.array([], dtype=np.int32)

    # Compute distances only for valid points
    valid_vectors = base_vectors[valid_indices]
    distances = compute_l2_distances(query, valid_vectors)

    # Find k nearest neighbors
    k_actual = min(k, len(valid_indices))
    nearest_indices = np.argpartition(distances, k_actual - 1)[:k_actual]
    nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]

    # Map back to original indices
    result_ids = valid_indices[nearest_indices]

    return result_ids.astype(np.int32)


def save_ivecs(groundtruth, output_path):
    """
    Save groundtruth to .ivecs file

    Format: for each query, write [k: int (4 bytes)] followed by [k IDs: int (4 bytes each)]
    """
    with open(output_path, 'wb') as f:
        for gt in groundtruth:
            k = len(gt)
            f.write(struct.pack('i', k))
            for id in gt:
                f.write(struct.pack('i', id))

    print(f"Saved groundtruth to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate cube-filtered groundtruth (only same cube neighbors)'
    )
    parser.add_argument(
        '--base', type=str, required=True,
        help='Path to base vectors .fvecs file'
    )
    parser.add_argument(
        '--query', type=str, required=True,
        help='Path to query .fvecs file'
    )
    parser.add_argument(
        '--cube-base', type=str, required=True,
        help='Path to base cube_ids .bin file'
    )
    parser.add_argument(
        '--cube-query', type=str, required=True,
        help='Path to query cube_ids .bin file'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output groundtruth .ivecs file'
    )
    parser.add_argument(
        '--k', type=int, default=100,
        help='Number of nearest neighbors (default: 100)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading base vectors from: {args.base}")
    base_vectors, base_dim = read_fvecs(args.base)
    print(f"Base vectors: {base_vectors.shape}")

    print(f"\nLoading query vectors from: {args.query}")
    query_vectors, query_dim = read_fvecs(args.query)
    print(f"Query vectors: {query_vectors.shape}")

    print(f"\nLoading base cube_ids from: {args.cube_base}")
    base_cube_ids = load_cube_ids(args.cube_base)
    print(f"Base cube_ids: {len(base_cube_ids)}")

    print(f"\nLoading query cube_ids from: {args.cube_query}")
    query_cube_ids = load_cube_ids(args.cube_query)
    print(f"Query cube_ids: {len(query_cube_ids)}")

    # Verify dimensions
    assert len(base_cube_ids) == len(base_vectors), \
        f"Base cube_ids ({len(base_cube_ids)}) doesn't match base vectors ({len(base_vectors)})"
    assert len(query_cube_ids) == len(query_vectors), \
        f"Query cube_ids ({len(query_cube_ids)}) doesn't match query vectors ({len(query_vectors)})"

    # Compute cube-filtered groundtruth
    print(f"\nComputing cube-filtered groundtruth with K={args.k}...")
    groundtruth = []

    # Count statistics
    same_cube_counts = []

    for i in range(len(query_vectors)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(query_vectors)}")

        query = query_vectors[i]
        query_cube_id = query_cube_ids[i]

        # Count how many base vectors are in same cube
        same_cube_count = np.sum(base_cube_ids == query_cube_id)
        same_cube_counts.append(same_cube_count)

        # Find KNN within same cube
        result_ids = cube_filtered_knn(
            query, base_vectors, base_cube_ids, query_cube_id, args.k
        )

        groundtruth.append(result_ids)

    # Print statistics
    same_cube_counts = np.array(same_cube_counts)
    print(f"\nCube-filtered groundtruth computation complete!")
    print(f"  Same-cube base vectors: mean={same_cube_counts.mean():.1f}, "
          f"std={same_cube_counts.std():.1f}, min={same_cube_counts.min()}, max={same_cube_counts.max()}")

    result_sizes = [len(gt) for gt in groundtruth]
    print(f"  Result sizes: mean={np.mean(result_sizes):.1f}, "
          f"std={np.std(result_sizes):.1f}, min={np.min(result_sizes)}, max={np.max(result_sizes)}")

    # Show example
    print(f"\nExample (query 0):")
    print(f"  Query cube_id: {query_cube_ids[0]}")
    print(f"  Same-cube base vectors: {same_cube_counts[0]}")
    print(f"  Result size: {len(groundtruth[0])}")

    # Save groundtruth
    save_ivecs(groundtruth, args.output)

    print(f"\nDone!")


if __name__ == '__main__':
    main()
