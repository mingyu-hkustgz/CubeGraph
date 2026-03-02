#!/usr/bin/env python3
"""
Groundtruth Generator for Filtered Vector Search

Computes groundtruth (K nearest neighbors) for filtered queries using brute-force search.
"""

import numpy as np
import struct
import argparse
import json
from pathlib import Path
import heapq


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


def load_metadata(metadata_path):
    """
    Load metadata from binary file

    Format: [n: size_t (8 bytes)] [d: size_t (8 bytes)] [vectors...]
    """
    with open(metadata_path, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        d = struct.unpack('Q', f.read(8))[0]

        metadata = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            metadata[i] = np.frombuffer(f.read(4 * d), dtype=np.float32)

    return metadata


def load_filters(filter_path):
    """Load filters from JSON file"""
    with open(filter_path, 'r') as f:
        filters = json.load(f)
    return filters


def apply_filter(metadata, filter_params, filter_type):
    """
    Apply filter to metadata and return mask of valid points

    Args:
        metadata: numpy array of shape (n, d)
        filter_params: dict with filter parameters
        filter_type: 'range' or 'radius'

    Returns:
        boolean mask of shape (n,)
    """
    if filter_type == 'range':
        min_bounds = np.array(filter_params['min_bounds'])
        max_bounds = np.array(filter_params['max_bounds'])
        mask = np.all((metadata >= min_bounds) & (metadata <= max_bounds), axis=1)

    elif filter_type == 'radius':
        center = np.array(filter_params['center'])
        radius = filter_params['radius']
        distances = np.linalg.norm(metadata - center, axis=1)
        mask = distances <= radius

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return mask


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


def brute_force_filtered_search(query, base_vectors, mask, k):
    """
    Brute-force filtered KNN search

    Args:
        query: numpy array of shape (d,)
        base_vectors: numpy array of shape (n, d)
        mask: boolean mask of shape (n,) indicating valid points
        k: number of nearest neighbors

    Returns:
        numpy array of k nearest neighbor IDs
    """
    # Get valid indices
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        # No valid points, return empty result
        return np.array([], dtype=np.int32)

    # Compute distances only for valid points
    valid_vectors = base_vectors[valid_indices]
    distances = compute_l2_distances(query, valid_vectors)

    # Find k nearest neighbors
    k_actual = min(k, len(valid_indices))
    nearest_indices = np.argpartition(distances, k_actual-1)[:k_actual]
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
    parser = argparse.ArgumentParser(description='Generate groundtruth for filtered queries')
    parser.add_argument('--base', type=str, required=True, help='Path to base vectors .fvecs file')
    parser.add_argument('--query', type=str, required=True, help='Path to query .fvecs file')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata .bin file')
    parser.add_argument('--filters', type=str, required=True, help='Path to filters .json file')
    parser.add_argument('--output', type=str, required=True, help='Output groundtruth .ivecs file')
    parser.add_argument('--k', type=int, default=100, help='Number of nearest neighbors')

    args = parser.parse_args()

    # Load data
    print(f"Loading base vectors from: {args.base}")
    base_vectors, base_dim = read_fvecs(args.base)
    print(f"Base vectors: {base_vectors.shape}")

    print(f"\nLoading query vectors from: {args.query}")
    query_vectors, query_dim = read_fvecs(args.query)
    print(f"Query vectors: {query_vectors.shape}")

    print(f"\nLoading metadata from: {args.metadata}")
    metadata = load_metadata(args.metadata)
    print(f"Metadata: {metadata.shape}")

    print(f"\nLoading filters from: {args.filters}")
    filters = load_filters(args.filters)
    print(f"Number of filters: {len(filters)}")

    # Verify dimensions
    assert base_vectors.shape[0] == metadata.shape[0], "Base vectors and metadata must have same number of points"
    assert len(filters) == len(query_vectors), "Number of filters must match number of queries"

    # Compute groundtruth
    print(f"\nComputing groundtruth with K={args.k}...")
    groundtruth = []

    for i, filter_info in enumerate(filters):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(filters)}")

        query_id = filter_info['query_id']
        filter_type = filter_info['filter_type']
        filter_params = filter_info['params']

        # Apply filter
        mask = apply_filter(metadata, filter_params, filter_type)
        num_valid = np.sum(mask)

        # Brute-force search
        query = query_vectors[query_id]
        result_ids = brute_force_filtered_search(query, base_vectors, mask, args.k)

        groundtruth.append(result_ids)

        if i == 0:
            print(f"\nExample (query {query_id}):")
            print(f"  Filter type: {filter_type}")
            print(f"  Valid points: {num_valid} ({num_valid/len(metadata)*100:.2f}%)")
            print(f"  Result size: {len(result_ids)}")

    # Print statistics
    result_sizes = [len(gt) for gt in groundtruth]
    print(f"\nGroundtruth computation complete!")
    print(f"Average result size: {np.mean(result_sizes):.1f} ± {np.std(result_sizes):.1f}")
    print(f"Min: {np.min(result_sizes)}, Max: {np.max(result_sizes)}")

    # Save groundtruth
    save_ivecs(groundtruth, args.output)


if __name__ == '__main__':
    main()
