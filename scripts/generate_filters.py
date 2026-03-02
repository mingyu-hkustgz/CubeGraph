#!/usr/bin/env python3
"""
Query Filter Generator for Filtered Vector Search

Generates query filters with specific selectivity levels (1%, 5%, 10%, 30%)
Supports both multi-dimensional range filters and radius filters.
"""

import numpy as np
import struct
import argparse
import json
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


def compute_selectivity(metadata, filter_params, filter_type='range'):
    """
    Compute the selectivity (percentage of points that pass the filter)

    Args:
        metadata: numpy array of shape (n, d)
        filter_params: dict with filter parameters
        filter_type: 'range' or 'radius'

    Returns:
        selectivity as a fraction (0.0 to 1.0)
    """
    n = len(metadata)

    if filter_type == 'range':
        min_bounds = np.array(filter_params['min_bounds'])
        max_bounds = np.array(filter_params['max_bounds'])

        # Check which points are within the range
        within_range = np.all((metadata >= min_bounds) & (metadata <= max_bounds), axis=1)
        count = np.sum(within_range)

    elif filter_type == 'radius':
        center = np.array(filter_params['center'])
        radius = filter_params['radius']

        # Compute Euclidean distance from center
        distances = np.linalg.norm(metadata - center, axis=1)
        count = np.sum(distances <= radius)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return count / n


def generate_range_filter(metadata, target_selectivity, seed=None):
    """
    Generate a range filter with target selectivity

    Args:
        metadata: numpy array of shape (n, d)
        target_selectivity: target selectivity (0.0 to 1.0)
        seed: random seed

    Returns:
        dict with 'min_bounds' and 'max_bounds'
    """
    if seed is not None:
        np.random.seed(seed)

    n, d = metadata.shape

    # Compute data range for each dimension
    data_min = metadata.min(axis=0)
    data_max = metadata.max(axis=0)
    data_range = data_max - data_min

    # Binary search for the right range size
    # Start with a range that covers sqrt(target_selectivity) in each dimension
    range_fraction = np.sqrt(target_selectivity)

    best_params = None
    best_selectivity_diff = float('inf')

    # Try multiple random centers
    for _ in range(20):
        # Random center point
        center = np.random.uniform(data_min + data_range * range_fraction / 2,
                                   data_max - data_range * range_fraction / 2,
                                   size=d)

        # Binary search for range size
        low, high = 0.01, 2.0
        for _ in range(15):
            mid = (low + high) / 2
            half_range = data_range * range_fraction * mid / 2

            min_bounds = np.maximum(center - half_range, data_min)
            max_bounds = np.minimum(center + half_range, data_max)

            params = {'min_bounds': min_bounds.tolist(), 'max_bounds': max_bounds.tolist()}
            selectivity = compute_selectivity(metadata, params, 'range')

            if abs(selectivity - target_selectivity) < best_selectivity_diff:
                best_selectivity_diff = abs(selectivity - target_selectivity)
                best_params = params

            if selectivity < target_selectivity:
                low = mid
            else:
                high = mid

    return best_params


def generate_radius_filter(metadata, target_selectivity, seed=None):
    """
    Generate a radius filter with target selectivity

    Args:
        metadata: numpy array of shape (n, d)
        target_selectivity: target selectivity (0.0 to 1.0)
        seed: random seed

    Returns:
        dict with 'center' and 'radius'
    """
    if seed is not None:
        np.random.seed(seed)

    n, d = metadata.shape

    # Compute data range
    data_min = metadata.min(axis=0)
    data_max = metadata.max(axis=0)
    data_range = data_max - data_min
    max_distance = np.linalg.norm(data_range)

    best_params = None
    best_selectivity_diff = float('inf')

    # Try multiple random centers
    for _ in range(20):
        # Random center point
        center = np.random.uniform(data_min, data_max, size=d)

        # Binary search for radius
        low, high = 0.0, max_distance
        for _ in range(15):
            mid = (low + high) / 2

            params = {'center': center.tolist(), 'radius': float(mid)}
            selectivity = compute_selectivity(metadata, params, 'radius')

            if abs(selectivity - target_selectivity) < best_selectivity_diff:
                best_selectivity_diff = abs(selectivity - target_selectivity)
                best_params = params

            if selectivity < target_selectivity:
                low = mid
            else:
                high = mid

    return best_params


def save_filters(filters, output_path):
    """
    Save filters to JSON file

    Format:
    [
        {
            "query_id": 0,
            "filter_type": "range" or "radius",
            "params": {...}
        },
        ...
    ]
    """
    with open(output_path, 'w') as f:
        json.dump(filters, f, indent=2)

    print(f"Saved {len(filters)} filters to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate query filters with specific selectivity')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata .bin file')
    parser.add_argument('--query', type=str, required=True, help='Path to query .fvecs file')
    parser.add_argument('--output', type=str, required=True, help='Output filter file (.json)')
    parser.add_argument('--selectivity', type=float, required=True,
                       help='Target selectivity (e.g., 0.01 for 1%%)')
    parser.add_argument('--filter-type', type=str, choices=['range', 'radius'], default='range',
                       help='Filter type: range or radius')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from: {args.metadata}")
    metadata = load_metadata(args.metadata)
    print(f"Metadata shape: {metadata.shape}")
    print(f"Metadata range: [{metadata.min():.2f}, {metadata.max():.2f}]")

    # Load queries
    print(f"\nLoading queries from: {args.query}")
    queries, query_dim = read_fvecs(args.query)
    num_queries = len(queries)
    print(f"Number of queries: {num_queries}")
    print(f"Query dimension: {query_dim}")

    # Generate filters
    print(f"\nGenerating {args.filter_type} filters with {args.selectivity*100:.1f}% selectivity...")
    filters = []

    for i in range(num_queries):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_queries}")

        # Generate filter
        if args.filter_type == 'range':
            params = generate_range_filter(metadata, args.selectivity, seed=args.seed + i)
        else:
            params = generate_radius_filter(metadata, args.selectivity, seed=args.seed + i)

        # Verify selectivity
        actual_selectivity = compute_selectivity(metadata, params, args.filter_type)

        filters.append({
            'query_id': i,
            'filter_type': args.filter_type,
            'params': params,
            'target_selectivity': args.selectivity,
            'actual_selectivity': actual_selectivity
        })

    # Print statistics
    actual_selectivities = [f['actual_selectivity'] for f in filters]
    print(f"\nFilter generation complete!")
    print(f"Target selectivity: {args.selectivity*100:.2f}%")
    print(f"Actual selectivity: {np.mean(actual_selectivities)*100:.2f}% ± {np.std(actual_selectivities)*100:.2f}%")
    print(f"Min: {np.min(actual_selectivities)*100:.2f}%, Max: {np.max(actual_selectivities)*100:.2f}%")

    # Save filters
    save_filters(filters, args.output)


if __name__ == '__main__':
    main()

