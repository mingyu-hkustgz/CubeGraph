#!/usr/bin/env python3
"""
Metadata Generator for Filtered Vector Search

Generates synthetic metadata with different distributions:
- Uniform distribution
- Normal (Gaussian) distribution
- Clustered distribution
- Skewed distribution

Metadata format: binary file with [n, d, vector_0, vector_1, ..., vector_n-1]
"""

import numpy as np
import struct
import argparse
import os
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


def get_num_vectors(data_path):
    """Get number of vectors from .fvecs file"""
    vectors, dim = read_fvecs(data_path)
    return len(vectors), dim


def generate_uniform_metadata(n, attr_dim, min_val=0.0, max_val=100.0, seed=42):
    """
    Generate uniformly distributed metadata

    Args:
        n: Number of vectors
        attr_dim: Dimension of attribute space
        min_val: Minimum value
        max_val: Maximum value
        seed: Random seed

    Returns:
        numpy array of shape (n, attr_dim)
    """
    np.random.seed(seed)
    metadata = np.random.uniform(min_val, max_val, size=(n, attr_dim)).astype(np.float32)
    return metadata


def generate_normal_metadata(n, attr_dim, mean=50.0, std=15.0, seed=42):
    """
    Generate normally distributed metadata

    Args:
        n: Number of vectors
        attr_dim: Dimension of attribute space
        mean: Mean of distribution
        std: Standard deviation
        seed: Random seed

    Returns:
        numpy array of shape (n, attr_dim)
    """
    np.random.seed(seed)
    metadata = np.random.normal(mean, std, size=(n, attr_dim)).astype(np.float32)
    # Clip to reasonable range
    metadata = np.clip(metadata, 0.0, 100.0)
    return metadata


def generate_clustered_metadata(n, attr_dim, num_clusters=5, cluster_std=5.0, seed=42):
    """
    Generate clustered metadata (mixture of Gaussians)

    Args:
        n: Number of vectors
        attr_dim: Dimension of attribute space
        num_clusters: Number of clusters
        cluster_std: Standard deviation within each cluster
        seed: Random seed

    Returns:
        numpy array of shape (n, attr_dim)
    """
    np.random.seed(seed)

    # Generate cluster centers
    cluster_centers = np.random.uniform(10.0, 90.0, size=(num_clusters, attr_dim))

    # Assign each point to a cluster
    cluster_assignments = np.random.randint(0, num_clusters, size=n)

    # Generate points around cluster centers
    metadata = np.zeros((n, attr_dim), dtype=np.float32)
    for i in range(n):
        cluster_id = cluster_assignments[i]
        center = cluster_centers[cluster_id]
        metadata[i] = np.random.normal(center, cluster_std, size=attr_dim)

    # Clip to reasonable range
    metadata = np.clip(metadata, 0.0, 100.0)
    return metadata


def generate_skewed_metadata(n, attr_dim, skew_factor=2.0, seed=42):
    """
    Generate skewed metadata (exponential-like distribution)

    Args:
        n: Number of vectors
        attr_dim: Dimension of attribute space
        skew_factor: Skewness factor
        seed: Random seed

    Returns:
        numpy array of shape (n, attr_dim)
    """
    np.random.seed(seed)

    # Generate exponential distribution and transform
    metadata = np.random.exponential(scale=skew_factor, size=(n, attr_dim)).astype(np.float32)

    # Normalize to [0, 100] range
    metadata = (metadata / metadata.max()) * 100.0
    return metadata


def save_metadata(metadata, output_path):
    """
    Save metadata to binary file

    Format:
        [n: size_t] [d: size_t] [vector_0: float[d]] [vector_1: float[d]] ...
    """
    n, d = metadata.shape

    with open(output_path, 'wb') as f:
        # Write n and d
        f.write(struct.pack('Q', n))  # size_t (8 bytes)
        f.write(struct.pack('Q', d))  # size_t (8 bytes)

        # Write metadata vectors
        for i in range(n):
            f.write(metadata[i].tobytes())

    print(f"Saved metadata: {output_path}")
    print(f"  Shape: {metadata.shape}")
    print(f"  Min: {metadata.min():.2f}, Max: {metadata.max():.2f}")
    print(f"  Mean: {metadata.mean():.2f}, Std: {metadata.std():.2f}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic metadata for filtered vector search')
    parser.add_argument('--data', type=str, required=True, help='Path to .fvecs data file')
    parser.add_argument('--output-dir', type=str, default='./DATA', help='Output directory')
    parser.add_argument('--attr-dim', type=int, default=2, help='Attribute dimension')
    parser.add_argument('--distributions', nargs='+',
                       choices=['uniform', 'normal', 'clustered', 'skewed', 'all'],
                       default=['all'], help='Distributions to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Get number of vectors
    print(f"Reading data from: {args.data}")
    n, vec_dim = get_num_vectors(args.data)
    print(f"Number of vectors: {n}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Attribute dimension: {args.attr_dim}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset name from data path
    dataset_name = Path(args.data).stem.replace('_base', '')

    # Generate distributions
    distributions = args.distributions
    if 'all' in distributions:
        distributions = ['uniform', 'normal', 'clustered', 'skewed']

    for dist in distributions:
        print(f"\n{'='*60}")
        print(f"Generating {dist} distribution metadata...")
        print(f"{'='*60}")

        if dist == 'uniform':
            metadata = generate_uniform_metadata(n, args.attr_dim, seed=args.seed)
            output_path = output_dir / f"{dataset_name}_metadata_uniform_{args.attr_dim}d.bin"

        elif dist == 'normal':
            metadata = generate_normal_metadata(n, args.attr_dim, seed=args.seed)
            output_path = output_dir / f"{dataset_name}_metadata_normal_{args.attr_dim}d.bin"

        elif dist == 'clustered':
            metadata = generate_clustered_metadata(n, args.attr_dim, seed=args.seed)
            output_path = output_dir / f"{dataset_name}_metadata_clustered_{args.attr_dim}d.bin"

        elif dist == 'skewed':
            metadata = generate_skewed_metadata(n, args.attr_dim, seed=args.seed)
            output_path = output_dir / f"{dataset_name}_metadata_skewed_{args.attr_dim}d.bin"

        save_metadata(metadata, output_path)


if __name__ == '__main__':
    main()
