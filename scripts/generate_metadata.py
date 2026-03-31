#!/usr/bin/env python3
"""
Generate metadata files with various dimensions and distributions.
Metadata format: [n: size_t][d: size_t][vector_0: float[d]]...[vector_n-1: float[d]]
"""

import numpy as np
import argparse
import os

def save_metadata(path, data):
    """Save metadata in binary format: [n, d, then n*d floats]"""
    n, d = data.shape
    with open(path, 'wb') as f:
        # Write n and d as size_t (unsigned long)
        f.write(np.array(n, dtype=np.uint64).tobytes())
        f.write(np.array(d, dtype=np.uint64).tobytes())
        # Write data as floats
        f.write(data.astype(np.float32).tobytes())
    print(f"Saved {n} vectors of dimension {d} to {path}")

def generate_uniform(n, d, range_min=0.0, range_max=1.0):
    """Uniform distribution in [range_min, range_max]"""
    return np.random.uniform(range_min, range_max, size=(n, d)).astype(np.float32)

def generate_normal(n, d, mean=0.5, std=0.15):
    """Normal distribution around mean with given std"""
    data = np.random.normal(mean, std, size=(n, d)).astype(np.float32)
    # Clip to [0, 1] range
    return np.clip(data, 0.0, 1.0)

def generate_clustered(n, d, n_clusters=5, seed=42):
    """Clustered distribution - points around random cluster centers"""
    np.random.seed(seed)
    cluster_centers = np.random.uniform(0.1, 0.9, size=(n_clusters, d)).astype(np.float32)
    data = []
    per_cluster = n // n_clusters
    for i in range(n_clusters):
        cluster_size = per_cluster + (1 if i < n % n_clusters else 0)
        cluster_data = np.random.normal(
            cluster_centers[i], 0.05, size=(cluster_size, d)
        ).astype(np.float32)
        cluster_data = np.clip(cluster_data, 0.0, 1.0)
        data.append(cluster_data)
    return np.vstack(data)

def generate_skewed(n, d, seed=42):
    """Skewed distribution - more points in lower ranges"""
    np.random.seed(seed)
    # Use beta distribution to create skewness
    data = np.random.beta(0.5, 0.5, size=(n, d)).astype(np.float32)
    return data

def generate_hollow(n, d, inner_radius=0.2, outer_radius=0.5, seed=42):
    """Hollow sphere/ball - points on the surface of a hypersphere"""
    np.random.seed(seed)
    # Generate points on unit sphere surface, then scale
    data = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms  # Normalize to unit sphere

    # Scale to hollow ball between inner and outer radius
    radii = np.random.uniform(inner_radius, outer_radius, size=(n, 1))
    data = data * radii

    # Shift to center at 0.5
    data = data + 0.5
    return np.clip(data, 0.0, 1.0)

def generate_3d_uniform(n):
    """3D uniform distribution"""
    return generate_uniform(n, 3)

def generate_3d_normal(n):
    """3D normal distribution"""
    return generate_normal(n, 3)

def generate_3d_clustered(n):
    """3D clustered distribution"""
    return generate_clustered(n, 3)

def generate_3d_skewed(n):
    """3D skewed distribution"""
    return generate_skewed(n, 3)

def generate_3d_hollow(n):
    """3D hollow sphere distribution"""
    return generate_hollow(n, 3)

def generate_4d_uniform(n):
    """4D uniform distribution"""
    return generate_uniform(n, 4)

def generate_4d_normal(n):
    """4D normal distribution"""
    return generate_normal(n, 4)

def generate_4d_clustered(n):
    """4D clustered distribution"""
    return generate_clustered(n, 4)

def generate_4d_skewed(n):
    """4D skewed distribution"""
    return generate_skewed(n, 4)

def generate_4d_hollow(n):
    """4D hollow hypersphere distribution"""
    return generate_hollow(n, 4)

def main():
    parser = argparse.ArgumentParser(description='Generate metadata files')
    parser.add_argument('--output-dir', '-o', default='./DATA/sift',
                        help='Output directory')
    parser.add_argument('--n', type=int, default=1000000,
                        help='Number of vectors')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n = args.n

    # Generate 2D metadata (existing distributions for reference)
    print("\n=== Generating 2D metadata ===")
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_uniform_2d.bin'),
                  generate_uniform(n, 2))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_normal_2d.bin'),
                  generate_normal(n, 2))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_clustered_2d.bin'),
                  generate_clustered(n, 2))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_skewed_2d.bin'),
                  generate_skewed(n, 2))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_hollow_2d.bin'),
                  generate_hollow(n, 2))

    # Generate 3D metadata
    print("\n=== Generating 3D metadata ===")
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_uniform_3d.bin'),
                  generate_3d_uniform(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_normal_3d.bin'),
                  generate_3d_normal(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_clustered_3d.bin'),
                  generate_3d_clustered(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_skewed_3d.bin'),
                  generate_3d_skewed(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_hollow_3d.bin'),
                  generate_3d_hollow(n))

    # Generate 4D metadata
    print("\n=== Generating 4D metadata ===")
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_uniform_4d.bin'),
                  generate_4d_uniform(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_normal_4d.bin'),
                  generate_4d_normal(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_clustered_4d.bin'),
                  generate_4d_clustered(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_skewed_4d.bin'),
                  generate_4d_skewed(n))
    save_metadata(os.path.join(args.output_dir, 'sift_metadata_hollow_4d.bin'),
                  generate_4d_hollow(n))

    print("\nDone!")

if __name__ == '__main__':
    main()
