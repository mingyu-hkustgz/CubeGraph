#!/usr/bin/env python3
"""
Generate small synthetic test data for cube index testing
"""

import numpy as np
import struct
from pathlib import Path

def write_fvecs(filename, vectors):
    """Write vectors in .fvecs format"""
    with open(filename, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack('i', dim))
            f.write(struct.pack('f' * dim, *vec))

def write_metadata(filename, metadata):
    """Write metadata in .bin format"""
    n, d = metadata.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', n))
        f.write(struct.pack('Q', d))
        for i in range(n):
            f.write(metadata[i].astype(np.float32).tobytes())

def main():
    # Create output directory
    output_dir = Path('DATA')
    output_dir.mkdir(exist_ok=True)

    print("Generating synthetic test data...")

    # Parameters
    n_base = 10000  # Number of base vectors
    n_query = 100   # Number of query vectors
    vec_dim = 128   # Vector dimension
    attr_dim = 2    # Attribute dimension

    np.random.seed(42)

    # Generate base vectors (random normalized vectors)
    print(f"  Generating {n_base} base vectors (dim={vec_dim})...")
    base_vectors = np.random.randn(n_base, vec_dim).astype(np.float32)
    base_vectors = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)

    # Generate query vectors
    print(f"  Generating {n_query} query vectors (dim={vec_dim})...")
    query_vectors = np.random.randn(n_query, vec_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Generate metadata (uniform distribution in [0, 100])
    print(f"  Generating metadata (dim={attr_dim})...")
    metadata = np.random.uniform(0, 100, size=(n_base, attr_dim)).astype(np.float32)

    # Save files
    print("  Saving files...")
    write_fvecs(output_dir / 'test_base.fvecs', base_vectors)
    write_fvecs(output_dir / 'test_query.fvecs', query_vectors)
    write_metadata(output_dir / 'test_metadata_2d.bin', metadata)

    print("\nTest data generated successfully!")
    print(f"  Base vectors: {output_dir / 'test_base.fvecs'} ({n_base} vectors, dim={vec_dim})")
    print(f"  Query vectors: {output_dir / 'test_query.fvecs'} ({n_query} vectors, dim={vec_dim})")
    print(f"  Metadata: {output_dir / 'test_metadata_2d.bin'} ({n_base} points, dim={attr_dim})")
    print(f"\nFile sizes:")
    print(f"  Base: {(output_dir / 'test_base.fvecs').stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Query: {(output_dir / 'test_query.fvecs').stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Metadata: {(output_dir / 'test_metadata_2d.bin').stat().st_size / 1024:.2f} KB")

if __name__ == '__main__':
    main()
