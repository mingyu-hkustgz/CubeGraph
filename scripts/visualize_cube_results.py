#!/usr/bin/env python3
"""
Visualize Cube Index Results

Creates visualizations of the cube index structure and search results
using the SIFT dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import struct
import sys
from pathlib import Path


def read_fvecs(filename):
    """Read .fvecs file format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            d = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * d, f.read(4 * d))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def read_binary_metadata(filename):
    """Read binary metadata file"""
    with open(filename, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        d = struct.unpack('Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32)
        return data.reshape(n, d)


def plot_metadata_distribution(metadata, output_path):
    """Plot metadata distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(metadata[:, 0], metadata[:, 1], alpha=0.1, s=1)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Metadata Distribution (Uniform 2D)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(metadata[:, 0], bins=50, alpha=0.7, label='Dim 1')
    ax.hist(metadata[:, 1], bins=50, alpha=0.7, label='Dim 2')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('Metadata Histogram')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cube_structure(metadata, output_path):
    """Plot cube structure with threshold"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot all points
    ax.scatter(metadata[:, 0], metadata[:, 1], alpha=0.05, s=1, c='blue')

    # Simulated cube boundaries (based on 8x8 grid for 64 cubes)
    num_cubes = 8
    cube_size = 100.0 / num_cubes

    for i in range(num_cubes + 1):
        ax.axhline(i * cube_size, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(i * cube_size, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    # Draw filter region
    filter_box = patches.Rectangle((20, 20), 60, 60, linewidth=3,
                                     edgecolor='green', facecolor='green', alpha=0.2)
    ax.add_patch(filter_box)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Cube Structure with Filter Region\n(Green: Query Region [20,80]×[20,80])', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_filter_region(output_path):
    """Plot filter regions used in testing"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Range filter
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    filter_box = patches.Rectangle((20, 20), 60, 60, linewidth=3,
                                     edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(filter_box)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Range Filter\n[20,80] × [20,80]')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Radius filter
    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    circle = patches.Circle((50, 50), 30, linewidth=3,
                           edgecolor='blue', facecolor='lightblue', alpha=0.5)
    ax.add_patch(circle)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Radius Filter\nCenter=(50,50), Radius=30')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_edge_memory_layout(output_path):
    """Visualize the edge memory layout"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    y_pos = 0.5

    # Intra-cube edges
    rect = patches.Rectangle((0, y_pos - 0.2), 4, 0.4, linewidth=2,
                             edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(rect)
    ax.text(2, y_pos, 'Intra-cube Links\n(M0 = 64)', ha='center', va='center', fontsize=10)

    # Cross-cube edges
    rect = patches.Rectangle((5, y_pos - 0.2), 2, 0.4, linewidth=2,
                             edgecolor='black', facecolor='lightcoral', alpha=0.7)
    ax.add_patch(rect)
    ax.text(6, y_pos, 'Cross-cube Links\n(2*d*M_cross_per_dir)', ha='center', va='center', fontsize=10)

    # Label
    rect = patches.Rectangle((8, y_pos - 0.2), 1, 0.4, linewidth=2,
                             edgecolor='black', facecolor='lightgreen', alpha=0.7)
    ax.add_patch(rect)
    ax.text(8.5, y_pos, 'Label', ha='center', va='center', fontsize=10)

    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Node Memory Layout at Level 0\n(2D: M=32, M_cross=16)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths
    base_path = '../DATA/sift/sift_base.fvecs'
    metadata_path = '../DATA/sift/sift_metadata_uniform_2d.bin'
    output_dir = Path('visualizations/cube_index')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    try:
        metadata = read_binary_metadata(metadata_path)
        print(f"Metadata shape: {metadata.shape}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    print("Creating visualizations...")

    # 1. Metadata distribution
    plot_metadata_distribution(metadata, output_dir / 'metadata_distribution.png')

    # 2. Cube structure
    plot_cube_structure(metadata, output_dir / 'cube_structure.png')

    # 3. Filter regions
    plot_filter_region(output_dir / 'filter_regions.png')

    # 4. Edge memory layout
    plot_edge_memory_layout(output_dir / 'edge_memory_layout.png')

    print("\nAll visualizations saved to:", output_dir)


if __name__ == '__main__':
    main()
