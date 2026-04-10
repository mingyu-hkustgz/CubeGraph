#!/usr/bin/env python3
"""
Visualize metadata distribution.
- 2D: direct scatter plot
- 3D: 3D scatter plot with multiple angles
- 4D: PCA projection to 2D + pairwise projections
"""

import numpy as np
import argparse
import struct
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def load_metadata(path):
    """Load metadata from binary format: [n: uint64][d: uint64][data: float[n*d]]"""
    with open(path, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        d = struct.unpack('Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    return data

def plot_2d(data, output_path, title):
    """Plot 2D metadata as scatter plot"""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(data[:, 0], data[:, 1], alpha=0.15, s=1, c='navy')
    ax.set_xlabel('Dimension 0', fontweight='bold')
    ax.set_ylabel('Dimension 1', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=40, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D plot to {output_path}")

def plot_3d(data, output_path, title):
    """Plot 3D metadata as 3D scatter plot with multiple viewing angles"""
    # Sample if too many points
    if len(data) > 10000:
        indices = np.random.choice(len(data), 10000, replace=False)
        sample = data[indices]
    else:
        sample = data

    fig = plt.figure(figsize=(16, 14))

    # Four viewing angles
    angles = [(30, 45), (30, 135), (60, 45), (15, 90)]
    titles = ['View 1 (elev=30°, az=45°)', 'View 2 (elev=30°, az=135°)',
              'View 3 (elev=60°, az=45°)', 'View 4 (elev=15°, az=90°)']

    for i, (elev, az) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], alpha=0.15, s=2, c='navy')
        ax.set_xlabel('Dimension 0', fontweight='bold')
        ax.set_ylabel('Dimension 1', fontweight='bold')
        ax.set_zlabel('Dimension 2', fontweight='bold')
        ax.set_title(titles[i], fontweight='bold', pad=10)
        ax.view_init(elev=elev, azim=az)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

    plt.suptitle(title, fontweight='bold', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D plot to {output_path}")

def plot_4d_pca(data, output_path, title):
    """Plot 4D metadata using PCA projection to 2D"""
    from sklearn.decomposition import PCA

    # Sample if too many points
    if len(data) > 10000:
        indices = np.random.choice(len(data), 10000, replace=False)
        sample = data[indices]
    else:
        sample = data

    # PCA to 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(sample)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], alpha=0.2, s=2, c='navy')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax.set_title(f'{title}\n(PCA projection, total var explained: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%)', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 4D PCA plot to {output_path}")

def plot_4d_pairwise(data, output_path, title):
    """Plot 4D metadata as pairwise 2D projections (6 subplots)"""
    d = data.shape[1]

    # Sample if too many points
    if len(data) > 10000:
        indices = np.random.choice(len(data), 10000, replace=False)
        sample = data[indices]
    else:
        sample = data

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(sample[:, i], sample[:, j], alpha=0.15, s=2, c='navy')
        ax.set_xlabel(f'Dimension {i}', fontweight='bold')
        ax.set_ylabel(f'Dimension {j}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(title, fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 4D pairwise plot to {output_path}")

def plot_distribution_comparison(output_dir):
    """Plot all distributions for comparison"""
    base_path = os.path.join(output_dir, 'sift_metadata_')

    distributions = ['uniform', 'normal', 'clustered', 'skewed', 'hollow']
    dims = [2, 3, 4]

    for dim in dims:
        fig, axes = plt.subplots(1, len(distributions), figsize=(25, 5))

        for idx, dist in enumerate(distributions):
            filename = f'{base_path}{dist}_{dim}d.bin'

            if not os.path.exists(filename):
                print(f"Skipping missing file: {filename}")
                continue

            data = load_metadata(filename)

            # Sample for plotting
            if len(data) > 100000:
                indices = np.random.choice(len(data), 100000, replace=False)
                sample = data[indices]
            else:
                sample = data

            if dim == 2:
                axes[idx].scatter(sample[:, 0], sample[:, 1], alpha=0.2, s=2, c='navy')
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
                axes[idx].set_aspect('equal')
                axes[idx].grid(True, alpha=0.3, linestyle='--')
            else:
                # For 3D/4D, just show histogram of first dimension
                axes[idx].hist(sample[:, 0], bins=50, alpha=0.7, color='navy', edgecolor='white')
                axes[idx].set_xlim(0, 1)
                axes[idx].grid(True, alpha=0.3, linestyle='--')

            axes[idx].set_title(f'{dist.capitalize()} ({dim}D)', fontweight='bold')
            axes[idx].set_xlabel('Value', fontweight='bold')
            axes[idx].tick_params(labelsize=11)

        plt.suptitle(f'Metadata Distributions - {dim}D', fontweight='bold', fontsize=16)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'metadata_distributions_{dim}d.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize metadata distribution')
    parser.add_argument('--input', '-i', required=True,
                        help='Input metadata file')
    parser.add_argument('--output', '-o',
                        help='Output image file (default: <input>.png)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all distributions in the input directory')
    args = parser.parse_args()

    if args.compare:
        # Compare all distributions in the directory
        input_dir = os.path.dirname(args.input) if os.path.dirname(args.input) else './DATA/sift'
        plot_distribution_comparison(input_dir)
        return

    # Single file visualization
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    data = load_metadata(args.input)
    dim = data.shape[1]
    n = data.shape[0]

    # Generate output filename if not provided
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f'{base}.png'

    title = f'Metadata: {os.path.basename(args.input)} (n={n}, dim={dim})'

    print(f"Loaded {n} vectors of dimension {dim}")

    if dim == 2:
        plot_2d(data, output_path, title)
    elif dim == 3:
        plot_3d(data, output_path, title)
    elif dim == 4:
        # Create both PCA and pairwise projections
        base, ext = os.path.splitext(output_path)
        plot_4d_pca(data, f'{base}_pca.png', title)
        plot_4d_pairwise(data, f'{base}_pairwise.png', title)
    else:
        print(f"Error: Unsupported dimension {dim} (supported: 2, 3, 4)")
        sys.exit(1)

if __name__ == '__main__':
    main()
