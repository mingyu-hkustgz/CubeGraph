#!/usr/bin/env python3
"""
Metadata Visualizer for Filtered Vector Search

Visualizes metadata distributions in 2D and 3D:
- Scatter plots
- Density plots
- Distribution histograms
- Cluster analysis
"""

import numpy as np
import struct
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def load_metadata(metadata_path):
    """Load metadata from binary file"""
    with open(metadata_path, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        d = struct.unpack('Q', f.read(8))[0]

        metadata = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            metadata[i] = np.frombuffer(f.read(4 * d), dtype=np.float32)

    return metadata


def plot_2d_scatter(metadata, title, output_path):
    """Plot 2D scatter plot"""
    if metadata.shape[1] < 2:
        print(f"Skipping 2D scatter plot: metadata dimension is {metadata.shape[1]}")
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(metadata[:, 0], metadata[:, 1], alpha=0.5, s=1)
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_2d_density(metadata, title, output_path):
    """Plot 2D density heatmap"""
    if metadata.shape[1] < 2:
        print(f"Skipping 2D density plot: metadata dimension is {metadata.shape[1]}")
        return

    plt.figure(figsize=(10, 8))
    plt.hexbin(metadata[:, 0], metadata[:, 1], gridsize=50, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_scatter(metadata, title, output_path):
    """Plot 3D scatter plot"""
    if metadata.shape[1] < 3:
        print(f"Skipping 3D scatter plot: metadata dimension is {metadata.shape[1]}")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(metadata[:, 0], metadata[:, 1], metadata[:, 2], alpha=0.5, s=1)
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.set_zlabel('Dimension 2')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_histograms(metadata, title, output_path):
    """Plot histograms for each dimension"""
    n_dims = metadata.shape[1]
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_dims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_dims):
        axes[i].hist(metadata[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(f'Dimension {i}')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Dimension {i} Distribution')
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_statistics(metadata, title, output_path):
    """Plot statistical summary"""
    n_dims = metadata.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Box plot
    axes[0, 0].boxplot([metadata[:, i] for i in range(n_dims)],
                       labels=[f'Dim {i}' for i in range(n_dims)])
    axes[0, 0].set_title('Box Plot')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)

    # Violin plot
    parts = axes[0, 1].violinplot([metadata[:, i] for i in range(n_dims)],
                                   positions=range(n_dims), showmeans=True)
    axes[0, 1].set_title('Violin Plot')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_xticks(range(n_dims))
    axes[0, 1].set_xticklabels([f'Dim {i}' for i in range(n_dims)])
    axes[0, 1].grid(True, alpha=0.3)

    # Statistics table
    stats_data = []
    for i in range(n_dims):
        stats_data.append([
            f'Dim {i}',
            f'{metadata[:, i].min():.2f}',
            f'{metadata[:, i].max():.2f}',
            f'{metadata[:, i].mean():.2f}',
            f'{metadata[:, i].std():.2f}',
            f'{np.median(metadata[:, i]):.2f}'
        ])

    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    table = axes[1, 0].table(cellText=stats_data,
                             colLabels=['Dim', 'Min', 'Max', 'Mean', 'Std', 'Median'],
                             cellLoc='center',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 0].set_title('Statistics Summary')

    # Correlation heatmap (if multiple dimensions)
    if n_dims > 1:
        corr = np.corrcoef(metadata.T)
        im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_xticks(range(n_dims))
        axes[1, 1].set_yticks(range(n_dims))
        axes[1, 1].set_xticklabels([f'Dim {i}' for i in range(n_dims)])
        axes[1, 1].set_yticklabels([f'Dim {i}' for i in range(n_dims)])
        axes[1, 1].set_title('Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 1])

        # Add correlation values
        for i in range(n_dims):
            for j in range(n_dims):
                text = axes[1, 1].text(j, i, f'{corr[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)
    else:
        axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(metadata_dict, output_path):
    """Plot comparison of different distributions"""
    n_dists = len(metadata_dict)
    fig, axes = plt.subplots(2, n_dists, figsize=(5*n_dists, 10))

    if n_dists == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, metadata) in enumerate(metadata_dict.items()):
        # 2D scatter
        if metadata.shape[1] >= 2:
            axes[0, idx].scatter(metadata[:, 0], metadata[:, 1], alpha=0.5, s=1)
            axes[0, idx].set_xlabel('Dimension 0')
            axes[0, idx].set_ylabel('Dimension 1')
            axes[0, idx].set_title(f'{name} - 2D Scatter')
            axes[0, idx].grid(True, alpha=0.3)

        # Histogram of first dimension
        axes[1, idx].hist(metadata[:, 0], bins=50, alpha=0.7, edgecolor='black')
        axes[1, idx].set_xlabel('Dimension 0')
        axes[1, idx].set_ylabel('Frequency')
        axes[1, idx].set_title(f'{name} - Histogram')
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('Distribution Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize metadata distributions')
    parser.add_argument('--metadata', nargs='+', required=True, help='Path(s) to metadata file(s)')
    parser.add_argument('--output-dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--compare', action='store_true', help='Create comparison plots')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_dict = {}
    for metadata_path in args.metadata:
        print(f"\nLoading: {metadata_path}")
        metadata = load_metadata(metadata_path)
        name = Path(metadata_path).stem
        metadata_dict[name] = metadata
        print(f"  Shape: {metadata.shape}")
        print(f"  Min: {metadata.min():.2f}, Max: {metadata.max():.2f}")
        print(f"  Mean: {metadata.mean():.2f}, Std: {metadata.std():.2f}")

    # Generate plots for each metadata file
    for name, metadata in metadata_dict.items():
        print(f"\n{'='*60}")
        print(f"Generating plots for: {name}")
        print(f"{'='*60}")

        # 2D scatter
        plot_2d_scatter(metadata, f'{name} - 2D Scatter',
                       output_dir / f'{name}_scatter_2d.png')

        # 2D density
        plot_2d_density(metadata, f'{name} - 2D Density',
                       output_dir / f'{name}_density_2d.png')

        # 3D scatter
        plot_3d_scatter(metadata, f'{name} - 3D Scatter',
                       output_dir / f'{name}_scatter_3d.png')

        # Histograms
        plot_histograms(metadata, f'{name} - Histograms',
                       output_dir / f'{name}_histograms.png')

        # Statistics
        plot_statistics(metadata, f'{name} - Statistics',
                       output_dir / f'{name}_statistics.png')

    # Comparison plot
    if args.compare and len(metadata_dict) > 1:
        print(f"\n{'='*60}")
        print("Generating comparison plot...")
        print(f"{'='*60}")
        plot_comparison(metadata_dict, output_dir / 'comparison.png')

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
