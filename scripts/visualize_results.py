#!/usr/bin/env python3
"""
Query Result Visualizer for Filtered Vector Search

Visualizes query results:
- Query regions and result distributions
- Performance metrics (latency, recall)
- Filter selectivity analysis
"""

import numpy as np
import struct
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_metadata(metadata_path):
    """Load metadata from binary file"""
    with open(metadata_path, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        d = struct.unpack('Q', f.read(8))[0]

        metadata = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            metadata[i] = np.frombuffer(f.read(4 * d), dtype=np.float32)

    return metadata


def plot_query_results_2d(metadata, query_center, query_radius, result_ids, title, output_path):
    """Plot query results in 2D"""
    if metadata.shape[1] < 2:
        print(f"Skipping 2D query plot: metadata dimension is {metadata.shape[1]}")
        return

    plt.figure(figsize=(12, 10))

    # Plot all points
    plt.scatter(metadata[:, 0], metadata[:, 1], alpha=0.3, s=10, c='gray', label='All points')

    # Plot result points
    if len(result_ids) > 0:
        result_metadata = metadata[result_ids]
        plt.scatter(result_metadata[:, 0], result_metadata[:, 1],
                   alpha=0.8, s=30, c='red', label=f'Results ({len(result_ids)})')

    # Plot query region
    if query_radius is not None:
        circle = plt.Circle((query_center[0], query_center[1]), query_radius,
                           color='blue', fill=False, linewidth=2, linestyle='--',
                           label=f'Query region (r={query_radius:.1f})')
        plt.gca().add_patch(circle)

    # Plot query center
    plt.scatter([query_center[0]], [query_center[1]], s=200, c='blue',
               marker='*', edgecolors='black', linewidths=2, label='Query center', zorder=5)

    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_query_results_range_2d(metadata, bbox_min, bbox_max, result_ids, title, output_path):
    """Plot range query results in 2D"""
    if metadata.shape[1] < 2:
        print(f"Skipping 2D range query plot: metadata dimension is {metadata.shape[1]}")
        return

    plt.figure(figsize=(12, 10))

    # Plot all points
    plt.scatter(metadata[:, 0], metadata[:, 1], alpha=0.3, s=10, c='gray', label='All points')

    # Plot result points
    if len(result_ids) > 0:
        result_metadata = metadata[result_ids]
        plt.scatter(result_metadata[:, 0], result_metadata[:, 1],
                   alpha=0.8, s=30, c='red', label=f'Results ({len(result_ids)})')

    # Plot query bounding box
    from matplotlib.patches import Rectangle
    width = bbox_max[0] - bbox_min[0]
    height = bbox_max[1] - bbox_min[1]
    rect = Rectangle((bbox_min[0], bbox_min[1]), width, height,
                     linewidth=2, edgecolor='blue', facecolor='none',
                     linestyle='--', label='Query region')
    plt.gca().add_patch(rect)

    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_performance_metrics(results_dict, output_path):
    """Plot performance metrics comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results_dict.keys())
    latencies = [results_dict[m]['latency'] for m in methods]
    recalls = [results_dict[m].get('recall', 0) for m in methods]
    throughputs = [1000000.0 / results_dict[m]['latency'] for m in methods]  # QPS
    result_counts = [results_dict[m]['num_results'] for m in methods]

    # Latency comparison
    axes[0, 0].bar(methods, latencies, color='skyblue', edgecolor='black')
    axes[0, 0].set_ylabel('Latency (μs)')
    axes[0, 0].set_title('Query Latency')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Throughput comparison
    axes[0, 1].bar(methods, throughputs, color='lightgreen', edgecolor='black')
    axes[0, 1].set_ylabel('Throughput (QPS)')
    axes[0, 1].set_title('Query Throughput')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Recall comparison
    if any(r > 0 for r in recalls):
        axes[1, 0].bar(methods, recalls, color='lightcoral', edgecolor='black')
        axes[1, 0].set_ylabel('Recall (%)')
        axes[1, 0].set_title('Search Recall')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No recall data available',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')

    # Result count comparison
    axes[1, 1].bar(methods, result_counts, color='plum', edgecolor='black')
    axes[1, 1].set_ylabel('Number of Results')
    axes[1, 1].set_title('Results Returned')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.suptitle('Performance Metrics Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_filter_selectivity(metadata, filters, output_path):
    """Plot filter selectivity analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Radius selectivity
    radii = np.linspace(5, 50, 20)
    center = np.mean(metadata, axis=0)
    selectivities = []

    for r in radii:
        if metadata.shape[1] >= 2:
            dists = np.sqrt(np.sum((metadata[:, :2] - center[:2])**2, axis=1))
            selected = np.sum(dists <= r)
        else:
            dists = np.abs(metadata[:, 0] - center[0])
            selected = np.sum(dists <= r)
        selectivities.append(100.0 * selected / len(metadata))

    axes[0, 0].plot(radii, selectivities, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Radius')
    axes[0, 0].set_ylabel('Selectivity (%)')
    axes[0, 0].set_title('Radius Filter Selectivity')
    axes[0, 0].grid(True, alpha=0.3)

    # Range selectivity (varying range size)
    range_sizes = np.linspace(10, 50, 20)
    selectivities = []

    for size in range_sizes:
        bbox_min = center - size/2
        bbox_max = center + size/2
        selected = np.sum(np.all((metadata >= bbox_min) & (metadata <= bbox_max), axis=1))
        selectivities.append(100.0 * selected / len(metadata))

    axes[0, 1].plot(range_sizes, selectivities, marker='s', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Range Size')
    axes[0, 1].set_ylabel('Selectivity (%)')
    axes[0, 1].set_title('Range Filter Selectivity')
    axes[0, 1].grid(True, alpha=0.3)

    # Distribution of distances from center
    if metadata.shape[1] >= 2:
        dists = np.sqrt(np.sum((metadata[:, :2] - center[:2])**2, axis=1))
    else:
        dists = np.abs(metadata[:, 0] - center[0])

    axes[1, 0].hist(dists, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Distance from Center')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distance Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative selectivity
    sorted_dists = np.sort(dists)
    cumulative = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists) * 100

    axes[1, 1].plot(sorted_dists, cumulative, linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Distance from Center')
    axes[1, 1].set_ylabel('Cumulative Selectivity (%)')
    axes[1, 1].set_title('Cumulative Distance Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Filter Selectivity Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize query results')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--results', type=str, help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--query-center', nargs='+', type=float, help='Query center coordinates')
    parser.add_argument('--query-radius', type=float, help='Query radius')
    parser.add_argument('--bbox-min', nargs='+', type=float, help='Bounding box minimum')
    parser.add_argument('--bbox-max', nargs='+', type=float, help='Bounding box maximum')
    parser.add_argument('--result-ids', nargs='+', type=int, default=[], help='Result IDs')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print(f"Loading metadata: {args.metadata}")
    metadata = load_metadata(args.metadata)
    print(f"  Shape: {metadata.shape}")

    # Plot filter selectivity
    print("\nGenerating filter selectivity analysis...")
    plot_filter_selectivity(metadata, None, output_dir / 'filter_selectivity.png')

    # Plot query results if provided
    if args.query_center and args.query_radius:
        print("\nGenerating radius query visualization...")
        query_center = np.array(args.query_center)
        plot_query_results_2d(metadata, query_center, args.query_radius,
                             args.result_ids, 'Radius Query Results',
                             output_dir / 'query_radius_results.png')

    if args.bbox_min and args.bbox_max:
        print("\nGenerating range query visualization...")
        bbox_min = np.array(args.bbox_min)
        bbox_max = np.array(args.bbox_max)
        plot_query_results_range_2d(metadata, bbox_min, bbox_max,
                                    args.result_ids, 'Range Query Results',
                                    output_dir / 'query_range_results.png')

    # Load and plot performance results if provided
    if args.results:
        print(f"\nLoading results: {args.results}")
        with open(args.results, 'r') as f:
            results_dict = json.load(f)
        plot_performance_metrics(results_dict, output_dir / 'performance_metrics.png')

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
