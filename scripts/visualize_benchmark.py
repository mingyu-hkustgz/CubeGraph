#!/usr/bin/env python3
"""
Visualize CubeGraph Benchmark Results

Creates comparison plots for recall vs QPS across different algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path


def parse_log_file(filepath):
    """Parse a log file in format: 'recall QPS'"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    recall = float(parts[0])
                    qps = float(parts[1])
                    results.append((recall, qps))
                except ValueError:
                    continue
    return results


def plot_results(results_dir, dataset, output_path):
    """Plot recall vs QPS for different algorithms"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        'hnsw': '#1f77b4',  # blue
        'cube': '#ff7f0e',   # orange
        'grid': '#2ca02c',   # green
        'rtree': '#d62728'   # red
    }

    markers = {
        'hnsw': 'o',
        'cube': 's',
        'grid': '^',
        'rtree': 'D'
    }

    # Find all log files
    algos = ['hnsw', 'cube', 'grid', 'rtree']
    recall_data = {}
    qps_data = {}

    for algo in algos:
        log_path = Path(results_dir) / f'{dataset}-{algo}.log'
        if log_path.exists():
            data = parse_log_file(log_path)
            if data:
                recalls = [d[0] for d in data]
                qps = [d[1] for d in data]
                recall_data[algo] = recalls
                qps_data[algo] = qps

    # Plot each algorithm
    for algo in recall_data.keys():
        ax.scatter(qps_data[algo], recall_data[algo],
                  c=colors.get(algo, 'gray'),
                  marker=markers.get(algo, 'o'),
                  s=100, label=algo.upper(), alpha=0.8)

        # Connect points with line
        if len(qps_data[algo]) > 1:
            # Sort by QPS
            sorted_pairs = sorted(zip(qps_data[algo], recall_data[algo]))
            sorted_qps = [p[0] for p in sorted_pairs]
            sorted_recall = [p[1] for p in sorted_pairs]
            ax.plot(sorted_qps, sorted_recall,
                   c=colors.get(algo, 'gray'), linestyle='--', alpha=0.5)

    ax.set_xlabel('QPS (queries per second)', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title(f'{dataset.upper()} - Recall vs QPS Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    if qps_data:
        max_qps = max(max(v) for v in qps_data.values())
        ax.set_xlim(0, max_qps * 1.1)

    if recall_data:
        max_recall = max(max(v) for v in recall_data.values())
        ax.set_ylim(0, min(100, max_recall * 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize CubeGraph benchmark results')
    parser.add_argument('--results-dir', type=str, default='results/recall@20',
                        help='Directory containing results')
    parser.add_argument('--dataset', type=str, default='sift',
                        help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path')

    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        output_path = f'{args.results_dir}/{args.dataset}/comparison.png'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plot_results(args.results_dir, args.dataset, output_path)


if __name__ == '__main__':
    main()
