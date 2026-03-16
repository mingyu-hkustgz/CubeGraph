#!/usr/bin/env python3
"""
Visualization script for QPS-Recall curves.
Reads query result logs from results/recall@{K}/{dataset}/ and generates plots.
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Set up matplotlib for better looking plots
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def parse_log_file(filepath):
    """Parse a log file and extract recall, QPS, and ratio values."""
    recalls = []
    qps_values = []
    ratios = []

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
                    ratio = float(parts[2]) if len(parts) >= 3 else 1.0
                    recalls.append(recall)
                    qps_values.append(qps)
                    ratios.append(ratio)
                except ValueError:
                    continue

    return np.array(recalls), np.array(qps_values), np.array(ratios)


def get_datasets_from_set_sh(set_sh_path='set.sh'):
    """Extract dataset names from set.sh."""
    datasets = []
    if os.path.exists(set_sh_path):
        with open(set_sh_path, 'r') as f:
            for line in f:
                # Match patterns like: export datasets=("sift" "gist" ...)
                if 'datasets=(' in line:
                    # Extract all quoted strings
                    matches = re.findall(r'"([^"]+)"', line)
                    datasets.extend(matches)
    return datasets if datasets else ['sift']  # Default to sift if not found


def get_k_values_from_results(result_path):
    """Get all K values (recall@K) from results directory."""
    k_values = []
    pattern = os.path.join(result_path, 'recall@*')
    for dir_path in glob.glob(pattern):
        match = re.search(r'recall@(\d+)', dir_path)
        if match:
            k_values.append(int(match.group(1)))
    return sorted(k_values)


def get_methods_from_logs(result_path, dataset, k):
    """Get all method names from log files for a given dataset and K."""
    results_dir = os.path.join(result_path, f'recall@{k}', dataset)
    if not os.path.exists(results_dir):
        return []

    methods = []
    pattern = os.path.join(results_dir, '*.log')
    for log_file in glob.glob(pattern):
        # Extract method name from filename: dataset-method.log -> method
        basename = os.path.basename(log_file)
        # Remove dataset prefix and .log suffix
        if basename.endswith('.log'):
            method = basename[:-4]  # Remove .log
            if method.startswith(dataset + '-'):
                method = method[len(dataset) + 1:]
            methods.append(method)
    return sorted(methods)


def plot_qps_recall_curve(result_path, dataset, k, methods, output_dir):
    """Plot QPS-Recall curves for all methods."""
    if not methods:
        print(f"  No methods found for {dataset} @ recall@{k}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    for idx, method in enumerate(methods):
        log_file = os.path.join(result_path, f'recall@{k}', dataset, f'{dataset}-{method}.log')

        if not os.path.exists(log_file):
            continue

        recalls, qps_values, ratios = parse_log_file(log_file)

        if len(recalls) == 0:
            continue

        # Sort by recall for proper line plotting
        sort_idx = np.argsort(recalls)
        recalls = recalls[sort_idx]
        qps_values = qps_values[sort_idx]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax.plot(recalls, qps_values, marker=marker, markersize=8,
                label=method, color=color, linewidth=2, markevery=1)

    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('QPS (queries per second)')
    ax.set_title(f'QPS-Recall Curve - {dataset.upper()} (recall@{k})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Set x-axis to show percentage format
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    output_file = os.path.join(output_dir, f'{dataset}_recall{k}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize QPS-Recall curves')
    parser.add_argument('--result-path', type=str, default='./results',
                        help='Path to results directory')
    parser.add_argument('--figure-path', type=str, default='./figure',
                        help='Path to save figure output')
    parser.add_argument('--set-sh', type=str, default='./set.sh',
                        help='Path to set.sh for dataset list')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to plot (overrides set.sh)')
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                        help='Specific K values to plot (e.g., 20 100)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Specific methods to plot (e.g., hnsw hnsw-cube)')
    args = parser.parse_args()

    # Get datasets from set.sh or use provided
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = get_datasets_from_set_sh(args.set_sh)

    # Get K values from results or use provided
    if args.k_values:
        k_values = args.k_values
    else:
        k_values = get_k_values_from_results(args.result_path)

    if not k_values:
        print("No K values found in results directory")
        return

    print(f"Datasets: {datasets}")
    print(f"K values: {k_values}")

    # Ensure figure directory exists
    os.makedirs(args.figure_path, exist_ok=True)

    for dataset in datasets:
        # Create dataset-specific figure directory
        dataset_figure_dir = os.path.join(args.figure_path, dataset)
        os.makedirs(dataset_figure_dir, exist_ok=True)

        for k in k_values:
            if args.methods:
                methods = args.methods
            else:
                methods = get_methods_from_logs(args.result_path, dataset, k)

            if not methods:
                print(f"No log files found for {dataset} @ recall@{k}")
                continue

            print(f"Processing {dataset} @ recall@{k}...")
            plot_qps_recall_curve(args.result_path, dataset, k, methods, dataset_figure_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
