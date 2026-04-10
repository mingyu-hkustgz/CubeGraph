#!/usr/bin/env python3
"""
Extract recall@20 and Qps data from experiment log files.
Outputs .dat files for use in TikZ/PGFPlots figures.
"""

import os
import re
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results/recall@20")
OUTPUT_DIR = Path("figure")
RATIOS_2D = [0.01, 0.02, 0.05, 0.10]
RATIOS_3D_4D = [0.05, 0.10, 0.01, 0.02]
DATASETS = ["sift", "msmarc10m", "yfcc"]
DIMENSIONS = ["2d", "3d", "4d"]

def parse_cube_file(filepath):
    """Parse CG (CubeGraph) file: [recall qps ratio]"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    recall = float(parts[0])
                    qps = float(parts[1])
                    data.append((recall, qps))
                except ValueError:
                    continue
    return data

def parse_acorn_file(filepath):
    """Parse ACORN file: [efSearch recall qps]"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    ef = float(parts[0])
                    recall = float(parts[1])
                    qps = float(parts[2])
                    data.append((recall, qps))
                except ValueError:
                    continue
    return data

def find_files(dataset, method, ratio, dim):
    """Find log files for given dataset, method, ratio, and dimension."""
    dirpath = RESULTS_DIR / dataset

    # Format ratio for different file naming conventions
    ratio_2f = f"{ratio:.2f}"  # e.g., "0.01", "0.05", "0.10"
    ratio_6f = f"{ratio:.6f}"  # e.g., "0.010000", "0.050000", "0.100000"

    if method == "CG":
        # CG: *hnsw-cube-merge-layer-{dim}-{ratio}.log
        patterns = [
            f"{dataset}-hnsw-cube-merge-layer-uniform_{dim}-{ratio_2f}.log",
            f"{dataset}-hnsw-cube-merge-layer-real_{dim}-{ratio_2f}.log",
            f"{dataset}-hnsw-cube-merge-layer-{dim}-{ratio_2f}.log",  # SIFT style (no dim prefix)
        ]
        for p in patterns:
            fp = dirpath / p
            if fp.exists():
                return fp

    elif method == "ACORN":
        # ACORN: *acorn_1_*_{dim}_{ratio_6f}.log
        patterns = [
            f"{dataset}_acorn_1_uniform_{dim}_{ratio_6f}.log",
            f"{dataset}_acorn_1_real_{dim}_{ratio_6f}.log",
        ]
        for p in patterns:
            fp = dirpath / p
            if fp.exists():
                return fp

    elif method == "POST":
        # POST: *hnsw-post-*_{dim}-{ratio}.log
        patterns = [
            f"{dataset}-hnsw-post-uniform_{dim}-{ratio_2f}.log",
            f"{dataset}-hnsw-post-real_{dim}-{ratio_2f}.log",
        ]
        for p in patterns:
            fp = dirpath / p
            if fp.exists():
                return fp

    return None

def extract_all_data():
    """Extract data for all datasets and methods."""
    all_data = {}

    for dataset in DATASETS:
        all_data[dataset] = {}
        for dim in DIMENSIONS:
            all_data[dataset][dim] = {}
            # Use different ratio sets for 2D vs 3D/4D
            ratios = RATIOS_2D if dim == "2d" else RATIOS_3D_4D
            for ratio in ratios:
                ratio_key = f"{ratio:.2f}"
                all_data[dataset][dim][ratio_key] = {}

                for method in ["CG", "ACORN", "POST"]:
                    filepath = find_files(dataset, method, ratio, dim)
                    if filepath:
                        if method == "ACORN":
                            data = parse_acorn_file(filepath)
                        else:
                            data = parse_cube_file(filepath)

                        if data:
                            all_data[dataset][dim][ratio_key][method] = data
                            print(f"Found {dataset}/{dim}/{method}/{ratio_key}: {len(data)} points from {filepath.name}")

    return all_data

def write_dat_files(all_data):
    """Write .dat files for each dataset/method/ratio/dimension combination."""
    for dataset, dims in all_data.items():
        for dim, ratios in dims.items():
            for ratio_str, methods in ratios.items():
                for method, data in methods.items():
                    if data:
                        outdir = OUTPUT_DIR / dataset
                        outdir.mkdir(parents=True, exist_ok=True)
                        outfile = outdir / f"data_{method}_{dim}_{ratio_str.replace('.', '_')}.dat"
                        # Sort by recall (ascending) to ensure monotonic curves
                        data_sorted = sorted(data, key=lambda x: x[0])
                        with open(outfile, 'w') as f:
                            for recall, qps in data_sorted:
                                f.write(f"{recall:.4f} {qps:.2f}\n")

if __name__ == "__main__":
    print("Extracting recall@20 vs Qps data...")
    all_data = extract_all_data()
    write_dat_files(all_data)
    print("\nDone! Data files written to figure/{dataset}/")
