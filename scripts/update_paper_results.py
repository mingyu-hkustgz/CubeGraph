#!/usr/bin/env python3
"""
Parse benchmark logs from results/ and update paper/*.tex files.
Uses position-aware replacement to ensure correct subplot blocks are updated.
"""

import os
import re
import json
import glob
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "recall@20"
TIME_LOG_DIR = ROOT / "results" / "time-log"
PAPER_DIR = ROOT / "paper"


def parse_standard_log(filepath):
    """Parse logs with format: recall qps ratio"""
    points = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                recall = float(parts[0])
                qps = float(parts[1])
                ratio = float(parts[2])
            except ValueError:
                continue
            points.append((recall, qps, ratio))
    return points


def parse_acorn_log(filepath):
    """Parse ACORN logs with format: param recall qps"""
    points = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                param = float(parts[0])
                recall = float(parts[1])
                qps = float(parts[2])
            except ValueError:
                continue
            points.append((recall, qps))
    return points


def get_standard_points(filepath):
    """Get (recall, qps) points from standard logs, filtering ratio <= 1.01"""
    if not filepath.exists():
        return None
    points = parse_standard_log(filepath)
    # Filter ratio <= 1.01
    filtered = [(recall, qps) for recall, qps, ratio in points if ratio <= 1.01]
    # Deduplicate by recall (rounded to 2 decimals), keeping highest QPS
    best_qps = {}
    for recall, qps in filtered:
        r = round(recall, 2)
        if r not in best_qps or qps > best_qps[r][1]:
            best_qps[r] = (recall, qps)
    result = list(best_qps.values())
    result.sort(key=lambda x: x[0])
    return result


def get_acorn_points(filepath):
    """Get (recall, qps) points from ACORN logs"""
    if not filepath.exists():
        return None
    points = parse_acorn_log(filepath)
    # Deduplicate by recall (rounded to 2 decimals), keeping highest QPS
    best_qps = {}
    for recall, qps in points:
        r = round(recall, 2)
        if r not in best_qps or qps > best_qps[r][1]:
            best_qps[r] = (recall, qps)
    result = list(best_qps.values())
    result.sort(key=lambda x: x[0])
    return result


def format_coordinates(points):
    """Format points as tikz coordinate string"""
    if not points:
        return ""
    lines = []
    for recall, qps in points:
        lines.append(f"    ({recall:.2f}, {qps:.2f})")
    return "\n".join(lines)


def parse_time_log(filepath):
    """Parse time log and return median of values > 10 seconds"""
    if not filepath.exists():
        return None
    values = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("index_time_sec:"):
                try:
                    val = float(line.split(":", 1)[1].strip())
                    if val > 10:
                        values.append(val)
                except ValueError:
                    continue
    if not values:
        return None
    return statistics.median(values)


# ============================================================================
# Position-aware block replacement
# ============================================================================

def replace_next_block(content, label, new_coords, start_pos=0):
    """
    Find the next occurrence of a comment label after start_pos,
    then replace the plot coordinates block that follows it.
    Returns (new_content, end_pos_of_replacement) or (None, None) if not found.
    """
    # Find the comment line
    escaped_label = re.escape(label)
    # Pattern to find the comment line
    comment_pattern = rf'%\s*{escaped_label}\s*\n'

    comment_match = re.search(comment_pattern, content[start_pos:])
    if not comment_match:
        return None, None

    comment_abs_start = start_pos + comment_match.start()
    comment_abs_end = start_pos + comment_match.end()

    # Now find the plot coordinates block starting after the comment
    # Pattern: optional \addplot[...] line, then plot coordinates { ... };
    block_pattern = r'(?:\\addplot\[[^\]]*\]\s*\n)?plot\s+coordinates\s*\{[^}]*\};'

    block_match = re.search(block_pattern, content[comment_abs_end:])
    if not block_match:
        return None, None

    block_abs_start = comment_abs_end + block_match.start()
    block_abs_end = comment_abs_end + block_match.end()

    # Check if there's an \addplot line in the match
    block_text = content[block_abs_start:block_abs_end]
    addplot_match = re.match(r'(\\addplot\[[^\]]*\]\s*\n)', block_text)

    if addplot_match:
        addplot_line = addplot_match.group(1)
        new_block = f"{addplot_line}plot coordinates {{\n{new_coords}\n}};"
    else:
        new_block = f"plot coordinates {{\n{new_coords}\n}};"

    new_content = content[:block_abs_start] + new_block + content[block_abs_end:]
    new_end_pos = block_abs_start + len(new_block)

    return new_content, new_end_pos


# ============================================================================
# Log path helpers
# ============================================================================

def get_cg_log(dataset, dim, ratio):
    """Get CG log path"""
    if dataset == "yfcc":
        meta = f"real_{dim}"
    else:
        meta = f"uniform_{dim}"
    return RESULTS_DIR / dataset / f"{dataset}-hnsw-cube-merge-layer-{meta}-{ratio}.log"


def get_post_log(dataset, dim, ratio):
    """Get POST log path"""
    if dataset == "yfcc":
        meta = f"real_{dim}"
    else:
        meta = f"uniform_{dim}"
    return RESULTS_DIR / dataset / f"{dataset}-hnsw-post-{meta}-{ratio}.log"


def get_acorn_log(dataset, dim, ratio):
    """Get ACORN log path"""
    if dataset == "deep100m":
        return None
    if dataset == "yfcc":
        meta = f"real_{dim}"
    else:
        meta = f"uniform_{dim}"
    pattern = f"{dataset}_acorn_*_{meta}_{float(ratio):.6f}.log"
    matches = list((RESULTS_DIR / dataset).glob(pattern))
    if matches:
        return matches[0]
    return None


def get_irange_log(dataset, dim, ratio):
    """Get iRange/KDTree log path"""
    if dataset == "deep100m":
        return None
    if dataset == "yfcc":
        meta = f"real_{dim}"
    else:
        meta = f"uniform_{dim}"
    return RESULTS_DIR / dataset / f"{dataset}-kdtree-partition-{meta}-{ratio}.log"


def get_polygon_log(dataset, dim, ratio, n):
    """Get Polygon-N log path"""
    return RESULTS_DIR / dataset / f"{dataset}-hnsw-cube-merge-layer-polygon-uniform_{dim}-{n}-{ratio}.log"


def get_radius_log(dataset, dim, ratio):
    """Get Radius log path"""
    return RESULTS_DIR / dataset / f"{dataset}-hnsw-cube-merge-layer-radius-uniform_{dim}-{ratio}.log"


def get_compose_log(dataset, dim, ratio):
    """Get Compose log path"""
    if dim != "2d":
        return None
    return RESULTS_DIR / dataset / f"{dataset}-hnsw-cube-complex-uniform_{dim}-{ratio}-0.30.log"


# ============================================================================
# Update functions
# ============================================================================

def update_exp1():
    """Update Exp-1.tex"""
    tex_path = PAPER_DIR / "Exp-1.tex"
    with open(tex_path) as f:
        content = f.read()

    configs = [
        ("sift", "2d", "0.01"), ("sift", "2d", "0.02"), ("sift", "2d", "0.05"), ("sift", "2d", "0.10"),
        ("msmarc10m", "2d", "0.01"), ("msmarc10m", "2d", "0.02"), ("msmarc10m", "2d", "0.05"), ("msmarc10m", "2d", "0.10"),
        ("yfcc", "2d", "0.01"), ("yfcc", "2d", "0.02"), ("yfcc", "2d", "0.05"), ("yfcc", "2d", "0.10"),
        ("deep100m", "2d", "0.05"), ("deep100m", "2d", "0.10"), ("deep100m", "2d", "0.20"), ("deep100m", "2d", "0.40"),
    ]

    methods = [
        ("cg", "CG (CubeGraph)"),
        ("acorn", "ACORN"),
        ("post", "POST"),
        ("irange", "iRange (KD-Tree)"),
    ]

    pos = 0
    for dataset, dim, ratio in configs:
        for method_key, method_label in methods:
            if method_key == "cg":
                log_path = get_cg_log(dataset, dim, ratio)
            elif method_key == "post":
                log_path = get_post_log(dataset, dim, ratio)
            elif method_key == "acorn":
                log_path = get_acorn_log(dataset, dim, ratio)
            elif method_key == "irange":
                log_path = get_irange_log(dataset, dim, ratio)
            else:
                log_path = None

            if log_path is None:
                continue

            if method_key == "acorn":
                points = get_acorn_points(log_path)
            else:
                points = get_standard_points(log_path)

            if points is None:
                print(f"Exp-1: {dataset} {dim} r={ratio} {method_key}: log not found")
                continue
            if not points:
                print(f"Exp-1: {dataset} {dim} r={ratio} {method_key}: no valid points")
                continue

            new_coords = format_coordinates(points)
            new_content, new_pos = replace_next_block(content, method_label, new_coords, pos)

            if new_content is None:
                print(f"Exp-1: {dataset} {dim} r={ratio} {method_key}: BLOCK NOT FOUND")
                continue

            content = new_content
            pos = new_pos
            print(f"Exp-1: {dataset} {dim} r={ratio} {method_key}: updated with {len(points)} points")

    with open(tex_path, "w") as f:
        f.write(content)
    print("Exp-1.tex updated.")


def update_exp2():
    """Update Exp-2.tex"""
    tex_path = PAPER_DIR / "Exp-2.tex"
    with open(tex_path) as f:
        content = f.read()

    configs = [
        ("sift", "3d", "0.02"), ("sift", "3d", "0.05"),
        ("sift", "4d", "0.02"), ("sift", "4d", "0.05"),
        ("msmarc10m", "3d", "0.02"), ("msmarc10m", "3d", "0.05"),
        ("msmarc10m", "4d", "0.02"), ("msmarc10m", "4d", "0.05"),
        ("yfcc", "3d", "0.01"), ("yfcc", "3d", "0.02"), ("yfcc", "3d", "0.05"), ("yfcc", "3d", "0.10"),
    ]

    methods = [
        ("cg", "CG (CubeGraph)"),
        ("post", "POST"),
        ("irange", "iRange (KD-Tree)"),
    ]

    pos = 0
    for dataset, dim, ratio in configs:
        for method_key, method_label in methods:
            if method_key == "cg":
                log_path = get_cg_log(dataset, dim, ratio)
            elif method_key == "post":
                log_path = get_post_log(dataset, dim, ratio)
            elif method_key == "irange":
                log_path = get_irange_log(dataset, dim, ratio)
            else:
                log_path = None

            if log_path is None:
                continue

            points = get_standard_points(log_path)
            if points is None:
                print(f"Exp-2: {dataset} {dim} r={ratio} {method_key}: log not found")
                continue
            if not points:
                print(f"Exp-2: {dataset} {dim} r={ratio} {method_key}: no valid points")
                continue

            new_coords = format_coordinates(points)
            new_content, new_pos = replace_next_block(content, method_label, new_coords, pos)

            if new_content is None:
                print(f"Exp-2: {dataset} {dim} r={ratio} {method_key}: BLOCK NOT FOUND")
                continue

            content = new_content
            pos = new_pos
            print(f"Exp-2: {dataset} {dim} r={ratio} {method_key}: updated with {len(points)} points")

    with open(tex_path, "w") as f:
        f.write(content)
    print("Exp-2.tex updated.")


def update_exp3():
    """Update Exp-3.tex"""
    tex_path = PAPER_DIR / "Exp-3.tex"
    with open(tex_path) as f:
        content = f.read()

    configs = [
        ("sift", "2d", "0.05"), ("sift", "2d", "0.10"),
        ("sift", "3d", "0.10"), ("sift", "4d", "0.10"),
    ]

    pos = 0
    for dataset, dim, ratio in configs:
        # Cube
        log_path = get_cg_log(dataset, dim, ratio)
        if dim == "3d":
            label = "Cube-3D (baseline)"
        elif dim == "4d":
            label = "Cube-4D (baseline)"
        else:
            label = "Cube (baseline)"
        points = get_standard_points(log_path)
        if points:
            new_coords = format_coordinates(points)
            new_content, new_pos = replace_next_block(content, label, new_coords, pos)
            if new_content:
                content = new_content
                pos = new_pos
                print(f"Exp-3: {dataset} {dim} r={ratio} cube: updated with {len(points)} points")

        # Polygon-3,4,5 (only for 2d)
        if dim == "2d":
            for n in ["3", "4", "5"]:
                log_path = get_polygon_log(dataset, dim, ratio, n)
                label = f"Polygon-{n}"
                points = get_standard_points(log_path)
                if points:
                    new_coords = format_coordinates(points)
                    new_content, new_pos = replace_next_block(content, label, new_coords, pos)
                    if new_content:
                        content = new_content
                        pos = new_pos
                        print(f"Exp-3: {dataset} {dim} r={ratio} polygon-{n}: updated with {len(points)} points")

        # Radius
        log_path = get_radius_log(dataset, dim, ratio)
        if dim == "3d":
            label = "Radius-3D"
        elif dim == "4d":
            label = "Radius-4D"
        else:
            label = "Radius"
        points = get_standard_points(log_path)
        if points:
            new_coords = format_coordinates(points)
            new_content, new_pos = replace_next_block(content, label, new_coords, pos)
            if new_content:
                content = new_content
                pos = new_pos
                print(f"Exp-3: {dataset} {dim} r={ratio} radius: updated with {len(points)} points")

        # Compose (only for 2d)
        if dim == "2d":
            log_path = get_compose_log(dataset, dim, ratio)
            label = "Compose (Cube - Radius)"
            points = get_standard_points(log_path)
            if points:
                new_coords = format_coordinates(points)
                new_content, new_pos = replace_next_block(content, label, new_coords, pos)
                if new_content:
                    content = new_content
                    pos = new_pos
                    print(f"Exp-3: {dataset} {dim} r={ratio} compose: updated with {len(points)} points")

    with open(tex_path, "w") as f:
        f.write(content)
    print("Exp-3.tex updated.")


def update_exp4():
    """Update Exp-4.tex indexing time table"""
    tex_path = PAPER_DIR / "Exp-4.tex"
    with open(tex_path) as f:
        content = f.read()

    datasets = ["sift", "yfcc", "msmarc10m", "deep100m"]
    methods = {
        r"$\CG$": "bench_hierarchical_cube",
        r"$\POST$": "bench_hnsw_post",
        r"$\ACORN$-$$\gamma$$": None,
        r"$\iRange$": "bench_kdtree_partition",
    }

    # Build time data
    times = {}
    for dataset in datasets:
        times[dataset] = {}
        for method_name, log_prefix in methods.items():
            if log_prefix is None:
                continue
            log_path = TIME_LOG_DIR / f"{dataset}-{log_prefix}.log"
            t = parse_time_log(log_path)
            if t is not None:
                times[dataset][method_name] = round(t)

    # Update table rows
    for method_name, log_prefix in methods.items():
        if log_prefix is None:
            continue

        cols = [method_name]
        for dataset in datasets:
            if dataset in times and method_name in times[dataset]:
                cols.append(str(times[dataset][method_name]))
            else:
                cols.append("--")

        new_row = "        " + " & ".join(cols) + r" \\"

        escaped_method = re.escape(method_name)
        pattern = rf"^\s*{escaped_method}\s+&.*\\\\\s*$"

        def row_replacer(m):
            return new_row

        content, count = re.subn(pattern, row_replacer, content, count=1, flags=re.MULTILINE)
        if count:
            print(f"Exp-4: Updated row for {method_name}")
        else:
            print(f"Exp-4: Could not find row for {method_name}")

    with open(tex_path, "w") as f:
        f.write(content)
    print("Exp-4.tex updated.")


if __name__ == "__main__":
    print("Updating paper results from benchmark logs...")
    print()

    print("=== Updating Exp-1.tex ===")
    update_exp1()
    print()

    print("=== Updating Exp-2.tex ===")
    update_exp2()
    print()

    print("=== Updating Exp-3.tex ===")
    update_exp3()
    print()

    print("=== Updating Exp-4.tex ===")
    update_exp4()
    print()

    print("Done!")
