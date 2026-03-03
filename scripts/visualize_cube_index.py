#!/usr/bin/env python3
"""
Visualize Cube-Based Index Structure

Creates diagrams showing:
1. Hierarchical cube structure
2. Cross-cube edge connectivity
3. Filtered search process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def plot_cube_hierarchy():
    """Plot hierarchical cube structure"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Root cube
    root = patches.Rectangle((0, 0), 100, 100, linewidth=3, edgecolor='black', facecolor='lightblue', alpha=0.3)
    ax.add_patch(root)
    ax.text(50, 105, 'Root Cube [0,100]×[0,100]', ha='center', fontsize=12, fontweight='bold')

    # Level 1 cubes (4 children in 2D)
    colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightcyan']
    labels = ['[0,50]×[0,50]', '[50,100]×[0,50]', '[0,50]×[50,100]', '[50,100]×[50,100]']
    positions = [(0, 0), (50, 0), (0, 50), (50, 50)]

    for i, (x, y) in enumerate(positions):
        cube = patches.Rectangle((x, y), 50, 50, linewidth=2, edgecolor='blue', facecolor=colors[i], alpha=0.4)
        ax.add_patch(cube)
        ax.text(x + 25, y + 25, f'Cube {i}\n{labels[i]}', ha='center', va='center', fontsize=9)

    # Level 2 cubes (subdivide one cube as example)
    sub_positions = [(0, 0), (25, 0), (0, 25), (25, 25)]
    for i, (x, y) in enumerate(sub_positions):
        cube = patches.Rectangle((x, y), 25, 25, linewidth=1, edgecolor='red', facecolor='white', alpha=0.6)
        ax.add_patch(cube)
        ax.text(x + 12.5, y + 12.5, f'Leaf\n{i}', ha='center', va='center', fontsize=7)

    # Add arrows showing hierarchy
    ax.annotate('', xy=(25, 50), xytext=(50, 95), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(12.5, 25), xytext=(25, 45), arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 115)
    ax.set_aspect('equal')
    ax.set_title('Hierarchical Cube Structure (2D)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig

def plot_cross_cube_edges():
    """Plot cross-cube edge connectivity"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw 3x3 grid of cubes
    cube_size = 30
    margin = 5
    colors = plt.cm.Set3(np.linspace(0, 1, 9))

    cubes = []
    for i in range(3):
        for j in range(3):
            x = j * (cube_size + margin)
            y = i * (cube_size + margin)
            cube = patches.Rectangle((x, y), cube_size, cube_size,
                                    linewidth=2, edgecolor='black',
                                    facecolor=colors[i*3 + j], alpha=0.5)
            ax.add_patch(cube)
            ax.text(x + cube_size/2, y + cube_size/2, f'C{i}{j}',
                   ha='center', va='center', fontsize=12, fontweight='bold')
            cubes.append((x, y))

    # Draw cross-cube edges (center cube to neighbors)
    center_idx = 4  # Cube C11
    center_x, center_y = cubes[center_idx]
    center_x += cube_size / 2
    center_y += cube_size / 2

    # Neighbors: left, right, top, bottom
    neighbor_indices = [3, 5, 1, 7]  # C10, C12, C01, C21
    directions = ['Left', 'Right', 'Top', 'Bottom']
    arrow_colors = ['red', 'blue', 'green', 'orange']

    for idx, direction, color in zip(neighbor_indices, directions, arrow_colors):
        neighbor_x, neighbor_y = cubes[idx]
        neighbor_x += cube_size / 2
        neighbor_y += cube_size / 2

        # Draw multiple edges (M_cross = 8)
        for k in range(3):  # Show 3 edges as example
            offset = (k - 1) * 2
            if direction in ['Left', 'Right']:
                start = (center_x, center_y + offset)
                end = (neighbor_x, neighbor_y + offset)
            else:
                start = (center_x + offset, center_y)
                end = (neighbor_x + offset, neighbor_y)

            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=color, alpha=0.6))

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='red', alpha=0.6, label='Left edges'),
        patches.Patch(facecolor='blue', alpha=0.6, label='Right edges'),
        patches.Patch(facecolor='green', alpha=0.6, label='Top edges'),
        patches.Patch(facecolor='orange', alpha=0.6, label='Bottom edges')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlim(-5, 3*(cube_size + margin))
    ax.set_ylim(-5, 3*(cube_size + margin))
    ax.set_aspect('equal')
    ax.set_title('Cross-Cube Edge Connectivity (2D)\nCenter cube C11 connected to 4 neighbors',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    ax.text(50, -10, 'Each arrow represents M_cross=8 edges\n(showing 3 for clarity)',
           ha='center', fontsize=10, style='italic')

    return fig

def plot_filtered_search():
    """Plot filtered search process"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw 4x4 grid of cubes
    cube_size = 20
    margin = 2

    for i in range(4):
        for j in range(4):
            x = j * (cube_size + margin)
            y = i * (cube_size + margin)

            # Determine cube type
            if (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                # Overlapping cubes
                color = 'lightgreen'
                alpha = 0.7
                label = 'O'
            else:
                # Non-overlapping cubes
                color = 'lightgray'
                alpha = 0.3
                label = ''

            cube = patches.Rectangle((x, y), cube_size, cube_size,
                                    linewidth=1, edgecolor='black',
                                    facecolor=color, alpha=alpha)
            ax.add_patch(cube)

            if label:
                ax.text(x + cube_size/2, y + cube_size/2, label,
                       ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw query filter region
    filter_x, filter_y = 25, 25
    filter_w, filter_h = 40, 40
    filter_rect = patches.Rectangle((filter_x, filter_y), filter_w, filter_h,
                                   linewidth=3, edgecolor='red',
                                   facecolor='none', linestyle='--')
    ax.add_patch(filter_rect)
    ax.text(filter_x + filter_w/2, filter_y + filter_h + 5,
           'Query Filter\n[25,65]×[25,65]',
           ha='center', fontsize=11, fontweight='bold', color='red')

    # Draw query point
    query_x, query_y = 45, 45
    ax.plot(query_x, query_y, 'r*', markersize=20, label='Query Point')

    # Draw some result points in overlapping cubes
    np.random.seed(42)
    for _ in range(15):
        # Random points in overlapping region
        px = np.random.uniform(filter_x, filter_x + filter_w)
        py = np.random.uniform(filter_y, filter_y + filter_h)
        ax.plot(px, py, 'bo', markersize=4, alpha=0.6)

    # Draw search path (example)
    path_points = [(45, 45), (40, 50), (50, 40), (35, 55), (55, 35)]
    for i in range(len(path_points) - 1):
        ax.plot([path_points[i][0], path_points[i+1][0]],
               [path_points[i][1], path_points[i+1][1]],
               'g-', linewidth=2, alpha=0.5)

    ax.set_xlim(-5, 4*(cube_size + margin) + 5)
    ax.set_ylim(-5, 4*(cube_size + margin) + 5)
    ax.set_aspect('equal')
    ax.set_title('Filtered Search Process\nGreen cubes overlap with query filter',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='lightgreen', alpha=0.7, label='Overlapping cubes (searched)'),
        patches.Patch(facecolor='lightgray', alpha=0.3, label='Non-overlapping cubes (skipped)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=15, label='Query point'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Result points'),
        plt.Line2D([0], [0], color='g', linewidth=2, label='Search path')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    return fig

def plot_edge_comparison():
    """Plot edge count comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2D comparison
    methods = ['HNSW\nStatic', 'HNSW\nCube']
    intra_edges = [32, 32]
    cross_edges = [0, 32]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x, intra_edges, width, label='Intra-cube edges', color='steelblue')
    bars2 = ax1.bar(x, cross_edges, width, bottom=intra_edges, label='Cross-cube edges', color='coral')

    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('Edge Count Comparison (2D)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (intra, cross) in enumerate(zip(intra_edges, cross_edges)):
        total = intra + cross
        ax1.text(i, total + 2, str(total), ha='center', fontsize=11, fontweight='bold')

    # 3D comparison
    intra_edges_3d = [32, 32]
    cross_edges_3d = [0, 48]

    bars1 = ax2.bar(x, intra_edges_3d, width, label='Intra-cube edges', color='steelblue')
    bars2 = ax2.bar(x, cross_edges_3d, width, bottom=intra_edges_3d, label='Cross-cube edges', color='coral')

    ax2.set_ylabel('Number of Edges', fontsize=12)
    ax2.set_title('Edge Count Comparison (3D)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (intra, cross) in enumerate(zip(intra_edges_3d, cross_edges_3d)):
        total = intra + cross
        ax2.text(i, total + 2, str(total), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig

def main():
    output_dir = Path('visualizations/cube_index')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating cube index visualizations...")

    # Generate plots
    print("  1. Hierarchical cube structure...")
    fig1 = plot_cube_hierarchy()
    fig1.savefig(output_dir / 'cube_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    print("  2. Cross-cube edge connectivity...")
    fig2 = plot_cross_cube_edges()
    fig2.savefig(output_dir / 'cross_cube_edges.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print("  3. Filtered search process...")
    fig3 = plot_filtered_search()
    fig3.savefig(output_dir / 'filtered_search.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)

    print("  4. Edge count comparison...")
    fig4 = plot_edge_comparison()
    fig4.savefig(output_dir / 'edge_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)

    print(f"\nVisualizations saved to: {output_dir}")
    print("Files created:")
    print("  - cube_hierarchy.png")
    print("  - cross_cube_edges.png")
    print("  - filtered_search.png")
    print("  - edge_comparison.png")

if __name__ == '__main__':
    main()
