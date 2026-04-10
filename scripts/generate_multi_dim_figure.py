#!/usr/bin/env python3
"""
Generate Exp-2-Multi-Dim.tex - TikZ figure for multi-dimensional (3D/4D) experiments.
"""

from pathlib import Path

# Configuration
OUTPUT_FILE = Path("figure/Exp-2-Multi-Dim.tex")
DATASETS = ["sift", "msmarc10m", "yfcc"]
DIMENSIONS = ["3d", "4d"]
RATIOS = [0.02, 0.05]
METHODS = ["CG", "POST"]

# Dataset display names
DATASET_NAMES = {
    "sift": "SIFT",
    "msmarc10m": "MSMARC10M",
    "yfcc": "YFCC"
}

# Layout: 3 datasets × 2 dimensions = 6 rows, 2 ratios = 2 columns
SUBPLOT_LAYOUT = []
for dataset in DATASETS:
    for dim in DIMENSIONS:
        row = []
        for ratio in RATIOS:
            row.append((dataset, dim, ratio))
        SUBPLOT_LAYOUT.append(row)

def load_data(dataset, dim, ratio, method):
    """Load (recall, qps) data from .dat file."""
    ratio_str = f"{ratio:.2f}".replace('.', '_')
    filepath = Path("figure") / dataset / f"data_{method}_{dim}_{ratio_str}.dat"
    if not filepath.exists():
        return []

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    recall = float(parts[0])
                    qps = float(parts[1])
                    data.append((recall, qps))
                except ValueError:
                    continue
    return data

def generate_coords(data):
    """Generate TikZ coordinate string."""
    if not data:
        return ""
    coords = []
    for recall, qps in data:
        coords.append(f"({recall:.2f}, {qps:.2f})")
    return "\n    ".join(coords)

def generate_tikz():
    """Generate the full TikZ figure."""

    lines = []
    lines.append(r"\begin{figure*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\begin{footnotesize}")

    # Legend
    lines.append(r"% Top legend")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"    \begin{customlegend}[legend columns=2,")
    lines.append(r"        legend entries={$\CG$-Cube,$\POST$},")
    lines.append(r"        legend style={at={(0.5,1.15)},anchor=north,draw=none,font=\scriptsize,column sep=0.4cm}]")
    lines.append(r"    \addlegendimage{line width=0.15mm,color=navy,mark=triangle,mark size=0.5mm}")
    lines.append(r"    \addlegendimage{line width=0.15mm,color=amaranth,mark=o,mark size=0.5mm}")
    lines.append(r"    \end{customlegend}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")
    lines.append(r"\vspace{0.2cm}")

    num_rows = len(SUBPLOT_LAYOUT)
    num_cols = len(SUBPLOT_LAYOUT[0]) if SUBPLOT_LAYOUT else 0

    # Generate subplots
    for row_idx, row in enumerate(SUBPLOT_LAYOUT):
        for col_idx, (dataset, dim, ratio) in enumerate(row):
            subfig_idx = row_idx * num_cols + col_idx + 1

            # Check if data exists
            has_data = any(load_data(dataset, dim, ratio, m) for m in METHODS)
            if not has_data:
                # Skip empty subplots
                continue

            lines.append(f"% Subplot {subfig_idx}: {dataset} {dim} ratio {ratio}")
            lines.append(f"\\subfloat[{DATASET_NAMES[dataset]} {dim.upper()} ratio {ratio:.2f}]{{\\vgap")
            lines.append(r"\begin{tikzpicture}[scale=0.85]")
            lines.append(r"\begin{axis}[")
            lines.append(r"    height=\columnwidth/2.60,")
            lines.append(r"    width=\columnwidth/1.80,")
            lines.append(r"    xlabel=recall@20(\%),")
            lines.append(r"    ylabel=Qps,")
            lines.append(r"    ymode=log,")
            lines.append(r"    xmin=85,")
            lines.append(r"    xmax=100.2,")
            lines.append(r"    label style={font=\scriptsize},")
            lines.append(r"    tick label style={font=\scriptsize},")
            lines.append(r"    title style={font=\scriptsize},")
            lines.append(r"    ymajorgrids=true,")
            lines.append(r"    xmajorgrids=true,")
            lines.append(r"    grid style=dashed,")
            lines.append(r"]")

            # Add CG data
            cg_data = load_data(dataset, dim, ratio, "CG")
            if cg_data:
                coords = generate_coords(cg_data)
                lines.append(r"% CG (CubeGraph)")
                lines.append(r"\addplot[line width=0.15mm,color=navy,mark=triangle,mark size=0.5mm]")
                lines.append(r"plot coordinates {")
                lines.append(f"    {coords}")
                lines.append(r"};")

            # Add POST data
            post_data = load_data(dataset, dim, ratio, "POST")
            if post_data:
                coords = generate_coords(post_data)
                lines.append(r"% POST")
                lines.append(r"\addplot[line width=0.15mm,color=amaranth,mark=o,mark size=0.5mm]")
                lines.append(r"plot coordinates {")
                lines.append(f"    {coords}")
                lines.append(r"};")

            lines.append(r"\end{axis}")
            lines.append(r"\end{tikzpicture}")

            if col_idx < num_cols - 1:
                lines.append(r"}\hspace{2mm}")
            else:
                lines.append(r"}")

        # Add vertical spacing between rows
        if row_idx < num_rows - 1:
            lines.append("")
            lines.append(r"\vspace{0.3cm}")
            lines.append("")

    lines.append("")
    lines.append(r"\caption{Search efficiency comparison across different dimensions (3D/4D) and filter ratios (recall@20 vs.\ Qps).}")
    lines.append(r"\label{fig:exp2-multi-dim}\vspace{-3ex}")
    lines.append(r"\end{footnotesize}")
    lines.append(r"\end{figure*}")

    return "\n".join(lines)

if __name__ == "__main__":
    print("Generating Exp-2-Multi-Dim.tex...")
    content = generate_tikz()
    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)
    print(f"Generated {OUTPUT_FILE}")
