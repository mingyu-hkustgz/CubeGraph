#!/usr/bin/env python3
"""
Generate TikZ figure from extracted data files.
"""

from pathlib import Path

# Configuration
OUTPUT_FILE = Path("figure/Exp-1-Recall-Qps.tex")
DATASETS = ["sift", "msmarc10m", "deep100m", "yfcc"]
RATIOS = [0.01, 0.05, 0.10, 0.20]
METHODS = ["CG", "ACORN", "POST"]

# Dataset display names
DATASET_NAMES = {
    "sift": "SIFT",
    "msmarc10m": "MSMARC10M",
    "deep100m": "Deep100M",
    "yfcc": "YFCC"
}

# # Subplot positions: (row, col) in 4x4 grid
SUBPLOT_LAYOUT = [
    [("sift", 0.01), ("sift", 0.02), ("sift", 0.05), ("sift", 0.10)],
    [("msmarc10m", 0.01), ("msmarc10m", 0.02), ("msmarc10m", 0.05), ("msmarc10m", 0.10)],
    [("yfcc", 0.01), ("yfcc", 0.02), ("yfcc", 0.05), ("yfcc", 0.10)],
]

# Subplot positions: (row, col) in 4x4 grid
# SUBPLOT_LAYOUT = [
#     [("sift", 0.01), ("sift", 0.05), ("sift", 0.10)],
#     [("msmarc10m", 0.01), ("msmarc10m", 0.05), ("msmarc10m", 0.10)],
#     [("deep100m", 0.01), ("deep100m", 0.05), ("deep100m", 0.10)],
#     [("yfcc", 0.01), ("yfcc", 0.05), ("yfcc", 0.10)],
# ]
def load_data(dataset, ratio, method):
    """Load (recall, qps) data from .dat file."""
    ratio_str = f"{ratio:.2f}".replace('.', '_')
    filepath = Path("figure") / dataset / f"data_{method}_{ratio_str}.dat"
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
    lines.append(r"    \begin{customlegend}[legend columns=3,")
    lines.append(r"        legend entries={$\CG$-Cube,$\ACORN$-$\gamma$,$\POST$},")
    lines.append(r"        legend style={at={(0.5,1.15)},anchor=north,draw=none,font=\scriptsize,column sep=0.4cm}]")
    lines.append(r"    \addlegendimage{line width=0.15mm,color=navy,mark=triangle,mark size=0.5mm}")
    lines.append(r"    \addlegendimage{line width=0.15mm,color=forestgreen,mark=diamond,mark size=0.5mm}")
    lines.append(r"    \addlegendimage{line width=0.15mm,color=amaranth,mark=*,mark size=0.5mm}")
    lines.append(r"    \end{customlegend}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")
    lines.append(r"\vspace{0.2cm}")

    # Generate 16 subplots in 4 rows
    for row_idx, row in enumerate(SUBPLOT_LAYOUT):
        for col_idx, (dataset, ratio) in enumerate(row):
            subfig_idx = row_idx * 4 + col_idx + 1
            lines.append(f"% Subplot {subfig_idx}: {dataset} ratio {ratio}")
            lines.append(f"\\subfloat[{DATASET_NAMES[dataset]} 2D ratio {ratio:.2f}]{{\\vgap")
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
            lines.append(r"    title style={font=\scriptsizesize},")
            lines.append(r"    ymajorgrids=true,")
            lines.append(r"    xmajorgrids=true,")
            lines.append(r"    grid style=dashed,")
            lines.append(r"]")

            # Add CG data
            cg_data = load_data(dataset, ratio, "CG")
            if cg_data:
                coords = generate_coords(cg_data)
                lines.append(r"% CG (CubeGraph)")
                lines.append(r"\addplot[line width=0.15mm,color=navy,mark=triangle,mark size=0.5mm]")
                lines.append(r"plot coordinates {")
                lines.append(f"    {coords}")
                lines.append(r"};")

            # Add ACORN data (skip for deep100m)
            if dataset != "deep100m":
                acorn_data = load_data(dataset, ratio, "ACORN")
                if acorn_data:
                    coords = generate_coords(acorn_data)
                    lines.append(r"% ACORN")
                    lines.append(r"\addplot[line width=0.15mm,color=forestgreen,mark=diamond,mark size=0.5mm]")
                    lines.append(r"plot coordinates {")
                    lines.append(f"    {coords}")
                    lines.append(r"};")

            # Add POST data
            post_data = load_data(dataset, ratio, "POST")
            if post_data:
                coords = generate_coords(post_data)
                lines.append(r"% POST")
                lines.append(r"\addplot[line width=0.15mm,color=amaranth,mark=o,mark size=0.5mm]")
                lines.append(r"plot coordinates {")
                lines.append(f"    {coords}")
                lines.append(r"};")

            lines.append(r"\end{axis}")
            lines.append(r"\end{tikzpicture}")

            if col_idx < 3:
                lines.append(r"}\hspace{2mm}")
            else:
                lines.append(r"}")

        # Add vertical spacing between rows
        if row_idx < 3:
            lines.append("")
            lines.append(r"\vspace{0.3cm}")
            lines.append("")

    lines.append("")
    lines.append(r"\caption{Search efficiency comparison across different datasets and filter ratios (recall@20 vs.\ Qps).}")
    lines.append(r"\label{fig:exp1-all-datasets}\vspace{-3ex}")
    lines.append(r"\end{footnotesize}")
    lines.append(r"\end{figure*}")

    return "\n".join(lines)

if __name__ == "__main__":
    print("Generating TikZ figure...")
    content = generate_tikz()
    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)
    print(f"Generated {OUTPUT_FILE}")
