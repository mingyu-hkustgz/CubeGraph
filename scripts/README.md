# Scripts for Data Synthesis and Visualization

This directory contains Python scripts for generating synthetic metadata and visualizing results for filtered vector search experiments.

## Scripts Overview

### 1. generate_metadata.py
Generates synthetic metadata with different distributions:
- **Uniform**: Uniformly distributed values in [0, 100]
- **Normal**: Gaussian distribution with mean=50, std=15
- **Clustered**: Mixture of Gaussians (5 clusters by default)
- **Skewed**: Exponential-like distribution

### 2. visualize_metadata.py
Visualizes metadata distributions:
- 2D/3D scatter plots
- Density heatmaps
- Histograms for each dimension
- Statistical summaries
- Correlation matrices
- Comparison plots

### 3. visualize_results.py
Visualizes query results and performance:
- Query regions and result distributions
- Performance metrics (latency, throughput, recall)
- Filter selectivity analysis
- Distance distributions

## Installation

Install required Python packages:

```bash
pip install numpy matplotlib seaborn
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

## Usage Examples

### Generate Metadata

#### Generate all distributions for SIFT dataset (2D attributes)
```bash
python scripts/generate_metadata.py \
    --data ./DATA/sift/sift_base.fvecs \
    --output-dir ./DATA/sift \
    --attr-dim 2 \
    --distributions all
```

#### Generate only uniform and normal distributions (3D attributes)
```bash
python scripts/generate_metadata.py \
    --data ./DATA/gist/gist_base.fvecs \
    --output-dir ./DATA/gist \
    --attr-dim 3 \
    --distributions uniform normal
```

#### Generate with custom seed
```bash
python scripts/generate_metadata.py \
    --data ./DATA/sift/sift_base.fvecs \
    --output-dir ./DATA/sift \
    --attr-dim 2 \
    --distributions clustered \
    --seed 123
```

### Visualize Metadata

#### Visualize single metadata file
```bash
python scripts/visualize_metadata.py \
    --metadata ./DATA/sift/sift_metadata_uniform_2d.bin \
    --output-dir ./visualizations/sift_uniform
```

#### Visualize and compare multiple distributions
```bash
python scripts/visualize_metadata.py \
    --metadata \
        ./DATA/sift/sift_metadata_uniform_2d.bin \
        ./DATA/sift/sift_metadata_normal_2d.bin \
        ./DATA/sift/sift_metadata_clustered_2d.bin \
    --output-dir ./visualizations/sift_comparison \
    --compare
```

### Visualize Query Results

#### Visualize filter selectivity
```bash
python scripts/visualize_results.py \
    --metadata ./DATA/sift/sift_metadata_uniform_2d.bin \
    --output-dir ./visualizations/sift_selectivity
```

#### Visualize radius query results
```bash
python scripts/visualize_results.py \
    --metadata ./DATA/sift/sift_metadata_uniform_2d.bin \
    --output-dir ./visualizations/sift_query \
    --query-center 50.0 50.0 \
    --query-radius 20.0 \
    --result-ids 10 25 37 42 58 91 103 ...
```

#### Visualize range query results
```bash
python scripts/visualize_results.py \
    --metadata ./DATA/sift/sift_metadata_uniform_2d.bin \
    --output-dir ./visualizations/sift_query \
    --bbox-min 20.0 30.0 \
    --bbox-max 40.0 50.0 \
    --result-ids 10 25 37 42 58 91 103 ...
```

#### Visualize performance metrics from JSON
```bash
python scripts/visualize_results.py \
    --metadata ./DATA/sift/sift_metadata_uniform_2d.bin \
    --results ./results/performance.json \
    --output-dir ./visualizations/performance
```

## Output Files

### generate_metadata.py
Generates binary metadata files:
- `{dataset}_metadata_uniform_{dim}d.bin`
- `{dataset}_metadata_normal_{dim}d.bin`
- `{dataset}_metadata_clustered_{dim}d.bin`
- `{dataset}_metadata_skewed_{dim}d.bin`

### visualize_metadata.py
Generates visualization images:
- `{name}_scatter_2d.png` - 2D scatter plot
- `{name}_density_2d.png` - 2D density heatmap
- `{name}_scatter_3d.png` - 3D scatter plot (if dim >= 3)
- `{name}_histograms.png` - Histograms for each dimension
- `{name}_statistics.png` - Statistical summary
- `comparison.png` - Comparison of multiple distributions

### visualize_results.py
Generates analysis images:
- `filter_selectivity.png` - Filter selectivity analysis
- `query_radius_results.png` - Radius query visualization
- `query_range_results.png` - Range query visualization
- `performance_metrics.png` - Performance comparison

## Metadata File Format

Binary format:
```
[n: uint64]           // Number of vectors
[d: uint64]           // Attribute dimension
[vector_0: float32[d]] // Attribute values for vector 0
[vector_1: float32[d]] // Attribute values for vector 1
...
[vector_n-1: float32[d]]
```

## Performance Results JSON Format

```json
{
    "method_name": {
        "latency": 123.45,      // Query latency in microseconds
        "num_results": 10,      // Number of results returned
        "recall": 95.5          // Recall percentage (optional)
    },
    ...
}
```

## Complete Workflow Example

```bash
# 1. Generate metadata for SIFT dataset
python scripts/generate_metadata.py \
    --data ./DATA/sift/sift_base.fvecs \
    --output-dir ./DATA/sift \
    --attr-dim 2 \
    --distributions all

# 2. Visualize all generated metadata
python scripts/visualize_metadata.py \
    --metadata \
        ./DATA/sift/sift_metadata_uniform_2d.bin \
        ./DATA/sift/sift_metadata_normal_2d.bin \
        ./DATA/sift/sift_metadata_clustered_2d.bin \
        ./DATA/sift/sift_metadata_skewed_2d.bin \
    --output-dir ./visualizations/sift_all \
    --compare

# 3. Build filtered index (using your C++ implementation)
./build/src/test_grid_index \
    ./DATA/sift/sift_base.fvecs \
    ./DATA/sift/sift_metadata_uniform_2d.bin \
    ./DATA/sift/sift_index_grid.bin \
    2

# 4. Run queries and collect results
# (Your C++ query code here)

# 5. Visualize query results
python scripts/visualize_results.py \
    --metadata ./DATA/sift/sift_metadata_uniform_2d.bin \
    --results ./results/performance.json \
    --output-dir ./visualizations/sift_results
```

## Tips

1. **Memory Usage**: For large datasets (>1M vectors), consider generating metadata in chunks or using lower precision.

2. **Visualization Performance**: For very large datasets, the scatter plots may be slow. Consider:
   - Sampling a subset of points for visualization
   - Using hexbin plots instead of scatter plots
   - Reducing the DPI for faster rendering

3. **Distribution Selection**:
   - **Uniform**: Good baseline for testing
   - **Normal**: Realistic for many real-world scenarios
   - **Clustered**: Tests performance with data locality
   - **Skewed**: Tests performance with imbalanced data

4. **Attribute Dimensions**:
   - 2D: Easy to visualize, good for initial testing
   - 3D: Still visualizable, more realistic
   - Higher dimensions: More realistic but harder to visualize

## Troubleshooting

### Import Error: No module named 'matplotlib'
```bash
pip install matplotlib seaborn numpy
```

### Memory Error when loading large datasets
Reduce the number of vectors or use sampling:
```python
# In the script, add sampling
metadata = metadata[::10]  # Sample every 10th vector
```

### Plots are too slow
Reduce DPI or sample data:
```bash
# Edit the script to change dpi=300 to dpi=150
# Or sample the data before plotting
```

## Advanced Usage

### Custom Distribution
Edit `generate_metadata.py` to add your own distribution:

```python
def generate_custom_metadata(n, attr_dim, seed=42):
    np.random.seed(seed)
    # Your custom distribution logic here
    metadata = ...
    return metadata
```

### Custom Visualization
Edit `visualize_metadata.py` to add custom plots:

```python
def plot_custom(metadata, title, output_path):
    # Your custom visualization logic here
    plt.figure(figsize=(10, 8))
    # ... plotting code ...
    plt.savefig(output_path, dpi=300)
    plt.close()
```

## References

- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/
- Seaborn documentation: https://seaborn.pydata.org/
