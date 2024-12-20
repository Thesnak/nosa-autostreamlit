# Data Visualization Generator

## Overview
The Data Visualization Generator creates interactive and insightful visualizations.

## Supported Visualizations
- Histograms
- Scatter Plots
- Box Plots
- Correlation Heatmaps

## Example Usage

```python
from nosa_autostreamlit.generators import DataVisualizationGenerator

# Create generator
generator = DataVisualizationGenerator()
generator.load_data(data)

# Create visualizations
generator.create_histogram(column='age')
generator.create_scatterplot(x_column='income', y_column='expenses')


## Visualization Methods
- create_histogram()
- create_scatterplot()
- create_boxplot()
- create_correlation_heatmap()
