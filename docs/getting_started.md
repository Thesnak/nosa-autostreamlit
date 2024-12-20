# Getting Started

## Machine Learning Example

```python
from nosa_autostreamlit.generators import AdvancedMachineLearningGenerator

# Create generator
generator = AdvancedMachineLearningGenerator()

# Load data
generator.load_data(
    data, 
    target_column='target', 
    problem_type='classification'
)

# Preprocess and train models
generator.advanced_preprocessing()
generator.train_multiple_models()
generator.generate_model_comparison_report()
```

## Data Visualization Example

```python
from nosa_autostreamlit.generators import DataVisualizationGenerator

# Create visualization generator
generator = DataVisualizationGenerator()
generator.load_data(data)
generator.create_histogram()
generator.create_scatterplot()
```

## Key Concepts
- Generators: Core components for ML and Visualization
- Preprocessing: Automated data preparation
- Model Training: Multiple algorithm support
- Visualization: Interactive data insights
