# Machine Learning Generator

## Overview
The Machine Learning Generator provides advanced capabilities for automated model training and evaluation.

## Key Features
- Supports classification and regression
- Advanced preprocessing
- Multiple model training
- Cross-validation
- Hyperparameter tuning

## Methods

### `load_data()`
Load and prepare data for machine learning.

**Parameters:**
- `source`: Data source (DataFrame or file path)
- `target_column`: Name of the target variable
- `problem_type`: 'classification' or 'regression'

### `advanced_preprocessing()`
Perform advanced data preprocessing.

**Parameters:**
- `scaling_method`: 'robust', 'quantile', or 'standard'
- `feature_selection`: Boolean to enable feature selection

### `train_multiple_models()`
Train multiple machine learning models.

### `generate_model_comparison_report()`
Create a comprehensive model performance report.
