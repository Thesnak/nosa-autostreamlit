# Nosa-autoStreamlit 🚀

## 📊 Automated Machine Learning and Data Visualization Framework

Nosa-autoStreamlit is an advanced Python framework that automates the creation of machine learning and data visualization Streamlit applications. Designed to simplify complex data science workflows with powerful, user-friendly tools.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.10.0+-green.svg)
![License](https://img.shields.io/github/license/thesnak/nosa-autostreamlit)

## ✨ Features

### 🤖 Machine Learning Generator
- Supports classification and regression problems
- Advanced preprocessing techniques
- Multiple machine learning models
- Cross-validation
- Hyperparameter tuning
- Model saving and loading

### 📈 Data Visualization Generator
- Multiple visualization types
- Interactive Plotly plots
- Easy-to-use interface

## 🚀 Quick Start

## Installation

### Install via pip

```bash
pip install nosa-autostreamlit
```
### Install from GitHub
```bash
pip install git+https://github.com/thesnak/nosa-autostreamlit.git
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nosa-autostreamlit.git
cd nosa-autostreamlit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 🔗 Links
- PyPI Package: https://pypi.org/project/nosa-autostreamlit/
- GitHub Repository: https://github.com/thesnak/nosa-autostreamlit


### Machine Learning Example

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

### Data Visualization Example

```python
from nosa_autostreamlit.generators import DataVisualizationGenerator

# Create generator
generator = DataVisualizationGenerator()

# Load data
generator.load_data(data)

# Create visualizations
generator.create_histogram()
generator.create_scatterplot()
generator.create_boxplot()
```

### 🛠 Key Components
- **`machine_learning_generator.py`**: Core ML functionality
- **`data_visualization_generator.py`**: Visualization tools
- **`advanced_ml_comparison.py`**: Example ML workflow
- **`advanced_data_viz_example.py`**: Example visualization workflow


### 📦 Dependencies
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

### 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

### 📄 License
Distributed under the MIT License. See LICENSE for more information.

### 📞 Contact
Your Name - mohamed.mahmoud0726@gmail.com

Project Link: https://github.com/thesnak/nosa-autostreamlit


Made with ❤️ by [Mohamed Mahmoud](https://github.com/thesnak)