import streamlit as st
import pandas as pd
from nosa_autostreamlit.generators import MachineLearningGenerator

def main():
    """
    Machine Learning Model Exploration Example
    """
    # Create a machine learning generator
    generator = MachineLearningGenerator()
    
    # Add title
    generator.add_component('title', title='Machine Learning Model Explorer')
    
    # Create sample classification dataset
    sample_data = pd.DataFrame({
        'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'target': [0, 0, 0, 0, 1, 1, 1, 1]
    })
    
    # Load data and specify target and features
    generator.load_data(sample_data, target_column='target', feature_columns=['feature1', 'feature2'])
    
    # Train logistic regression model
    generator.train_logistic_regression()
    
    # Generate and display model report
    generator.generate_model_report()

if __name__ == '__main__':
    main()