import streamlit as st
import pandas as pd
import numpy as np
from nosa_autostreamlit.generators import AdvancedMachineLearningGenerator
import os
def generate_synthetic_loan_data(n_samples=1000):
    """
    Generate synthetic loan default dataset
    """
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    credit_score = np.random.normal(700, 50, n_samples)
    
    # Create categorical features
    education_levels = ['High School', 'Bachelors', 'Masters', 'PhD']
    education = np.random.choice(education_levels, n_samples)
    
    employment_status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples)
    
    # Define loan default based on features
    loan_default = (
        (age < 25) | 
        (income < 30000) | 
        (credit_score < 600) | 
        (np.random.random(n_samples) < 0.1)
    ).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'education': education,
        'employment_status': employment_status,
        'loan_default': loan_default
    })
    
    return df

def main():
    """
    Advanced Machine Learning Model Comparison Example
    """
    # Create an advanced machine learning generator
    generator = AdvancedMachineLearningGenerator()
    
    # Add title
    generator.add_component('title', title='Loan Default Prediction Model Comparison')
    
    # Generate synthetic loan data
    sample_data = generate_synthetic_loan_data()
    
    try:
        # Load data and specify target and features
        generator.load_data(
            sample_data, 
            target_column='loan_default', 
            feature_columns=['age', 'income', 'credit_score'],
            categorical_columns=['education', 'employment_status'],
            problem_type='classification'
        )
        
        # Perform advanced preprocessing
        generator.advanced_preprocessing(
            scaling_method='robust', 
            feature_selection=True
        )
        
        # Train multiple models
        generator.train_multiple_models()
        
        # Generate and display model comparison report
        generator.generate_model_comparison_report()
        
        # Save trained models
        st.subheader("Model Saving")
        saved_models = generator.save_models()
        st.write("Saved Models:", saved_models)
        
        # Demonstrate model loading (optional)
        st.subheader("Model Loading Demo")
        loaded_models = generator.load_saved_models()
        st.write("Loaded Models:", list(loaded_models.keys()))
        
        # Cross-validate models
        cv_scores = generator.cross_validate_models()
        st.subheader("Cross-Validation Scores")
        st.write(cv_scores)
        
        # Hyperparameter tuning for a specific model
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
        tuning_results = generator.hyperparameter_tuning('Logistic Regression', param_grid)
        st.subheader("Hyperparameter Tuning Results")
        st.write(tuning_results)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
