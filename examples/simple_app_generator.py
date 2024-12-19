import streamlit as st
import pandas as pd
from nosa_autostreamlit.generators import BaseGenerator
from nosa_autostreamlit.utils import DataProcessor

def main():
    """
    Example Streamlit app generator
    """
    # Create a base generator
    generator = BaseGenerator()
    
    # Add a title component
    generator.add_component('title', title='Simple Data Visualization App')
    
    # Add a header component
    generator.add_component('header', header='Data Summary')
    
    # Create a sample dataset
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 45],
        'Salary': [50000, 60000, 75000, 90000, 100000]
    })
    
    # Add dataframe component
    generator.add_component('dataframe', data=sample_data)
    
    # Display data summary
    DataProcessor.display_data_summary(sample_data)

if __name__ == '__main__':
    main()