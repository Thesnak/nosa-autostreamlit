import streamlit as st
from nosa_autostreamlit.generators import DataVisualizationGenerator
import pandas as pd
def main():
    """
    Advanced Data Visualization Example
    """
    # Create a data visualization generator
    generator = DataVisualizationGenerator()
    
    # Add title
    generator.add_component('title', title='Advanced Data Visualization')
    
    # Load sample data (you can replace with your own dataset)
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi'],
        'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Salary': [50000, 60000, 75000, 90000, 100000, 110000, 120000, 130000],
        'Department': ['HR', 'Sales', 'Marketing', 'IT', 'Finance', 'HR', 'Sales', 'Marketing']
    }
    
    # Load the sample data
    generator.load_data(pd.DataFrame(sample_data))
    
    # Add visualizations
    generator.add_scatter_plot('Age', 'Salary', 'Age vs Salary')
    generator.add_bar_chart('Department', 'Salary', 'Average Salary by Department')
    generator.add_histogram('Age', 'Age Distribution')
    
    # Generate the app
    generator.generate_app()

if __name__ == '__main__':
    main()