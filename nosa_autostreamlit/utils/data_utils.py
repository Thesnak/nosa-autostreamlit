import pandas as pd
import streamlit as st
from typing import Union, List, Dict

class DataProcessor:
    """
    Utility class for data processing and manipulation
    """
    @staticmethod
    def load_data(source: Union[str, pd.DataFrame], 
                  file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from various sources
        
        :param source: File path or DataFrame
        :param file_type: Type of file to load (csv, excel, etc.)
        :return: Processed DataFrame
        """
        if isinstance(source, pd.DataFrame):
            return source
        
        if file_type == 'csv':
            return pd.read_csv(source)
        elif file_type == 'excel':
            return pd.read_excel(source)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    @staticmethod
    def display_data_summary(df: pd.DataFrame):
        """
        Display summary statistics of a DataFrame in Streamlit
        
        :param df: Input DataFrame
        """
        st.write("Data Summary")
        st.write(df.describe())
        
        st.write("Columns")
        st.write(df.columns.tolist())