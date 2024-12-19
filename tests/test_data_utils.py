import pytest
import pandas as pd
import io
from nosa_autostreamlit.utils import DataProcessor

def test_load_data_csv():
    # Create a sample CSV in memory
    csv_data = io.StringIO("""Name,Age,Salary
Alice,25,50000
Bob,30,60000""")
    
    # Load data
    df = DataProcessor.load_data(csv_data, file_type='csv')
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['Name', 'Age', 'Salary']

def test_load_data_dataframe():
    # Create a sample DataFrame
    sample_df = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Age': [25, 30]
    })
    
    # Load data
    df = DataProcessor.load_data(sample_df)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.equals(sample_df)

def test_load_data_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        DataProcessor.load_data('sample.txt', file_type='txt')