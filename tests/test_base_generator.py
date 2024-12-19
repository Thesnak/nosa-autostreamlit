import pytest
import pandas as pd
import streamlit as st
from nosa_autostreamlit.generators import BaseGenerator

def test_base_generator_initialization():
    # Test initialization with and without config
    generator1 = BaseGenerator()
    generator2 = BaseGenerator({'theme': 'dark'})
    
    assert generator1.config == {}
    assert generator2.config == {'theme': 'dark'}
    assert generator1.app_components == []

def test_add_component():
    generator = BaseGenerator()
    
    # Add different types of components
    generator.add_component('title', title='Test App')
    generator.add_component('header', header='Test Section')
    
    sample_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    generator.add_component('dataframe', data=sample_df)
    
    assert len(generator.app_components) == 3
    assert generator.app_components[0]['type'] == 'title'
    assert generator.app_components[1]['type'] == 'header'
    assert generator.app_components[2]['type'] == 'dataframe'