import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

class BaseGenerator:
    """
    Base class for generating Streamlit applications
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base generator with optional configuration
        
        :param config: Dictionary of configuration parameters
        """
        self.config = config or {}
        self.app_components = []
    
    def add_component(self, component_type: str, **kwargs):
        """
        Add a component to the Streamlit application
        
        :param component_type: Type of Streamlit component to add
        :param kwargs: Configuration parameters for the component
        """
        component = {
            'type': component_type,
            'params': kwargs
        }
        self.app_components.append(component)
    
    def generate_app(self):
        """
        Generate the Streamlit application based on added components
        """
        for component in self.app_components:
            self._render_component(component)
    
    def _render_component(self, component: Dict[str, Any]):
        """
        Render a specific Streamlit component
        
        :param component: Component dictionary with type and parameters
        """
        component_type = component['type']
        params = component['params']
        
        if component_type == 'title':
            st.title(params.get('title', 'Nosa-autoStreamlit App'))
        elif component_type == 'header':
            st.header(params.get('header', 'Section Header'))
        elif component_type == 'dataframe':
            data = params.get('data')
            if isinstance(data, pd.DataFrame):
                st.dataframe(data)
        # Add more component rendering logic as needed