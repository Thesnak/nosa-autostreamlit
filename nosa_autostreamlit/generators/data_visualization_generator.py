import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import Union, Dict, Any

from .base_generator import BaseGenerator
from ..utils import DataProcessor

class DataVisualizationGenerator(BaseGenerator):
    """
    Advanced generator for creating data visualization Streamlit apps
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Data Visualization Generator
        
        :param config: Configuration dictionary for customization
        """
        super().__init__(config)
        self.data = None
    
    def load_data(self, source: Union[str, pd.DataFrame], file_type: str = 'csv'):
        """
        Load and prepare data for visualization
        
        :param source: Data source (file path or DataFrame)
        :param file_type: Type of file to load
        :return: self for method chaining
        """
        self.data = DataProcessor.load_data(source, file_type)
        return self
    
    def add_scatter_plot(self, x_column: str, y_column: str, title: str = 'Scatter Plot'):
        """
        Create an interactive scatter plot
        
        :param x_column: Column for x-axis
        :param y_column: Column for y-axis
        :param title: Plot title
        :return: self for method chaining
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        fig = px.scatter(
            self.data, 
            x=x_column, 
            y=y_column, 
            title=title,
            labels={x_column: x_column, y_column: y_column}
        )
        
        self.add_component('plotly_chart', figure=fig)
        return self
    
    def add_bar_chart(self, category_column: str, value_column: str, title: str = 'Bar Chart'):
        """
        Create an interactive bar chart
        
        :param category_column: Column for categories
        :param value_column: Column for values
        :param title: Chart title
        :return: self for method chaining
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Aggregate data if needed
        aggregated_data = self.data.groupby(category_column)[value_column].sum().reset_index()
        
        fig = px.bar(
            aggregated_data, 
            x=category_column, 
            y=value_column, 
            title=title,
            labels={category_column: category_column, value_column: value_column}
        )
        
        self.add_component('plotly_chart', figure=fig)
        return self
    
    def add_histogram(self, column: str, title: str = 'Histogram'):
        """
        Create an interactive histogram
        
        :param column: Column to create histogram for
        :param title: Histogram title
        :return: self for method chaining
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        fig = px.histogram(
            self.data, 
            x=column, 
            title=title,
            labels={column: column}
        )
        
        self.add_component('plotly_chart', figure=fig)
        return self
    
    # Add to data_visualization_generator.py
    def add_box_plot(self, column: str, category_column: str = None, title: str = 'Box Plot'):
        """
        Create an interactive box plot
        
        :param column: Column to create box plot for
        :param category_column: Optional column to group by
        :param title: Plot title
        :return: self for method chaining
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if category_column:
            fig = px.box(
                self.data, 
                x=category_column, 
                y=column, 
                title=title,
                labels={column: column, category_column: category_column}
            )
        else:
            fig = px.box(
                self.data, 
                y=column, 
                title=title,
                labels={column: column}
            )
        
        self.add_component('plotly_chart', figure=fig)
        return self

    def add_violin_plot(self, column: str, category_column: str = None, title: str = 'Violin Plot'):
        """
        Create an interactive violin plot
        
        :param column: Column to create violin plot for
        :param category_column: Optional column to group by
        :param title: Plot title
        :return: self for method chaining
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if category_column:
            fig = px.violin(
                self.data, 
                x=category_column, 
                y=column, 
                title=title,
                labels={column: column, category_column: category_column}
            )
        else:
            fig = px.violin(
                self.data, 
                y=column, 
                title=title,
                labels={column: column}
            )
        
        self.add_component('plotly_chart', figure=fig)
        return self

    def add_heatmap(self, columns: list = None, title: str = 'Correlation Heatmap'):
        """
        Create a correlation heatmap
        
        :param columns: List of columns to include in heatmap (optional)
        :param title: Plot title
        :return: self for method chaining
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Select numeric columns
        if columns is None:
            columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Compute correlation matrix
        corr_matrix = self.data[columns].corr()
        
        fig = px.imshow(
            corr_matrix, 
            title=title,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=columns,
            y=columns,
            color_continuous_scale='RdBu_r'
        )
        
        self.add_component('plotly_chart', figure=fig)
        return self
        
    def generate_app(self):
            """
            Override base generate_app to handle Plotly charts
            """
            for component in self.app_components:
                if component['type'] == 'plotly_chart':
                    st.plotly_chart(component['params']['figure'], use_container_width=True)
                else:
                    self._render_component(component)