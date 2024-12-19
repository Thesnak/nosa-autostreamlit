import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    mean_squared_error, r2_score, 
    mean_absolute_error
)

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

from sklearn.model_selection import (
    cross_val_score, 
    GridSearchCV, 
    StratifiedKFold, 
    KFold
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import RobustScaler, QuantileTransformer


from .base_generator import BaseGenerator
from ..utils import DataProcessor

class AdvancedMachineLearningGenerator(BaseGenerator):
    """
    Advanced generator for machine learning model exploration and comparison
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
    
    # Update the load_data method to initialize preprocessor
    def load_data(self, source, target_column, 
                feature_columns=None, 
                problem_type='classification',
                categorical_columns=None):
        """
        Load and prepare data for machine learning with advanced preprocessing
        
        :param source: Data source (file path or DataFrame)
        :param target_column: Name of the target variable column
        :param feature_columns: List of feature column names
        :param problem_type: 'classification' or 'regression'
        :param categorical_columns: List of categorical column names
        :return: self for method chaining
        """
        # Load data
        self.data = DataProcessor.load_data(source)
        self.problem_type = problem_type
        
        # Prepare features and target
        if feature_columns is None:
            feature_columns = [col for col in self.data.columns if col != target_column]
        
        self.X = self.data[feature_columns]
        self.y = self.data[target_column]
        
        # Preprocessing
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = categorical_columns or []
        
        # Create default preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        return self    
    def train_multiple_models(self):
        """
        Train multiple models based on problem type
        """
        # Ensure data is loaded and preprocessed
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded first")
        
        # Ensure preprocessor is defined
        if not hasattr(self, 'preprocessor'):
            self.advanced_preprocessing()
        
        if self.problem_type == 'classification':
            classification_models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(random_state=42)
            }
            
            for name, model in classification_models.items():
                try:
                    # Create pipeline with preprocessor and model
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('classifier', model)
                    ])
                    
                    # Fit the pipeline
                    pipeline.fit(self.X_train, self.y_train)
                    
                    # Predict
                    y_pred = pipeline.predict(self.X_test)
                    
                    # Store results
                    self.models[name] = pipeline
                    self.model_results[name] = {
                        'report': classification_report(self.y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                    }
                except Exception as e:
                    print(f"Error training {name} model: {e}")
                    continue
        
        elif self.problem_type == 'regression':
            regression_models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42)
            }
            
            for name, model in regression_models.items():
                try:
                    # Create pipeline with preprocessor and model
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('regressor', model)
                    ])
                    
                    # Fit the pipeline
                    pipeline.fit(self.X_train, self.y_train)
                    
                    # Predict
                    y_pred = pipeline.predict(self.X_test)
                    
                    # Store results
                    self.models[name] = pipeline
                    self.model_results[name] = {
                        'mse': mean_squared_error(self.y_test, y_pred),
                        'mae': mean_absolute_error(self.y_test, y_pred),
                        'r2': r2_score(self.y_test, y_pred)
                    }
                except Exception as e:
                    print(f"Error training {name} model: {e}")
                    continue
        
        return self
        
    def generate_model_comparison_report(self):
        """
        Generate and visualize model comparison report
        """
        if self.problem_type == 'classification':
            # Create comparison DataFrame
            comparison_data = []
            for name, results in self.model_results.items():
                try:
                    comparison_data.append({
                        'Model': name,
                        'Accuracy': results['report'].get('accuracy', 0),
                        'Precision': results['report']['weighted avg'].get('precision', 0),
                        'Recall': results['report']['weighted avg'].get('recall', 0),
                        'F1-Score': results['report']['weighted avg'].get('f1-score', 0)
                    })
                except (KeyError, TypeError) as e:
                    print(f"Error processing model {name}: {e}")
                    continue
            
            # Convert to DataFrame safely
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Visualize model performance
                try:
                    fig = px.bar(
                        comparison_df, 
                        x='Model', 
                        y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        title='Model Performance Comparison'
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
            else:
                st.warning("No model results available for comparison")
        
        elif self.problem_type == 'regression':
            # Create comparison DataFrame
            comparison_data = []
            for name, results in self.model_results.items():
                try:
                    comparison_data.append({
                        'Model': name,
                        'MSE': results.get('mse', 0),
                        'MAE': results.get('mae', 0),
                        'R2': results.get('r2', 0)
                    })
                except (KeyError, TypeError) as e:
                    print(f"Error processing model {name}: {e}")
                    continue
            
            # Convert to DataFrame safely
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Visualize model performance
                try:
                    fig = px.bar(
                        comparison_df, 
                        x='Model', 
                        y=['MSE', 'MAE', 'R2'],
                        title='Regression Model Performance Comparison'
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
            else:
                st.warning("No model results available for comparison")
        return self    
    
    def advanced_preprocessing(self, scaling_method='robust', feature_selection=True):
        """
        Advanced preprocessing with multiple scaling and feature selection options
        
        :param scaling_method: 'robust', 'quantile', or 'standard'
        :param feature_selection: Whether to perform feature selection
        :return: self for method chaining
        """
        # Ensure basic data is loaded
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded first")

        # Identify numeric and categorical features dynamically
        # Use .columns.tolist() to ensure we have a list
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Prepare transformers for each feature type
        transformers = []

        # Numeric feature transformer
        if numeric_features:
            if scaling_method == 'robust':
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ])
            elif scaling_method == 'quantile':
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', QuantileTransformer(n_quantiles=100, output_distribution='normal'))
                ])
            else:  # standard
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
            
            transformers.append(('num', numeric_transformer, numeric_features))

        # Categorical feature transformer
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        # Feature selection
        if feature_selection and numeric_features:
            if self.problem_type == 'classification':
                feature_selector = SelectKBest(score_func=f_classif, k='all')
            else:
                feature_selector = SelectKBest(score_func=f_regression, k='all')
            
            transformers.append(('selector', feature_selector, numeric_features))

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Keep other columns as-is
        )
        
        return self
    def cross_validate_models(self, cv_method='stratified', n_splits=5):
        """
        Perform cross-validation on trained models
        
        :param cv_method: 'stratified' for classification, 'standard' for regression
        :param n_splits: Number of cross-validation splits
        :return: Dictionary of cross-validation scores
        """
        cv_scores = {}
        
        # Choose appropriate cross-validation strategy
        if cv_method == 'stratified' and self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Perform cross-validation for each model
        for name, model in self.models.items():
            if self.problem_type == 'classification':
                scores = cross_val_score(
                    model, 
                    self.X, 
                    self.y, 
                    cv=cv, 
                    scoring='accuracy'
                )
            else:
                scores = cross_val_score(
                    model, 
                    self.X, 
                    self.y, 
                    cv=cv, 
                    scoring='r2'
                )
            
            cv_scores[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
        
        return cv_scores

    def hyperparameter_tuning(self, model_name, param_grid):
        """
        Perform hyperparameter tuning for a specific model
        
        :param model_name: Name of the model to tune
        :param param_grid: Dictionary of hyperparameters to search
        :return: Best model and best parameters
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract base model from pipeline
        base_model = self.models[model_name].named_steps['classifier' if self.problem_type == 'classification' else 'regressor']
        
        # Scoring metric based on problem type
        scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
        
        # Create grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring=scoring, 
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        best_model = grid_search.best_estimator_
        
        # Replace model in pipeline
        if self.problem_type == 'classification':
            self.models[model_name].named_steps['classifier'] = best_model
        else:
            self.models[model_name].named_steps['regressor'] = best_model
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

    def save_models(self, directory=None):
        """
        Save trained models to a specified directory
        
        :param directory: Directory to save models. If None, uses a default Streamlit directory
        :return: Dictionary of saved model paths
        """
        # Create default directory if not specified
        if directory is None:
            directory = os.path.join(os.getcwd(), 'saved_models')
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Dictionary to store saved model paths
        saved_model_paths = {}
        
        # Save each trained model
        for name, model in self.models.items():
            try:
                # Create a safe filename
                safe_name = name.lower().replace(' ', '_')
                model_filename = os.path.join(directory, f'{safe_name}_model.joblib')
                
                # Save the model
                joblib.dump(model, model_filename)
                saved_model_paths[name] = model_filename
                
                st.success(f"Saved model: {name} to {model_filename}")
            except Exception as e:
                st.error(f"Error saving model {name}: {e}")
        
        return saved_model_paths
    
    def load_saved_models(self, directory=None):
        """
        Load previously saved models from a directory
        
        :param directory: Directory containing saved models. If None, uses default Streamlit directory
        :return: Dictionary of loaded models
        """
        # Create default directory if not specified
        if directory is None:
            directory = os.path.join(os.getcwd(), 'saved_models')
        
        # Dictionary to store loaded models
        loaded_models = {}
        
        # Check if directory exists
        if not os.path.exists(directory):
            st.warning(f"No saved models directory found at {directory}")
            return loaded_models
        
        # Load models
        for filename in os.listdir(directory):
            if filename.endswith('_model.joblib'):
                try:
                    model_path = os.path.join(directory, filename)
                    model = joblib.load(model_path)
                    
                    # Extract model name from filename
                    model_name = filename.replace('_model.joblib', '').replace('_', ' ').title()
                    
                    loaded_models[model_name] = model
                    st.success(f"Loaded model: {model_name} from {model_path}")
                except Exception as e:
                    st.error(f"Error loading model from {filename}: {e}")
        
        return loaded_models
