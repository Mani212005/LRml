
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ml import train_model, get_metrics, create_plots

@pytest.fixture
def sample_data():
    data = {
        'feature1': np.arange(10),
        'feature2': np.arange(10) * 2,
        'target': np.arange(10) * 3
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_with_outliers():
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], # Outlier at 100
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000], # Outlier at 1000
        'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100] # Outlier at 100
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_with_categorical():
    data = {
        'feature_num': np.arange(10),
        'feature_cat': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'target': np.arange(10) * 5
    }
    return pd.DataFrame(data)

def test_train_model(sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    model, X_test, y_test, _ = train_model(X, y, missing_strategy='Drop Rows', outlier_strategy='None', categorical_cols=[], model_type='Linear Regression', alpha=1.0, cv_folds=5)
    assert isinstance(model, LinearRegression)
    assert X_test.shape[0] == 2
    assert y_test.shape[0] == 2

def test_train_model_outlier_removal(sample_data_with_outliers):
    X = sample_data_with_outliers[['feature1', 'feature2']]
    y = sample_data_with_outliers['target']
    model, X_test, y_test, _ = train_model(X, y, missing_strategy='Drop Rows', outlier_strategy='Remove Outliers (IQR)', categorical_cols=[], model_type='Linear Regression', alpha=1.0, cv_folds=5)
    assert isinstance(model, LinearRegression)
    # After outlier removal, the dataset size should be smaller.
    # For the given sample_data_with_outliers, the last row should be removed.
    # So, 10 rows - 1 outlier = 9 rows. 80/20 split means 7 train, 2 test.
    assert X_test.shape[0] == 2 # 20% of 9 rows is 1.8, so it will be 2 due to rounding
    assert y_test.shape[0] == 2

def test_train_model_categorical_encoding(sample_data_with_categorical):
    X = sample_data_with_categorical[['feature_num', 'feature_cat']]
    y = sample_data_with_categorical['target']
    model, X_test, y_test, _ = train_model(X, y, missing_strategy='Drop Rows', outlier_strategy='None', categorical_cols=['feature_cat'], model_type='Linear Regression', alpha=1.0, cv_folds=5)
    assert isinstance(model, LinearRegression)
    # Original X has 2 columns. After one-hot encoding 'feature_cat' (3 unique values, drop_first=True), it should have 1 + 2 = 3 columns.
    # The X_test will have 3 columns as well.
    assert X_test.shape[1] == 3

def test_get_metrics(sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    model, X_test, y_test, _ = train_model(X, y, missing_strategy='Drop Rows', outlier_strategy='None', categorical_cols=[], model_type='Linear Regression', alpha=1.0, cv_folds=5)
    metrics = get_metrics(model, X_test, y_test)
    assert 'coef' in metrics
    assert 'intercept' in metrics
    assert 'r2_score' in metrics
    assert 'mae' in metrics
    assert 'mse' in metrics

def test_create_plots(sample_data):
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    model, X_test, y_test, _ = train_model(X, y, missing_strategy='Drop Rows', outlier_strategy='None', categorical_cols=[], model_type='Linear Regression', alpha=1.0, cv_folds=5)
    plots = create_plots(model, X_test, y_test)
    assert "actual_vs_predicted" in plots
    assert "residuals" in plots
