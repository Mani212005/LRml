
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
import plotly.express as px
import config

def train_model(X, y, missing_strategy, outlier_strategy, categorical_cols, model_type, alpha, cv_folds):
    # One-hot encode categorical columns
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Handle missing values
    if missing_strategy == 'Drop Rows':
        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data[y.name]
    elif missing_strategy == 'Mean Imputation':
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
    elif missing_strategy == 'Median Imputation':
        X = X.fillna(X.median())
        y = y.fillna(y.median())

    # Handle outliers
    if outlier_strategy == 'Remove Outliers (IQR)':
        # Combine X and y for consistent outlier removal
        data = pd.concat([X, y], axis=1)
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        X = data[X.columns]
        y = data[y.name]

    # Select model based on user choice
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'Lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("Invalid model type selected.")

    # Perform Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=config.CV_FOLDS, scoring='r2')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, X_test, y_test, cv_scores

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics_dict = {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "r2_score": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "mse": metrics.mean_squared_error(y_test, y_pred)
    }
    return metrics_dict

def create_plots(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Predicted vs Actual Values Plot with Regression Line
    fig_actual_vs_predicted = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Predicted vs Actual Values', trendline="ols")
    fig_actual_vs_predicted.add_shape(
        type='line',
        x0=y_test.min(),
        y0=y_test.min(),
        x1=y_test.max(),
        y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )

    # Residual Plot
    fig_residuals = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title='Residual Plot')
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

    return {
        "actual_vs_predicted": fig_actual_vs_predicted,
        "residuals": fig_residuals
    }
