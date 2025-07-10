# Linear Regression Model Trainer

This is a Streamlit web application that allows users to train a Linear Regression model with customizable data preprocessing, model selection, and hyperparameter tuning. It provides interactive visualizations and performance metrics to help users understand their data and model.


![Image_Alt](https://github.com/Mani212005/LRml/blob/472332cbbf84af53793f21f3267f6fc12561ab8f/Screenshot%202025-07-10%20150331.png)

## Features

### 1. Dataset Upload & Preprocessing
-   **CSV File Upload:** Accept `.csv` file uploads.
-   **Robust CSV Parsing:** Users can specify CSV delimiter (comma, semicolon, tab) and encoding (UTF-8, Latin-1, ISO-8859-1).
-   **Missing Value Strategies:** Choose from 'Drop Rows', 'Mean Imputation', or 'Median Imputation'.
-   **Outlier Handling:** Option to 'Remove Outliers (IQR)' for numerical columns.

### 2. Data Exploration & Visualizations
-   **Data Preview:** Show a preview of the first few rows of the uploaded dataset.
-   **Optional Visualizations:** Users can select to view:
    -   **Outlier Visualizations (Box Plots):** Box plots for selected numeric columns to identify outliers.
    -   **Feature Distribution Plots (Histograms):** Histograms for selected numeric columns to understand data distribution.
    -   **Correlation Matrix Heatmap:** A heatmap showing the correlation between numeric columns.

### 3. Feature & Target Selection
-   **Auto-detection:** Automatically detects numeric and categorical columns.
-   **User Selection:** Allows users to select:
    -   One or more **feature columns (X)** (numeric and one-hot encoded categorical).
    -   One **target column (y)** (numeric only).
-   **Categorical Feature Handling:** Selected categorical columns are automatically one-hot encoded.

### 4. Model Training & Hyperparameter Tuning
-   **Model Selection:** Choose between 'Linear Regression', 'Ridge', and 'Lasso' models.
-   **Hyperparameter Tuning:** For Ridge and Lasso, a slider is available to adjust the `alpha` regularization parameter.
-   **Cross-Validation:** Implements k-fold cross-validation (default k=5) for robust performance evaluation.

### 5. Model Performance & Insights
-   **Performance Metrics:** Displays key metrics in a table format:
    -   Coefficients
    -   Intercept
    -   R² score
    -   Mean Absolute Error (MAE)
    -   Mean Squared Error (MSE)
    -   Cross-Validation R² (Mean and Standard Deviation)
-   **Visualizations:**
    -   **Predicted vs Actual Values Plot:** Scatter plot with a regression line.
    -   **Residual Plot:** Scatter plot of residuals vs predicted values.
    -   **Feature Importance Bar Chart:** Bar chart showing the magnitude of coefficients for each feature.

### 6. Download Options
-   **Download Trained Model:** Download the trained model as a `.pkl` file.
-   **Download Predictions:** Download the model's predictions on the test set as a `.csv` file.

## Tech Stack

-   **Frontend/UI:** Streamlit
-   **Backend/Logic:** Python
    -   `scikit-learn` for machine learning models.
    -   `Pandas` for data manipulation.
    -   `Plotly Express` for interactive visualizations.
    -   `NumPy` for numerical operations.
    -   `statsmodels` for trendline in plots.

## How to Run the App

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mani212005/LRml.git
    cd LRml
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.
    
    ![Image_Alt](https://github.com/Mani212005/LRml/blob/e5bd028b1f6e17933dcafc338177f8564873e83a/2.png)
    ![Image_Alt](https://github.com/Mani212005/LRml/blob/e5bd028b1f6e17933dcafc338177f8564873e83a/3.png)
    ![Image_Alt](https://github.com/Mani212005/LRml/blob/e5bd028b1f6e17933dcafc338177f8564873e83a/4.png)
    ![Image_Alt](https://github.com/Mani212005/LRml/blob/e5bd028b1f6e17933dcafc338177f8564873e83a/5.png)
    ![Image_Alt](https://github.com/Mani212005/LRml/blob/9184712cb6b65ca3da83ef2c988aedad1e712db9/6.png)
    
## Future Enhancements (Ideas)

-   More advanced outlier detection methods.
-   Support for other regression models (e.g., RandomForestRegressor, XGBoost).
-   More sophisticated hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV).
-   Saving/loading models directly within the app.
-   User authentication and persistent storage.
