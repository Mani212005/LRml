# 📘 SPEC: Linear Regression Model Training MVP

## 🎯 Objective

Build a minimal web-based tool that allows users to:
- Upload a CSV dataset
- Select features and target variable
- Train a Linear Regression model
- View model coefficients and performance metrics
- Visualize predictions vs actual values

This MVP is aimed at students, beginner ML practitioners, or data analysts who want to explore and train simple regression models without coding.

---

## 🔍 Problem Statement

New learners often struggle to understand how simple machine learning models work without getting bogged down in code. There is a need for a minimal UI that lets users interactively train and evaluate a **Linear Regression model** using their own datasets.

---

## ✨ Core Features (Ordered by Importance)

### 1. 📤 Dataset Upload
- Accept `.csv` file upload
- Show a preview of first 5–10 rows
- Validate that data is tabular, contains numeric columns, and is not empty

### 2. 📊 Feature & Target Selection
- Auto-detect numeric columns
- Allow user to select:
  - One or more **feature columns (X)**
  - One **target column (y)**

### 3. 🤖 Train Linear Regression Model
- Train scikit-learn's `LinearRegression` model on selected data
- Handle missing values gracefully (warn user or drop rows)
- Split dataset into train/test (80/20 split)

### 4. 📈 Display Model Performance
- Show:
  - Coefficients
  - Intercept
  - R² score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

### 5. 📉 Visualizations
- Plot **Predicted vs Actual values** for test set
- Optional: Residual plot

---

## 📥 Inputs

- CSV file uploaded by the user
- User selections for:
  - Feature columns (numeric only)
  - Target column (numeric only)

---

## 📤 Outputs

- Trained model (session-based; not downloadable for now)
- Numeric performance metrics
- Interactive visualizations (e.g., matplotlib, seaborn, or Plotly)

---

## 🔁 User Flow

1. User opens the web app
2. Uploads CSV file
3. Selects feature and target columns
4. Clicks “Train Model”
5. App trains the model and displays:
   - Coefficients
   - Metrics
   - Plots
6. User optionally uploads a new dataset to start over

---

## 🧱 Tech Stack

- **Frontend/UI**: Streamlit 
- **Backend/Logic**: Python + scikit-learn + Pandas + Matplotlib/Plotly

---

## 🧠 Edge Cases

- Non-numeric columns selected → block with warning
- Missing target/feature selection → disable training button
- Missing values in data → handle with a simple strategy (e.g., drop rows)
- Dataset too large (>10MB) → display warning or restrict upload
- Dataset with fewer than 10 rows → block model training

---

## 🛠️ Notes

- No login or user auth in MVP
- No file storage or model persistence
- No hyperparameter tuning or regularization (pure `LinearRegression`)
- Keep UI simple: max 1–2 pages, avoid complexity
- Aim for clean code separation: UI logic and ML logic in separate modules

