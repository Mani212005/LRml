
import streamlit as st
import pandas as pd
import plotly.express as px
from ml import train_model, get_metrics, create_plots
import config

st.set_page_config(layout="wide", page_title="Linear Regression App 📈")

st.title("Linear Regression Model Trainer 📈")
st.write("Welcome! This app helps you train a Linear Regression model with customizable preprocessing steps. Follow the steps below to get started:")

# Step 1: Dataset Upload
st.header("1. Upload Your Dataset 📂")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

if uploaded_file is not None:
    # Add a reset button
    if st.button("Reset App 🔄", key="reset_button"):
        st.session_state.clear()
        st.rerun()

    with st.expander("CSV Configuration & Preprocessing Options ⚙️", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delimiter = st.selectbox("CSV Delimiter", (',', ';', '\t'), key="delimiter_select")
        with col2:
            encoding = st.selectbox("Encoding", ('utf-8', 'latin1', 'iso-8859-1'), key="encoding_select")
        with col3:
            missing_strategy = st.selectbox("Missing Values", ('Drop Rows', 'Mean Imputation', 'Median Imputation'), key="missing_strategy_select")
        with col4:
            outlier_strategy = st.selectbox("Outlier Handling", ('None', 'Remove Outliers (IQR)'), key="outlier_strategy_select")

    try:
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        st.write("### Data Preview 📊")
        st.write(df.head())

        with st.expander("Explore Data Visualizations (Optional) 📊"):
            selected_outlier_cols = st.multiselect("Select columns for Outlier Visualizations (Box Plots) 📈", numeric_cols)
            selected_distribution_cols = st.multiselect("Select columns for Feature Distribution Plots (Histograms) 📈", numeric_cols)
            show_correlation_viz = st.checkbox("Show Correlation Matrix Heatmap 📈")

            if selected_outlier_cols:
                st.write("### Outlier Visualizations 📊")
                for col in selected_outlier_cols:
                    fig = px.box(df, y=col, title=f'Box Plot of {col}', height=400, width=600)
                    st.plotly_chart(fig)

            if selected_distribution_cols:
                st.write("### Feature Distributions 📊")
                for col in selected_distribution_cols:
                    fig = px.histogram(df, x=col, title=f'Distribution of {col}', height=400, width=600)
                    st.plotly_chart(fig)

            if show_correlation_viz:
                st.write("### Correlation Matrix Heatmap 📈")
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix", height=600, width=800)
                st.plotly_chart(fig_corr)

        # Step 2: Feature & Target Selection
        st.header("2. Select Features and Target 🎯")

        if df.empty:
            st.warning("The uploaded file is empty.")
        elif len(df.columns) == 0:
            st.warning("The uploaded file has no columns.")
        else:
            if not numeric_cols and not categorical_cols:
                st.warning("No numeric or categorical columns found in the dataset.")
            else:
                if not numeric_cols:
                    st.warning("No numeric columns found. Linear Regression requires numeric features.")

                selected_categorical_cols = []
                if categorical_cols:
                    with st.expander("Categorical Feature Handling 🗂️"):
                        selected_categorical_cols = st.multiselect("Select categorical columns for One-Hot Encoding", categorical_cols)

                feature_cols = st.multiselect("Select feature columns (X) (Numeric and One-Hot Encoded Categorical)", numeric_cols + selected_categorical_cols)
                target_col = st.selectbox("Select target column (y) (Numeric Only)", numeric_cols)

                # Step 3: Train Model
                st.header("3. Train Linear Regression Model 🧠")

                with st.expander("Model Selection & Hyperparameters ⚙️", expanded=True):
                    model_type = st.selectbox("Select Model Type", ('Linear Regression', 'Ridge', 'Lasso'), key="model_type_select")
                    alpha = None
                    if model_type in ['Ridge', 'Lasso']:
                        alpha = st.slider(f"Alpha for {model_type}", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                    
                    

                if st.button("Train Model ✨", key="train_button"):
                    if not feature_cols:
                        st.warning("⚠️ Please select at least one feature column.")
                    elif not target_col:
                        st.warning("⚠️ Please select a target column.")
                    elif target_col in feature_cols:
                        st.warning("⚠️ Target column cannot be a feature column. Please select a different target or remove it from features.")
                    else:
                        X = df[feature_cols]
                        y = df[target_col]

                        with st.spinner("Training model... This might take a moment! ⏳"):
                            model, X_test, y_test, cv_scores = train_model(X, y, missing_strategy, outlier_strategy, selected_categorical_cols, model_type, alpha, config.CV_FOLDS)
                        metrics = get_metrics(model, X_test, y_test)
                        
                        # Step 4: Display Model Performance
                        st.header("4. Model Performance 📊")
                        st.write("Here are the key performance indicators for your trained model:")
                        
                        metrics_df = pd.DataFrame({
                            "Metric": ["Coefficients", "Intercept", "R² Score", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Cross-Validation R² (Mean)", "Cross-Validation R² (Std)"],
                            "Value": [
                                f"{metrics['coef']:.4f}" if isinstance(metrics['coef'], (int, float)) else f"[{', '.join([f'{c:.4f}' for c in metrics['coef']])}]",
                                f"{metrics['intercept']:.4f}",
                                f"{metrics['r2_score']:.4f}",
                                f"{metrics['mae']:.4f}",
                                f"{metrics['mse']:.4f}",
                                f"{cv_scores.mean():.4f}",
                                f"{cv_scores.std():.4f}"
                            ]
                        })
                        st.dataframe(metrics_df, hide_index=True)

                        # Step 5: Visualizations
                        st.header("5. Visualizations 📈")
                        st.write("Explore the relationship between predicted and actual values:")
                        plots = create_plots(model, X_test, y_test)
                        st.plotly_chart(plots['actual_vs_predicted'], use_container_width=True)

                        st.write("### Residual Plot 📉")
                        st.plotly_chart(plots['residuals'], use_container_width=True)

                        st.write("### Feature Importance (Coefficients) 📊")
                        if hasattr(model, 'coef_') and len(feature_cols) > 0:
                            coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
                            coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()
                            coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)
                            fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Importance', height=400, width=600)
                            st.plotly_chart(fig_coef)
                        else:
                            st.info("Coefficients are not available for this model type or no features were selected.")

                        # Step 6: Download Options
                        st.header("6. Download Options 💾")
                        
                        # Download Trained Model
                        import pickle
                        model_bytes = pickle.dumps(model)
                        st.download_button(
                            label="Download Trained Model (.pkl)",
                            data=model_bytes,
                            file_name="linear_regression_model.pkl",
                            mime="application/octet-stream"
                        )

                        # Download Predictions
                        y_pred = model.predict(X_test)
                        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                        csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions (.csv)",
                            data=csv_predictions,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

    except Exception as e:
        st.error(f"An error occurred: {e} 🐞 Please check your data and selections.")
