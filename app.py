"""
ML Assignment 2 - Streamlit Web Application
Heart Disease Classification
Interactive ML Model Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 72px;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        padding: 20px 0;
        margin: 0;
        line-height: 1.2;
    }

    /* Sub header styling */
    .sub-header {
        font-size: 32px;
        font-weight: bold;
        color: #0068C9;
        padding: 10px 0;
    }

    /* Metric box styling */
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }

    /* Smooth button hover effect */
    .stButton>button {
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Download button hover effect */
    .stDownloadButton>button {
        transition: all 0.3s ease;
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 style="font-size: 30 px; font-weight: 600; color: #FF4B4B; text-align: center; padding: 5px 0; margin: 0; line-height: 1.2;">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Classification Model Comparison")

# Sidebar
st.sidebar.title("⚙️ Configuration")

# ====================================
# FEATURE 1: DATASET UPLOAD
# ====================================
st.sidebar.markdown("### 📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# ====================================
# FEATURE 2: MODEL SELECTION
# ====================================
st.sidebar.markdown("### 🤖 Select Model")

model_options = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'K-Nearest Neighbors (kNN)': 'knn_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(model_options.keys())
)

# Sidebar Footer with student details
st.sidebar.markdown("<br>" * 3, unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='position: fixed; bottom: 20px; left: 20px; width: 300px; font-size: 15px; color: #b0b0b0; line-height: 1.6;'>
    <p style='margin: 0; font-weight: bold; color: #e0e0e0;'>Student:</p>
    <p style='margin: 0;'>Deepti Yashwant Walde</p>
    <p style='margin: 0;'>ID: 2025AA05185</p>
</div>
""", unsafe_allow_html=True)

# Load training data info
@st.cache_data
def load_training_info():
    """Load information about the training dataset"""
    if os.path.exists('heart_disease.csv'):
        df = pd.read_csv('heart_disease.csv')
        return df
    return None

# Load model and scaler
@st.cache_resource
def load_model_and_scaler(model_name):
    """Load the selected model and scaler"""
    try:
        # Load scaler
        scaler_path = 'model/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = pickle.load(open(scaler_path, 'rb'))
        else:
            scaler = None

        # Load model
        model_filename = model_options[model_name]
        model_path = f'model/{model_filename}'

        if os.path.exists(model_path):
            model = pickle.load(open(model_path, 'rb'))
            return model, scaler, None
        else:
            return None, None, f"Model file not found: {model_path}"
    except Exception as e:
        return None, None, str(e)

# Main Content
if uploaded_file is not None:
    # Load uploaded data
    try:
        test_data = pd.read_csv(uploaded_file)

        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(test_data))
        with col2:
            st.metric("Total Features", len(test_data.columns))
        with col3:
            if 'target' in test_data.columns:
                st.metric("Has Target Column", "Yes ✓")
            else:
                st.metric("Has Target Column", "No ✗")

        # Show data preview
        with st.expander("📊 View Data Preview"):
            st.dataframe(test_data, height=400, use_container_width=True)

        # Load model
        model, scaler, error = load_model_and_scaler(selected_model_name)

        if error:
            st.error(f"❌ Error loading model: {error}")
            if "xgboost" in error.lower():
                st.info("💡 XGBoost model not found. Please train the model first by running model_training.py on BITS Virtual Lab.")
        elif model is None:
            st.error("❌ Model could not be loaded. Please train the models first.")
        else:
            st.success(f"✅ Model loaded: {selected_model_name}")

            # Prepare data
            if 'target' in test_data.columns:
                X_test = test_data.drop('target', axis=1)
                y_test = test_data['target']
                has_target = True
            else:
                X_test = test_data
                y_test = None
                has_target = False

            # Scale data if needed
            if selected_model_name in ['Logistic Regression', 'K-Nearest Neighbors (kNN)', 'Naive Bayes']:
                if scaler:
                    X_test_processed = scaler.transform(X_test)
                else:
                    st.warning("⚠️ Scaler not found. Using unscaled data.")
                    X_test_processed = X_test
            else:
                X_test_processed = X_test

            # Make predictions
            try:
                y_pred = model.predict(X_test_processed)
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

                # Display predictions
                st.markdown("---")
                st.markdown('<h2 style="font-size: 32px; font-weight: bold; color: #0068C9; padding: 10px 0;">🎯 Prediction Results</h2>', unsafe_allow_html=True)

                # Prediction distribution
                pred_df = pd.DataFrame({
                    'Prediction': ['No Disease (0)', 'Disease (1)'],
                    'Count': [(y_pred == 0).sum(), (y_pred == 1).sum()],
                    'Percentage': [(y_pred == 0).sum()/len(y_pred)*100, (y_pred == 1).sum()/len(y_pred)*100]
                })

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.dataframe(pred_df.style.format({'Percentage': '{:.2f}%'}))

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#00CC96', '#FF4B4B']
                    ax.bar(pred_df['Prediction'], pred_df['Count'], color=colors)
                    ax.set_ylabel('Count')
                    ax.set_title('Prediction Distribution')
                    plt.xticks(rotation=0)
                    st.pyplot(fig)

                # ====================================
                # FEATURE 3: DISPLAY METRICS (if target available)
                # ====================================
                if has_target and y_test is not None:
                    st.markdown("---")
                    st.markdown('<h2 style="font-size: 32px; font-weight: bold; color: #0068C9; padding: 10px 0;">📊 Model Performance Metrics</h2>', unsafe_allow_html=True)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    mcc = matthews_corrcoef(y_test, y_pred)

                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
                        st.metric("📈 AUC Score", f"{auc:.4f}", f"{auc*100:.2f}%")

                    with col2:
                        st.metric("🔍 Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
                        st.metric("🎪 Recall", f"{recall:.4f}", f"{recall*100:.2f}%")

                    with col3:
                        st.metric("⚖️ F1 Score", f"{f1:.4f}", f"{f1*100:.2f}%")
                        st.metric("🔢 MCC Score", f"{mcc:.4f}")

                    # ====================================
                    # FEATURE 4: CONFUSION MATRIX & CLASSIFICATION REPORT
                    # ====================================
                    st.markdown("---")
                    st.markdown('<h2 style="font-size: 32px; font-weight: bold; color: #0068C9; padding: 10px 0;">📉 Detailed Analysis</h2>', unsafe_allow_html=True)

                    # Model Insights - Show first at the top
                    st.markdown("### 💡 Model Insights")

                    col_insight1, col_insight2 = st.columns(2)

                    with col_insight1:
                        if accuracy >= 0.8:
                            st.success(f"✅ Excellent performance! Accuracy: {accuracy:.2%}")
                        elif accuracy >= 0.7:
                            st.info(f"✓ Good performance. Accuracy: {accuracy:.2%}")
                        else:
                            st.warning(f"⚠️ Room for improvement. Accuracy: {accuracy:.2%}")

                    with col_insight2:
                        # Best metric
                        metrics_dict = {
                            'Accuracy': accuracy,
                            'AUC': auc,
                            'Precision': precision,
                            'Recall': recall,
                            'F1': f1
                        }
                        best_metric = max(metrics_dict, key=metrics_dict.get)
                        st.info(f"🏆 Best Metric: {best_metric} ({metrics_dict[best_metric]:.4f})")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Confusion Matrix and Classification Report in columns
                    col1, col2 = st.columns(2)

                    with col1:
                        # Confusion Matrix
                        st.markdown("#### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=['No Disease', 'Disease'],
                                   yticklabels=['No Disease', 'Disease'],
                                   cbar_kws={'label': 'Count'})
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                        ax.set_title(f'Confusion Matrix - {selected_model_name}')
                        st.pyplot(fig)

                        # Show confusion matrix values in a cleaner format
                        st.markdown("**Confusion Matrix Breakdown:**")
                        breakdown_col1, breakdown_col2 = st.columns(2)
                        with breakdown_col1:
                            st.metric("True Negatives (TN)", cm[0,0])
                            st.metric("False Negatives (FN)", cm[1,0])
                        with breakdown_col2:
                            st.metric("False Positives (FP)", cm[0,1])
                            st.metric("True Positives (TP)", cm[1,1])

                    with col2:
                        # Classification Report
                        st.markdown("#### Classification Report")
                        report = classification_report(y_test, y_pred,
                                                      target_names=['No Disease', 'Disease'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

                # Download predictions
                st.markdown("---")
                st.markdown("### 💾 Download Predictions")

                result_df = test_data.copy()
                result_df['Predicted_Class'] = y_pred
                result_df['Prediction_Probability'] = y_pred_proba

                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")
                st.info("Please ensure your CSV file has the correct features.")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to get started")

    # Show sample data
    training_data = load_training_info()
    if training_data is not None:
        # Expected Data Format Section - Show first
        with st.expander("📋 View Required Features"):
            st.markdown("""
            **Patient Data Required:**

            1. **age** - Patient age
            2. **sex** - Gender (0=Female, 1=Male)
            3. **cp** - Chest pain type (0-3)
            4. **trestbps** - Resting blood pressure (mm Hg)
            5. **chol** - Cholesterol level (mg/dl)
            6. **fbs** - Fasting blood sugar >120 mg/dl (0=No, 1=Yes)
            7. **restecg** - Resting ECG results (0-2)
            8. **thalach** - Maximum heart rate achieved
            9. **exang** - Exercise induced angina (0=No, 1=Yes)
            10. **oldpeak** - ST depression value
            11. **slope** - ST segment slope (0-2)
            12. **ca** - Number of major vessels (0-3)
            13. **thal** - Thalassemia type (0-3)
            14. **target** (optional) - Actual diagnosis (0=No Disease, 1=Disease)
            """)

        # Show sample data - Show second
        with st.expander("👁️ View Sample Training Data"):
            st.dataframe(training_data.head(20), height=400, use_container_width=True)

        # Download sample template
        st.markdown("### 📥 Download Sample Template")
        sample_template = training_data.drop('target', axis=1).head(5)
        csv = sample_template.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv,
            file_name="sample_template.csv",
            mime="text/csv"
        )

    # Show model info
    st.markdown("### 🤖 Available Models")
    st.markdown("""
    Six classification models are available for prediction:

    1. **Logistic Regression** - Linear classifier with probabilistic output
    2. **Decision Tree** - Rule-based tree classifier
    3. **K-Nearest Neighbors (kNN)** - Instance-based learning algorithm
    4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Gradient boosting ensemble method
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>ML Assignment 2 - Heart Disease Classification System</strong></p>
    <p>BITS Pilani - M.Tech (AIML/DSE) | Machine Learning</p>
</div>
""", unsafe_allow_html=True)
