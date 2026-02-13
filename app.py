"""
ML Assignment 2 - Streamlit Web Application
Heart Disease Classification
Interactive ML Model Evaluation
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
from matplotlib.patches import Patch

# ====================================
# CONSTANTS
# ====================================
# Color Scheme
COLORS = {
    'excellent': '#43e97b',
    'good': '#4facfe',
    'fair': '#ffa726',
    'poor': '#ff6b6b',
    'primary': '#667eea',
    'secondary': '#764ba2',
    'tn': '#43e97b',  # True Negative
    'fp': '#ffa726',  # False Positive
    'fn': '#ff6b6b',  # False Negative
    'tp': '#667eea',  # True Positive
}

# Model file mapping
MODEL_OPTIONS = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'K-Nearest Neighbors (kNN)': 'knn_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl'
}

# Models that require scaling
MODELS_REQUIRING_SCALING = ['Logistic Regression', 'K-Nearest Neighbors (kNN)', 'Naive Bayes']

# ====================================
# HELPER FUNCTIONS - UTILITIES
# ====================================
def get_performance_level(score, metric_name):
    """Get performance level and color based on score"""
    if metric_name == "MCC":
        if score >= 0.7: return "Excellent", COLORS['excellent']
        elif score >= 0.5: return "Good", COLORS['good']
        elif score >= 0.3: return "Fair", COLORS['fair']
        else: return "Poor", COLORS['poor']
    else:
        if score >= 0.9: return "Excellent", COLORS['excellent']
        elif score >= 0.8: return "Good", COLORS['good']
        elif score >= 0.7: return "Fair", COLORS['fair']
        else: return "Poor", COLORS['poor']

def get_confidence_level(prob):
    """Determine confidence level from probability"""
    if prob < 0.4: return 'Low Confidence'
    elif prob < 0.7: return 'Medium Confidence'
    else: return 'High Confidence'

def get_risk_category(prob):
    """Determine risk category from probability"""
    if prob < 0.3: return 'Low Risk'
    elif prob < 0.7: return 'Medium Risk'
    else: return 'High Risk'

# ====================================
# HELPER FUNCTIONS - UI COMPONENTS
# ====================================
def create_section_header(icon, title, description):
    """Create a consistent section header with gradient background"""
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, #ffffff 100%);
                padding: 10px; border-radius: 15px; margin: 20px 0;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
        <h2 style="color: #2d1b4e; font-size: 32px; font-weight: bold; margin: 0; text-align: center;">
            {icon} {title}
        </h2>
    </div>
    ''', unsafe_allow_html=True)

    if description:
        st.markdown(f"""
        <p style="text-align: center; color: #aaa; font-size: 14px; margin: -10px 0 20px 0;">
        {description}
        </p>
        """, unsafe_allow_html=True)

def create_styled_classification_report(df):
    """Create HTML table for classification report"""
    html = """
    <style>
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }
    .report-table thead tr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .report-table thead th {
        padding: 15px;
        font-size: 15px;
        letter-spacing: 0.5px;
    }
    .report-table tbody tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .report-table tbody tr:nth-child(1),
    .report-table tbody tr:nth-child(2) {
        background-color: rgba(102, 126, 234, 0.15);
    }
    .report-table tbody tr:nth-child(3) {
        background-color: rgba(118, 75, 162, 0.2);
        font-weight: 600;
    }
    .report-table tbody tr:nth-child(4),
    .report-table tbody tr:nth-child(5) {
        background-color: rgba(102, 126, 234, 0.1);
    }
    .report-table tbody th {
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        color: #ddd;
        background-color: rgba(102, 126, 234, 0.2);
    }
    .report-table tbody td {
        padding: 12px 15px;
        text-align: center;
        color: #fff;
        font-weight: 500;
    }
    .report-table tbody td.high-value {
        background-color: rgba(102, 126, 234, 0.4);
        font-weight: 700;
    }
    </style>
    <table class="report-table">
        <thead>
            <tr>
                <th></th>
    """

    # Add column headers
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    # Add data rows
    for idx, row in df.iterrows():
        html += f"<tr><th>{idx}</th>"
        for i, val in enumerate(row):
            # Highlight high values
            cell_class = "high-value" if val > 0.85 and df.columns[i] != 'support' else ""
            html += f"<td class='{cell_class}'>{val:.4f}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    return html

def display_dataset_info(test_data):
    """Display dataset information cards"""
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

def create_prediction_distribution_table(no_disease_count, disease_count, no_disease_pct, disease_pct):
    """Create HTML table for prediction distribution"""
    return f"""
    <div class="pred-table-wrapper">
        <table class="pred-table">
            <thead>
                <tr>
                    <th>Prediction</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>✅ No Disease (0)</td>
                    <td><strong>{no_disease_count}</strong></td>
                    <td><strong>{no_disease_pct:.1f}%</strong></td>
                </tr>
                <tr>
                    <td>⚠️ Disease (1)</td>
                    <td><strong>{disease_count}</strong></td>
                    <td><strong>{disease_pct:.1f}%</strong></td>
                </tr>
            </tbody>
        </table>
    </div>
    """

def create_prediction_bar_chart(no_disease_count, disease_count):
    """Create bar chart for prediction distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS['excellent'], COLORS['poor']]
    predictions = ['No Disease (0)', 'Disease (1)']
    counts = [no_disease_count, disease_count]
    bars = ax.bar(predictions, counts, color=colors, edgecolor='white', linewidth=2)

    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def create_metric_card_html(name, value, description, level, color, progress_pct, percentage_display):
    """Create HTML for a single metric card"""
    return f"""
    <div class="metric-card" style="border-left-color: {color};">
        <div>
            <div class="metric-header">{name}</div>
            <div>
                <span class="metric-value" style="color: {color};">{value}</span>
                <span class="metric-percentage">{percentage_display}</span>
            </div>
        </div>
        <div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%; background: {color};"></div>
            </div>
            <span class="metric-level" style="background: {color};">{level}</span>
            <div class="metric-description">{description}</div>
        </div>
    </div>
    """

def create_metrics_comparison_chart(accuracy, precision, recall, f1, auc, mcc, model_name):
    """Create horizontal bar chart comparing all metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score', 'MCC\n(normalized)']
    metric_values = [accuracy, precision, recall, f1, auc, (mcc + 1) / 2]

    # Color based on performance
    colors_chart = []
    for val in metric_values:
        if val >= 0.9: colors_chart.append(COLORS['excellent'])
        elif val >= 0.8: colors_chart.append(COLORS['good'])
        elif val >= 0.7: colors_chart.append(COLORS['fair'])
        else: colors_chart.append(COLORS['poor'])

    bars = ax.barh(metric_names, metric_values, color=colors_chart, edgecolor='white', linewidth=2)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        width = bar.get_width()
        if i == 5:  # MCC
            label = f'{mcc:.4f}'
        else:
            label = f'{val:.4f} ({val*100:.1f}%)'
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               label, ha='left', va='center', fontweight='bold', fontsize=10)

    # Styling
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_title(f'Performance Metrics Overview - {model_name}',
                fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Reference lines
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0.8, color='blue', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.3, linewidth=1)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['excellent'], label='Excellent (≥0.9)'),
        Patch(facecolor=COLORS['good'], label='Good (0.8-0.9)'),
        Patch(facecolor=COLORS['fair'], label='Fair (0.7-0.8)'),
        Patch(facecolor=COLORS['poor'], label='Poor (<0.7)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    return fig

def create_confusion_matrix_chart(cm, model_name):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
               xticklabels=['No Disease', 'Disease'],
               yticklabels=['No Disease', 'Disease'],
               cbar_kws={'label': 'Count'},
               linewidths=3, linecolor='white',
               annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_cm_breakdown_card(label, value, description, card_class):
    """Create confusion matrix breakdown card HTML"""
    return f"""
    <div class="cm-card {card_class}">
        <div class="cm-label">{label}</div>
        <div class="cm-value">{value}</div>
        <div class="cm-desc">{description}</div>
    </div>
    """

# ====================================
# HELPER FUNCTIONS - VISUALIZATIONS
# ====================================
def display_prediction_results(y_pred):
    """Display prediction distribution section"""
    # Prediction distribution
    no_disease_count = (y_pred == 0).sum()
    disease_count = (y_pred == 1).sum()
    no_disease_pct = (no_disease_count / len(y_pred)) * 100
    disease_pct = (disease_count / len(y_pred)) * 100

    col1, col2 = st.columns([1, 2])

    with col1:
        # Prediction table
        table_html = create_prediction_distribution_table(
            no_disease_count, disease_count, no_disease_pct, disease_pct
        )
        st.markdown(table_html, unsafe_allow_html=True)

    with col2:
        # Bar chart
        fig = create_prediction_bar_chart(no_disease_count, disease_count)
        st.pyplot(fig)
        plt.close(fig)  # Close figure to free memory

def display_performance_metrics(y_test, y_pred, y_pred_proba):
    """Display all performance metrics"""
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics_data = [
        ("🎯 Accuracy", accuracy, "Overall correctness of predictions"),
        ("🔍 Precision", precision, "Accuracy of positive predictions"),
        ("🎪 Recall", recall, "Ability to find all positive cases"),
        ("⚖️ F1 Score", f1, "Balance between precision and recall"),
        ("📈 AUC Score", auc, "Ability to distinguish between classes"),
        ("🔢 MCC", mcc, "Overall quality (-1 to +1 scale)")
    ]

    # Display in 3 columns
    col1, col2, col3 = st.columns(3)

    for idx, (name, value, description) in enumerate(metrics_data):
        metric_type = "MCC" if "MCC" in name else "Standard"
        level, color = get_performance_level(value, metric_type)

        # Format display based on metric type
        if "MCC" in name:
            value_display = f"{value:.4f}"
            percentage_display = f"(Range: -1 to +1)"
            progress_pct = ((value + 1) / 2) * 100
        else:
            value_display = f"{value:.4f}"
            percentage_display = f"({value*100:.2f}%)"
            progress_pct = value * 100

        card_html = create_metric_card_html(
            name, value_display, description, level, color, progress_pct, percentage_display
        )

        # Distribute across 3 columns
        if idx % 3 == 0:
            col1.markdown(card_html, unsafe_allow_html=True)
        elif idx % 3 == 1:
            col2.markdown(card_html, unsafe_allow_html=True)
        else:
            col3.markdown(card_html, unsafe_allow_html=True)

    return accuracy, precision, recall, f1, auc, mcc

def display_detailed_analysis(y_test, y_pred, selected_model_name):
    """Display confusion matrix and classification report"""
    # Model Insights
    accuracy = accuracy_score(y_test, y_pred)
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
        auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

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

    # Confusion Matrix and Breakdown
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### 🎨 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig = create_confusion_matrix_chart(cm, selected_model_name)
        st.pyplot(fig)
        plt.close(fig)  # Close figure

    with col2:
        st.markdown("#### 📊 Confusion Matrix Breakdown")

        # 2x2 grid
        breakdown_row1_col1, breakdown_row1_col2 = st.columns(2)

        with breakdown_row1_col1:
            st.markdown(create_cm_breakdown_card(
                "✅ True Negatives (TN)", cm[0,0],
                "Correctly predicted No Disease", "cm-card-tn"
            ), unsafe_allow_html=True)

        with breakdown_row1_col2:
            st.markdown(create_cm_breakdown_card(
                "⚡ False Positives (FP)", cm[0,1],
                "False alarms", "cm-card-fp"
            ), unsafe_allow_html=True)

        breakdown_row2_col1, breakdown_row2_col2 = st.columns(2)

        with breakdown_row2_col1:
            st.markdown(create_cm_breakdown_card(
                "⚠️ False Negatives (FN)", cm[1,0],
                "Missed Disease cases", "cm-card-fn"
            ), unsafe_allow_html=True)

        with breakdown_row2_col2:
            st.markdown(create_cm_breakdown_card(
                "🎯 True Positives (TP)", cm[1,1],
                "Correctly detected Disease", "cm-card-tp"
            ), unsafe_allow_html=True)

    # Classification Report
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Classification Report")
    st.markdown("""
    <p style="color: #888; font-size: 13px; margin: 5px 0 15px 0;">
    Shows precision, recall, and F1-score calculated separately for "No Disease" and "Disease" classes. This reveals if the model is better at detecting one class over the other.
    </p>
    """, unsafe_allow_html=True)

    report = classification_report(y_test, y_pred,
                                  target_names=['No Disease', 'Disease'],
                                  output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.markdown(create_styled_classification_report(report_df), unsafe_allow_html=True)

# ====================================
# CACHED FUNCTIONS
# ====================================
@st.cache_data
def load_training_info():
    """Load information about the training dataset"""
    if os.path.exists('heart_disease.csv'):
        return pd.read_csv('heart_disease.csv')
    return None

@st.cache_resource
def load_model_and_scaler(model_name):
    """Load the selected model and scaler"""
    try:
        # Load scaler
        scaler_path = 'model/scaler.pkl'
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        # Load model
        model_filename = MODEL_OPTIONS[model_name]
        model_path = f'model/{model_filename}'

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, scaler, None
        else:
            return None, None, f"Model file not found: {model_path}"
    except Exception as e:
        return None, None, str(e)

# ====================================
# PAGE CONFIGURATION
# ====================================
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="🫀",
    layout="wide"
)

# ====================================
# GLOBAL CSS STYLES
# ====================================
st.markdown("""
    <style>
    /* Animated gradient background for sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d1b4e 100%);
    }

    /* Heartbeat animation */
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        10%, 30% { transform: scale(1.1); }
        20%, 40% { transform: scale(1); }
    }

    .heartbeat {
        animation: heartbeat 2s ease-in-out infinite;
        display: inline-block;
    }

    /* Colorful metric cards */
    div[data-testid="stMetric"] {
        font-size: 24px;
        font-weight: bold;
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* Different border colors for each metric card */
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) div[data-testid="stMetric"]) div[data-testid="stMetric"]:nth-of-type(1) {
        border-left-color: #43e97b;
    }
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2) div[data-testid="stMetric"]) div[data-testid="stMetric"]:nth-of-type(1) {
        border-left-color: #667eea;
    }
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(3) div[data-testid="stMetric"]) div[data-testid="stMetric"]:nth-of-type(1) {
        border-left-color: #764ba2;
    }
    div[data-testid="stMetric"] label {
        color: #ddd;
        font-size: 14px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 32px;
        font-weight: bold;
    }

    /* Button animations */
    .stButton>button, .stDownloadButton>button {
        transition: all 0.4s ease;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Colorful expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        font-weight: 600;
    }

    /* Message styling */
    .stSuccess {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #00d084;
        box-shadow: 0 5px 15px rgba(67, 233, 123, 0.3);
    }

    .stInfo {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #0095ff;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
    }

    .stWarning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 5px 15px rgba(250, 112, 154, 0.3);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }

    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* Glowing divider */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #FF4B4B, transparent);
        margin: 30px 0;
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Prediction table styling */
    .pred-table-wrapper { margin-top: -15px; }
    .pred-table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }
    .pred-table th {
        background-color: #667eea;
        color: white;
        padding: 10px;
        font-weight: 600;
        text-align: center;
        font-size: 14px;
    }
    .pred-table td {
        padding: 10px;
        text-align: center;
        color: #333;
        font-size: 14px;
        border: 1px solid #e0e0e0;
    }
    .pred-table tbody tr:nth-child(1) td:first-child { text-align: left; }
    .pred-table tbody tr:nth-child(2) td:first-child { text-align: left; }
    .pred-table tbody tr:nth-child(1) { background-color: #e8f5e9; }
    .pred-table tbody tr:nth-child(2) { background-color: #ffebee; }

    /* Metric cards styling */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 25px 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-left: 6px solid;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .metric-header {
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #333;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 8px 0;
        line-height: 1;
    }
    .metric-percentage {
        font-size: 20px;
        color: #666;
        margin-left: 8px;
        font-weight: 600;
    }
    .metric-description {
        font-size: 13px;
        color: #777;
        margin-top: 12px;
        line-height: 1.4;
    }
    .metric-level {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 12px;
        color: white;
    }
    .progress-bar {
        width: 100%;
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        margin-top: 12px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease;
    }

    /* Confusion Matrix breakdown cards */
    .cm-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 8px;
        border-left: 5px solid;
        height: 100%;
    }
    .cm-card-tn { border-left-color: #43e97b; }
    .cm-card-fp { border-left-color: #ffa726; }
    .cm-card-fn { border-left-color: #ff6b6b; }
    .cm-card-tp { border-left-color: #667eea; }
    .cm-label {
        font-size: 13px;
        font-weight: 600;
        color: #ddd;
        margin-bottom: 8px;
    }
    .cm-value {
        font-size: 32px;
        font-weight: bold;
        color: #fff;
    }
    .cm-desc {
        font-size: 11px;
        color: #aaa;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================================
# TITLE
# ====================================
st.markdown('''
<div style="text-align: center; padding: 20px 0 5px 0;">
    <h1 style="font-size: 38px; font-weight: 700; margin: 0; padding: 10px 0;">
        <span class="heartbeat">🫀</span>
        <span style="color: #FF4B4B;">Heart Disease Prediction System</span>
    </h1>
    <p style="font-size: 30px; color: #4facfe; font-weight: 600; margin-top: 10px; margin-bottom: 0;">
        🤖 Interactive Model Evaluation
    </p>
</div>
''', unsafe_allow_html=True)

# ====================================
# SIDEBAR
# ====================================
st.sidebar.title("⚙️ Configuration")

# Dataset Upload
st.sidebar.markdown("### 📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Model Selection
st.sidebar.markdown("### 🤖 Select Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODEL_OPTIONS.keys())
)

# Sidebar Footer
st.sidebar.markdown("<br>" * 3, unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='position: fixed; bottom: 20px; left: 20px; width: 300px; font-size: 15px; color: #b0b0b0; line-height: 1.6;'>
    <p style='margin: 0; font-weight: bold; color: #e0e0e0;'>Student:</p>
    <p style='margin: 0;'>Deepti Yashwant Walde</p>
    <p style='margin: 0;'>ID: 2025AA05185</p>
</div>
""", unsafe_allow_html=True)

# ====================================
# MAIN CONTENT
# ====================================
if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

        # Display dataset info
        display_dataset_info(test_data)

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
            has_target = 'target' in test_data.columns
            if has_target:
                X_test = test_data.drop('target', axis=1)
                y_test = test_data['target']
            else:
                X_test = test_data
                y_test = None

            # Scale data if needed
            if selected_model_name in MODELS_REQUIRING_SCALING:
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

                # ====================================
                # PREDICTION RESULTS
                # ====================================
                st.markdown("---")
                create_section_header(
                    "🎯",
                    "Prediction Results",
                    "The model analyzed your data and made predictions for each patient. Below is the distribution of predicted outcomes."
                )

                display_prediction_results(y_pred)

                # ====================================
                # METRICS (if target available)
                # ====================================
                if has_target and y_test is not None:
                    st.markdown("---")
                    create_section_header(
                        "📊",
                        "Model Performance Metrics",
                        "Since your dataset includes actual labels, we can evaluate how well the model performed. Each metric measures a different aspect of model quality."
                    )

                    accuracy, precision, recall, f1, auc, mcc = display_performance_metrics(
                        y_test, y_pred, y_pred_proba
                    )

                    # Metrics comparison chart
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 📊 Metrics Comparison Chart")
                    st.markdown("""
                    <p style="color: #888; font-size: 13px; margin: 5px 0 15px 0;">
                    Visual comparison of all performance metrics to quickly identify strengths and weaknesses.
                    </p>
                    """, unsafe_allow_html=True)

                    fig = create_metrics_comparison_chart(
                        accuracy, precision, recall, f1, auc, mcc, selected_model_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)  # Close figure

                    st.markdown("""
                    <p style="color: #888; font-size: 12px; font-style: italic; text-align: center; margin-top: 10px;">
                    Note: MCC is normalized from [-1, +1] to [0, 1] scale for visualization purposes. Actual MCC value is shown in the label.
                    </p>
                    """, unsafe_allow_html=True)

                    # ====================================
                    # CONFUSION MATRIX & CLASSIFICATION REPORT
                    # ====================================
                    st.markdown("---")
                    create_section_header(
                        "📉",
                        "Detailed Analysis",
                        "Dive deeper into model performance with confusion matrix and classification report. See how the model performs for each individual class (No Disease vs Disease)."
                    )

                    display_detailed_analysis(y_test, y_pred, selected_model_name)

                # Download predictions
                st.markdown("---")
                st.markdown("### 📥 Download Predictions")

                result_df = test_data.copy()
                result_df['Predicted_Class'] = y_pred
                result_df['Prediction_Probability'] = y_pred_proba
                result_df['Predicted_Label'] = result_df['Predicted_Class'].map({
                    0: 'No Disease',
                    1: 'Disease'
                })
                result_df['Confidence_Level'] = result_df['Prediction_Probability'].apply(get_confidence_level)
                result_df['Risk_Category'] = result_df['Prediction_Probability'].apply(get_risk_category)

                if has_target:
                    result_df['Prediction_Match'] = (result_df['target'] == result_df['Predicted_Class']).map({
                        True: 'Correct',
                        False: 'Incorrect'
                    })

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
    # No file uploaded - show instructions
    st.markdown('''
    <div style="background: linear-gradient(135deg, #667eea 0%, #ffffff 100%);
                padding: 20px 30px; border-radius: 15px;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); margin: 10px 0 20px 0;">
        <p style="color: #2d1b4e; font-size: 18px; margin: 0; font-weight: 500;">
            👈 Upload a CSV file from the sidebar to get started with predictions
        </p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📚 Understanding Heart Disease Prediction")

    st.markdown("""
    This application uses machine learning to predict the presence of heart disease in patients based on various clinical features.
    The models have been trained on a comprehensive dataset containing medical test results and patient demographics.
    """)

    # Educational expanders
    with st.expander("🫀 What is Heart Disease Classification?"):
        st.markdown("""
        **Heart disease classification** is a binary prediction problem where the goal is to determine whether a patient has heart disease or not based on medical features.

        **Input Features Include:**
        - Age, gender, and vital signs (blood pressure, heart rate)
        - Cholesterol levels and blood sugar readings
        - ECG results and exercise-induced symptoms
        - Blood vessel imaging results

        **Output:**
        - **Class 0:** No heart disease detected
        - **Class 1:** Heart disease present

        Early prediction helps doctors make timely interventions and potentially save lives.
        """)

    with st.expander("🤖 Understanding the Machine Learning Models"):
        st.markdown("""
        This system implements **6 different classification algorithms**, each with unique strengths:

        ### Basic Models:

        **1. Logistic Regression**
        - A linear model that calculates probability of disease
        - Fast, interpretable, works well for linearly separable data
        - Good baseline model for comparison
        - **Best when:** You need interpretability and features have linear relationships

        **2. Decision Tree**
        - Creates a tree of yes/no questions about patient features
        - Easy to understand and visualize decision paths
        - Can capture non-linear patterns in data
        - **Best when:** You need explainable decisions and have categorical features

        **3. K-Nearest Neighbors (kNN)**
        - Classifies based on similarity to nearest training examples
        - "If 5 similar patients had disease, this patient likely does too"
        - Works well when similar cases have similar outcomes
        - **Best when:** You have sufficient training data and features are scaled

        **4. Naive Bayes**
        - Uses probability theory (Bayes' theorem) for classification
        - Fast and efficient, especially for large datasets
        - Assumes features are independent
        - **Best when:** Features are relatively independent and you need fast predictions

        ### Ensemble Models (Often More Powerful):

        **5. Random Forest**
        - Combines predictions from multiple decision trees
        - Like asking 100 doctors and taking majority vote
        - Reduces overfitting and improves stability
        - **Best when:** You have enough data and want to reduce variance

        **6. XGBoost**
        - Advanced gradient boosting algorithm
        - Builds trees sequentially, each correcting previous errors
        - Often achieves high accuracy in competitions
        - **Best when:** You have time to tune parameters and want maximum performance

        ### Important Note:
        **There is no "best" model for all datasets!** Model performance depends on:
        - Dataset size and quality
        - Feature types and relationships
        - Amount of noise in data
        - Whether problem is linearly separable
        - Computational resources available

        **Best Practice:** Try multiple models and compare their performance on YOUR specific data. The "best" model is the one that performs best on your validation set!
        """)

    with st.expander("📊 Understanding Evaluation Metrics"):
        st.markdown("""
        After making predictions, we evaluate model performance using these metrics:

        ### 🎯 Accuracy
        **What it means:** Percentage of correct predictions out of all predictions

        **Formula:** (Correct Predictions) / (Total Predictions)

        **Example:** If model predicts correctly for 85 out of 100 patients → Accuracy = 85%

        **When to use:** Good for balanced datasets where both classes are equally important

        ---

        ### 🔍 Precision
        **What it means:** Of all patients predicted to have disease, how many actually have it?

        **Formula:** (True Positives) / (True Positives + False Positives)

        **Example:** Model says 100 patients have disease, but only 80 actually do → Precision = 80%

        **When to use:** Important when false alarms are costly (unnecessary treatments, anxiety)

        ---

        ### 🎪 Recall (Sensitivity)
        **What it means:** Of all patients who actually have disease, how many did we catch?

        **Formula:** (True Positives) / (True Positives + False Negatives)

        **Example:** Out of 100 patients with disease, we detected 90 → Recall = 90%

        **When to use:** Critical in medical diagnosis where missing a disease is dangerous

        ---

        ### ⚖️ F1 Score
        **What it means:** Harmonic mean of Precision and Recall (balanced metric)

        **Formula:** 2 × (Precision × Recall) / (Precision + Recall)

        **When to use:** When you need balance between precision and recall

        ---

        ### 📈 AUC Score (Area Under ROC Curve)
        **What it means:** Measures model's ability to distinguish between classes

        **Range:** 0.5 (random guessing) to 1.0 (perfect classification)

        **Example:** AUC = 0.92 means 92% chance model ranks a random disease patient higher than a healthy patient

        **When to use:** Evaluating overall model discrimination ability

        ---

        ### 🔢 MCC (Matthews Correlation Coefficient)
        **What it means:** Correlation between predicted and actual classifications

        **Range:** -1 (total disagreement) to +1 (perfect prediction)

        **Why special:** Works well even with imbalanced datasets

        **When to use:** When you want a single, balanced metric that considers all four confusion matrix values
        """)

    with st.expander("🎨 Understanding Confusion Matrix"):
        st.markdown("""
        A confusion matrix is a table showing four types of prediction outcomes:

        ```
                         Predicted: Negative    Predicted: Positive
        Actually Negative:      TN                    FP
        Actually Positive:      FN                    TP
        ```

        ### Four Outcomes Explained:

        **✅ True Negatives (TN)**
        - Model correctly predicted the negative class (e.g., No Disease)
        - **Reality:** Patient is healthy
        - **Prediction:** Model said healthy
        - **Good outcome:** Correct prediction

        **⚡ False Positives (FP) - Type I Error**
        - Model incorrectly predicted the positive class
        - **Reality:** Patient is healthy
        - **Prediction:** Model said disease
        - **Problem:** False alarm - may cause unnecessary worry, tests, or treatment

        **⚠️ False Negatives (FN) - Type II Error**
        - Model incorrectly predicted the negative class
        - **Reality:** Patient has disease
        - **Prediction:** Model said healthy
        - **Problem:** Missed diagnosis - VERY DANGEROUS in medical context!

        **🎯 True Positives (TP)**
        - Model correctly predicted the positive class (e.g., Disease)
        - **Reality:** Patient has disease
        - **Prediction:** Model said disease
        - **Good outcome:** Successfully detected the disease

        ### How to Interpret:
        - **High TP and TN:** Model is performing well on both classes
        - **Low FN:** Critical for medical diagnosis (don't miss diseases!)
        - **Low FP:** Reduces false alarms and patient anxiety
        - **Perfect model:** Only TP and TN, no FP or FN

        ### Example Interpretation:
        If confusion matrix shows 400 TN, 100 FP, 50 FN, and 450 TP:
        - Model correctly identified 400 healthy patients (TN)
        - Model wrongly flagged 100 healthy patients as sick (FP)
        - Model missed 50 sick patients (FN) - concerning!
        - Model caught 450 sick patients correctly (TP)
        - **Accuracy:** (400+450)/1000 = 85%
        """)

    with st.expander("🚀 How to Use This Application"):
        st.markdown("""
        ### Step-by-Step Guide:

        **Step 1: Prepare Your Data**
        - Download the sample CSV template below
        - Ensure your data has all 13 required features
        - Format: Each row = one patient, each column = one feature

        **Step 2: Upload Your File**
        - Click the file uploader in the left sidebar
        - Select your CSV file
        - The app will show dataset statistics

        **Step 3: Select a Model**
        - Choose from 6 available classification models
        - Each model has different strengths
        - Try different models to compare results

        **Step 4: View Results**
        - **Prediction Distribution:** See how many patients predicted with/without disease
        - **Metrics Dashboard:** Evaluate model performance with 6 metrics
        - **Confusion Matrix:** Understand prediction accuracy in detail
        - **Classification Report:** See per-class performance

        **Step 5: Download Results**
        - Export predictions with probability scores
        - Save for further analysis or reporting

        ### Tips:
        - Start with Logistic Regression (simple baseline)
        - Try ensemble models (Random Forest, XGBoost) for best accuracy
        - Compare metrics across different models
        - Pay attention to Recall for medical diagnosis (don't miss diseases!)
        """)

    # Sample data and template
    training_data = load_training_info()
    if training_data is not None:
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

        with st.expander("👁️ View Sample Training Data"):
            st.dataframe(training_data.head(20), height=400, use_container_width=True)

        st.markdown("### 📥 Download Sample Template")
        sample_template = training_data.drop('target', axis=1).head(5)
        csv = sample_template.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv,
            file_name="sample_template.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: gray; font-size: 16px; font-weight: bold; margin: 0;'>
        ML Assignment 2 - Heart Disease Classification System
    </p>
    <p style='color: gray; font-size: 14px; margin-top: 10px;'>
        BITS Pilani - M.Tech (AIML/DSE) | Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)
