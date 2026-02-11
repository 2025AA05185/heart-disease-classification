# Heart Disease Classification - ML Assignment 2

## 🎯 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and accurate prediction of heart disease can significantly improve patient outcomes and enable timely medical intervention. This project aims to develop and compare multiple machine learning classification models to predict the presence of heart disease in patients based on various clinical and demographic features.

The primary objective is to:
- Build a robust classification system that can accurately predict heart disease
- Compare the performance of six different machine learning algorithms
- Deploy an interactive web application for real-time predictions
- Provide healthcare professionals with a tool to assist in diagnostic decision-making

By leveraging machine learning techniques, this system helps identify high-risk patients and supports medical practitioners in making data-driven decisions for better patient care.

---

## 📊 Dataset Description

### Dataset Overview
- **Name:** Heart Disease Dataset
- **Source:** Kaggle (originally from UCI Machine Learning Repository)
- **Problem Type:** Binary Classification
- **Total Instances:** 1,025 samples
- **Total Features:** 13 clinical and demographic features
- **Target Variable:** `target` (0 = No Heart Disease, 1 = Heart Disease Present)
- **Class Distribution:**
  - Class 0 (No Disease): 499 samples (48.7%)
  - Class 1 (Disease): 526 samples (51.3%)

### Feature Descriptions

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| **age** | Age of the patient in years | Numeric | 29-80 years |
| **sex** | Gender of the patient | Binary | 0 = Female, 1 = Male |
| **cp** | Chest pain type | Categorical | 0-3 (4 types) |
| **trestbps** | Resting blood pressure (mm Hg) | Numeric | 90-200 mm Hg |
| **chol** | Serum cholesterol (mg/dl) | Numeric | 120-570 mg/dl |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0 = False, 1 = True |
| **restecg** | Resting electrocardiographic results | Categorical | 0-2 (3 types) |
| **thalach** | Maximum heart rate achieved | Numeric | 70-210 bpm |
| **exang** | Exercise induced angina | Binary | 0 = No, 1 = Yes |
| **oldpeak** | ST depression induced by exercise | Numeric | 0.0-6.2 |
| **slope** | Slope of peak exercise ST segment | Categorical | 0-2 (3 types) |
| **ca** | Number of major vessels colored by fluoroscopy | Numeric | 0-3 vessels |
| **thal** | Thalassemia | Categorical | 0-3 (4 types) |

### Data Preprocessing
- **Train-Test Split:** 80% training (820 samples), 20% testing (205 samples)
- **Feature Scaling:** StandardScaler applied for distance-based models (Logistic Regression, kNN, Naive Bayes)
- **Missing Values:** None (dataset is complete)
- **Class Imbalance:** Dataset is well-balanced (48.7% vs 51.3%)

---

## 🤖 Models Used

Six classification algorithms were implemented and compared on the Heart Disease dataset:

### 1. **Logistic Regression**
A linear model that uses the logistic function to model binary outcomes. It's interpretable and works well for linearly separable data.

### 2. **Decision Tree**
A tree-based model that makes decisions by learning simple decision rules from features. Maximum depth was limited to 5 to prevent overfitting.

### 3. **K-Nearest Neighbors (kNN)**
An instance-based learning algorithm that classifies samples based on the majority class of their k=5 nearest neighbors.

### 4. **Naive Bayes**
A probabilistic classifier based on Bayes' theorem with the assumption of feature independence. Uses Gaussian distribution for continuous features.

### 5. **Random Forest (Ensemble)**
An ensemble method that builds multiple decision trees (100 estimators) and combines their predictions through majority voting.

### 6. **XGBoost (Ensemble)**
An advanced gradient boosting algorithm that builds trees sequentially, with each tree correcting errors of previous trees.

---

## 📈 Model Performance Comparison

The table below shows the performance of all six models across six evaluation metrics:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--------------|:---------|:----|:----------|:-------|:---|:----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.8732 | 0.9326 | 0.8624 | 0.8952 | 0.8785 | 0.7465 |
| kNN | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Bold values** indicate the best performance for each metric.

### Metric Definitions

- **Accuracy:** Overall correctness of the model (correctly predicted / total predictions)
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Ratio of correct positive predictions to total positive predictions (important for minimizing false positives)
- **Recall:** Ratio of correct positive predictions to actual positives (important for minimizing false negatives)
- **F1 Score:** Harmonic mean of precision and recall (balanced metric)
- **MCC (Matthews Correlation Coefficient):** Correlation between predicted and actual classes, ranges from -1 to +1

---

## 🔍 Observations on Model Performance

### 1. Logistic Regression
**Performance:** Strong baseline performance with 80.98% accuracy and 0.9298 AUC score.

**Observations:**
- Demonstrates solid performance as a linear classifier
- Achieves good balance between precision (0.7619) and recall (0.9143)
- Exceptional recall (91.43%) is valuable in medical diagnosis, minimizing false negatives
- High AUC of 0.9298 indicates excellent discrimination ability
- F1 score of 0.8312 shows good overall balance
- MCC of 0.6309 indicates substantial correlation with actual outcomes
- Computationally efficient and interpretable, suitable for clinical settings
- Well-suited for this dataset with linear decision boundaries

**Clinical Significance:** High recall minimizes risk of missing actual heart disease cases.

---

### 2. Decision Tree
**Performance:** Strong performance with 87.32% accuracy, significantly better than simple linear models.

**Observations:**
- Achieves 87.32% accuracy with interpretable tree structure
- Excellent AUC score (0.9326) shows good ranking ability
- High precision (0.8624) and recall (0.8952) indicate balanced performance
- MCC score of 0.7465 is the second-best among all models
- Tree structure provides clear decision rules for clinicians
- Maximum depth limitation (max_depth=5) prevents overfitting effectively
- Good balance between model complexity and performance

**Advantage:** Provides interpretable decision paths for medical practitioners.

---

### 3. K-Nearest Neighbors (kNN)
**Performance:** Very strong performance with 86.34% accuracy and highest AUC among non-ensemble models.

**Observations:**
- Achieves impressive AUC of 0.9629, best among traditional models
- Excellent precision (0.8738) minimizes false positives
- Good recall (0.8571) ensures most disease cases are detected
- Distance-based approach benefits significantly from feature scaling
- Performance with k=5 neighbors is well-optimized for this dataset
- MCC of 0.7269 indicates strong correlation with actual outcomes
- Instance-based learning captures local patterns effectively

**Trade-off:** Higher computational cost during prediction but excellent performance.

---

### 4. Naive Bayes
**Performance:** Good performance with 82.93% accuracy and solid generalization.

**Observations:**
- Achieves 82.93% accuracy despite naive independence assumption
- Strong recall (0.8762) ensures good disease detection
- AUC of 0.9043 shows good discrimination ability
- Very fast training and prediction times
- Probabilistic nature provides confidence scores for predictions
- MCC of 0.6602 indicates moderate-to-strong correlation
- Robust to irrelevant features due to probabilistic framework

**Strength:** Excellent balance between speed and performance for real-time applications.

---

### 5. Random Forest (Ensemble)
**Performance:** **Perfect performance** with 100% accuracy across all metrics - likely overfitting on test set.

**Observations:**
- Achieves perfect scores (1.0000) across all six metrics
- Perfect accuracy, precision, recall, F1, and MCC scores
- Combines predictions from 100 decision trees
- The perfect performance suggests possible overfitting or memorization
- Provides feature importance rankings for interpretability
- Handles non-linear relationships and feature interactions excellently
- Robust ensemble method that reduces variance

**Caution:** Perfect scores may indicate overfitting. Cross-validation recommended for production deployment.

**Advantage:** Provides feature importance scores identifying critical health indicators.

---

### 6. XGBoost (Ensemble)
**Performance:** **Perfect performance** with 100% accuracy across all metrics - likely overfitting on test set.

**Observations:**
- Achieves perfect scores (1.0000) across all six metrics
- Advanced gradient boosting with regularization
- Perfect classification on the test set
- Handles missing values and feature interactions automatically
- Provides built-in feature importance for interpretability
- The perfect performance suggests possible overfitting or data leakage
- State-of-the-art algorithm optimized for structured data

**Caution:** Perfect scores warrant investigation. Recommend cross-validation and larger test set.

**Insight:** Both ensemble models (Random Forest and XGBoost) achieving perfect scores suggests they may have memorized the test set patterns, indicating potential overfitting.

---

## 🏆 Overall Model Ranking

Based on the comprehensive evaluation:

1. **🥇 Random Forest & XGBoost (Tie)** - Perfect performance (1.0000 across all metrics) - but caution for overfitting
2. **🥈 Decision Tree** - Excellent balanced performance (Accuracy: 0.8732, MCC: 0.7465)
3. **🥉 kNN** - Very strong performance (Accuracy: 0.8634, AUC: 0.9629)
4. **Naive Bayes** - Good performance (Accuracy: 0.8293)
5. **Logistic Regression** - Solid baseline (Accuracy: 0.8098, excellent recall: 0.9143)

---

## 🚀 Deployment & Usage

### Interactive Web Application

The project includes an interactive Streamlit web application with the following features:

#### Features:
1. **📁 CSV Upload:** Upload test datasets for batch predictions
2. **🤖 Model Selection:** Choose from 6 different classification models
3. **📊 Metrics Dashboard:** View all 6 evaluation metrics with visual indicators
4. **📈 Confusion Matrix:** Visualize model performance with heatmap
5. **📋 Classification Report:** Detailed precision, recall, and F1 scores per class
6. **💾 Download Results:** Export predictions to CSV format

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Deployment to Streamlit Cloud

1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

---

## 📁 Project Structure

```
heart-disease-classification/
├── app.py                          # Streamlit web application
├── model_training.py               # Model training script
├── heart_disease.csv               # Dataset (1,025 samples)
├── test_data.csv                   # Sample test data
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
└── model/                          # Trained models
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── knn_model.pkl
    ├── naive_bayes_model.pkl
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    ├── scaler.pkl
    └── model_results.csv
```

---

## 🔧 Technologies Used

- **Python 3.8+**
- **Scikit-learn:** ML models and metrics
- **XGBoost:** Gradient boosting
- **Pandas & NumPy:** Data manipulation
- **Streamlit:** Web application framework
- **Matplotlib & Seaborn:** Data visualization

---

## 📝 Assignment Details

- **Course:** Machine Learning - M.Tech (AIML)
- **Institution:** BITS Pilani
- **Assignment:** Assignment 2
- **Total Marks:** 15
- **Submission Deadline:** 15-Feb-2026

---

## 👨‍💻 Author

**Student Name:** Deepti Yashwant Walde
**Student ID:** 2025AA05185
**Program:** Machine Learning - M.Tech (AIML)
**Institution:** BITS Pilani

---

## 📚 References

1. Heart Disease Dataset - Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. UCI Machine Learning Repository - Heart Disease Database
3. Scikit-learn Documentation: https://scikit-learn.org/
4. XGBoost Documentation: https://xgboost.readthedocs.io/
5. Streamlit Documentation: https://docs.streamlit.io/
