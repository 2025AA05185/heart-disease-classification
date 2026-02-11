"""
ML Assignment 2 - Model Training Script
Implements 6 classification models and calculates 6 evaluation metrics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import evaluation metrics
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

import pickle
import os

# Create model directory
os.makedirs('model', exist_ok=True)

print("="*80)
print("ML ASSIGNMENT 2 - MODEL TRAINING")
print("="*80)

# ========================
# 1. LOAD DATASET
# ========================
print("\n[1/5] Loading dataset...")
df = pd.read_csv('heart_disease.csv')
print(f"✓ Dataset loaded: {df.shape}")
print(f"  Features: {df.shape[1] - 1}")
print(f"  Instances: {df.shape[0]}")
print(f"  Target distribution:\n{df['target'].value_counts()}")

# ========================
# 2. PREPARE DATA
# ========================
print("\n[2/5] Preparing data...")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

print(f"✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# ========================
# 3. DEFINE MODELS
# ========================
print("\n[3/5] Initializing 6 classification models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

print(f"✓ {len(models)} models initialized")

# ========================
# 4. TRAIN & EVALUATE
# ========================
print("\n[4/5] Training models and calculating metrics...")
print("-"*80)

results = []

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")

    # Train model
    if model_name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
        # Use scaled data for these models
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Use original data for tree-based models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate 6 metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })

    # Print metrics
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")

    # Save model
    model_filename = f"model/{model_name.lower().replace(' ', '_')}_model.pkl"
    pickle.dump(model, open(model_filename, 'wb'))
    print(f"  ✓ Model saved: {model_filename}")

# ========================
# 5. RESULTS SUMMARY
# ========================
print("\n" + "="*80)
print("[5/5] RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model/model_results.csv', index=False)
print("\n✓ Results saved to 'model/model_results.csv'")

# Find best model
best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
best_auc = results_df.loc[results_df['AUC'].idxmax()]
best_f1 = results_df.loc[results_df['F1'].idxmax()]

print("\n" + "="*80)
print("BEST PERFORMING MODELS")
print("="*80)
print(f"Best Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
print(f"Best AUC:       {best_auc['Model']} ({best_auc['AUC']:.4f})")
print(f"Best F1 Score:  {best_f1['Model']} ({best_f1['F1']:.4f})")

print("\n" + "="*80)
print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll models saved in 'model/' directory")
print("Ready for Streamlit app development!")
