"""
train_model.py
==============
Automated end-to-end Machine Learning Pipeline for Diabetes Risk Prediction.

Capabilities:
1. Automated Dataset Discovery & Inspection (Features, Target, Types, Missing, Duplicates, Medically Invalid Values).
2. Data Cleaning & Preprocessing (Zero-value handling, Imputation, Scaling, Encoding).
3. Leakage-free Scikit-Learn Pipeline construction.
4. Model Training & Comparison (Logistic Regression, Decision Tree, Random Forest, KNN, SVM).
5. Comprehensive Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
6. Automatic Best Model Selection & Joblib Serialization.
7. Rich Visualizations with Matplotlib & Seaborn.
8. Metadata Export (JSON) for dynamic Flask UI & Chart.js Dashboard.
"""

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Styling for saved figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True


def find_dataset(dataset_dir="dataset"):
    """Find the CSV dataset in the dataset directory."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory '{dataset_dir}' does not exist.")
    
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in '{dataset_dir}'. Please place your Kaggle dataset CSV there."
        )
    
    # Prioritize 'diabetes.csv' if multiple exist, else take the first
    for f in csv_files:
        if "diabetes" in os.path.basename(f).lower():
            return f
    return csv_files[0]


def inspect_and_identify_target(df):
    """
    Inspects columns and identifies target/output column.
    Throws informative error if target cannot be identified with confidence.
    """
    candidate_names = [
        "outcome", "diabetes", "target", "class", "label",
        "diabetic", "diagnosis", "diabetes_risk", "diabetes_binary"
    ]
    
    col_lower_map = {col.lower().strip(): col for col in df.columns}
    target_col = None
    
    # 1. Check exact or partial name matches
    for candidate in candidate_names:
        for low_name, actual_name in col_lower_map.items():
            if candidate == low_name or (len(candidate) > 4 and candidate in low_name):
                target_col = actual_name
                break
        if target_col:
            break
            
    # 2. If not found by name, check if last column is binary or few classes
    if not target_col:
        last_col = df.columns[-1]
        unique_vals = df[last_col].dropna().unique()
        if len(unique_vals) <= 3:
            target_col = last_col
            
    if not target_col:
        cols_summary = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Could not confidently detect target column from available columns: [{cols_summary}]. "
            "Please rename your target column to 'Outcome' or 'Diabetes' or specify it."
        )
        
    return target_col


def clean_and_inspect_data(df, target_col):
    """
    Inspects missing values, duplicates, medically invalid zeros,
    and returns cleaned dataframe along with inspection summary.
    """
    raw_rows = len(df)
    
    # 1. Duplicates
    duplicate_count = int(df.duplicated().sum())
    df_clean = df.drop_duplicates().copy()
    cleaned_rows = len(df_clean)
    
    # 2. Identify numerical and categorical columns (excluding Pregnancies for universal male/female suitability)
    feature_cols = [c for c in df_clean.columns if c != target_col and "pregnan" not in c.lower()]
    numerical_cols = []
    categorical_cols = []
    
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_clean[c]):
            # If numeric but has very few string-like discrete values, still keep numeric if continuous
            numerical_cols.append(c)
        else:
            categorical_cols.append(c)
            
    # 3. Handle Medically Impossible Zeros
    # In clinical datasets (like Pima Indians Diabetes), 0 in Glucose, BP, SkinThickness,
    # Insulin, and BMI represent unrecorded/missing data.
    zero_replace_keywords = ["glucose", "bloodpressure", "pressure", "bp", "skinthickness", "skin", "insulin", "bmi"]
    invalid_zeros_found = {}
    
    for c in numerical_cols:
        col_clean_name = c.lower().replace("_", "").replace(" ", "")
        for kw in zero_replace_keywords:
            if kw in col_clean_name:
                zero_count = int((df_clean[c] == 0).sum())
                if zero_count > 0:
                    invalid_zeros_found[c] = zero_count
                    # Replace 0 with NaN so the pipeline imputer can handle it cleanly
                    df_clean.loc[df_clean[c] == 0, c] = np.nan
                break

    # 4. Target inspection and standardization to binary (0 and 1)
    # Handle string target (e.g. 'Yes'/'No', 'Positive'/'Negative', 'Diabetic'/'Non-Diabetic')
    y_raw = df_clean[target_col]
    unique_targets = y_raw.dropna().unique().tolist()
    
    target_mapping = {}
    if y_raw.dtype == object or str(y_raw.dtype) == 'category':
        mapping_dict = {}
        for val in unique_targets:
            s_val = str(val).strip().lower()
            if s_val in ['1', 'yes', 'positive', 'diabetic', 'true']:
                mapping_dict[val] = 1
            else:
                mapping_dict[val] = 0
        df_clean[target_col] = df_clean[target_col].map(mapping_dict)
        target_mapping = {str(k): int(v) for k, v in mapping_dict.items()}
    else:
        df_clean[target_col] = df_clean[target_col].astype(int)
        
    class_counts = df_clean[target_col].value_counts().to_dict()
    non_diabetic_count = int(class_counts.get(0, 0))
    diabetic_count = int(class_counts.get(1, 0))
    
    inspection_report = {
        "raw_record_count": raw_rows,
        "cleaned_record_count": cleaned_rows,
        "duplicate_records_removed": duplicate_count,
        "target_column": target_col,
        "target_mapping": target_mapping,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_values_by_col": df.isna().sum().to_dict(),
        "invalid_zeros_replaced_with_nan": invalid_zeros_found,
        "class_distribution": {
            "Non-Diabetic (0)": non_diabetic_count,
            "Diabetic (1)": diabetic_count
        }
    }
    
    return df_clean, inspection_report, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """
    Builds a leakage-free Scikit-Learn ColumnTransformer preprocessor.
    """
    transformers = []
    
    if numerical_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, numerical_cols))
        
    if categorical_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor


def get_feature_metadata(df, numerical_cols, categorical_cols):
    """
    Generates rich feature metadata (min, max, median, mean, units, labels, descriptions)
    to dynamically build the UI prediction form.
    """
    metadata = {}
    
    # Domain-aware metadata dictionary for common diabetes metrics
    info_map = {
        "pregnancies": {"label": "Pregnancies", "unit": "Count", "desc": "Number of times pregnant", "min": 0, "max": 20, "step": 1},
        "glucose": {"label": "Glucose Level", "unit": "mg/dL", "desc": "Plasma glucose concentration (2 hours in an OGTT)", "min": 50, "max": 250, "step": 1},
        "bloodpressure": {"label": "Blood Pressure", "unit": "mm Hg", "desc": "Diastolic blood pressure", "min": 40, "max": 140, "step": 1},
        "skinthickness": {"label": "Skin Thickness", "unit": "mm", "desc": "Triceps skin fold thickness", "min": 5, "max": 100, "step": 1},
        "insulin": {"label": "Insulin Level", "unit": "μU/mL", "desc": "2-Hour serum insulin level", "min": 10, "max": 900, "step": 1},
        "bmi": {"label": "Body Mass Index (BMI)", "unit": "kg/m²", "desc": "Body mass index = weight in kg / (height in m)²", "min": 10.0, "max": 65.0, "step": 0.1},
        "diabetespedigreefunction": {"label": "Diabetes Pedigree Function", "unit": "Score", "desc": "Genetic diabetes risk score based on family history", "min": 0.05, "max": 2.50, "step": 0.01},
        "age": {"label": "Age", "unit": "Years", "desc": "Patient age in years", "min": 18, "max": 120, "step": 1}
    }
    
    for col in numerical_cols:
        clean_key = col.lower().replace("_", "").replace(" ", "")
        matched_info = None
        for k, v in info_map.items():
            if k in clean_key:
                matched_info = v
                break
                
        series = df[col].dropna()
        col_min = float(series.min()) if not series.empty else 0.0
        col_max = float(series.max()) if not series.empty else 100.0
        col_median = float(series.median()) if not series.empty else 50.0
        col_mean = float(series.mean()) if not series.empty else 50.0
        
        label = matched_info["label"] if matched_info else col.replace("_", " ").title()
        unit = matched_info["unit"] if matched_info else ""
        desc = matched_info["desc"] if matched_info else f"Measured value for {label}"
        step = matched_info["step"] if matched_info else (0.1 if series.dtype == float else 1)
        
        metadata[col] = {
            "name": col,
            "type": "numerical",
            "label": label,
            "unit": unit,
            "desc": desc,
            "min": round(col_min, 2),
            "max": round(col_max, 2),
            "median": round(col_median, 2),
            "mean": round(col_mean, 2),
            "step": step,
            "default_value": round(col_median, 2)
        }
        
    for col in categorical_cols:
        series = df[col].dropna()
        unique_vals = [str(x) for x in series.unique()]
        label = col.replace("_", " ").title()
        metadata[col] = {
            "name": col,
            "type": "categorical",
            "label": label,
            "desc": f"Category for {label}",
            "options": unique_vals,
            "default_value": unique_vals[0] if unique_vals else ""
        }
        
    return metadata


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Trains and compares:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. K-Nearest Neighbors
    5. Support Vector Machine
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    }
    
    results = {}
    fitted_pipelines = {}
    
    print("\n" + "="*70)
    print("                 MODEL TRAINING & EVALUATION REPORT")
    print("="*70)
    
    for name, estimator in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Predicted probabilities
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc_val = float(roc_auc_score(y_test, y_proba))
        else:
            auc_val = 0.0
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc_val, 4),
            "confusion_matrix": cm,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist() if hasattr(pipe, "predict_proba") else []
        }
        fitted_pipelines[name] = pipe
        
        print(f"\n[{name}]")
        print(f"  Accuracy : {acc * 100:.2f}% | ROC-AUC : {auc_val:.4f}")
        print(f"  Precision: {prec * 100:.2f}% | Recall  : {rec * 100:.2f}% | F1-Score: {f1:.4f}")
        print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Select best model: prioritize ROC-AUC, then F1-score and Accuracy
    best_model_name = max(
        results.keys(),
        key=lambda m: (results[m]["roc_auc"] * 0.5 + results[m]["f1_score"] * 0.3 + results[m]["accuracy"] * 0.2)
    )
    
    print("\n" + "="*70)
    print(f"[*] SELECTED BEST MODEL: {best_model_name}")
    print(f"  Accuracy : {results[best_model_name]['accuracy'] * 100:.2f}%")
    print(f"  ROC-AUC  : {results[best_model_name]['roc_auc']:.4f}")
    print(f"  Recall   : {results[best_model_name]['recall'] * 100:.2f}%")
    print(f"  F1-Score : {results[best_model_name]['f1_score']:.4f}")
    print("="*70)
    
    return results, fitted_pipelines, best_model_name


def extract_feature_importance(pipeline, feature_names):
    """Extracts feature importance or coefficients if available."""
    clf = pipeline.named_steps['classifier']
    importances = {}
    
    if hasattr(clf, "feature_importances_"):
        raw_imp = clf.feature_importances_
        for f, imp in zip(feature_names, raw_imp):
            importances[f] = float(imp)
    elif hasattr(clf, "coef_"):
        raw_coef = np.abs(clf.coef_[0])
        total = np.sum(raw_coef) if np.sum(raw_coef) > 0 else 1.0
        norm_coef = raw_coef / total
        for f, imp in zip(feature_names, norm_coef):
            importances[f] = float(imp)
    else:
        # Fallback uniform
        for f in feature_names:
            importances[f] = round(1.0 / len(feature_names), 4)
            
    # Sort descending
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    return sorted_importances


def save_chart(fig, base_name, output_dir):
    """Saves both high-resolution PNG and lightweight WebP versions."""
    png_path = os.path.join(output_dir, f"{base_name}.png")
    webp_path = os.path.join(output_dir, f"{base_name}.webp")
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    try:
        fig.savefig(webp_path, dpi=160, bbox_inches='tight')
    except Exception:
        pass


def generate_eda_and_charts(df_clean, target_col, results, best_model_name,
                            fitted_pipeline, numerical_cols, feature_names,
                            output_dir="static/images"):
    """
    Generates optimized publication charts using Matplotlib & Seaborn.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Diabetes Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df_clean[target_col].value_counts()
    labels = ['Non-Diabetic (0)', 'Diabetic (1)']
    colors = ['#2563eb', '#ef4444']
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, width=0.5, edgecolor='none', alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 5, f'{int(h)} ({h/len(df_clean)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_title('Diabetes Class Distribution in Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Patient Count', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(counts) * 1.15)
    save_chart(fig, 'diabetes_distribution', output_dir)
    plt.close(fig)

    # 2. Key Numerical Distributions (Age, BMI, Glucose, etc. if available)
    plot_cols = [c for c in ['Glucose', 'BMI', 'Age', 'BloodPressure'] if c in df_clean.columns]
    if not plot_cols and numerical_cols:
        plot_cols = numerical_cols[:4]
        
    if plot_cols:
        n_plots = len(plot_cols)
        fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.2))
        if n_plots == 1:
            axes = [axes]
        for ax, col in zip(axes, plot_cols):
            sns.kdeplot(data=df_clean, x=col, hue=target_col, fill=True, common_norm=False,
                        palette={0: '#2563eb', 1: '#ef4444'}, ax=ax, alpha=0.4)
            ax.set_title(f'{col} by Outcome', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
        fig.suptitle('Key Physiological Feature Distributions by Diabetes Outcome', fontsize=14, fontweight='bold', y=1.03)
        save_chart(fig, 'feature_distributions', output_dir)
        plt.close(fig)

    # 3. Correlation Heatmap
    if len(numerical_cols) > 1:
        corr_cols = [c for c in numerical_cols if c in df_clean.columns]
        if target_col in df_clean.columns and df_clean[target_col].dtype in [int, float, np.int64, np.float64]:
            corr_cols = corr_cols + [target_col]
        corr_matrix = df_clean[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7.5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=0.5, linecolor='#f1f5f9', ax=ax, cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        save_chart(fig, 'correlation_heatmap', output_dir)
        plt.close(fig)

    # 4. Feature Importance
    importances = extract_feature_importance(fitted_pipeline, feature_names)
    fig, ax = plt.subplots(figsize=(8, 5))
    feats = list(importances.keys())
    scores = list(importances.values())
    y_pos = np.arange(len(feats))
    ax.barh(y_pos, scores, color='#0ea5e9', edgecolor='none', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=10, fontweight='medium')
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance Score', fontsize=11)
    ax.set_title(f'Feature Importance ({best_model_name})', fontsize=14, fontweight='bold', pad=15)
    for i, v in enumerate(scores):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, color='#0f172a')
    ax.set_xlim(0, max(scores) * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    save_chart(fig, 'feature_importance', output_dir)
    plt.close(fig)

    # 5. Confusion Matrix for Best Model
    best_cm = np.array(results[best_model_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Pred Non-Diabetic (0)', 'Pred Diabetic (1)'],
                yticklabels=['Actual Non-Diabetic (0)', 'Actual Diabetic (1)'],
                annot_kws={"size": 14, "fontweight": "bold"}, ax=ax)
    ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold', pad=12)
    save_chart(fig, 'confusion_matrix', output_dir)
    plt.close(fig)

    # 6. Model Performance Comparison
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    x = np.arange(len(model_names))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
    
    for i, (m, label, col) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [results[m_name][m] for m_name in model_names]
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=col)
        
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11)
    ax.set_title('Cross-Algorithm Performance Benchmark', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    save_chart(fig, 'model_comparison', output_dir)
    plt.close(fig)
    
    print(f"\nAll 6 high-resolution charts saved to '{output_dir}/'.")


def main():
    print("="*70)
    print("  DIABETES RISK PREDICTION — MACHINE LEARNING TRAINING PIPELINE")
    print("="*70)
    
    # 1. Dataset Discovery
    dataset_path = find_dataset("dataset")
    print(f"Found Dataset: {dataset_path}")
    
    # 2. Loading
    df = pd.read_csv(dataset_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 3. Identify Target
    target_col = inspect_and_identify_target(df)
    print(f"Identified Target Column: '{target_col}'")
    
    # 4. Clean & Inspect
    df_clean, inspection_report, numerical_cols, categorical_cols = clean_and_inspect_data(df, target_col)
    
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    if inspection_report["invalid_zeros_replaced_with_nan"]:
        print(f"Medically invalid 0s replaced with NaN: {inspection_report['invalid_zeros_replaced_with_nan']}")
    print(f"Class Breakdown: {inspection_report['class_distribution']}")
    
    # 5. Extract Feature Metadata for Flask UI
    feature_metadata = get_feature_metadata(df_clean, numerical_cols, categorical_cols)
    
    # 6. Feature Matrix & Target
    feature_names = numerical_cols + categorical_cols
    X = df_clean[feature_names]
    y = df_clean[target_col]
    
    # 7. Train / Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    
    # 8. Build Leakage-Free Preprocessor
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    # 9. Train & Evaluate All Models
    results, fitted_pipelines, best_model_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    best_pipeline = fitted_pipelines[best_model_name]
    
    # 10. Generate EDA & Visualizations
    generate_eda_and_charts(
        df_clean=df_clean,
        target_col=target_col,
        results=results,
        best_model_name=best_model_name,
        fitted_pipeline=best_pipeline,
        numerical_cols=numerical_cols,
        feature_names=feature_names,
        output_dir="static/images"
    )
    
    # 11. Extract Feature Importance for Metadata
    importances = extract_feature_importance(best_pipeline, feature_names)
    
    # 12. Save Best Model / Full Pipeline with Joblib
    os.makedirs("model", exist_ok=True)
    model_save_path = os.path.join("model", "diabetes_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nSaved Best Model Pipeline to: {model_save_path}")
    
    # 13. Save Comprehensive Metadata JSON
    # Strip numpy arrays for pure JSON serialization
    serialized_results = {}
    for m, m_data in results.items():
        serialized_results[m] = {
            "accuracy": m_data["accuracy"],
            "precision": m_data["precision"],
            "recall": m_data["recall"],
            "f1_score": m_data["f1_score"],
            "roc_auc": m_data["roc_auc"],
            "confusion_matrix": m_data["confusion_matrix"]
        }
        
    metadata = {
        "dataset_summary": {
            "source_file": os.path.basename(dataset_path),
            "raw_records": inspection_report["raw_record_count"],
            "cleaned_records": inspection_report["cleaned_record_count"],
            "duplicates_removed": inspection_report["duplicate_records_removed"],
            "train_records": len(X_train),
            "test_records": len(X_test),
            "target_column": target_col,
            "diabetic_cases": inspection_report["class_distribution"]["Diabetic (1)"],
            "non_diabetic_cases": inspection_report["class_distribution"]["Non-Diabetic (0)"],
            "invalid_zeros_handled": inspection_report["invalid_zeros_replaced_with_nan"]
        },
        "feature_order": feature_names,
        "features": feature_metadata,
        "best_model": {
            "name": best_model_name,
            "metrics": serialized_results[best_model_name],
            "feature_importance": importances
        },
        "model_comparison": serialized_results,
        "chart_data": {
            "class_distribution": {
                "labels": ["Non-Diabetic", "Diabetic"],
                "counts": [
                    inspection_report["class_distribution"]["Non-Diabetic (0)"],
                    inspection_report["class_distribution"]["Diabetic (1)"]
                ]
            },
            "model_comparison": {
                "models": list(serialized_results.keys()),
                "accuracy": [serialized_results[m]["accuracy"] for m in serialized_results],
                "precision": [serialized_results[m]["precision"] for m in serialized_results],
                "recall": [serialized_results[m]["recall"] for m in serialized_results],
                "f1_score": [serialized_results[m]["f1_score"] for m in serialized_results],
                "roc_auc": [serialized_results[m]["roc_auc"] for m in serialized_results]
            },
            "confusion_matrix": serialized_results[best_model_name]["confusion_matrix"],
            "feature_importance": {
                "features": list(importances.keys()),
                "scores": [round(s, 4) for s in importances.values()]
            }
        }
    }
    
    metadata_save_path = os.path.join("model", "model_metadata.json")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved Model Metadata JSON to: {metadata_save_path}")
    print("\nTraining and pipeline generation finished successfully!\n")


if __name__ == "__main__":
    main()
