# Diabetes Risk Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://palletsprojects.com/p/flask/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mithuilnj09/diabetes-risk-prediction-ml)

> **Problem Statement:**  
> *“To develop a Machine Learning model that predicts the risk of diabetes using patient health data for early detection and preventive healthcare.”*

---

## Project Overview

**Diabetes Risk Prediction Using Machine Learning** is an end-to-end full-stack clinical machine learning application. It pairs an automated, leakage-free Scikit-Learn data science training pipeline with a modern, responsive Flask web application.

Users can evaluate patient physiological biomarkers (such as Blood Glucose, Insulin, BMI, Diastolic Blood Pressure, and Age) and receive an instant, probabilistic risk assessment powered by the top-performing champion model.

---

## Key Features

1. **Automated Dataset Discovery & Schema Inspection**:
   - Dynamically analyzes dataset features, identifies target outcome columns, checks duplicate records, and separates numerical vs categorical variables.
   - Detects medically impossible zeroes in clinical metrics (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) and treats them as missing values to prevent distortion.

2. **Leakage-Free Preprocessing Pipeline**:
   - Fits all imputations (median) and feature scaling (`StandardScaler`) strictly on training partitions (`ColumnTransformer` + `Pipeline`).
   - Encapsulates the entire workflow inside `model/diabetes_model.pkl` so predictions on new inputs require zero manual transformation in Flask.

3. **Multi-Model Benchmark & Champion Selection**:
   - Trains and evaluates 5 classification algorithms:
     1. **Logistic Regression**
     2. **Decision Tree Classifier**
     3. **Random Forest Classifier**
     4. **K-Nearest Neighbors (KNN)**
     5. **Support Vector Machine (SVM)**
   - Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
   - Automatically selects the best model based on discriminative performance.

4. **Modern Healthcare AI UI**:
   - **Landing Page (`/`)**: Project overview, clinical objective, ML workflow steps, performance summary, and quick navigation.
   - **Prediction Form (`/predict`)**: Dynamically generated input fields with medical units, ranges, real-time client validation, and one-click demo autofill buttons (Healthy vs High Risk).
   - **Result Card (`/result`)**: Clear risk badge (High Risk / Low Risk), confidence probability gauge (e.g. `78.4%`), contributing factor analysis, submitted parameter summary, and medical consultation advice.
   - **ML Dashboard (`/dashboard`)**: Diagnostic KPI cards, interactive Chart.js visualizations (Class Distribution, Algorithm Benchmark, Feature Importance), confusion matrix breakdown, and full-resolution Seaborn/Matplotlib chart gallery.
   - **Session Prediction History**: Session-based logging of recent patient assessments with a **Clear History** button. No sensitive patient data is stored permanently.

5. **REST API**:
   - `/api/predict`: JSON API for headless inferences and integrations.
   - `/api/metadata`: JSON endpoint exposing trained model metadata, metrics, and chart data.

---

## Project Structure

```text
ML PBL/
├── dataset/
│   └── diabetes.csv              # Kaggle Diabetes dataset
├── model/
│   ├── diabetes_model.pkl        # Serialized Scikit-Learn Pipeline
│   └── model_metadata.json       # Feature metadata, metrics & chart data
├── static/
│   ├── css/
│   │   └── style.css             # Modern Healthcare dark-mode design system
│   ├── js/
│   │   └── script.js             # Client validation, demo fill, Chart.js logic
│   └── images/                   # High-res Seaborn/Matplotlib charts
│       ├── diabetes_distribution.png
│       ├── feature_distributions.png
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       ├── confusion_matrix.png
│       └── model_comparison.png
├── templates/
│   ├── base.html                 # Master layout with navigation & disclaimer
│   ├── index.html                # Modern landing page
│   ├── predict.html              # Dynamic clinical prediction form
│   ├── result.html               # Prediction result & probability card
│   └── dashboard.html            # Analytics dashboard & session history
├── app.py                        # Flask web application & REST endpoints
├── train_model.py                # Automated data science & ML pipeline script
├── requirements.txt              # Project dependencies
├── README.md                     # Comprehensive project documentation
└── .gitignore                    # Git ignore file
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- `pip` package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training Pipeline
To inspect the dataset, train and benchmark all 5 models, generate static EDA charts, and save the model pipeline:
```bash
python train_model.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## Machine Learning Benchmark Results

On the stratified holdout test cohort (154 patient records):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **72.73%** | **58.57%** | **75.93%** | **0.6613** | **0.8194** |
| Logistic Regression | 73.38% | 60.32% | 70.37% | 0.6496 | 0.8126 |
| Support Vector Machine | 72.73% | 59.09% | 72.22% | 0.6500 | 0.8139 |
| K-Nearest Neighbors | 72.73% | 62.00% | 57.41% | 0.5962 | 0.7931 |
| Decision Tree | 72.73% | 61.11% | 61.11% | 0.6111 | 0.7706 |

---

## Medical & Educational Disclaimer

> **Notice:** This application is developed for educational and research purposes and should not be used as a substitute for professional medical diagnosis. Please consult a qualified healthcare professional for medical evaluation and personalized clinical care.
