"""
app.py
======
Flask Web Application for Diabetes Risk Prediction Using Machine Learning.
Loads the trained Scikit-Learn Pipeline and serves interactive UI pages and APIs.
"""

import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "diabetes-prediction-super-secret-key-2026")

# Global references for pipeline and metadata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "diabetes_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

pipeline = None
metadata = None


def load_model_and_metadata():
    """Load model pipeline and metadata safely."""
    global pipeline, metadata
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file '{METADATA_PATH}' not found. Please run 'python train_model.py' first."
        )
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Please run 'python train_model.py' first."
        )
    pipeline = joblib.load(MODEL_PATH)


# Load at startup
try:
    load_model_and_metadata()
    print("[Flask] Successfully loaded Machine Learning Pipeline and Model Metadata.")
except Exception as e:
    print(f"[Flask] Warning: Failed to load model on startup ({e}). Run train_model.py.")


@app.context_processor
def inject_global_data():
    """Inject common metadata into all Jinja templates."""
    return {
        "current_year": datetime.now().year,
        "best_model_name": metadata["best_model"]["name"] if metadata else "Machine Learning Model",
        "dataset_name": metadata["dataset_summary"]["source_file"] if metadata else "Kaggle Dataset",
        "accuracy": round(metadata["best_model"]["metrics"]["accuracy"] * 100, 1) if metadata else 0.0,
        "roc_auc": round(metadata["best_model"]["metrics"]["roc_auc"], 3) if metadata else 0.0,
    }


def compute_prediction_result(user_inputs, save_to_session=True):
    """Unified prediction logic for both Web Form and REST/AJAX API."""
    feature_order = metadata.get("feature_order", [])
    features_dict = metadata.get("features", {})
    invalid_handled = metadata["dataset_summary"].get("invalid_zeros_handled", {})
    
    processed_row = {}
    for feat in feature_order:
        val = user_inputs.get(feat, 0.0)
        try:
            val_float = float(val)
            if feat in invalid_handled and val_float == 0:
                processed_row[feat] = np.nan
            else:
                processed_row[feat] = val_float
        except (ValueError, TypeError):
            processed_row[feat] = val

    input_df = pd.DataFrame([processed_row], columns=feature_order)
    pred_class = int(pipeline.predict(input_df)[0])
    
    prob_high = 0.5
    prob_low = 0.5
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(input_df)[0]
        prob_low = float(probas[0])
        prob_high = float(probas[1])
    elif hasattr(pipeline, "decision_function"):
        dec = float(pipeline.decision_function(input_df)[0])
        prob_high = 1 / (1 + np.exp(-dec))
        prob_low = 1 - prob_high
    else:
        prob_high = 1.0 if pred_class == 1 else 0.0
        prob_low = 1.0 - prob_high
        
    risk_label = "High" if pred_class == 1 else "Low"
    risk_percentage = round(prob_high * 100, 1)
    
    # Clinical Risk Factors Evaluation
    risk_factors = []
    if "Glucose" in user_inputs and float(user_inputs["Glucose"]) >= 140:
        risk_factors.append(f"Elevated blood glucose ({user_inputs['Glucose']} mg/dL) exceeds normal fasting thresholds.")
    if "BMI" in user_inputs and float(user_inputs["BMI"]) >= 30:
        risk_factors.append(f"BMI ({user_inputs['BMI']} kg/m²) indicates clinical obesity range.")
    if "BloodPressure" in user_inputs and float(user_inputs["BloodPressure"]) >= 85:
        risk_factors.append(f"Diastolic blood pressure ({user_inputs['BloodPressure']} mm Hg) indicates stage 1 hypertension.")
    if "Age" in user_inputs and float(user_inputs["Age"]) >= 45:
        risk_factors.append(f"Patient age ({int(float(user_inputs['Age']))} yrs) is a recognized clinical demographic risk factor.")
    if "DiabetesPedigreeFunction" in user_inputs and float(user_inputs["DiabetesPedigreeFunction"]) >= 0.6:
        risk_factors.append(f"Genetic diabetes pedigree score ({user_inputs['DiabetesPedigreeFunction']}) indicates elevated hereditary predisposition.")
        
    result_data = {
        "prediction": risk_label,
        "predicted_class": pred_class,
        "probability": risk_percentage,
        "prob_low": round(prob_low * 100, 1),
        "prob_high": risk_percentage,
        "risk_factors": risk_factors,
        "inputs": user_inputs,
        "features_meta": features_dict,
        "model_name": metadata["best_model"]["name"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if save_to_session:
        if "prediction_history" not in session:
            session["prediction_history"] = []
            
        history = session["prediction_history"]
        history_item = {
            "id": len(history) + 1,
            "timestamp": result_data["timestamp"],
            "prediction": f"Diabetes Risk: {risk_label}",
            "risk_level": risk_label,
            "probability": f"{risk_percentage}%",
            "summary": ", ".join([f"{k}: {v}" for k, v in list(user_inputs.items())[:4]])
        }
        history.insert(0, history_item)
        session["prediction_history"] = history[:30]
        session["last_result"] = result_data
        session.modified = True
        
    return result_data


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    """Home landing page."""
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form page (GET) and form submission handler (POST)."""
    if request.method == "POST":
        if not pipeline or not metadata:
            flash("Model is not loaded. Please train the model first.", "danger")
            return redirect(url_for("predict"))
            
        feature_order = metadata.get("feature_order", [])
        features_meta = metadata.get("features", {})
        
        user_inputs = {}
        validation_errors = []
        
        for feat in feature_order:
            val_str = request.form.get(feat, "").strip()
            feat_info = features_meta.get(feat, {})
            
            if val_str == "":
                validation_errors.append(f"Field '{feat_info.get('label', feat)}' is required.")
                continue
                
            if feat_info.get("type") == "numerical":
                try:
                    num_val = float(val_str)
                    user_inputs[feat] = num_val
                except ValueError:
                    validation_errors.append(f"Field '{feat_info.get('label', feat)}' must be a valid number.")
            else:
                user_inputs[feat] = val_str
                
        if validation_errors:
            for err in validation_errors:
                flash(err, "danger")
            return render_template("predict.html", metadata=metadata, submitted_values=request.form)
            
        try:
            compute_prediction_result(user_inputs, save_to_session=True)
            return redirect(url_for("result"))
        except Exception as e:
            flash(f"Error executing prediction: {str(e)}", "danger")
            return render_template("predict.html", metadata=metadata, submitted_values=request.form)
            
    return render_template("predict.html", metadata=metadata, submitted_values={})


@app.route("/result")
def result():
    """Prediction result card page."""
    result_data = session.get("last_result")
    if not result_data:
        flash("No active prediction found. Please submit the form first.", "info")
        return redirect(url_for("predict"))
    return render_template("result.html", result=result_data, metadata=metadata)


@app.route("/dashboard")
def dashboard():
    """ML Analytics and Performance Dashboard."""
    history = session.get("prediction_history", [])
    return render_template("dashboard.html", metadata=metadata, history=history)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear session prediction history."""
    session.pop("prediction_history", None)
    flash("Prediction history cleared successfully.", "success")
    return redirect(request.referrer or url_for("dashboard"))


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST endpoint for asynchronous predictions (supports live HUD & in-place prediction)."""
    if not pipeline or not metadata:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
        
    try:
        data = request.get_json(force=True)
        save_hist = data.pop("_save_history", False)
        
        result_data = compute_prediction_result(data, save_to_session=bool(save_hist))
        
        return jsonify({
            "success": True,
            "prediction": result_data["prediction"],
            "predicted_class": result_data["predicted_class"],
            "probability": result_data["probability"],
            "prob_low": result_data["prob_low"],
            "prob_high": result_data["prob_high"],
            "risk_factors": result_data["risk_factors"],
            "model": result_data["model_name"],
            "timestamp": result_data["timestamp"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/metadata")
def api_metadata():
    """Returns trained model metadata and chart data as JSON."""
    if not metadata:
        return jsonify({"error": "Metadata not loaded"}), 500
    return jsonify(metadata)


@app.after_request
def add_performance_headers(response):
    """Enables browser caching for static assets (CSS, JS, WebP images)."""
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
