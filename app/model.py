"""
Model loading and prediction logic.
Loads saved artifacts once at startup and exposes a predict function.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Paths — relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Global holders (loaded once)
_model = None
_scaler = None
_label_encoder = None
_feature_columns = None


def load_artifacts():
    """Load all model artifacts into memory. Called once at app startup."""
    global _model, _scaler, _label_encoder, _feature_columns

    print("Loading model artifacts...")

    _model = joblib.load(os.path.join(MODELS_DIR, "stacking_model.joblib"))
    print(f"  Model loaded.")

    _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    print(f"  Scaler loaded.")

    _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    print(f"  Label encoder loaded. Classes: {list(_label_encoder.classes_)}")

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "r") as f:
        _feature_columns = json.load(f)
    print(f"  Feature columns loaded ({len(_feature_columns)} features).")

    print("All artifacts loaded successfully!")


def predict(features_df: pd.DataFrame) -> dict:
    """
    Takes an engineered features DataFrame (1 row, 31 columns)
    and returns prediction result.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")

    # Ensure correct column order
    features_df = features_df[_feature_columns]

    # Predict
    pred_encoded = _model.predict(features_df)[0]
    pred_label = _label_encoder.inverse_transform([pred_encoded])[0]

    # Probabilities
    pred_proba = _model.predict_proba(features_df)[0]
    proba_dict = {
        _label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }

    return {
        "predicted_load_type": pred_label,
        "confidence": round(float(max(pred_proba)), 4),
        "probabilities": proba_dict,
    }
