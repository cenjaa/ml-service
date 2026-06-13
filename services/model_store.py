"""
services/model_store.py
────────────────────────
Loads persisted ML model artefacts from disk into the shared
core.state globals so all API routes can access them.
"""
import os
import joblib

import core.state as state
from config import MODELS_DIR


def load_models():
    """
    Attempt to load all four model files from MODELS_DIR into memory.
    Silently skips if models don't exist yet (before first training run).
    """
    near_path = os.path.join(MODELS_DIR, "svm_near.pkl")
    if not os.path.exists(near_path):
        print("ℹ️  No trained models found — run /train first.")
        return

    try:
        state.svm_near  = joblib.load(os.path.join(MODELS_DIR, "svm_near.pkl"))
        state.svm_far   = joblib.load(os.path.join(MODELS_DIR, "svm_far.pkl"))
        state.pca       = joblib.load(os.path.join(MODELS_DIR, "pca_transformer.pkl"))
        state.label_map = joblib.load(os.path.join(MODELS_DIR, "label_map.pkl"))
        print("🧠 Models loaded into memory.")
    except Exception as e:
        print(f"⚠️  Failed to load models: {e}")
