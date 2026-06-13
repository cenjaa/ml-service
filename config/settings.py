"""
config/settings.py
──────────────────
Loads all configuration from environment variables or a YAML fallback.
"""
import os
import yaml


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config() -> dict:
    """
    Priority:
      1. Environment variables  (Docker / VPS deployment)
      2. config.yaml next to the repo root  (local dev)
      3. Hard-coded localhost defaults
    """
    if os.environ.get("MINIO_HOST"):
        return {
            "minio": {
                "endpoint": f"{os.environ['MINIO_HOST']}:{os.environ.get('MINIO_PORT', '9000')}",
                "access_key": os.environ.get("MINIO_ACCESS_KEY", "admin"),
                "secret_key": os.environ.get("MINIO_SECRET_KEY", ""),
                "bucket_name": os.environ.get("MINIO_BUCKET", "attendance"),
                "use_ssl": os.environ.get("MINIO_USE_SSL", "false").lower() == "true",
            }
        }

    config_path = os.path.join(BASE_DIR, "..", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    return {
        "minio": {
            "endpoint": "localhost:9000",
            "access_key": "admin",
            "secret_key": "",
            "bucket_name": "attendance",
            "use_ssl": False,
        }
    }


# ── Shared paths ────────────────────────────────────────────────
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEBUG_DIR   = os.path.join(BASE_DIR, "debug_output")

# ── ML hyper-parameters ─────────────────────────────────────────
IMG_SIZE       = (100, 100)
MIN_CONFIDENCE = 0.65

# ── Feature flags ───────────────────────────────────────────────
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").lower() == "true"
DEMO_MODE  = os.environ.get("DEMO_MODE",  "false").lower() == "true"
