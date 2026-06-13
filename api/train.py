"""
api/train.py
─────────────
Training trigger and status endpoints.

POST /train         — kick off a background training job
GET  /train/status  — poll training progress
"""
import threading

from fastapi import APIRouter

import core.state as state
from config import load_config
from services.storage import S3Client
from services.trainer import run_training

router = APIRouter()

_config = load_config()


@router.post("/train", tags=["Training"])
def start_training():
    """
    Start a training job in a background daemon thread.
    Returns immediately with {"status": "started"} or an error if
    a training job is already running.
    """
    if state.training_state["is_training"]:
        return {"status": "error", "message": "A training job is already running."}

    s3 = S3Client(_config["minio"])
    threading.Thread(target=run_training, args=(s3,), daemon=True).start()
    return {"status": "started"}


@router.get("/train/status", tags=["Training"])
def get_training_status():
    """Return the current training state (progress, status, message)."""
    return state.training_state
