"""
core/state.py
─────────────
Shared in-memory state and loaded model globals.
Using a module-level singleton keeps state consistent across
all route handlers without needing a dependency-injection framework.
"""
import threading
import cv2

# ── Thread safety lock ──────────────────────────────────────────
lock = threading.Lock()

# ── Training progress ───────────────────────────────────────────
training_state: dict = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "",
    "error": None,
}

# ── Loaded ML models (populated by services/model_store.py) ────
svm_near   = None
svm_far    = None
pca        = None
label_map  = None

# ── OpenCV cascade classifiers ──────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
