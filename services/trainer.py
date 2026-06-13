"""
services/trainer.py
────────────────────
Background training pipeline:
  1. Download dataset from MinIO
  2. Load + augment images
  3. Clean pixels with RPCA
  4. Fit shared PCA subspace
  5. Train near-distance and far-distance SVM classifiers
  6. Persist models locally and upload back to MinIO
"""
import os
import shutil
import traceback
import numpy as np
import cv2
import joblib
from typing import Optional
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import core.state as state
from config import MODELS_DIR, DATASET_DIR
from core.rpca import R_PCA
from services.model_store import load_models
from services.storage import S3Client
from utils.image import generic_undistort, align_face, augment_and_extract


# ── Internal helpers ────────────────────────────────────────────
def _set_state(status: str, message: str, progress: int = 0, error: str = None):
    with state.lock:
        state.training_state.update(
            {"status": status, "message": message, "progress": progress, "error": error}
        )


def _rpca_clean_batch(X_hybrid: np.ndarray) -> np.ndarray:
    """
    Apply RPCA single-vector cleaning to every row's pixel block (first 10 000 dims).
    Pixels are scaled back to [0, 255] before cleaning and normalised to [0, 1] after,
    to match the training convention established during dataset preparation.
    """
    rpca = R_PCA(np.ones((1, 1)))
    cleaned = []
    for row in X_hybrid:
        pixels          = row[:10000] * 255.0   # restore to [0, 255]
        lbp             = row[10000:]
        cleaned_pixels  = rpca.clean_image(pixels, iterations=5).flatten() / 255.0
        cleaned.append(np.concatenate([cleaned_pixels, lbp]))
    return np.array(cleaned)


# ── Main training function (runs in a daemon thread) ────────────
def run_training(s3_client: Optional[S3Client]):
    """
    Execute the full training pipeline.
    Updates `core.state.training_state` throughout for progress tracking.
    """
    try:
        with state.lock:
            state.training_state["is_training"] = True

        # Step 1 — Sync dataset
        _set_state("training", "Syncing dataset from MinIO...", 5)
        if s3_client:
            s3_client.download_dataset(DATASET_DIR)

        # Step 2 — Load images from disk
        _set_state("training", "Loading images...", 15)
        data_images, labels = [], []
        folders = [
            f for f in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, f))
        ]
        for folder_id in folders:
            try:
                uid = int(folder_id)
            except ValueError:
                continue
            path = os.path.join(DATASET_DIR, folder_id)
            for img_name in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_name), 0)
                if img is None:
                    continue
                img = generic_undistort(img)
                face_roi = align_face(img, (0, 0, img.shape[1], img.shape[0]))
                data_images.append(face_roi)
                labels.append(uid)

        if len(set(labels)) < 2:
            raise ValueError("Need at least 2 users to train.")

        # Step 3 — Augmentation
        _set_state("training", "Augmenting data...", 30)
        X_near, y_near = [], []
        X_far,  y_far  = [], []
        for img, lbl in zip(data_images, labels):
            augs = augment_and_extract(img)          # [orig, flip, zoom-in, zoom-out]
            X_near.extend([augs[0], augs[1], augs[2]])   # near: orig + flip + zoom-in
            y_near.extend([lbl,     lbl,     lbl    ])
            X_far.extend( [augs[0], augs[1], augs[3]])   # far:  orig + flip + zoom-out
            y_far.extend( [lbl,     lbl,     lbl    ])

        # Step 4 — RPCA cleaning
        _set_state("training", "Cleaning pixel vectors via RPCA (IALM)...", 45)
        X_n_clean = _rpca_clean_batch(np.array(X_near))
        X_f_clean = _rpca_clean_batch(np.array(X_far))

        # Step 5 — PCA dimensionality reduction (shared subspace)
        _set_state("training", "Fitting shared PCA subspace...", 60)
        pca_model = PCA(n_components=0.98, whiten=True)
        pca_model.fit(np.vstack([X_n_clean, X_f_clean]))

        # Step 6 — SVM classifiers
        _set_state("training", "Training SVM classifiers...", 80)
        svm_n = SVC(C=10.0, gamma="scale", kernel="rbf", probability=True).fit(
            pca_model.transform(X_n_clean), y_near
        )
        svm_f = SVC(C=10.0, gamma="scale", kernel="rbf", probability=True).fit(
            pca_model.transform(X_f_clean), y_far
        )

        # Step 7 — Persist
        _set_state("training", "Saving models...", 90)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(svm_n,                            os.path.join(MODELS_DIR, "svm_near.pkl"))
        joblib.dump(svm_f,                            os.path.join(MODELS_DIR, "svm_far.pkl"))
        joblib.dump(pca_model,                        os.path.join(MODELS_DIR, "pca_transformer.pkl"))
        joblib.dump({u: u for u in set(labels)},      os.path.join(MODELS_DIR, "label_map.pkl"))

        if s3_client:
            s3_client.upload_models(MODELS_DIR)

        # Step 8 — Clean up dataset and hot-reload models
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
        load_models()

        _set_state("completed", "Training successful", 100)
        print("✅ Training complete.")

    except Exception as e:
        traceback.print_exc()
        _set_state("error", str(e), 0, error=str(e))

    finally:
        with state.lock:
            state.training_state["is_training"] = False
