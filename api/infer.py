"""
api/infer.py
─────────────
Face recognition inference endpoint.

POST /api/infer
  Body:  multipart/form-data  { file: <image> }
  Returns: JSON with detection result, user_id, and confidence.
"""
import os
import time
import traceback

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

import core.state as state
from config import IMG_SIZE, MIN_CONFIDENCE, DEBUG_MODE, DEMO_MODE, DEBUG_DIR
from core.rpca import R_PCA
from utils.image import generic_undistort, align_face, extract_lbp_features

router = APIRouter()


@router.post("/api/infer", tags=["Inference"])
async def infer_face(file: UploadFile = File(...)):
    """
    Run face recognition on an uploaded image frame.

    - Detects faces using Haar cascade.
    - Selects the largest face.
    - Runs the RPCA + PCA + SVM pipeline.
    - Switches between near/far SVM classifier based on face width.
    """
    if state.svm_near is None or state.pca is None:
        return {"status": "error", "message": "Models not loaded. Run /train first."}

    try:
        contents = await file.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "message": "Invalid or unreadable image."}

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = state.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return {"status": "success", "detected": False}

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])  # largest face

        # Demo mode — always return user 1 (useful for frontend demos)
        if DEMO_MODE:
            return {
                "status":     "success",
                "detected":   True,
                "user_id":    1,
                "confidence": 99.9,
                "box":        [int(x), int(y), int(w), int(h)],
            }

        with state.lock:
            # ── Pre-processing ─────────────────────────────────
            gray_u   = generic_undistort(gray)
            face_roi = align_face(gray_u, (x, y, w, h))

            # ── RPCA pixel cleaning ────────────────────────────
            rpca            = R_PCA(np.ones((1, 1)))
            raw_pixels      = face_roi.flatten() / 255.0
            cleaned_pixels  = rpca.clean_image(raw_pixels * 255.0, iterations=5).flatten() / 255.0

            # ── Hybrid feature vector ──────────────────────────
            lbp_features = extract_lbp_features(face_roi)
            feat         = np.concatenate([cleaned_pixels, lbp_features])

            # ── PCA → SVM classification ───────────────────────
            pca_feat = state.pca.transform(feat.reshape(1, -1))
            clf      = state.svm_near if w > 130 else state.svm_far
            probs    = clf.predict_proba(pca_feat)[0]
            max_p    = float(np.max(probs))

            # ── Optional debug image ───────────────────────────
            if DEBUG_MODE:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                predicted_label = clf.classes_[np.argmax(probs)]
                timestamp       = int(time.time())
                cleaned_img     = (cleaned_pixels.reshape(IMG_SIZE) * 255).astype(np.uint8)
                comparison      = np.hstack([face_roi, cleaned_img])
                debug_path      = os.path.join(DEBUG_DIR, f"infer_{timestamp}_{predicted_label}.jpg")
                cv2.imwrite(debug_path, comparison)
                print(f"📸 [DEBUG] {debug_path}")

        # ── Below-threshold → unknown face ─────────────────────
        if max_p < MIN_CONFIDENCE:
            return {"status": "success", "detected": False, "confidence": round(max_p * 100, 2)}

        user_id = int(clf.classes_[np.argmax(probs)])
        return {
            "status":     "success",
            "detected":   True,
            "user_id":    state.label_map.get(user_id, user_id),
            "confidence": round(max_p * 100, 2),
            "box":        [int(x), int(y), int(w), int(h)],
        }

    except Exception:
        traceback.print_exc()
        return {"status": "error", "message": "Inference failed. Check server logs."}
