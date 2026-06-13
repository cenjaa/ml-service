"""
utils/image.py
──────────────
Low-level image processing helpers:
  - generic_undistort  — removes barrel distortion from webcam frames
  - align_face         — eye-alignment + crop + resize
  - extract_lbp        — grid-based Local Binary Pattern histogram
  - augment_and_extract — data augmentation + hybrid feature vector
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from config import IMG_SIZE
from core.state import eye_cascade


# ── Lens undistortion ────────────────────────────────────────────
def generic_undistort(img: np.ndarray) -> np.ndarray:
    """
    Apply a generic barrel-distortion correction to a grayscale frame.
    Uses a heuristic camera matrix scaled to the image dimensions.
    """
    h, w = img.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0,     1]], dtype=np.float32)
    dist = np.array([-0.15, 0.02, 0, 0, 0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, new_K)


# ── Face alignment ───────────────────────────────────────────────
def align_face(img: np.ndarray, face_box: tuple) -> np.ndarray:
    """
    Detect eyes inside the face ROI, rotate to horizontal alignment,
    then resize to IMG_SIZE.

    Args:
        img:      Grayscale image (already undistorted).
        face_box: (x, y, w, h) bounding box from the face detector.

    Returns:
        Aligned, resized face patch of shape IMG_SIZE.
    """
    x, y, w, h = face_box
    face_roi = img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(face_roi, 1.05, 2, minSize=(w // 10, h // 10))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        c1 = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
        c2 = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
        angle = np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0]))
        M = cv2.getRotationMatrix2D((float(w) / 2.0, float(h) / 2.0), float(angle), 1.0)
        face_roi = cv2.warpAffine(face_roi, M, (int(w), int(h)), flags=cv2.INTER_CUBIC)

    return cv2.resize(face_roi, IMG_SIZE)


# ── LBP feature extraction ───────────────────────────────────────
def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """
    Compute a grid-based LBP histogram (10×10 cells, uniform pattern).

    Args:
        img: Grayscale face patch of shape IMG_SIZE.

    Returns:
        1-D float32 array of length 10 * 10 * (n_points + 2).
    """
    radius, n_points = 1, 8
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    h, w = img.shape
    gh, gw = h // 10, w // 10
    features = []
    for i in range(10):
        for j in range(10):
            cell = lbp[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            hist, _ = np.histogram(cell, bins=np.arange(0, n_points + 3), density=True)
            features.extend(hist)
    return np.array(features, dtype=np.float32)


# ── Training augmentation ────────────────────────────────────────
def augment_and_extract(img: np.ndarray) -> list:
    """
    Generate 4 augmented views (original, flip, zoom-in, zoom-out)
    and return their hybrid feature vectors [pixels/255 || LBP].

    Args:
        img: Grayscale face patch of shape IMG_SIZE.

    Returns:
        List of 4 float32 feature vectors.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Zoom-in crop
    z_in = 0.8
    y1 = int(cy - h * z_in / 2); y2 = int(cy + h * z_in / 2)
    x1 = int(cx - w * z_in / 2); x2 = int(cx + w * z_in / 2)
    zoomed_in = cv2.resize(img[y1:y2, x1:x2], (w, h))

    # Zoom-out pad
    z_out = 1.2
    h_new, w_new = int(h * z_out), int(w * z_out)
    big = cv2.resize(img, (w_new, h_new))
    y1_o = (h_new - h) // 2; x1_o = (w_new - w) // 2
    zoomed_out = big[y1_o:y1_o + h, x1_o:x1_o + w]

    images = [img, cv2.flip(img, 1), zoomed_in, zoomed_out]
    return [
        np.concatenate([aug.flatten() / 255.0, extract_lbp_features(aug)])
        for aug in images
    ]
