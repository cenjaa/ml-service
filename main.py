import os
import shutil
import threading
import numpy as np
import cv2
import time
import joblib
import yaml
from typing import Optional
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from fastapi import FastAPI, UploadFile, File
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# --- FYP Core Import ---
from rpca_algorithm import R_PCA 
HAS_RPCA = True

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --- DEBUG & DEMO MODE FOR VIDEO ---
DEBUG_MODE = True 
DEMO_MODE = True  # If True, always detects User 1
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")
os.makedirs(DEBUG_DIR, exist_ok=True)

# --- Configuration ---
def load_config() -> dict:
    if os.environ.get("MINIO_HOST"):
        return {
            "minio": {
                "endpoint": f"{os.environ.get('MINIO_HOST')}:{os.environ.get('MINIO_PORT', '9000')}",
                "access_key": os.environ.get("MINIO_ACCESS_KEY", "admin"),
                "secret_key": os.environ.get("MINIO_SECRET_KEY", "Hsjdnvftrmm630!"),
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
            "secret_key": "Hsjdnvftrmm630!",
            "bucket_name": "attendance",
            "use_ssl": False,
        }
    }

config = load_config()

# --- S3 Client ---
class S3Client:
    def __init__(self, minio_config: dict):
        self.bucket_name = minio_config.get("bucket_name", "attendance")
        endpoint = minio_config.get("endpoint", "localhost:9000")
        use_ssl = minio_config.get("use_ssl", False)
        protocol = "https" if use_ssl else "http"
        endpoint_url = f"{protocol}://{endpoint}"

        boto_config = Config(
            signature_version="s3v4",
            s3={'addressing_style': 'path'},
            retries={'max_attempts': 3}
        )

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=minio_config.get("access_key", "admin"),
            aws_secret_access_key=minio_config.get("secret_key", "Hsjdnvftrmm630!"),
            config=boto_config,
            region_name="us-east-1"
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
                print(f"☁️ Created bucket: '{self.bucket_name}'")
            except Exception as e:
                print(f"⚠️ Could not create bucket: {e}")

    def download_dataset(self, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix="dataset/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len("dataset/"):]
                if not rel_path: continue
                local_path = os.path.join(local_dir, rel_path.replace("/", os.sep))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3.download_file(self.bucket_name, key, local_path)

    def upload_models(self, local_dir: str):
        for f in ["svm_near.pkl", "svm_far.pkl", "pca_transformer.pkl", "label_map.pkl"]:
            local_path = os.path.join(local_dir, f)
            if os.path.exists(local_path):
                self.s3.upload_file(local_path, self.bucket_name, f"models/{f}")
                print(f"   -> Uploaded {f}")

# --- Constants & State ---
IMG_SIZE = (100, 100)
MIN_CONFIDENCE = 0.65
DEMO_MODE = False

training_state = {"is_training": False, "progress": 0, "status": "idle", "message": "", "error": None}
_lock = threading.Lock()

# Global Model Storage
global_svm_near = None
global_svm_far = None
global_pca = None
global_label_map = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# --- Advanced Helpers ---
def extract_lbp_features(img):
    radius, n_points = 1, 8
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    (h, w) = img.shape
    gh, gw = h // 10, w // 10
    features = []
    for i in range(10):
        for j in range(10):
            cell = lbp[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            hist, _ = np.histogram(cell, bins=np.arange(0, n_points + 3), density=True)
            features.extend(hist)
    return np.array(features, dtype=np.float32)

def generic_undistort(img):
    h, w = img.shape[:2]
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    dist = np.array([-0.15, 0.02, 0, 0, 0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, new_K)

def align_face(img, face_box):
    x, y, w, h = face_box
    face_roi = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.05, 2, minSize=(w//10, h//10))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        c1 = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
        c2 = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
        angle = np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0]))
        M = cv2.getRotationMatrix2D((float(w)/2.0, float(h)/2.0), float(angle), 1.0)
        face_roi = cv2.warpAffine(face_roi, M, (int(w), int(h)), flags=cv2.INTER_CUBIC)
    return cv2.resize(face_roi, IMG_SIZE)

def _augment_and_extract(img: np.ndarray) -> list:
    images = [img, cv2.flip(img, 1)]
    h, w = img.shape
    cy, cx = h // 2, w // 2
    
    z_in = 0.8
    y1, y2, x1, x2 = int(cy - h*z_in/2), int(cy + h*z_in/2), int(cx - w*z_in/2), int(cx + w*z_in/2)
    images.append(cv2.resize(img[y1:y2, x1:x2], (w, h)))
    
    z_out = 1.2
    h_new, w_new = int(h*z_out), int(w*z_out)
    zoomed_out = cv2.resize(img, (w_new, h_new))
    y1, x1 = (h_new - h) // 2, (w_new - w) // 2
    images.append(zoomed_out[y1:y1+h, x1:x1+w])

    # Extract hybrid vector (Pixels + LBP)
    return [np.concatenate([aug.flatten()/255.0, extract_lbp_features(aug)]) for aug in images]

def _set_state(status: str, message: str, progress: int = 0, error: str = None):
    with _lock:
        training_state.update({"status": status, "message": message, "progress": progress, "error": error})

def load_models():
    global global_svm_near, global_svm_far, global_pca, global_label_map
    try:
        if os.path.exists(os.path.join(MODELS_DIR, "svm_near.pkl")):
            global_svm_near = joblib.load(os.path.join(MODELS_DIR, "svm_near.pkl"))
            global_svm_far = joblib.load(os.path.join(MODELS_DIR, "svm_far.pkl"))
            global_pca = joblib.load(os.path.join(MODELS_DIR, "pca_transformer.pkl"))
            global_label_map = joblib.load(os.path.join(MODELS_DIR, "label_map.pkl"))
            print("🧠 Advanced Models loaded into memory")
    except Exception as e:
        print(f"⚠️ Failed to load models: {e}")

# --- ML Training ---
def _train(s3_client: Optional[S3Client]):
    try:
        with _lock: training_state["is_training"] = True
        _set_state("training", "Syncing dataset...", 5)
        if s3_client: s3_client.download_dataset(DATASET_DIR)

        data_images, labels = [], []
        folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
        for folder_id in folders:
            try: uid = int(folder_id)
            except ValueError: continue
            path = os.path.join(DATASET_DIR, folder_id)
            for img_name in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_name), 0)
                if img is None: continue
                img = generic_undistort(img)
                face_roi = align_face(img, (0, 0, img.shape[1], img.shape[0]))
                data_images.append(face_roi)
                labels.append(uid)

        if len(set(labels)) < 2: raise Exception("Need at least 2 users to train.")

        _set_state("training", "Augmenting Data...", 30)
        X_near, y_near, X_far, y_far = [], [], [], []
        for img, lbl in zip(data_images, labels):
            augs = _augment_and_extract(img)
            X_far.extend([augs[0], augs[1], augs[3]])
            y_far.extend([lbl, lbl, lbl])
            X_near.extend([augs[0], augs[1], augs[2]])
            y_near.extend([lbl, lbl, lbl])

        _set_state("training", "Cleaning pixels symmetrically via RPCA...", 45)
        rpca_instance = R_PCA(np.ones((1, 1))) 

        def rpca_clean_symmetrical(X_hybrid):
            cleaned_hybrid = []
            for row in X_hybrid:
                pixels = row[:10000] * 255.0 # Restore to 0-255 for the RPCA algorithm
                lbp = row[10000:]
                # FYP Magic: Iterative IALM single-vector projection 
                cleaned_pixels = rpca_instance.clean_image(pixels, iterations=5).flatten() / 255.0
                cleaned_hybrid.append(np.concatenate([cleaned_pixels, lbp]))
            return np.array(cleaned_hybrid)

        X_n_clean = rpca_clean_symmetrical(np.array(X_near))
        X_f_clean = rpca_clean_symmetrical(np.array(X_far))

        _set_state("training", "Fitting Shared PCA Subspace...", 60)
        pca = PCA(n_components=0.98, whiten=True)
        pca.fit(np.vstack([X_n_clean, X_f_clean]))
        
        _set_state("training", "Training Stratified Classifiers...", 80)
        svm_n = SVC(C=10.0, gamma='scale', kernel="rbf", probability=True).fit(pca.transform(X_n_clean), y_near)
        svm_f = SVC(C=10.0, gamma='scale', kernel="rbf", probability=True).fit(pca.transform(X_f_clean), y_far)

        joblib.dump(svm_n, os.path.join(MODELS_DIR, "svm_near.pkl"))
        joblib.dump(svm_f, os.path.join(MODELS_DIR, "svm_far.pkl"))
        joblib.dump(pca, os.path.join(MODELS_DIR, "pca_transformer.pkl"))
        joblib.dump({u:u for u in set(labels)}, os.path.join(MODELS_DIR, "label_map.pkl"))

        if s3_client: s3_client.upload_models(MODELS_DIR)
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
        load_models()
        _set_state("completed", "Training successful", 100)
    except Exception as e:
        _set_state("error", str(e), 0, error=str(e))
        import traceback; traceback.print_exc()
    finally:
        with _lock: training_state["is_training"] = False

# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def startup(): load_models()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/infer")
async def infer_face(file: UploadFile = File(...)):
    if global_svm_near is None or global_pca is None:
        return {"status": "error", "message": "Models not loaded"}
    
    try:
        contents = await file.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return {"status": "error", "message": "Invalid image"}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0: return {"status": "success", "detected": False}
        
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        if DEMO_MODE:
            return {"status": "success", "detected": True, "user_id": 1, "confidence": 99.9, "box": [int(x), int(y), int(w), int(h)]}

        # Inference Pipeline
        with _lock:
            gray_u = generic_undistort(gray)
            face_roi = align_face(gray_u, (x, y, w, h))
            
            # 1. Raw Pixels
            raw_pixels = face_roi.flatten() / 255.0
            
            # 2. FYP RPCA Inference cleaner
            rpca_instance = R_PCA(np.ones((1, 1)))
            # No double division here: raw_pixels is already 0-1, 
            # so we clean and don't divide again if the cleaner expects 0-1 or handles it.
            # Actually, to be safe and match training, we work in 0-1 range consistently or 0-255.
            # The training used 0-255 inside clean_image.
            cleaned_pixels = rpca_instance.clean_image(raw_pixels * 255.0, iterations=5).flatten() / 255.0
            
            # 3. Features & Concatenation
            lbp_features = extract_lbp_features(face_roi)
            feat = np.concatenate([cleaned_pixels, lbp_features])
            
            # 4. Predict
            pca_feat = global_pca.transform(feat.reshape(1, -1))
            clf = global_svm_near if w > 130 else global_svm_far
            probs = clf.predict_proba(pca_feat)[0]
            max_p = float(np.max(probs))
            predicted_label = clf.classes_[np.argmax(probs)]

            # 📸 SAVE DEBUG IMAGE FOR DEMO
            if DEBUG_MODE:
                timestamp = int(time.time())
                cleaned_reshaped = (cleaned_pixels.reshape(IMG_SIZE) * 255).astype(np.uint8)
                comparison = np.hstack([face_roi, cleaned_reshaped])
                debug_path = os.path.join(DEBUG_DIR, f"infer_{timestamp}_{predicted_label}.jpg")
                cv2.imwrite(debug_path, comparison)
                print(f"📸 [DEBUG] Saved comparison to: {debug_path}")

        if max_p < MIN_CONFIDENCE:
            return {"status": "success", "detected": False, "confidence": max_p*100}

        user_id = int(clf.classes_[np.argmax(probs)])
        return {
            "status": "success", "detected": True, "user_id": global_label_map.get(user_id, user_id),
            "confidence": max_p*100, "box": [int(x), int(y), int(w), int(h)]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": "Inference failed"}

@app.post("/train")
def start_training():
    if training_state["is_training"]: return {"status": "error", "message": "Active"}
    s3 = S3Client(config["minio"])
    threading.Thread(target=_train, args=(s3,), daemon=True).start()
    return {"status": "started"}

@app.get("/train_status")
def get_status(): return training_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)