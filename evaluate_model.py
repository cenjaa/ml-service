import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse your existing logic
from main import align_face, generic_undistort, extract_lbp_features, _augment_and_extract, load_config, S3Client
from rpca_algorithm import R_PCA

DATASET_DIR = "dataset"
IMG_SIZE = (100, 100)

def evaluate():
    print("🚀 Starting Confusion Matrix Analysis...")
    
    # 0. Sync with S3/MinIO
    print("☁️ Syncing dataset from S3/MinIO...")
    config = load_config()
    s3 = S3Client(config["minio"])
    s3.download_dataset(DATASET_DIR)
    
    data_images, labels = [], []
    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    # 1. Load Data
    for folder_id in folders:
        try: uid = int(folder_id)
        except: continue
        path = os.path.join(DATASET_DIR, folder_id)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name), 0)
            if img is None: continue
            img = generic_undistort(img)
            face_roi = align_face(img, (0, 0, img.shape[1], img.shape[0]))
            data_images.append(face_roi)
            labels.append(uid)

    if not data_images:
        print("❌ No data found in dataset folder!")
        return

    # 2. Split into Train (80%) and Test (20%)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        data_images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"📊 Dataset split: {len(X_train_raw)} train, {len(X_test_raw)} test.")

    # 3. Augment and Clean Training Data
    print("🪄 Augmenting and Cleaning Training Data (RPCA)...")
    X_train_final, y_train_final = [], []
    rpca = R_PCA(np.ones((1, 1)))
    
    for img, lbl in zip(X_train_raw, y_train_raw):
        augs = _augment_and_extract(img) # Returns [pixels+lbp, ...]
        for feat in augs:
            # Clean pixels in the hybrid vector
            px = feat[:10000] * 255.0
            lbp = feat[10000:]
            cleaned_px = rpca.clean_image(px, iterations=5).flatten() / 255.0
            X_train_final.append(np.concatenate([cleaned_px, lbp]))
            y_train_final.append(lbl)

    # 4. Prepare and Clean Test Data (No Augmentation for Test!)
    print("🔍 Preparing Test Data...")
    X_test_final = []
    for img in X_test_raw:
        px = (img.flatten() / 255.0) * 255.0
        lbp = extract_lbp_features(img)
        cleaned_px = rpca.clean_image(px, iterations=5).flatten() / 255.0
        X_test_final.append(np.concatenate([cleaned_px, lbp]))

    # 5. Fit PCA & SVM
    print("🧠 Training Model...")
    pca = PCA(n_components=0.98, whiten=True)
    X_train_pca = pca.fit_transform(X_train_final)
    X_test_pca = pca.transform(X_test_final)

    svm = SVC(C=10, kernel="rbf", probability=True)
    svm.fit(X_train_pca, y_train_final)

    # 6. Predict and Analyze
    print("📈 Generating Metrics...")
    y_pred = svm.predict(X_test_pca)
    
    acc = accuracy_score(y_test_raw, y_pred)
    prec = precision_score(y_test_raw, y_pred, average='weighted')
    rec = recall_score(y_test_raw, y_pred, average='weighted')
    
    print("\n" + "="*30)
    print(f"✅ ACCURACY:  {acc*100:.2f}%")
    print(f"✅ PRECISION: {prec*100:.2f}%")
    print(f"✅ RECALL:    {rec*100:.2f}%")
    print("="*30 + "\n")

    print("📝 Classification Report:")
    print(classification_report(y_test_raw, y_pred))

    # 7. Plot Confusion Matrix
    cm = confusion_matrix(y_test_raw, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(list(set(labels))), 
                yticklabels=sorted(list(set(labels))))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
    plt.savefig('confusion_matrix.png')
    print("🖼️ Confusion Matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate()
