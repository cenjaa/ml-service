import os
import shutil
import traceback
import numpy as np
import cv2
import joblib
from typing import Optional
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import pygad

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
        pixels          = row[:10000] * 255.0 
        lbp             = row[10000:]
        cleaned_pixels  = rpca.clean_image(pixels, iterations=5).flatten() / 255.0
        cleaned.append(np.concatenate([cleaned_pixels, lbp]))
    return np.array(cleaned)


def _ga_optimize_svm(X_train: np.ndarray, y_train: list,
                     num_generations: int = 15, sol_per_pop: int = 10):
    """
    Genetic Algorithm to optimise SVM hyperparameters C and gamma.

    Each individual encodes two genes on a log-10 scale:
        gene[0]  ->  C     = 10^gene[0]   (search range 0.01 -- 1 000)
        gene[1]  ->  gamma  = 10^gene[1]   (search range 0.0001 -- 10)

    Fitness is the mean accuracy of 3-fold stratified cross-validation.
    Returns (best_C, best_gamma, best_fitness).
    """
    y_arr = np.array(y_train)

    def fitness_func(ga_instance, solution, solution_idx):
        C     = 10 ** solution[0]
        gamma = 10 ** solution[1]
        try:
            clf    = SVC(C=C, gamma=gamma, kernel="rbf")
            scores = cross_val_score(clf, X_train, y_arr, cv=3, scoring="accuracy")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def on_generation(ga_instance):
        gen  = ga_instance.generations_completed
        best = ga_instance.best_solution()[1]
        print(f"   -> [GA] Generation {gen}/{num_generations}  "
              f"best fitness = {best:.4f}")
        _set_state(
            "training",
            f"GA-SVM: generation {gen}/{num_generations} "
            f"(best accuracy {best:.2%})...",
            65 + int(15 * gen / num_generations),
        )

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=2,
        gene_space=[
            {"low": -2, "high": 3},   # log10(C):     0.01  ... 1 000
            {"low": -4, "high": 1},   # log10(gamma): 0.0001 ... 10
        ],
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=50,
        on_generation=on_generation,
        suppress_warnings=True,
    )
    ga.run()

    best_solution, best_fitness, _ = ga.best_solution()
    best_C     = 10 ** best_solution[0]
    best_gamma = 10 ** best_solution[1]

    return best_C, best_gamma, best_fitness


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
            augs = augment_and_extract(img)
            X_near.extend([augs[0], augs[1], augs[2]]) 
            y_near.extend([lbl,     lbl,     lbl    ])
            X_far.extend( [augs[0], augs[1], augs[3]])
            y_far.extend( [lbl,     lbl,     lbl    ])

        # Step 4 — RPCA cleaning
        _set_state("training", "Cleaning pixel vectors via RPCA (IALM)...", 45)
        X_n_clean = _rpca_clean_batch(np.array(X_near))
        X_f_clean = _rpca_clean_batch(np.array(X_far))

        # Step 5 — PCA dimensionality reduction (shared subspace)
        _set_state("training", "Fitting shared PCA subspace...", 60)
        pca_model = PCA(n_components=0.98, whiten=True)
        pca_model.fit(np.vstack([X_n_clean, X_f_clean]))

        # Step 6 — GA-SVM: Optimise hyperparameters via Genetic Algorithm
        #   - Initialise GA population
        #   - Evolution loop: evaluate fitness (SVM CV accuracy),
        #     selection, crossover, mutation
        #   - Repeat until max generation reached
        #   - Extract best hyperparameters (C, gamma)
        _set_state("training", "Initialising GA population for SVM optimisation...", 65)
        X_combined_pca = pca_model.transform(np.vstack([X_n_clean, X_f_clean]))
        y_combined = y_near + y_far

        best_C, best_gamma, best_fitness = _ga_optimize_svm(
            X_combined_pca, y_combined
        )
        print(f"\U0001f9ec [GA-SVM] Best params: C={best_C:.4f}, "
              f"gamma={best_gamma:.6f}, fitness={best_fitness:.4f}")

        # Step 7 — Train final SVM classifiers with GA-optimised params
        _set_state(
            "training",
            f"Training final SVMs with GA params "
            f"(C={best_C:.2f}, \u03b3={best_gamma:.4f})...",
            82,
        )
        svm_n = SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True).fit(
            pca_model.transform(X_n_clean), y_near
        )
        svm_f = SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True).fit(
            pca_model.transform(X_f_clean), y_far
        )

        # Step 8 — Persist
        _set_state("training", "Saving models...", 90)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(svm_n,                            os.path.join(MODELS_DIR, "svm_near.pkl"))
        joblib.dump(svm_f,                            os.path.join(MODELS_DIR, "svm_far.pkl"))
        joblib.dump(pca_model,                        os.path.join(MODELS_DIR, "pca_transformer.pkl"))
        joblib.dump({u: u for u in set(labels)},      os.path.join(MODELS_DIR, "label_map.pkl"))

        if s3_client:
            s3_client.upload_models(MODELS_DIR)

        # Step 9 — Clean up dataset and hot-reload models
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
