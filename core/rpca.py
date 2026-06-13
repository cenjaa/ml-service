"""
core/rpca.py
────────────
Robust PCA via Inexact Augmented Lagrange Multiplier (IALM).
Moved from rpca_algorithm.py — no logic changes.
"""
import numpy as np
from numpy.linalg import norm, svd


class R_PCA:
    def __init__(self, D, mu=None, lmbda=None):
        """
        Robust PCA via Inexact Augmented Lagrange Multiplier (IALM).
        Faster and more robust than the exact version.
        """
        self.D = D
        self.m, self.n = D.shape
        self.S = np.zeros(D.shape)
        self.Y = np.zeros(D.shape)

        # Heuristic initialisation
        self.lmbda = lmbda if lmbda else 1 / np.sqrt(max(self.m, self.n))
        self.mu    = mu    if mu    else 1.25 / norm(self.D, 2)
        self.rho   = 1.5  # Growth rate for mu

    # ── Full IALM solver (used during training) ────────────────
    def fit(self, tol=1E-7, max_iter=1000):
        """IALM solver for full matrix decomposition."""
        iter_count = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        d_norm = norm(self.D, "fro")

        print(f"   -> [IALM] Starting decomposition on {self.D.shape} matrix...")

        while (err > tol) and (iter_count < max_iter):
            Lk = self.svd_thresholding(self.D - Sk + (1 / self.mu) * Yk, 1 / self.mu)
            Sk = self.soft_thresholding(self.D - Lk + (1 / self.mu) * Yk, self.lmbda / self.mu)
            Z  = self.D - Lk - Sk
            Yk = Yk + self.mu * Z
            self.mu = self.mu * self.rho
            err = norm(Z, "fro") / d_norm
            iter_count += 1
            if iter_count % 10 == 0 or iter_count == 1:
                print(f"   -> [IALM] Iteration {iter_count}: Error {err:.7f}")

        self.L = Lk
        self.S = Sk
        print(f"   -> [IALM] Converged after {iter_count} iterations. Final Error: {err:.7f}")
        return Lk, Sk

    # ── Fast single-vector cleaner (used during inference) ─────
    @staticmethod
    def clean_image(img_vec, iterations: int = 5) -> np.ndarray:
        """
        Strip sparse noise (glare/shadows) from a single image vector.
        Runs a fixed number of IALM iterations for speed.

        Args:
            img_vec:    1-D numpy array, pixel values in [0, 255].
            iterations: Number of IALM steps (default 5).

        Returns:
            Cleaned vector as uint8.
        """
        d      = img_vec.reshape(-1, 1).astype(np.float32)
        s      = np.zeros(d.shape)
        y      = np.zeros(d.shape)
        mu     = 1.25 / (norm(d, 2) + 1e-6)
        lmbda  = 1 / np.sqrt(max(d.shape))
        l      = np.zeros(d.shape)

        for _ in range(iterations):
            # Low-rank update — for a vector SVD reduces to a scalar norm
            temp_l = d - s + (1 / mu) * y
            n = norm(temp_l)
            l = max(n - 1 / mu, 0) * (temp_l / n) if n > 0 else np.zeros(d.shape)

            # Sparse update
            s = (np.sign(d - l + (1 / mu) * y)
                 * np.maximum(np.abs(d - l + (1 / mu) * y) - lmbda / mu, 0))

            # Multiplier update
            y  = y + mu * (d - l - s)
            mu *= 1.5

        return l.flatten().astype(np.uint8)

    # ── Helpers ────────────────────────────────────────────────
    def svd_thresholding(self, X, tau):
        U, S, V = svd(X, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.soft_thresholding(S, tau)), V))

    @staticmethod
    def soft_thresholding(x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
