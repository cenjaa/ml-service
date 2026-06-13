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

        # Heuristic initialization
        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(max(self.m, self.n))

        if mu:
            self.mu = mu
        else:
            self.mu = 1.25 / norm(self.D, 2)
            
        self.rho = 1.5  # Growth rate for mu

    def fit(self, tol=1E-7, max_iter=1000):
        """
        IALM solver for faster convergence.
        """
        iter_count = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        
        d_norm = norm(self.D, 'fro')

        print(f"   -> [IALM] Starting optimized decomposition on {self.D.shape} matrix...")

        while (err > tol) and (iter_count < max_iter):
            # 1. Update L (Low-Rank)
            Lk = self.svd_thresholding(self.D - Sk + (1/self.mu) * Yk, 1/self.mu)

            # 2. Update S (Sparse)
            Sk = self.soft_thresholding(self.D - Lk + (1/self.mu) * Yk, self.lmbda/self.mu)

            # 3. Update Y (Lagrange Multiplier) and mu
            Z = self.D - Lk - Sk
            Yk = Yk + self.mu * Z
            self.mu = self.mu * self.rho

            # 4. Calculate Error
            err = norm(Z, 'fro') / d_norm

            iter_count += 1
            if iter_count % 10 == 0 or iter_count == 1:
                print(f"   -> [IALM] Iteration {iter_count}: Error {err:.7f}")

        self.L = Lk
        self.S = Sk
        print(f"   -> [IALM] Converged after {iter_count} iterations. Final Error: {err:.7f}")
        return Lk, Sk

    @staticmethod
    def clean_image(img_vec, iterations=5):
        """
        Specialized fast cleaning for a single image vector (inference).
        Runs a few iterations to strip away sparse noise (glare/shadows).
        """
        # Reshape to a column matrix for RPCA logic
        d = img_vec.reshape(-1, 1).astype(np.float32)
        s = np.zeros(d.shape)
        y = np.zeros(d.shape)
        mu = 1.25 / (norm(d, 2) + 1e-6)
        lmbda = 1 / np.sqrt(max(d.shape))
        
        l = np.zeros(d.shape)
        
        for _ in range(iterations):
            # 1. Low-rank update (SVD of vector is just scaling/thresholding)
            # For a vector, SVD is simple: norm is the only singular value
            temp_l = d - s + (1/mu) * y
            n = norm(temp_l)
            if n > 0:
                l = max(n - 1/mu, 0) * (temp_l / n)
            else:
                l = np.zeros(d.shape)
                
            # 2. Sparse update
            s = np.sign(d - l + (1/mu) * y) * np.maximum(np.abs(d - l + (1/mu) * y) - lmbda/mu, 0)
            
            # 3. Multiplier update
            y = y + mu * (d - l - s)
            mu *= 1.5
            
        return l.flatten().astype(np.uint8)

    def svd_thresholding(self, X, tau):
        U, S, V = svd(X, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.soft_thresholding(S, tau)), V))

    def soft_thresholding(self, x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
