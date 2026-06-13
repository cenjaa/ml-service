"""
rpca_algorithm.py — DEPRECATED shim.
The R_PCA class has moved to core/rpca.py.
This file is kept only for backwards compatibility with evaluate_model.py.
"""
from core.rpca import R_PCA  # noqa: F401

__all__ = ["R_PCA"]
