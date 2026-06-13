"""api/__init__.py"""
from .health import router as health_router
from .infer  import router as infer_router
from .train  import router as train_router

__all__ = ["health_router", "infer_router", "train_router"]
