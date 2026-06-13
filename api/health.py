"""
api/health.py
─────────────
Simple liveness / readiness endpoints.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
def health():
    """Returns 200 OK when the service is alive."""
    return {"status": "ok"}
