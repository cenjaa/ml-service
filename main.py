import os
from fastapi import FastAPI

from config import MODELS_DIR, DATASET_DIR, DEBUG_DIR
from services.model_store import load_models
from api import health_router, infer_router, train_router

# ── Ensure required directories exist ──────────────────────────
for directory in (MODELS_DIR, DATASET_DIR, DEBUG_DIR):
    os.makedirs(directory, exist_ok=True)

# ── App factory ────────────────────────────────────────────────
app = FastAPI(
    title="Face Recognition ML Service",
    description="RPCA + PCA + SVM face recognition pipeline for the Attendance System.",
    version="1.0.0",
)

# ── Register routers ───────────────────────────────────────────
app.include_router(health_router)
app.include_router(infer_router)
app.include_router(train_router)


# ── Startup event ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_models()


# ── Local development entry point ──────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)