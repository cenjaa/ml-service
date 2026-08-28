# 🤖 ML Service — Face Recognition Attendance System

Machine learning microservice built with **FastAPI** that provides face recognition capabilities using a **RPCA + PCA + LDA + SVM** pipeline. Handles face inference (identification) and model training/retraining.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI |
| Server | Uvicorn |
| ML Pipeline | scikit-learn (PCA, LDA, SVM) |
| Image Processing | OpenCV, scikit-image |
| Optimization | PyGAD (Genetic Algorithm) |
| Object Storage | MinIO (via boto3) |
| Serialization | joblib |
| Containerization | Docker |

## Architecture

```
ml-service/
├── main.py                 # FastAPI app entry point
├── api/                    # API route handlers
│   ├── __init__.py
│   ├── health.py           # Health check endpoint
│   ├── infer.py            # Face inference (identification)
│   └── train.py            # Model training/retraining
├── core/                   # Core ML algorithms
│   ├── __init__.py
│   ├── rpca.py             # Robust PCA implementation
│   └── state.py            # Application state management
├── services/               # Business logic services
│   ├── __init__.py
│   ├── model_store.py      # Model loading & persistence
│   ├── storage.py          # MinIO storage operations
│   └── trainer.py          # Training pipeline orchestration
├── config/                 # Configuration
│   ├── __init__.py
│   └── settings.py         # Environment & YAML config loader
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── image.py            # Image preprocessing utilities
├── evaluate_model.py       # Model evaluation & metrics
├── rpca_algorithm.py       # RPCA algorithm reference
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build config
├── deploy/                 # Deployment config
│   ├── docker-compose.yml  # ML service stack
│   └── deploy.sh           # Deployment script
└── .env.example            # Environment variable template
```

## ML Pipeline

The face recognition pipeline consists of the following stages:

```
Input Image
    │
    ▼
┌──────────────────┐
│  Preprocessing   │  Resize to 100×100, grayscale conversion
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      RPCA        │  Robust PCA for noise/occlusion removal
│  (Low-rank +     │  Separates clean face from sparse noise
│   Sparse)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      PCA         │  Dimensionality reduction
│  (Eigenfaces)    │  Extracts principal components
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      LDA         │  Linear Discriminant Analysis
│  (Fisherfaces)   │  Maximizes class separability
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      SVM         │  Support Vector Machine classifier
│  (Multi-class)   │  Final face identification
└────────┬─────────┘
         │
         ▼
   Predicted Label
   + Confidence Score
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/infer` | Identify a face from an uploaded image |
| `POST` | `/train` | Trigger model retraining |

## Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- MinIO instance (for dataset/model storage)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### Configuration

Edit `.env` with your MinIO credentials:
```env
MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET=attendance
MINIO_USE_SSL=false
```

### Run Locally

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The service will start on `http://localhost:8001`.

### Run with Docker

```bash
# Create the shared Docker network (first time only)
docker network create app_network

# Start the ML service
cd deploy
docker compose up -d --build
```

## Model Files

Trained model files (~45MB total) are **not included** in the repository. They are stored separately and loaded at startup:

| File | Description | Size |
|------|-------------|------|
| `pca_transformer.pkl` | PCA transformation matrix | ~35 MB |
| `lda_transformer.pkl` | LDA transformation matrix | ~680 KB |
| `svm_model.pkl` | SVM classifier | ~110 KB |
| `svm_far.pkl` | SVM for far-distance faces | ~4.5 MB |
| `svm_near.pkl` | SVM for near-distance faces | ~4.5 MB |
| `label_map.pkl` | Label encoding map | ~52 B |

To train new models, use the `/train` endpoint or place dataset images in the `dataset/` directory.

## Related Services

- [**Backend**](https://github.com/cenjaa/backend-facerecognition) — Go REST API service
- [**Frontend**](https://github.com/cenjaa/frontend-facerecognition) — React-based kiosk UI
