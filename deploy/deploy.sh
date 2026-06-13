#!/usr/bin/env bash
# =============================================================
#  deploy/deploy.sh — ML Service Stack
#  Run this on your VPS to deploy or update the ML service.
#
#  NOTE: Run the backend stack first (backend-facerecognition/deploy)
#        so the shared app_network and MinIO are already up.
#
#  Usage:
#    bash deploy.sh            # Start / update (no image rebuild)
#    bash deploy.sh --build    # Force rebuild of Docker image
#    bash deploy.sh --down     # Stop container (no data to lose)
# =============================================================

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$DEPLOY_DIR/docker-compose.yml"
BUILD_FLAG=""
DOWN_MODE=false

for arg in "$@"; do
  case $arg in
    --build) BUILD_FLAG="--build" ;;
    --down)  DOWN_MODE=true ;;
  esac
done

echo "=============================================="
echo "  ML Service — VPS Deploy"
echo "=============================================="

# ── Sanity checks ──────────────────────────────────────────────
if [ ! -f "$DEPLOY_DIR/.env" ]; then
  echo "ERROR: .env not found in deploy/!"
  echo "  cp $DEPLOY_DIR/.env.example $DEPLOY_DIR/.env"
  echo "  Then fill in your MinIO credentials."
  exit 1
fi

# ── Verify shared network exists ──────────────────────────────
if ! docker network inspect app_network &>/dev/null; then
  echo "ERROR: Docker network 'app_network' not found!"
  echo "  Deploy the backend stack first:"
  echo "    cd ../backend-facerecognition/deploy && bash deploy.sh"
  exit 1
fi

# ── Tear down ─────────────────────────────────────────────────
if [ "$DOWN_MODE" = true ]; then
  echo "[DOWN] Stopping ML service..."
  docker compose -f "$COMPOSE_FILE" down
  echo "✅  ML service stopped."
  exit 0
fi

echo "[1/3] Pulling latest code..."
git -C "$DEPLOY_DIR/.." pull --ff-only 2>/dev/null || echo "  Skipping git pull."

echo "[2/3] Building / updating Docker image..."
docker compose -f "$COMPOSE_FILE" up -d $BUILD_FLAG

echo "[3/3] Container status:"
docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "✅  ML service deploy complete!"
echo "  Logs:  docker compose -f $COMPOSE_FILE logs -f ml-service"
