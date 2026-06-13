"""
services/storage.py
────────────────────
S3/MinIO client wrapper.
Handles bucket creation, dataset download, and model upload.
"""
import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


class S3Client:
    """Thin wrapper around boto3 for MinIO object storage."""

    def __init__(self, minio_config: dict):
        self.bucket_name = minio_config.get("bucket_name", "attendance")
        endpoint         = minio_config.get("endpoint", "localhost:9000")
        use_ssl          = minio_config.get("use_ssl", False)
        protocol         = "https" if use_ssl else "http"

        boto_config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 3},
        )

        self.s3 = boto3.client(
            "s3",
            endpoint_url=f"{protocol}://{endpoint}",
            aws_access_key_id=minio_config.get("access_key", "admin"),
            aws_secret_access_key=minio_config.get("secret_key", ""),
            config=boto_config,
            region_name="us-east-1",
        )
        self._ensure_bucket()

    # ── Bucket management ───────────────────────────────────────
    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
                print(f"☁️  Created bucket: '{self.bucket_name}'")
            except Exception as e:
                print(f"⚠️  Could not create bucket: {e}")

    # ── Dataset sync ────────────────────────────────────────────
    def download_dataset(self, local_dir: str):
        """Download all objects under the 'dataset/' prefix to local_dir."""
        os.makedirs(local_dir, exist_ok=True)
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix="dataset/"):
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                rel_path = key[len("dataset/"):]
                if not rel_path:
                    continue
                local_path = os.path.join(local_dir, rel_path.replace("/", os.sep))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3.download_file(self.bucket_name, key, local_path)
                print(f"   ↓ {key}")

    # ── Model upload ────────────────────────────────────────────
    def upload_models(self, local_dir: str):
        """Upload trained model artefacts to the 'models/' prefix."""
        model_files = [
            "svm_near.pkl",
            "svm_far.pkl",
            "pca_transformer.pkl",
            "label_map.pkl",
        ]
        for fname in model_files:
            local_path = os.path.join(local_dir, fname)
            if os.path.exists(local_path):
                self.s3.upload_file(local_path, self.bucket_name, f"models/{fname}")
                print(f"   ↑ Uploaded {fname}")
