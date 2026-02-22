"""
download_and_train.py
=====================
Run this to download the Kaggle dataset, train + cache the ensemble model.
The model is always retrained and the pickle file is replaced on every run.
All training metrics and artifacts are logged to MLflow automatically.

Usage:
    python download_and_train.py           # always retrains + overwrites pickle
    python download_and_train.py --skip    # skip retrain, load existing pickle
    python download_and_train.py --no-mlflow  # train without MLflow logging

Kaggle credentials must be stored in a .env file:
    KAGGLE_USERNAME=<your_username>
    KAGGLE_KEY=<your_api_key>
"""

import sys
import os
import pickle
from dotenv import load_dotenv
import kagglehub
from classifier import EnsembleIRClassifier, KAGGLE_DATASET, MODEL_PATH


def main():
    # -----------------------------
    # Load .env file
    # -----------------------------
    load_dotenv()

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file")

    # Explicitly set environment variables for kagglehub
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    # Flags
    skip_retrain = "--skip" in sys.argv
    log_mlflow   = "--no-mlflow" not in sys.argv

    print("=" * 60)
    print("  Smart-Support MVR — Dataset Download & Training")
    print(f"  MLflow logging: {'ON' if log_mlflow else 'OFF'}")
    print("=" * 60)

    # -----------------------------
    # Step 1: Download dataset
    # -----------------------------
    print(f"\n[Step 1] Downloading dataset: {KAGGLE_DATASET}")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"  → Files saved to: {path}\n")

    # -----------------------------
    # Step 2: Train model
    # -----------------------------
    if skip_retrain and os.path.exists(MODEL_PATH):
        print(f"[Step 2] --skip flag set: loading existing model from '{MODEL_PATH}'")
        clf = EnsembleIRClassifier()
        clf.load_or_train(force_retrain=False, log_mlflow=log_mlflow)
    else:
        if os.path.exists(MODEL_PATH):
            print(f"[Step 2] Removing existing model: {MODEL_PATH}")
            os.remove(MODEL_PATH)
        print("[Step 2] Training ensemble IR classifier (fresh) …")
        clf = EnsembleIRClassifier()
        clf.load_or_train(force_retrain=True, log_mlflow=log_mlflow)

    print(f"\n✓ Model saved to: {MODEL_PATH}")
    print("✓ Done. Start the server with:  uvicorn main:app --reload")

    if log_mlflow:
        print("\n📊 View training metrics and visualizations:")
        print("   mlflow ui --port 5000")
        print("   → http://localhost:5000")
        print("\n📈 Generate dashboard: python mlops_dashboard.py")


if __name__ == "__main__":
    main()

