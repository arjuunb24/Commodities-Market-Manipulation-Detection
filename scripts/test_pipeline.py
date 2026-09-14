"""
scripts/test_pipeline.py
========================
A simple script to test the Phase 3 DetectorPipeline on a simulation dataset.
"""

import sys
import yaml
import logging
from pathlib import Path

# Adjust path to find src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detector.pipeline import DetectorPipeline
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def main():
    # 1. Load the detector config
    config_path = REPO_ROOT / "configs" / "detector_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Point to the dataset via command line or use a default
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = REPO_ROOT / "data" / "runs" / "runs" / "20260914T102011Z_42"
    
    if not run_dir.exists():
        logger.error(f"Directory not found: {run_dir}. Did you delete it?")
        return

    logger.info(f"Initializing pipeline with config: {config_path.name}")
    pipeline = DetectorPipeline(config)

    # 3. Load and Preprocess (Generate X features and y labels)
    X, y = pipeline.load_and_preprocess(run_dir)
    logger.info(f"Generated Feature Matrix (X) shape: {X.shape}")
    logger.info(f"Generated Label Matrix (y) shape: {y.shape}")
    
    if X.empty:
        logger.warning("Feature matrix is empty. Exiting.")
        return

    # 4. Train the models (IsolationForest and XGBoost)
    logger.info("Training models...")
    pipeline.train(X, y)

    # 5. Score the dataset
    logger.info("Scoring dataset...")
    results = pipeline.score(X)
    
    logger.info(f"Scoring complete. Results shape: {results.shape}")
    print("\n--- Top 5 Most Anomalous Windows (Unsupervised) ---")
    print(results.sort_values("iso_score", ascending=False).head(5)[["iso_score", "iso_flag", "ensemble_flag"]])
    
    print("\n--- Top 5 Highest Probabilities of Spoofing (Supervised) ---")
    if "prob_spoofing" in results.columns:
        print(results.sort_values("prob_spoofing", ascending=False).head(5)[["prob_spoofing", "sup_flag", "ensemble_flag"]])

if __name__ == "__main__":
    main()
