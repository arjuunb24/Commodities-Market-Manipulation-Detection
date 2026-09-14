"""
src/detector
============
Module 2: Detection Layer. 
Extracts features and scores manipulation probability.
"""

from src.detector.features import FeatureEngine
from src.detector.anomaly import AnomalyDetector
from src.detector.classifiers import SupervisedDetector
from src.detector.pipeline import DetectorPipeline

__all__ = [
    "FeatureEngine",
    "AnomalyDetector",
    "SupervisedDetector",
    "DetectorPipeline"
]
