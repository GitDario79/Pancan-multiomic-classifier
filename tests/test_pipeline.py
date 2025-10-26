import os, joblib, pandas as pd
from pathlib import Path

def test_artifacts_exist():
    assert Path("artifacts/toy/model.joblib").exists(), "Model artifact missing"

def test_metrics_non_empty():
    # run after evaluation
    assert Path("figures/confusion_matrix.png").exists(), "Confusion matrix not created"
