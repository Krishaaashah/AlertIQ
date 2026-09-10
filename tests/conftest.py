"""
Shared pytest fixtures for the AlertIQ test suite.

Fixtures build small, deterministic, synthetic data in-memory so the suite
runs in seconds without needing the full PaySim download or a trained model.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make `src` and `api` importable when running `pytest` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "sample_paysim.csv",
)


@pytest.fixture(scope="session")
def raw_sample_df() -> pd.DataFrame:
    """Load the small synthetic PaySim-shaped sample shipped in dataset/, unmodified."""
    if not os.path.exists(SAMPLE_DATA_PATH):
        pytest.skip(
            "dataset/sample_paysim.csv not found — run "
            "'python dataset/generate_sample.py' first."
        )
    return pd.read_csv(SAMPLE_DATA_PATH)


@pytest.fixture(scope="session")
def encoded_sample_df() -> pd.DataFrame:
    """Sample data run through the real src.data_loader.load_and_clean()."""
    if not os.path.exists(SAMPLE_DATA_PATH):
        pytest.skip(
            "dataset/sample_paysim.csv not found — run "
            "'python dataset/generate_sample.py' first."
        )
    from src.data_loader import load_and_clean
    return load_and_clean(SAMPLE_DATA_PATH)


@pytest.fixture
def synthetic_eval_df() -> pd.DataFrame:
    """
    A small hand-built 'post ML model' evaluation frame, i.e. the shape of
    data the RL environment and gating evaluator expect: one row per alert,
    with a calibrated prob_fraud already attached.
    """
    rng = np.random.default_rng(0)
    n = 200
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:10] = 1  # 5% fraud rate, deliberately front-loaded for easy slicing
    rng.shuffle(is_fraud)

    # Fraud cases get a higher (but not perfect) prob_fraud so tests can
    # exercise both "the model was right" and "the model was wrong" paths.
    prob_fraud = np.where(
        is_fraud == 1,
        rng.uniform(0.6, 0.99, n),
        rng.uniform(0.0, 0.4, n),
    )

    df = pd.DataFrame({
        "amount": rng.lognormal(9, 1.5, n).round(2),
        "isFraud": is_fraud,
        "prob_fraud": prob_fraud,
        "prob_benign": 1 - prob_fraud,
        "rule_count": rng.integers(1, 4, n),
        "alert_reason": rng.choice(["R1", "R2", "R3", "R4"], n),
    })
    return df
