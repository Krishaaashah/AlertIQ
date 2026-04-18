# AlertIQ: Reinforcement Learning for AML Alert Governance

## Overview
AlertIQ is an end-to-end Machine Learning and Reinforcement Learning pipeline designed to counteract "Alert Fatigue" in Anti-Money Laundering (AML) and financial fraud tracking systems. By dynamically predicting fraud probability and using a Deep Q-Network to govern alert triaging, the system successfully filters out 97%+ of false positives while maintaining regulatory complaint fraud recall.

## Project Architecture

### Phase 1: Raw Data Ingestion & EDA
* **Dataset:** Simulated PaySim mobile money transaction data (approx. 6.3 million rows).
* **Imbalance Challenge:** Only 0.13% of transactions constitute actual fraud.
* **Function:** Evaluates distributions, network pathways, and time-series patterns.

### Phase 2: Rule-Based Alert Generation
* **Function:** Simulates a legacy bank SOC (Security Operations Center) using hardcoded behavioral rules (e.g., "High-Value Transfer," "Rapid Account Drain").
* **Result:** Successfully catches 96.48% of fraud, but generates a 99.61% False Positive Rate. Requires over 2 million manual reviews, simulating severe Alert Fatigue.

### Phase 3: Machine Learning Suppression (The Foundation)
* **Model:** Cost-Sensitive Logistic Regression equipped with Platt Scaling calibration.
* **Metrics:** Achieves a heavily calibrated probability mapping (ROC-AUC: 0.9961, Brier Score: 0.0088).
* **Function:** Serves as the probabilistic foundation for the RL agent, separating structural noise from potential risks.

### Phase 4: Reinforcement Learning Decision Engine
* **Objective:** Replace static threshold logic with a dynamic actor capable of optimizing for true economic cost (analyst salary vs fraud fines).
* **Implementation:** Deep Q-Network (DQN) with Balanced Experience Replay.
* **Results:** The agent learned a highly efficient suppression policy, eliminating 97.88% of false-positive workload while successfully escalating 97.48% of actual fraud.

### Phase 5: MLOps, Drift & Explainability
* **Drift Detection:** A Population Stability Index (PSI) and Kullback-Leibler (KL) Divergence monitor tracks shifts in feature distribution, capable of triggering "Safe Mode" overrides if incoming data strays from training boundaries.
* **Explainability:** SHAP/LIME integration provides localized feature-importance mapping, allowing SOC managers to understand the exact mathematical reasoning behind every RL intervention.

## FastAPI Backend Integration
The `api/` directory orchestrates the model deployments:
* `/api/alerts/evaluate`: Processes live JSON transactions through Phase 3 and Phase 4 logic.
* `/api/drift/status`: Continuously monitors ingestion payloads.
* `/api/explainability/feature_importance`: Reverses the suppression decision into human-readable rule vectors.

## Installation & Execution
```bash
# Setup Virtual Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the API Server
uvicorn api.main:app --reload

# Run End-to-End Pipeline Evaluation
python run_eda.py
```
