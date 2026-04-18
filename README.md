# AlertIQ: Risk-Constrained Reinforcement Learning Framework for Adaptive AML Alert Governance

> An intelligent, multi-layered AI system that optimizes AML alert handling by learning **when to suppress false positives and when to escalate real threats** — using reinforcement learning at the decision layer.

---

## Problem Statement

Anti-Money Laundering (AML) monitoring systems intentionally **over-alert** because missing a true fraud case is catastrophically costly. This floods analysts with alerts, the vast majority of which are **false positives** (~95-99%).

```
Transaction Data → Rule-Based Monitoring → Thousands of Alerts → Human Review → Most dismissed
```

**AlertIQ solves this** by learning from historical patterns to intelligently suppress benign alerts while preserving near-perfect fraud recall.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AlertIQ Pipeline                             │
├─────────────┬───────────────┬──────────────┬───────────────────┤
│  Phase 2    │   Phase 3     │   Phase 4    │   Phase 5 & 6     │
│  Data &     │   Suppression │   RL Decision│   Safety &        │
│  Alerts     │   Model       │   Policy     │   Deployment      │
├─────────────┼───────────────┼──────────────┼───────────────────┤
│ PaySim Data │ Cost-Sensitive│ DQN Agent    │ Drift Detection   │
│ Rule Engine │ LogReg +      │ Contextual   │ (PSI + KL)        │
│ (4 Rules)   │ Platt Scaling │ Bandit Agent │                   │
│ Analyst Sim │               │              │ Explainability    │
│ Trust Score │ Calibrated    │ ε-Greedy     │ Engine            │
│             │ Probabilities │ Exploration  │                   │
│ Train/Test  │               │ vs Static    │ FastAPI Backend   │
│ Split       │ Threshold     │ Baseline     │                   │
│             │ Evaluation    │ Comparison   │ REST API          │
└─────────────┴───────────────┴──────────────┴───────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **Rule-Based Alert Engine** | 4 configurable rules mimicking real AML monitoring |
| **Analyst Trust Simulation** | Junior/Senior analysts with evolving trust scores |
| **Cost-Sensitive ML** | Logistic Regression with 50:1 fraud miss penalty |
| **Platt Calibration** | CalibratedClassifierCV for reliable probabilities |
| **DQN Agent** | Deep Q-Network with experience replay + target network |
| **Contextual Bandit** | Interpretable linear agent with feature weights |
| **Asymmetric Rewards** | -50 for suppressing fraud vs +1 for suppressing FP |
| **Drift Detection** | PSI + KL Divergence with automatic fallback mode |
| **Explainability** | Human-readable, audit-ready decision explanations |
| **REST API** | FastAPI backend with Swagger docs |

---

## Quick Start

### 1. Setup

```bash
# Clone and install
cd RL_Project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Place Data

Download [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `PaySim.csv` in `data/`.

### 3. Run Full Pipeline

```bash
python run_all.py
```

Or run phases individually:

```bash
python run_phase2.py          # Data → Alerts → Analyst Sim → Split
python run_phase3.py          # Train suppression model
python run_phase4_rl.py       # Train RL agents
python run_phase5_drift.py    # Test drift detection
```

### 4. Start API

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Structure

```
RL_Project/
├── run_phase2.py              # Data pipeline runner
├── run_phase3.py              # Model training runner
├── run_phase4_rl.py           # RL training runner
├── run_phase5_drift.py        # Drift testing runner
├── run_all.py                 # Full pipeline orchestrator
├── evaluate_metrics.py        # Comprehensive metrics evaluation
├── requirements.txt
├── README.md
│
├── src/                       # Core library
│   ├── __init__.py
│   ├── config.py              # All configuration constants
│   ├── data_loader.py         # Load & clean PaySim
│   ├── alert_rules.py         # Rule-based alert generation (R1-R4)
│   ├── analyst_simulator.py   # Analyst feedback with trust scoring
│   ├── dataset_builder.py     # Stratified train/test split
│   ├── suppression_model.py   # Cost-sensitive LogReg + Platt calibration
│   ├── gating_evaluator.py    # Static threshold evaluation
│   ├── metrics.py             # Metrics utilities
│   ├── rl_environment.py      # MDP environment (state/action/reward)
│   ├── rl_agent.py            # DQN + Contextual Bandit agents
│   ├── rl_trainer.py          # Training + policy comparison
│   ├── rl_pipeline.py         # End-to-end RL pipeline
│   ├── drift_detector.py      # PSI + KL drift detection + fallback
│   └── explainability.py      # Decision explanation engine
│
├── api/                       # FastAPI backend
│   ├── __init__.py
│   ├── main.py                # App entry point + model loading
│   ├── schemas.py             # Pydantic request/response models
│   └── routes/
│       ├── alerts.py          # POST /api/alerts/evaluate, /batch
│       ├── pipeline.py        # POST /api/pipeline/run
│       ├── drift.py           # POST /api/drift/check, /simulate
│       └── models.py          # GET /api/models/info, /comparison
│
├── data/                      # Raw dataset
│   └── PaySim.csv
├── models/                    # Trained models
│   └── suppression_model.pkl
└── outputs/                   # Pipeline outputs
    ├── train_feedback.csv
    ├── test_feedback.csv
    ├── threshold_report.csv
    └── rl_outputs/
        ├── dqn_agent.pkl
        ├── bandit_agent.pkl
        ├── policy_comparison.csv
        ├── training_curves.png
        └── evaluation_report.txt
```

---

## MDP Formulation

| Component | Details |
|---|---|
| **State** | `[prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload_level]` |
| **Actions** | `0 = SUPPRESS`, `1 = ESCALATE` |
| **Reward** | Suppress Fraud: **-50**, Suppress Benign: **+1**, Escalate Fraud: **+1**, Escalate Benign: **-0.1** |
| **Objective** | Maximize cumulative reward while maintaining ≥99% fraud recall |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/api/alerts/evaluate` | Evaluate single alert |
| `POST` | `/api/alerts/batch` | Evaluate batch of alerts |
| `GET` | `/api/alerts/stats` | System statistics |
| `POST` | `/api/pipeline/run` | Trigger pipeline execution |
| `GET` | `/api/pipeline/status` | Pipeline status |
| `GET` | `/api/pipeline/results` | Latest results |
| `POST` | `/api/drift/check` | Run drift detection |
| `POST` | `/api/drift/simulate` | Simulate drift (demo) |
| `GET` | `/api/drift/status` | Drift monitoring status |
| `GET` | `/api/models/info` | Model metadata |
| `GET` | `/api/models/comparison` | RL vs baseline results |
| `GET` | `/api/models/training-curves` | Training curves image |

---

## Configuration

All parameters live in `src/config.py`:

```python
# Reward structure
RL_REWARD_SUPPRESS_FRAUD = -50.0    # CATASTROPHIC
RL_REWARD_SUPPRESS_BENIGN = 1.0     # Desired
RL_REWARD_ESCALATE_FRAUD = 1.0      # Correct
RL_REWARD_ESCALATE_BENIGN = -0.1    # Acceptable

# Drift thresholds
DRIFT_PSI_WARNING = 0.1
DRIFT_PSI_CRITICAL = 0.25

# Cost sensitivity
MISSED_FRAUD_COST = 50              # 50:1 cost ratio
```

---

## Technology Stack

- **ML**: scikit-learn (Logistic Regression, CalibratedClassifierCV)
- **RL**: PyTorch (DQN), NumPy (Contextual Bandit)
- **Backend**: FastAPI + Uvicorn
- **Data**: PaySim synthetic financial dataset
- **Visualization**: Matplotlib, Seaborn
- **Python**: 3.8+

---

## Novelty

This project is **not** a standalone fraud classifier. Its contribution is a **system-level integration**:

1. Rule-based alert generation (realistic AML simulation)
2. Analyst feedback learning with trust scoring
3. Cost-sensitive probability modeling with Platt calibration
4. **RL-based decision optimization** (DQN + Contextual Bandit)
5. Drift detection with automatic safety fallback
6. Compliance-ready explainability
7. Production-grade REST API

The RL agent replaces static thresholds with an adaptive policy that learns contextually optimal suppression decisions.

---

## References

- Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015 (DQN)
- Li et al., "A contextual-bandit approach to personalized news article recommendation", JMLR 2010
- Elkan, "The foundations of cost-sensitive learning", SIGKDD 2001
- PaySim: E. Lopez-Rojas, "Applying PAYSIM financial simulator", 2016
