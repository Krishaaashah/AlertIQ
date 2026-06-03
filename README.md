# AlertIQ

Risk-constrained reinforcement learning for adaptive AML alert governance.

AlertIQ is an end-to-end AML alert suppression and escalation system. It combines rule-based monitoring, analyst-feedback simulation, calibrated fraud-risk modeling, reinforcement learning, drift detection, explainability, a FastAPI backend, and a React dashboard.

The goal is simple: reduce false-positive analyst workload while protecting fraud recall and keeping every decision explainable enough for audit review.


## Problem Statement

AML systems generate a huge number of alerts, and almost all of them are false positives. Analysts spend time reviewing non-fraud cases, productivity drops, and the risk of missing real fraud increases.

<img width="2366" height="902" alt="image" src="https://github.com/user-attachments/assets/f26f6952-2bd1-4307-8ceb-9a86bae99911" />

## Why AlertIQ

Traditional AML systems intentionally over-alert because missing a true fraud case is far more expensive than reviewing a benign transaction. That safety-first posture creates severe alert fatigue: analysts spend most of their time dismissing false positives.

AlertIQ treats alert governance as a decision problem instead of only a classification problem. A calibrated model estimates fraud probability, and a reinforcement learning policy learns when to suppress low-risk alerts or escalate cases for review under an asymmetric reward structure.

<img width="1305" height="737" alt="image" src="https://github.com/user-attachments/assets/c5ca3cae-ace5-4562-87b0-a826452dd0d3" />

## Core Results

| Area | Result |
| --- | --- |
| Raw dataset | 6.36M PaySim transactions |
| Rule alerts generated | 2.05M alerts |
| Fraud capture from rules | 96.48% |
| ML model | Cost-sensitive Logistic Regression + Platt calibration |
| ROC-AUC | 0.9961 |
| Best static threshold | 0.80 |
| Fraud recall at threshold | 0.9933 |
| False-positive reduction at threshold | 0.9627 |
| RL fraud catch rate | 0.9748 |
| RL false-positive reduction | 0.9788 |
| RL workload savings | 0.9751 |

## Objectives
<img width="792" height="726" alt="image" src="https://github.com/user-attachments/assets/dcd3e367-9dbf-4440-ae8c-9651560d9887" />


The project success criteria are:

- Capture at least 95% fraud using rule-based detection.
- Build an ML suppression model to reduce false alerts.
- Reduce workload by at least 90%.
- Maintain high fraud recall for compliance safety.
- Add reinforcement learning, drift detection, API deployment, and dashboard visibility.

## 5-Phase Pipeline

<img width="2708" height="1688" alt="image" src="https://github.com/user-attachments/assets/08e90fef-7986-4182-81a8-5967e40576d3" />
<img width="2544" height="1688" alt="image" src="https://github.com/user-attachments/assets/adae6162-c96a-4cc9-9de4-f0c5b9d0725f" />


1. **Data and rules**
   Load PaySim transactions, engineer AML features, and generate rule-based alerts.

2. **ML suppression**
   Train a cost-sensitive suppression model with calibrated fraud probabilities.

3. **RL optimization**
   Build the alert decision environment and train a DQN policy against static thresholds.

4. **Drift and safety**
   Monitor PSI and KL divergence, then trigger fallback safety behavior under critical drift.

5. **API and deployment**
   Serve alert decisions through FastAPI and expose monitoring through the React dashboard.

## System Components

| Layer | What it does |
| --- | --- |
| Rule engine | Generates AML alerts from high-value, burst, risky-type, and balance-drain rules |
| Analyst simulator | Creates labeled analyst feedback with junior/senior trust scoring |
| Suppression model | Produces calibrated `prob_fraud` values for the decision layer |
| Threshold evaluator | Benchmarks static gating thresholds from 0.50 to 0.99 |
| RL environment | Models alert governance as suppress/escalate decisions with asymmetric costs |
| DQN agent | Learns adaptive suppression policy using replay and target network stabilization |
| Contextual bandit | Provides a lightweight interpretable baseline |
| Drift detector | Tracks PSI and KL divergence for safety monitoring |
| Explainability engine | Produces audit-readable decision explanations |
| FastAPI backend | Serves evaluation, drift, model, and pipeline endpoints |
| React dashboard | Displays live metrics, charts, research flow, and model status |

## Quick Start

## Setup

### Backend Setup

```bash
git clone https://github.com/sahilawatramani/AlertIQ.git
cd AlertIQ

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the PaySim dataset from Kaggle and place it at:

```text
data/PaySim.csv
```

Run the complete pipeline:

```bash
python run_all.py
```

Or run phases individually:

```bash
python run_eda.py
python run_phase2.py
python run_phase3.py
python run_phase4_rl.py
python run_phase5_drift.py
```

Start the API:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Open the API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will print the local dashboard URL, usually [http://localhost:5173](http://localhost:5173).

## Phase Outputs

### Phase 1: Rule-Based Alert Engine

<img width="1332" height="464" alt="image" src="https://github.com/user-attachments/assets/c0d3bd0e-8208-4357-abb1-5e6d35077765" />
<img width="990" height="330" alt="image" src="https://github.com/user-attachments/assets/1e43b46d-e847-4c67-a5ba-e4440cc6fedb" />


Phase 1 loads PaySim transactions, engineers features, applies AML rules, and creates train/test alert feedback datasets.

### Phase 2: ML Suppression Model

<img width="1863" height="794" alt="image" src="https://github.com/user-attachments/assets/00b5ff68-f376-4c4e-a01f-1bd28f2a5ff0" />
<img width="594" height="462" alt="image" src="https://github.com/user-attachments/assets/788d12c9-b84e-4a80-8e92-f6ab685adc4f" />
<img width="1767" height="742" alt="image" src="https://github.com/user-attachments/assets/e781bc01-fdac-4db2-b6a6-75e86be50b7d" />
<img width="1811" height="716" alt="image" src="https://github.com/user-attachments/assets/f7199043-8444-447b-8a43-dcf94d246485" />


Phase 2 trains the calibrated suppression model and evaluates thresholds for fraud recall, false-positive reduction, and workload savings.

### Phase 3: Reinforcement Learning
<img width="1853" height="760" alt="image" src="https://github.com/user-attachments/assets/e9b01d2a-78f1-4d5e-b559-6e54432c17a4" />

Phase 3 trains the DQN policy and compares it against static thresholds and a contextual bandit baseline.

### Phase 4: Drift Detection and Safety

<img width="548" height="292" alt="image" src="https://github.com/user-attachments/assets/49c7c99b-5668-49ef-8cd1-193d99f9514d" />


Phase 4 monitors incoming data distribution shifts using PSI and KL divergence, then activates fallback behavior when drift becomes critical.

### Phase 5: MLOps and Deployment

<img width="878" height="354" alt="image" src="https://github.com/user-attachments/assets/ee191c65-b58d-4cae-9c3e-c71992c569c2" />


Phase 5 exposes the backend API, dashboard, model information, drift endpoints, and deployment-ready monitoring surfaces.

## API Surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Root health check |
| `GET` | `/health` | Detailed service health |
| `POST` | `/api/alerts/evaluate` | Evaluate one alert |
| `POST` | `/api/alerts/batch` | Evaluate alerts in batch |
| `GET` | `/api/alerts/stats` | Alert statistics |
| `POST` | `/api/pipeline/run` | Trigger pipeline execution |
| `GET` | `/api/pipeline/status` | Pipeline status |
| `GET` | `/api/pipeline/results` | Latest pipeline results |
| `POST` | `/api/drift/check` | Check drift on incoming data |
| `POST` | `/api/drift/simulate` | Simulate drift scenarios |
| `GET` | `/api/drift/status` | Current drift status |
| `GET` | `/api/models/info` | Model metadata |
| `GET` | `/api/models/comparison` | RL vs baseline comparison |
| `GET` | `/api/models/training-curves` | Training curve image |

## RL Formulation

| Component | Definition |
| --- | --- |
| State | `[prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload_level]` |
| Actions | `0 = SUPPRESS`, `1 = ESCALATE` |
| Reward | Suppress fraud: `-50`, suppress benign: `+1`, escalate fraud: `+1`, escalate benign: `-0.1` |
| Objective | Maximize cumulative reward while preserving high fraud recall |

## Folder Structure

```text
AlertIQ/
|-- api/                         # FastAPI backend
|   |-- main.py
|   |-- schemas.py
|   `-- routes/
|-- frontend/                    # React + Vite dashboard
|   |-- src/
|   `-- package.json
|-- src/                         # Core ML/RL pipeline
|   |-- alert_rules.py
|   |-- analyst_simulator.py
|   |-- data_loader.py
|   |-- dataset_builder.py
|   |-- suppression_model.py
|   |-- rl_agent.py
|   |-- rl_environment.py
|   |-- rl_trainer.py
|   |-- drift_detector.py
|   `-- explainability.py
|-- outputs/                     # Generated charts and reports
|-- docs/readme-assets/          # Images extracted from the project PPTX
|-- run_all.py
|-- run_eda.py
|-- run_phase2.py
|-- run_phase3.py
|-- run_phase4_rl.py
|-- run_phase5_drift.py
|-- evaluate_metrics.py
`-- requirements.txt
```


## Technology Stack

| Area | Tools |
| --- | --- |
| Machine learning | scikit-learn, NumPy, pandas |
| Reinforcement learning | PyTorch |
| Calibration and evaluation | Platt scaling, ROC-AUC, PR curves, confusion analysis |
| Drift monitoring | PSI, KL divergence |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React, Vite |
| Visualization | Matplotlib, Seaborn |


## References

- Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015.
- Li et al., "A contextual-bandit approach to personalized news article recommendation", JMLR, 2010.
- Elkan, "The foundations of cost-sensitive learning", SIGKDD, 2001.
- PaySim: E. Lopez-Rojas, "Applying PAYSIM financial simulator", 2016.
