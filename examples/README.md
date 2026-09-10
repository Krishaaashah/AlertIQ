# AlertIQ — Examples

Small, self-contained scripts that exercise the real project code end to
end, in seconds, without needing the full 6.36M-row Kaggle dataset or a
trained model. Each one prints its own output so you can read it top to
bottom.

Run them from the project root:

```bash
python examples/01_rule_engine_walkthrough.py
python examples/02_drift_detection_demo.py
python examples/03_decision_explainability_walkthrough.py
```

For example 4, start the API in one terminal, then run the client in another:

```bash
uvicorn api.main:app --reload --port 8000
python examples/04_call_api_client.py
```

| Script | What it shows |
| --- | --- |
| `01_rule_engine_walkthrough.py` | Phase 1 — loading raw transactions and generating rule-based alerts (R1-R4) |
| `02_drift_detection_demo.py` | Phase 5 — PSI/KL drift scoring, both a stable batch and a simulated drifted one, and the resulting safety fallback |
| `03_decision_explainability_walkthrough.py` | Phases 3/4/6 — running alerts through the adaptive-threshold policy and generating an audit-readable explanation for each decision |
| `04_call_api_client.py` | Calling the live FastAPI backend the way an external system would, using only the standard library |

All four run correctly with **no trained model on disk** — the suppression
model and RL agent fall back to sane defaults (0.5 probability, static
threshold) when their `.pkl` files aren't present, which is exactly what
you'll see right after a fresh clone. To see the full trained pipeline in
action, follow the "Full pipeline" instructions in the root `README.md`.
