"""
Example 4 — Calling the live API

Demonstrates hitting a running AlertIQ API instance the way an external
banking system would: POST a transaction's features, get back a decision.

Uses only the Python standard library (urllib) so it has zero extra
dependencies beyond a running server.

Start the server first, in a separate terminal:
    uvicorn api.main:app --reload --port 8000

Then run this script:
    python examples/04_call_api_client.py
"""

import json
import sys
import urllib.error
import urllib.request

API_URL = "http://localhost:8000"

# A high-risk-looking transaction: large CASH_OUT that drains almost the
# entire sender balance. See api/schemas.py -> AlertFeatures for the full
# field list.
SAMPLE_ALERT = {
    "amount": 250000.0,
    "step": 12,
    "rule_count": 2,
    "alert_reason": "R1",
    "balance_drain_ratio": 0.95,
    "oldbalanceOrg": 260000.0,
    "newbalanceOrig": 13000.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 250000.0,
    "type_CASH_OUT": 1,
    "type_DEBIT": 0,
    "type_PAYMENT": 0,
    "type_TRANSFER": 0,
}


def post_json(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{API_URL}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    try:
        health = get_json("/health")
    except urllib.error.URLError:
        print(f"Could not reach {API_URL}.")
        print("Start the server first: uvicorn api.main:app --reload --port 8000")
        sys.exit(1)

    print("Server health:", json.dumps(health, indent=2))

    print("\nEvaluating one alert via POST /api/alerts/evaluate ...")
    decision = post_json("/api/alerts/evaluate", SAMPLE_ALERT)
    print(json.dumps(decision, indent=2))

    print("\nEvaluating the same alert twice via POST /api/alerts/batch ...")
    batch_result = post_json("/api/alerts/batch", {"alerts": [SAMPLE_ALERT, SAMPLE_ALERT]})
    print(f"Suppressed: {batch_result['suppressed']}, Escalated: {batch_result['escalated']}")
    print(f"Summary: {json.dumps(batch_result['summary'], indent=2)}")


if __name__ == "__main__":
    main()
