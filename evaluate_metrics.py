"""
AlertIQ — Comprehensive Metrics Evaluation
Extracts detailed metrics from Phase 2 (Data Pipeline) and Phase 3 (Suppression Model)
to benchmark against industry standards.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
RAW_PATH   = r"D:\College\RL_Project\data\PaySim.csv"
TRAIN_PATH = r"D:\College\RL_Project\outputs\train_feedback.csv"
TEST_PATH  = r"D:\College\RL_Project\outputs\test_feedback.csv"
MODEL_PATH = r"D:\College\RL_Project\models\suppression_model.pkl"

SEPARATOR = "\n" + "=" * 70


def phase1_raw_data_metrics():
    """Metrics on the raw PaySim dataset before any processing."""
    print(SEPARATOR)
    print("  PHASE 1: RAW DATA PROFILING (PaySim.csv)")
    print(SEPARATOR)

    df = pd.read_csv(RAW_PATH)
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    # Class balance
    fraud_count = df['isFraud'].sum()
    total = len(df)
    fraud_rate = fraud_count / total
    print(f"\n🔢 Class Distribution:")
    print(f"   Legitimate : {total - fraud_count:>10,}  ({(1 - fraud_rate) * 100:.4f}%)")
    print(f"   Fraudulent : {fraud_count:>10,}  ({fraud_rate * 100:.4f}%)")
    print(f"   Fraud Rate : 1 in {int(1 / fraud_rate):,}")

    # isFlaggedFraud comparison
    flagged = df['isFlaggedFraud'].sum()
    if fraud_count > 0:
        flagged_recall = df[(df['isFraud'] == 1) & (df['isFlaggedFraud'] == 1)].shape[0] / fraud_count
    else:
        flagged_recall = 0
    print(f"\n🚩 PaySim's Built-in isFlaggedFraud:")
    print(f"   Flagged count  : {flagged:,}")
    print(f"   Recall on fraud: {flagged_recall:.4f}  (captures {flagged_recall * 100:.2f}% of real fraud)")

    # Transaction type breakdown
    print(f"\n📂 Transaction Type Breakdown:")
    type_stats = df.groupby('type').agg(
        count=('type', 'size'),
        fraud_count=('isFraud', 'sum')
    )
    type_stats['fraud_rate'] = type_stats['fraud_count'] / type_stats['count']
    type_stats['pct_of_total'] = type_stats['count'] / total * 100
    print(type_stats.to_string())

    os.makedirs('outputs', exist_ok=True)
    report_path_1 = 'outputs/transaction_type_report.csv'
    type_stats.to_csv(report_path_1)
    
    plt.figure(figsize=(10, 6))
    plt.bar(type_stats.index.astype(str), type_stats['count'], color='#1f77b4', label='Total Transactions')
    plt.bar(type_stats.index.astype(str), type_stats['fraud_count'], color='#d62728', label='Fraud Transactions')
    plt.yscale('log')
    plt.title('Transaction Volume vs Fraud by Type (Log Scale)')
    plt.ylabel('Count (Log)')
    plt.xlabel('Transaction Type')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('outputs/transaction_type_chart.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n📁 Saved Phase 1 report to: {report_path_1} and chart to: outputs/transaction_type_chart.png")

    # Amount statistics
    print(f"\n💰 Transaction Amount Stats:")
    print(f"   Mean   : ${df['amount'].mean():>14,.2f}")
    print(f"   Median : ${df['amount'].median():>14,.2f}")
    print(f"   Max    : ${df['amount'].max():>14,.2f}")
    print(f"   Std    : ${df['amount'].std():>14,.2f}")

    fraud_amt = df[df['isFraud'] == 1]['amount']
    legit_amt = df[df['isFraud'] == 0]['amount']
    print(f"\n   Fraud avg amount    : ${fraud_amt.mean():>14,.2f}")
    print(f"   Legit avg amount    : ${legit_amt.mean():>14,.2f}")

    # Missing values
    missing = df.isnull().sum()
    print(f"\n⚠️  Missing Values: {missing.sum()} total")

    return df


def phase2_alert_generation_metrics(raw_df):
    """Metrics from the alert generation & analyst simulation pipeline."""
    print(SEPARATOR)
    print("  PHASE 2: ALERT GENERATION & ANALYST SIMULATION")
    print(SEPARATOR)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    alerts_df = pd.concat([train_df, test_df], ignore_index=True)

    total_txns = len(raw_df)
    total_alerts = len(alerts_df)
    alert_rate = total_alerts / total_txns

    print(f"\n📢 Alert Generation Stats:")
    print(f"   Total transactions : {total_txns:>10,}")
    print(f"   Alerts generated   : {total_alerts:>10,}")
    print(f"   Alert rate         : {alert_rate * 100:.2f}%")

    # How many real frauds are in the alert set?
    fraud_in_alerts = alerts_df['isFraud'].sum()
    fraud_in_raw = raw_df['isFraud'].sum() if 'isFraud' in raw_df.columns else 0
    alert_fraud_capture = fraud_in_alerts / fraud_in_raw if fraud_in_raw > 0 else 0

    print(f"\n🎯 Alert Rule Effectiveness:")
    print(f"   Frauds in raw data       : {int(fraud_in_raw):>10,}")
    print(f"   Frauds captured by alerts: {int(fraud_in_alerts):>10,}")
    print(f"   Fraud capture rate       : {alert_fraud_capture * 100:.2f}%")

    # False positive rate of the rule engine
    non_fraud_alerts = total_alerts - fraud_in_alerts
    alert_precision = fraud_in_alerts / total_alerts if total_alerts > 0 else 0
    alert_fpr = non_fraud_alerts / total_alerts if total_alerts > 0 else 0

    print(f"\n📈 Alert Quality (Rule Engine):")
    print(f"   True positives (fraud alerts)  : {int(fraud_in_alerts):>10,}")
    print(f"   False positives (benign alerts): {int(non_fraud_alerts):>10,}")
    print(f"   Alert precision (fraud/alerts) : {alert_precision * 100:.4f}%")
    print(f"   False positive rate            : {alert_fpr * 100:.2f}%")
    print(f"   ⚡ This means {alert_fpr * 100:.1f}% of all alerts are NOT fraud — this is the problem we solve!")

    # Alert reason distribution
    if 'alert_reason' in alerts_df.columns:
        print(f"\n📋 Alert Reason Distribution:")
        reason_stats = alerts_df.groupby('alert_reason').agg(
            count=('alert_reason', 'size'),
            fraud_count=('isFraud', 'sum')
        )
        reason_stats['fraud_rate'] = reason_stats['fraud_count'] / reason_stats['count'] * 100
        reason_stats['pct_of_alerts'] = reason_stats['count'] / total_alerts * 100
        print(reason_stats.to_string())

        report_path_2 = 'outputs/alert_reason_report.csv'
        reason_stats.to_csv(report_path_2)
        
        plt.figure(figsize=(8, 5))
        plt.bar(reason_stats.index.astype(str), reason_stats['count'], color='#2ca02c')
        plt.title('Alerts Generated by Rule')
        plt.ylabel('Number of Alerts')
        for i, val in enumerate(reason_stats['count']):
            plt.text(i, val, f'{int(val):,}', ha='center', va='bottom')
        plt.savefig('outputs/alert_reason_chart.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\n📁 Saved Alert Reason report/chart to outputs/")

    # Analyst simulation metrics
    print(f"\n👤 Analyst Simulation Stats:")
    analyst_dist = alerts_df['analyst_type'].value_counts()
    for atype, count in analyst_dist.items():
        print(f"   {atype:>8s} analysts: {count:>10,} ({count / total_alerts * 100:.1f}%)")

    # Analyst decision distribution
    decision_dist = alerts_df['analyst_decision'].value_counts()
    print(f"\n📝 Analyst Decisions:")
    for decision, count in decision_dist.items():
        print(f"   {decision:>10s}: {count:>10,} ({count / total_alerts * 100:.2f}%)")

    decision_df = pd.DataFrame({'decision': decision_dist.index, 'count': decision_dist.values})
    decision_df.to_csv('outputs/analyst_performance_report.csv', index=False)
    
    plt.figure(figsize=(6, 6))
    plt.pie(decision_dist.values, labels=decision_dist.index, autopct='%1.1f%%', startangle=90, colors=['#a1c9f4', '#ffb482'])
    plt.title('Analyst Decisions (Escalate vs Dismiss)')
    plt.savefig('outputs/analyst_decision_chart.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n📁 Saved Analyst Decision report/chart to outputs/")

    # Key: Did analysts correctly escalate fraud?
    fraud_alerts = alerts_df[alerts_df['isFraud'] == 1]
    fraud_escalated = (fraud_alerts['analyst_decision'] == 'escalate').sum()
    fraud_dismissed = (fraud_alerts['analyst_decision'] == 'dismiss').sum()
    fraud_recall_analyst = fraud_escalated / len(fraud_alerts) if len(fraud_alerts) > 0 else 0

    print(f"\n🔍 Analyst Performance on Real Fraud:")
    print(f"   Fraud escalated (correct) : {fraud_escalated:>10,}")
    print(f"   Fraud dismissed (MISSED)  : {fraud_dismissed:>10,}")
    print(f"   Fraud recall by analysts  : {fraud_recall_analyst * 100:.4f}%")

    # Non-fraud handling
    nonfraud_alerts = alerts_df[alerts_df['isFraud'] == 0]
    nonfraud_dismissed = (nonfraud_alerts['analyst_decision'] == 'dismiss').sum()
    nonfraud_escalated = (nonfraud_alerts['analyst_decision'] == 'escalate').sum()
    nonfraud_dismiss_rate = nonfraud_dismissed / len(nonfraud_alerts) if len(nonfraud_alerts) > 0 else 0

    print(f"\n✅ Analyst Performance on Non-Fraud:")
    print(f"   Non-fraud dismissed (correct)   : {nonfraud_dismissed:>10,}")
    print(f"   Non-fraud escalated (wasted)    : {nonfraud_escalated:>10,}")
    print(f"   Non-fraud correct dismiss rate  : {nonfraud_dismiss_rate * 100:.4f}%")

    # Train/test split
    print(f"\n📦 Train/Test Split:")
    print(f"   Train : {len(train_df):>10,}  ({len(train_df) / total_alerts * 100:.1f}%)")
    print(f"   Test  : {len(test_df):>10,}  ({len(test_df) / total_alerts * 100:.1f}%)")
    print(f"   Train fraud rate: {train_df['isFraud'].mean() * 100:.4f}%")
    print(f"   Test  fraud rate: {test_df['isFraud'].mean() * 100:.4f}%")

    return alerts_df


def phase3_suppression_model_metrics():
    """Detailed ML model evaluation metrics."""
    print(SEPARATOR)
    print("  PHASE 3: SUPPRESSION MODEL EVALUATION")
    print(SEPARATOR)

    if not os.path.exists(MODEL_PATH):
        print("  ⚠️  Model not found. Run phase3.py first.")
        return

    model = joblib.load(MODEL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    y_true_fraud = df_test['isFraud'].values
    y_true_dismiss = (df_test['analyst_decision'] == 'dismiss').astype(int).values

    drop_cols = ['analyst_decision', 'analyst_type', 'analyst_weight', 'analyst_id', 'isFraud']
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    # Predictions
    dismiss_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ── 3A: Classification Performance (on analyst decision prediction) ──
    print(f"\n📊 3A: Model Classification (Predicting Analyst Dismiss Decision)")
    print(f"{'─' * 60}")
    print(classification_report(y_true_dismiss, y_pred, target_names=['Escalate', 'Dismiss']))

    cm = confusion_matrix(y_true_dismiss, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"   Confusion Matrix:")
    print(f"                    Predicted Escalate  Predicted Dismiss")
    print(f"   Actual Escalate  {tn:>18,}  {fp:>17,}")
    print(f"   Actual Dismiss   {fn:>18,}  {tp:>17,}")

    # ── 3B: Probabilistic Calibration ──
    print(f"\n📊 3B: Probability Calibration")
    print(f"{'─' * 60}")
    auc = roc_auc_score(y_true_dismiss, dismiss_proba)
    ap  = average_precision_score(y_true_dismiss, dismiss_proba)
    brier = brier_score_loss(y_true_dismiss, dismiss_proba)
    logloss = log_loss(y_true_dismiss, dismiss_proba)

    print(f"   ROC-AUC Score          : {auc:.6f}")
    print(f"   Average Precision (AP) : {ap:.6f}")
    print(f"   Brier Score            : {brier:.6f}  (lower is better, 0 = perfect)")
    print(f"   Log Loss               : {logloss:.6f}  (lower is better)")

    # Calibration check
    try:
        fraction_pos, mean_predicted = calibration_curve(y_true_dismiss, dismiss_proba, n_bins=10)
        print(f"\n   Calibration Table (10 bins):")
        print(f"   {'Predicted Prob':>15s}  {'Actual Fraction':>16s}")
        for mp, fp_val in zip(mean_predicted, fraction_pos):
            bar = '█' * int(fp_val * 30)
            print(f"   {mp:>15.4f}  {fp_val:>16.4f}  {bar}")
    except Exception:
        print("   (Could not compute calibration curve)")

    # ── 3C: Confidence Gating at Multiple Thresholds ──
    print(f"\n📊 3C: Confidence Gating — Suppress vs. Escalate")
    print(f"{'─' * 60}")
    print(f"   {'Threshold':>10s}  {'Suppressed':>11s}  {'Escalated':>10s}  {'Fraud Recall':>13s}  {'FP Reduction':>13s}  {'Fraud Leaked':>13s}")

    for thr in [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]:
        suppress = dismiss_proba >= thr
        escalated = ~suppress

        fraud_mask = y_true_fraud == 1
        nonfraud_mask = y_true_fraud == 0

        fraud_recall = escalated[fraud_mask].mean() if fraud_mask.sum() > 0 else 1.0
        fp_reduction = suppress[nonfraud_mask].mean() if nonfraud_mask.sum() > 0 else 0.0
        fraud_leaked = suppress[fraud_mask].sum()

        print(f"   {thr:>10.2f}  {suppress.sum():>11,}  {escalated.sum():>10,}  {fraud_recall:>13.5f}  {fp_reduction:>13.5f}  {fraud_leaked:>13,}")

    # ── 3D: Cost-Sensitivity Analysis ──
    print(f"\n📊 3D: Cost-Sensitivity Analysis (at threshold = 0.95)")
    print(f"{'─' * 60}")
    thr = 0.95
    suppress = dismiss_proba >= thr

    # Industry standard costs
    cost_per_review = 15          # $ per analyst alert review
    cost_missed_fraud = 50_000    # $ penalty per missed fraud case
    cost_false_escalation = 50    # $ for unnecessary escalation

    alerts_suppressed = suppress.sum()
    alerts_remaining = (~suppress).sum()
    fraud_leaked = suppress[y_true_fraud == 1].sum()
    fp_suppressed = suppress[y_true_fraud == 0].sum()

    savings = fp_suppressed * cost_per_review
    fraud_cost = fraud_leaked * cost_missed_fraud
    net_savings = savings - fraud_cost

    print(f"   Alerts suppressed       : {alerts_suppressed:>10,}")
    print(f"   Alerts still reviewed   : {alerts_remaining:>10,}")
    print(f"   FP alerts removed       : {fp_suppressed:>10,}")
    print(f"   Fraud cases leaked      : {fraud_leaked:>10,}")
    print(f"   ───────────────────────────────────────")
    print(f"   Review savings          : ${savings:>12,.0f}  ({fp_suppressed:,} × ${cost_per_review})")
    print(f"   Fraud leak penalty      : ${fraud_cost:>12,.0f}  ({fraud_leaked:,} × ${cost_missed_fraud:,})")
    print(f"   Net savings             : ${net_savings:>12,.0f}  {'✅ POSITIVE' if net_savings > 0 else '❌ NEGATIVE'}")

    # ── 3E: Probability Distribution Analysis ──
    print(f"\n📊 3E: Dismiss Probability Distribution")
    print(f"{'─' * 60}")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"   Percentile distribution of P(dismiss):")
    for p in percentiles:
        val = np.percentile(dismiss_proba, p)
        print(f"   P{p:<3d}: {val:.6f}")

    print(f"\n   Mean P(dismiss)   : {dismiss_proba.mean():.6f}")
    print(f"   Median P(dismiss): {np.median(dismiss_proba):.6f}")
    print(f"   Std P(dismiss)   : {dismiss_proba.std():.6f}")

    # High-confidence buckets
    print(f"\n   High-confidence breakdown:")
    for cutoff in [0.9, 0.95, 0.99]:
        count = (dismiss_proba >= cutoff).sum()
        pct = count / len(dismiss_proba) * 100
        print(f"   P(dismiss) >= {cutoff}: {count:>10,} alerts ({pct:.2f}%)")


def industry_benchmark():
    """Print industry benchmarks for comparison."""
    print(SEPARATOR)
    print("  INDUSTRY BENCHMARKS & COMPARISON")
    print(SEPARATOR)

    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                  INDUSTRY STANDARD BENCHMARKS                  │
    ├──────────────────────────┬──────────────────────────────────────┤
    │ Metric                   │ Industry Standard                   │
    ├──────────────────────────┼──────────────────────────────────────┤
    │ Alert false positive rate│ 90-99% (typical AML systems)        │
    │ Fraud recall (minimum)   │ ≥ 95% (regulatory requirement)      │
    │ ROC-AUC (good model)     │ ≥ 0.85                              │
    │ Brier Score (calibrated) │ ≤ 0.10                              │
    │ FP Reduction target      │ 30-60% (practical goal)             │
    │ Cost per alert review    │ $15-$50 per alert                   │
    │ Missed fraud penalty     │ $50K-$500K+ per case                │
    │ Model must be            │ Explainable (regulatory)            │
    │ Drift monitoring         │ Required (Basel/FATF guidance)       │
    └──────────────────────────┴──────────────────────────────────────┘

    Notes:
    • Logistic Regression is preferred in AML for its interpretability ✅
    • Cost-sensitive learning is expected — current model lacks this ⚠️
    • Confidence gating is a valid approach for safe suppression ✅
    • RL decision layer will add adaptive threshold optimization ⏳
    • Drift detection is mandatory for production compliance ⏳
    • SHAP/LIME explanations are needed for audit trails ⏳
    """)


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  AlertIQ — Comprehensive Metrics Evaluation")
    print("█" * 70)

    raw_df = phase1_raw_data_metrics()
    phase2_alert_generation_metrics(raw_df)
    phase3_suppression_model_metrics()
    industry_benchmark()

    print(SEPARATOR)
    print("  EVALUATION COMPLETE")
    print(SEPARATOR)
