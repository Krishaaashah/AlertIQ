"""
AlertIQ — Comprehensive EDA & Evaluation Dashboard
Generates publication-ready visualizations and metrics for every pipeline phase.

Outputs:
  outputs/eda/           — All EDA charts and data profiles
  outputs/evaluation/    — Model evaluation plots (ROC, PR, calibration, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss, f1_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

from src.config import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    SUPPRESSION_MODEL_PATH, OUTPUT_DIR, RL_OUTPUT_DIR,
    EDA_OUTPUT_DIR, EVAL_OUTPUT_DIR
)

# ─── Style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = {'fraud': '#e74c3c', 'benign': '#2ecc71', 'primary': '#3498db',
          'secondary': '#9b59b6', 'accent': '#f39c12', 'dark': '#2c3e50'}

SEP = "\n" + "=" * 70


def ensure_dirs():
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 1: RAW DATA EDA
# ═════════════════════════════════════════════════════════════════════════
def phase1_eda():
    print(SEP)
    print("  PHASE 1 — RAW DATA EDA (PaySim)")
    print(SEP)

    df = pd.read_csv(RAW_DATA_PATH)
    n = len(df)
    fraud_n = df['isFraud'].sum()
    fraud_rate = fraud_n / n

    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraud: {fraud_n:,} ({fraud_rate*100:.4f}%) — 1 in {int(1/fraud_rate):,}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    # ── 1A: Class imbalance pie + bar ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie
    labels = ['Legitimate', 'Fraudulent']
    sizes = [n - fraud_n, fraud_n]
    explode = (0, 0.1)
    axes[0].pie(sizes, explode=explode, labels=labels, autopct='%1.3f%%',
                colors=[COLORS['benign'], COLORS['fraud']],
                shadow=True, startangle=90, textprops={'fontsize': 11})
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')

    # Bar by transaction type
    type_fraud = df.groupby('type')['isFraud'].agg(['sum', 'count'])
    type_fraud['benign'] = type_fraud['count'] - type_fraud['sum']
    type_fraud = type_fraud.sort_values('count', ascending=True)

    y = range(len(type_fraud))
    axes[1].barh(y, type_fraud['benign'], color=COLORS['benign'], label='Benign')
    axes[1].barh(y, type_fraud['sum'], left=type_fraud['benign'],
                 color=COLORS['fraud'], label='Fraud')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(type_fraud.index)
    axes[1].set_xlabel('Count')
    axes[1].set_title('Transactions by Type', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_xscale('log')
    for i, (_, row) in enumerate(type_fraud.iterrows()):
        axes[1].text(row['count'] * 1.1, i, f"{int(row['count']):,}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '01_class_distribution.png'), bbox_inches='tight')
    plt.close()

    # ── 1B: Amount distributions ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fraud_amounts = df[df['isFraud'] == 1]['amount']
    legit_amounts = df[df['isFraud'] == 0]['amount'].sample(n=min(50000, n-fraud_n), random_state=42)

    axes[0].hist(np.log1p(legit_amounts), bins=60, alpha=0.7, color=COLORS['benign'],
                 label=f'Legitimate (μ=${legit_amounts.mean():,.0f})', density=True)
    axes[0].hist(np.log1p(fraud_amounts), bins=60, alpha=0.7, color=COLORS['fraud'],
                 label=f'Fraud (μ=${fraud_amounts.mean():,.0f})', density=True)
    axes[0].set_xlabel('log(1 + amount)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Transaction Amount Distribution (Log Scale)', fontsize=13, fontweight='bold')
    axes[0].legend()

    # Box plot
    box_data = pd.DataFrame({
        'Amount': pd.concat([fraud_amounts, legit_amounts.head(len(fraud_amounts))]),
        'Class': ['Fraud'] * len(fraud_amounts) + ['Legitimate'] * min(len(fraud_amounts), len(legit_amounts))
    })
    sns.boxplot(data=box_data, x='Class', y='Amount', hue='Class', palette=[COLORS['fraud'], COLORS['benign']],
                ax=axes[1], showfliers=False, legend=False)
    axes[1].set_title('Amount by Class (outliers hidden)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Amount ($)')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '02_amount_distributions.png'), bbox_inches='tight')
    plt.close()

    # ── 1C: Temporal patterns ──
    fig, ax = plt.subplots(figsize=(12, 5))
    step_fraud = df.groupby('step')['isFraud'].sum()
    step_all = df.groupby('step').size()
    step_rate = step_fraud / step_all

    ax.fill_between(step_rate.index, step_rate.values, alpha=0.3, color=COLORS['fraud'])
    ax.plot(step_rate.index, step_rate.values, color=COLORS['fraud'], linewidth=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Fraud Rate')
    ax.set_title('Fraud Rate Over Time', fontsize=14, fontweight='bold')
    ax.axhline(y=fraud_rate, color=COLORS['dark'], linestyle='--', alpha=0.5, label=f'Overall rate = {fraud_rate:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '03_temporal_fraud_rate.png'), bbox_inches='tight')
    plt.close()

    # ── 1D: Feature correlations ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'isFlaggedFraud' in numeric_cols:
        numeric_cols.remove('isFlaggedFraud')

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '04_correlation_matrix.png'), bbox_inches='tight')
    plt.close()

    # ── 1E: Balance drain analysis ──
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sample = df.sample(n=min(100000, n), random_state=42)
    fraud_mask = df_sample['isFraud'] == 1
    ax.scatter(df_sample.loc[~fraud_mask, 'oldbalanceOrg'],
               df_sample.loc[~fraud_mask, 'newbalanceOrig'],
               alpha=0.05, s=2, c=COLORS['benign'], label='Legitimate')
    ax.scatter(df_sample.loc[fraud_mask, 'oldbalanceOrg'],
               df_sample.loc[fraud_mask, 'newbalanceOrig'],
               alpha=0.5, s=8, c=COLORS['fraud'], label='Fraud')
    ax.set_xlabel('Original Balance')
    ax.set_ylabel('New Balance')
    ax.set_title('Balance Before vs After (Fraud Drains to $0)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, df_sample['oldbalanceOrg'].quantile(0.99))
    ax.set_ylim(0, df_sample['newbalanceOrig'].quantile(0.99))
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '05_balance_drain.png'), bbox_inches='tight')
    plt.close()

    print(f"  ✅ Phase 1 EDA saved to {EDA_OUTPUT_DIR}/01-05_*.png")
    return df


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2: ALERT GENERATION & ANALYST SIMULATION EDA
# ═════════════════════════════════════════════════════════════════════════
def phase2_eda(raw_df):
    print(SEP)
    print("  PHASE 2 — ALERT GENERATION & ANALYST SIMULATION")
    print(SEP)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    alerts_df = pd.concat([train_df, test_df], ignore_index=True)

    total_txns = len(raw_df)
    total_alerts = len(alerts_df)
    fraud_in_alerts = alerts_df['isFraud'].sum()
    fraud_in_raw = raw_df['isFraud'].sum()
    alert_rate = total_alerts / total_txns
    fraud_capture_rate = fraud_in_alerts / fraud_in_raw if fraud_in_raw > 0 else 0

    print(f"  Alerts generated: {total_alerts:,} / {total_txns:,} = {alert_rate*100:.2f}%")
    print(f"  Fraud capture rate: {fraud_capture_rate*100:.2f}%")
    print(f"  Alert FP rate: {(1 - fraud_in_alerts/total_alerts)*100:.2f}%")

    # ── 2A: Alert generation funnel ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Funnel
    stages = ['All Txns', 'Flagged\n(Alerts)', 'True Fraud\nin Alerts']
    values = [total_txns, total_alerts, fraud_in_alerts]
    colors = [COLORS['primary'], COLORS['accent'], COLORS['fraud']]
    bars = axes[0].bar(stages, values, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_yscale('log')
    axes[0].set_title('Alert Generation Funnel', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Count (log)')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val * 1.3,
                     f'{val:,}', ha='center', fontsize=10, fontweight='bold')

    # Alert reasons
    if 'alert_reason' in alerts_df.columns:
        reason_stats = alerts_df.groupby('alert_reason').agg(
            count=('alert_reason', 'size'),
            fraud_count=('isFraud', 'sum')
        )
        reason_stats['fraud_rate'] = reason_stats['fraud_count'] / reason_stats['count'] * 100

        x = range(len(reason_stats))
        width = 0.35
        axes[1].bar([i - width/2 for i in x], reason_stats['count'],
                    width, label='Total Alerts', color=COLORS['primary'])
        axes[1].bar([i + width/2 for i in x], reason_stats['fraud_count'],
                    width, label='Contains Fraud', color=COLORS['fraud'])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(reason_stats.index)
        axes[1].set_title('Alerts by Rule', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].set_ylabel('Count')

        # Fraud rate per rule
        axes[2].bar(reason_stats.index, reason_stats['fraud_rate'], color=COLORS['secondary'])
        axes[2].set_title('Fraud Rate by Alert Rule (%)', fontsize=13, fontweight='bold')
        axes[2].set_ylabel('Fraud Rate (%)')
        for i, (idx, row) in enumerate(reason_stats.iterrows()):
            axes[2].text(i, row['fraud_rate'] + 0.1, f"{row['fraud_rate']:.2f}%",
                        ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '06_alert_generation.png'), bbox_inches='tight')
    plt.close()

    # ── 2B: Analyst simulation analysis ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Decision distribution
    if 'analyst_decision' in alerts_df.columns:
        decisions = alerts_df['analyst_decision'].value_counts()
        axes[0].pie(decisions.values, labels=decisions.index, autopct='%1.1f%%',
                    colors=[COLORS['benign'], COLORS['fraud']], startangle=90,
                    textprops={'fontsize': 11})
        axes[0].set_title('Analyst Decision Distribution', fontsize=13, fontweight='bold')

    # Analyst type accuracy on fraud
    if 'analyst_type' in alerts_df.columns:
        fraud_alerts = alerts_df[alerts_df['isFraud'] == 1]
        if len(fraud_alerts) > 0:
            analyst_acc = fraud_alerts.groupby('analyst_type').apply(
                lambda x: (x['analyst_decision'] == 'escalate').mean()
            )
            axes[1].bar(analyst_acc.index, analyst_acc.values * 100,
                       color=[COLORS['primary'], COLORS['secondary']])
            axes[1].set_title('Fraud Escalation Rate by Analyst Type', fontsize=13, fontweight='bold')
            axes[1].set_ylabel('Fraud Recall (%)')
            axes[1].set_ylim(95, 100.5)
            for i, (idx, val) in enumerate(analyst_acc.items()):
                axes[1].text(i, val * 100 + 0.1, f"{val*100:.2f}%", ha='center', fontsize=10)

    # Train/test balance check
    train_fraud = train_df['isFraud'].mean()
    test_fraud = test_df['isFraud'].mean()
    axes[2].bar(['Train', 'Test'], [train_fraud * 100, test_fraud * 100],
               color=[COLORS['primary'], COLORS['accent']])
    axes[2].set_title('Fraud Rate: Train vs Test', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Fraud Rate (%)')
    for i, v in enumerate([train_fraud * 100, test_fraud * 100]):
        axes[2].text(i, v + 0.01, f"{v:.3f}%", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, '07_analyst_simulation.png'), bbox_inches='tight')
    plt.close()

    # ── Metrics table ──
    metrics = {
        'Total Transactions': f'{total_txns:,}',
        'Alerts Generated': f'{total_alerts:,}',
        'Alert Rate': f'{alert_rate*100:.2f}%',
        'Fraud in Alerts': f'{int(fraud_in_alerts):,}',
        'Fraud Capture Rate': f'{fraud_capture_rate*100:.2f}%',
        'Alert FP Rate': f'{(1 - fraud_in_alerts/total_alerts)*100:.2f}%',
        'Train Size': f'{len(train_df):,}',
        'Test Size': f'{len(test_df):,}',
        'Train Fraud Rate': f'{train_fraud*100:.4f}%',
        'Test Fraud Rate': f'{test_fraud*100:.4f}%',
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(os.path.join(EDA_OUTPUT_DIR, 'phase2_metrics.csv'), index=False)

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"  ✅ Phase 2 EDA saved to {EDA_OUTPUT_DIR}/06-07_*.png")
    return alerts_df


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 3: SUPPRESSION MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════
def phase3_evaluation():
    print(SEP)
    print("  PHASE 3 — SUPPRESSION MODEL EVALUATION")
    print(SEP)

    if not os.path.exists(SUPPRESSION_MODEL_PATH):
        print("  ⚠️  Model not found. Run run_phase3.py first.")
        return

    model = joblib.load(SUPPRESSION_MODEL_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    y_true_fraud = df_test['isFraud'].values

    drop_cols = ['analyst_decision', 'analyst_type', 'analyst_weight', 'analyst_id', 'isFraud']
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    prob_benign = model.predict_proba(X_test)[:, 0]
    prob_fraud = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Core metrics (Measuring ability to predict Fraud correctly)
    auc_roc = roc_auc_score(y_true_fraud, prob_fraud)
    ap = average_precision_score(y_true_fraud, prob_fraud)
    brier = brier_score_loss(y_true_fraud, prob_fraud)
    logloss_val = log_loss(y_true_fraud, prob_fraud)
    f1 = f1_score(y_true_fraud, y_pred)
    mcc = matthews_corrcoef(y_true_fraud, y_pred)

    print(f"  ROC-AUC:            {auc_roc:.6f}")
    print(f"  Average Precision:  {ap:.6f}")
    print(f"  Brier Score:        {brier:.6f}")
    print(f"  Log Loss:           {logloss_val:.6f}")
    print(f"  F1 Score:           {f1:.6f}")
    print(f"  MCC:                {mcc:.6f}")

    # ── 3A: ROC + PR Curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true_fraud, prob_fraud)
    axes[0].plot(fpr, tpr, color=COLORS['primary'], linewidth=2,
                 label=f'ROC (AUC = {auc_roc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color=COLORS['primary'])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve for Fraud Detection', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(y_true_fraud, prob_fraud)
    axes[1].plot(recall, precision, color=COLORS['secondary'], linewidth=2,
                 label=f'PR (AP = {ap:.4f})')
    axes[1].fill_between(recall, precision, alpha=0.1, color=COLORS['secondary'])
    no_skill = y_true_fraud.mean()
    axes[1].axhline(y=no_skill, color='k', linestyle='--', alpha=0.3, label=f'No Skill ({no_skill:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '08_roc_pr_curves.png'), bbox_inches='tight')
    plt.close()

    # ── 3B: Confusion Matrix + Calibration ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true_fraud, y_pred)
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
                xticklabels=['Benign', 'Fraud'], yticklabels=['Benign', 'Fraud'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Calibration curve
    try:
        fraction_pos, mean_predicted = calibration_curve(y_true_fraud, prob_fraud, n_bins=10)
        axes[1].plot(mean_predicted, fraction_pos, 's-', color=COLORS['primary'],
                     linewidth=2, markersize=8, label='Model Calibration')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
        axes[1].fill_between(mean_predicted, fraction_pos, mean_predicted, alpha=0.15,
                            color=COLORS['fraud'])
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Fraction of Positives')
        axes[1].set_title(f'Calibration Curve (Brier={brier:.4f})', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Calibration error: {e}', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '09_confusion_calibration.png'), bbox_inches='tight')
    plt.close()

    # ── 3C: Probability Distribution ──
    fig, ax = plt.subplots(figsize=(12, 5))
    # Analyzing the distribution of the Benign Probability score explicitly used for thresholds
    fraud_proba = prob_benign[y_true_fraud == 1]
    benign_proba = prob_benign[y_true_fraud == 0]

    ax.hist(benign_proba, bins=80, alpha=0.7, color=COLORS['benign'],
            label=f'Benign (n={len(benign_proba):,})', density=True)
    ax.hist(fraud_proba, bins=80, alpha=0.7, color=COLORS['fraud'],
            label=f'Fraud (n={len(fraud_proba):,})', density=True)
    ax.axvline(x=0.95, color=COLORS['dark'], linestyle='--', linewidth=2,
               label='Threshold = 0.95')
    ax.set_xlabel('P(dismiss)')
    ax.set_ylabel('Density')
    ax.set_title('Dismiss Probability Distribution by True Label', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '10_probability_distribution.png'), bbox_inches='tight')
    plt.close()

    # ── 3D: Threshold Sweep Analysis ──
    thresholds = np.arange(0.50, 1.00, 0.01)
    results = []
    for thr in thresholds:
        suppress = prob_benign >= thr
        fraud_mask = y_true_fraud == 1
        nonfraud_mask = y_true_fraud == 0

        fraud_recall = (~suppress)[fraud_mask].mean() if fraud_mask.sum() > 0 else 1.0
        fp_reduction = suppress[nonfraud_mask].mean() if nonfraud_mask.sum() > 0 else 0.0
        fraud_leaked = suppress[fraud_mask].sum()
        suppressed_pct = suppress.mean()

        results.append({
            'threshold': thr,
            'fraud_recall': fraud_recall,
            'fp_reduction': fp_reduction,
            'fraud_leaked': fraud_leaked,
            'suppressed_pct': suppressed_pct
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(EVAL_OUTPUT_DIR, 'threshold_sweep.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(results_df['threshold'], results_df['fraud_recall'] * 100,
                 color=COLORS['fraud'], linewidth=2, label='Fraud Recall (%)')
    axes[0].plot(results_df['threshold'], results_df['fp_reduction'] * 100,
                 color=COLORS['benign'], linewidth=2, label='FP Reduction (%)')
    axes[0].axvline(x=0.95, color=COLORS['dark'], linestyle='--', alpha=0.5, label='t=0.95')
    axes[0].set_xlabel('Suppression Threshold')
    axes[0].set_ylabel('Rate (%)')
    axes[0].set_title('Fraud Recall vs FP Reduction Tradeoff', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cost analysis
    cost_review = 15
    cost_missed = 50000
    results_df['savings'] = results_df['fp_reduction'] * nonfraud_mask.sum() * cost_review
    results_df['penalty'] = results_df['fraud_leaked'] * cost_missed
    results_df['net'] = results_df['savings'] - results_df['penalty']

    axes[1].plot(results_df['threshold'], results_df['net'] / 1e6,
                 color=COLORS['accent'], linewidth=2, label='Net Savings ($M)')
    axes[1].fill_between(results_df['threshold'],
                         results_df['net'] / 1e6,
                         alpha=0.2, color=COLORS['accent'])
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0.95, color=COLORS['dark'], linestyle='--', alpha=0.5, label='t=0.95')
    axes[1].set_xlabel('Suppression Threshold')
    axes[1].set_ylabel('Net Savings ($M)')
    axes[1].set_title('Cost-Benefit Analysis', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '11_threshold_analysis.png'), bbox_inches='tight')
    plt.close()

    # ── Save metrics ──
    metrics_dict = {
        'ROC-AUC': f'{auc_roc:.6f}',
        'Average Precision': f'{ap:.6f}',
        'Brier Score': f'{brier:.6f}',
        'Log Loss': f'{logloss_val:.6f}',
        'F1 Score': f'{f1:.6f}',
        'MCC': f'{mcc:.6f}',
    }
    pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value']).to_csv(
        os.path.join(EVAL_OUTPUT_DIR, 'phase3_metrics.csv'), index=False
    )

    print(f"  ✅ Phase 3 evaluation saved to {EVAL_OUTPUT_DIR}/08-11_*.png")
    return metrics_dict


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 4: RL TRAINING EVALUATION
# ═════════════════════════════════════════════════════════════════════════
def phase4_evaluation():
    print(SEP)
    print("  PHASE 4 — RL AGENT EVALUATION")
    print(SEP)

    # Check for comparison results
    comparison_path = os.path.join(RL_OUTPUT_DIR, 'policy_comparison.csv')
    report_path = os.path.join(RL_OUTPUT_DIR, 'evaluation_report.txt')

    if not os.path.exists(comparison_path):
        print("  ⚠️  RL results not found. Run run_phase4_rl.py first.")
        return

    comp_df = pd.read_csv(comparison_path)
    print(f"\n  Policy Comparison Results:")
    print(comp_df.to_string(index=False))

    # ── 4A: Policy comparison radar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar comparison
    metrics_cols = ['Fraud Catch Rate', 'FP Reduction Rate', 'Workload Savings']
    x = np.arange(len(metrics_cols))
    width = 0.12

    rl_rows = comp_df[comp_df['Policy'] == 'RL Agent']
    static_rows = comp_df[comp_df['Policy'] == 'Static Threshold']

    for i, (_, row) in enumerate(rl_rows.iterrows()):
        vals = [row[m] for m in metrics_cols]
        axes[0].bar(x - width/2, vals, width, label=f"RL ({row['Parameter']})",
                   color=COLORS['fraud'], alpha=0.9)

    for i, (_, row) in enumerate(static_rows.iterrows()):
        vals = [row[m] for m in metrics_cols]
        axes[0].bar(x + width * (i+1) - width/2, vals, width,
                   label=f"Static t={row['Parameter']}", alpha=0.7)

    axes[0].set_xticks(x + width * len(static_rows)/2)
    axes[0].set_xticklabels(metrics_cols, rotation=15)
    axes[0].set_ylabel('Rate')
    axes[0].set_title('RL vs Static Policy Comparison', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Reward comparison
    axes[1].barh(comp_df['Policy'] + ' (' + comp_df['Parameter'].astype(str) + ')',
                 comp_df['Avg Reward'],
                 color=[COLORS['fraud'] if p == 'RL Agent' else COLORS['primary']
                        for p in comp_df['Policy']])
    axes[1].set_xlabel('Average Reward')
    axes[1].set_title('Total Reward by Policy', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '12_rl_policy_comparison.png'), bbox_inches='tight')
    plt.close()

    # ── Check for training curves ──
    curves_path = os.path.join(RL_OUTPUT_DIR, 'training_curves.png')
    if os.path.exists(curves_path):
        print(f"  ✅ Training curves: {curves_path}")

    # Save metrics
    if len(rl_rows) > 0:
        rl_best = rl_rows.iloc[0]
        metrics = {
            'RL Fraud Catch Rate': f"{rl_best['Fraud Catch Rate']:.4f}",
            'RL FP Reduction': f"{rl_best['FP Reduction Rate']:.4f}",
            'RL Workload Savings': f"{rl_best['Workload Savings']:.4f}",
            'RL Avg Reward': f"{rl_best['Avg Reward']:.2f}",
        }
        # Static best
        if len(static_rows) > 0:
            best_static = static_rows.loc[static_rows['Avg Reward'].idxmax()]
            metrics['Best Static Reward'] = f"{best_static['Avg Reward']:.2f}"
            metrics['Best Static Threshold'] = str(best_static['Parameter'])
            metrics['RL Improvement'] = f"{((rl_best['Avg Reward'] - best_static['Avg Reward']) / abs(best_static['Avg Reward']) * 100):.1f}%" if best_static['Avg Reward'] != 0 else 'N/A'

        pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv(
            os.path.join(EVAL_OUTPUT_DIR, 'phase4_metrics.csv'), index=False
        )
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print(f"  ✅ Phase 4 evaluation saved to {EVAL_OUTPUT_DIR}/12_*.png")


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 5: DRIFT DETECTION EVALUATION
# ═════════════════════════════════════════════════════════════════════════
def phase5_evaluation():
    print(SEP)
    print("  PHASE 5 — DRIFT DETECTION EVALUATION")
    print(SEP)

    from src.drift_detector import DriftDetector, simulate_drift

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    detector = DriftDetector(train_df)
    print(f"  Monitoring {len(detector.feature_columns)} features")

    # Run drift checks at different intensities
    intensities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    drift_results = []

    for intensity in intensities:
        if intensity == 0:
            target = test_df
        else:
            target = simulate_drift(test_df, drift_intensity=intensity)

        report = detector.check_drift(target)
        drift_results.append({
            'intensity': intensity,
            'severity': report['overall_severity'],
            'avg_psi': report['avg_psi'],
            'max_psi': report['max_psi'],
            'avg_kl': report['avg_kl_divergence'],
            'features_drifted': report['features_drifted'],
            'fallback_bias': detector.get_fallback_action_bias()
        })
        print(f"  Drift intensity {intensity:.1f} → {report['overall_severity']} "
              f"(PSI={report['avg_psi']:.4f}, drifted={report['features_drifted']})")

    drift_df = pd.DataFrame(drift_results)
    drift_df.to_csv(os.path.join(EVAL_OUTPUT_DIR, 'drift_analysis.csv'), index=False)

    # ── 5A: Drift analysis charts ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PSI vs intensity
    axes[0].plot(drift_df['intensity'], drift_df['avg_psi'], 'o-',
                 color=COLORS['primary'], linewidth=2, markersize=8, label='Avg PSI')
    axes[0].plot(drift_df['intensity'], drift_df['max_psi'], 's--',
                 color=COLORS['fraud'], linewidth=2, markersize=8, label='Max PSI')
    axes[0].axhline(y=0.1, color=COLORS['accent'], linestyle=':', label='Warning (0.1)')
    axes[0].axhline(y=0.25, color=COLORS['fraud'], linestyle=':', label='Critical (0.25)')
    axes[0].set_xlabel('Drift Intensity')
    axes[0].set_ylabel('PSI')
    axes[0].set_title('PSI vs Drift Intensity', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KL divergence
    axes[1].plot(drift_df['intensity'], drift_df['avg_kl'], 'o-',
                 color=COLORS['secondary'], linewidth=2, markersize=8)
    axes[1].set_xlabel('Drift Intensity')
    axes[1].set_ylabel('Avg KL Divergence')
    axes[1].set_title('KL Divergence vs Drift Intensity', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Features drifted
    severity_colors = {'STABLE': COLORS['benign'], 'WARNING': COLORS['accent'],
                       'CRITICAL': COLORS['fraud']}
    bar_colors = [severity_colors.get(s, COLORS['dark']) for s in drift_df['severity']]
    axes[2].bar(drift_df['intensity'].astype(str), drift_df['features_drifted'],
                color=bar_colors)
    axes[2].set_xlabel('Drift Intensity')
    axes[2].set_ylabel('Features Drifted')
    axes[2].set_title('Drifted Features Count', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '13_drift_analysis.png'), bbox_inches='tight')
    plt.close()

    print(f"  ✅ Phase 5 evaluation saved to {EVAL_OUTPUT_DIR}/13_*.png")


# ═════════════════════════════════════════════════════════════════════════
#  BENCHMARK SUMMARY
# ═════════════════════════════════════════════════════════════════════════
def generate_benchmark_summary():
    print(SEP)
    print("  BENCHMARK SUMMARY")
    print(SEP)

    # Collect all phase metrics
    benchmarks = []

    # Phase 3 metrics
    p3_path = os.path.join(EVAL_OUTPUT_DIR, 'phase3_metrics.csv')
    if os.path.exists(p3_path):
        p3 = pd.read_csv(p3_path)
        for _, row in p3.iterrows():
            benchmarks.append({'Phase': 'P3: Suppression Model', 'Metric': row['Metric'],
                             'Our Value': row['Value'], 'Benchmark': ''})

    # Industry benchmarks
    industry = [
        ('P3: Suppression Model', 'ROC-AUC', '≥ 0.85'),
        ('P3: Suppression Model', 'Brier Score', '≤ 0.10'),
        ('P4: RL Agent', 'Fraud Recall', '≥ 95%'),
        ('P4: RL Agent', 'FP Reduction', '30-60%'),
        ('P5: Drift Detection', 'PSI Sensitivity', 'Detects at 0.1+'),
    ]

    # Phase 4 metrics
    p4_path = os.path.join(EVAL_OUTPUT_DIR, 'phase4_metrics.csv')
    if os.path.exists(p4_path):
        p4 = pd.read_csv(p4_path)
        for _, row in p4.iterrows():
            benchmarks.append({'Phase': 'P4: RL Agent', 'Metric': row['Metric'],
                             'Our Value': row['Value'], 'Benchmark': ''})

    # Add industry benchmarks
    for phase, metric, benchmark in industry:
        for b in benchmarks:
            if b['Phase'] == phase and metric in b['Metric']:
                b['Benchmark'] = benchmark

    if benchmarks:
        bench_df = pd.DataFrame(benchmarks)
        bench_df.to_csv(os.path.join(EVAL_OUTPUT_DIR, 'benchmark_summary.csv'), index=False)
        print(bench_df.to_string(index=False))

    # ── Summary figure ──
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    table_data = []
    headers = ['Phase', 'Metric', 'Our Value', 'Industry Benchmark']

    # Gather data
    p3_file = os.path.join(EVAL_OUTPUT_DIR, 'phase3_metrics.csv')
    p4_file = os.path.join(EVAL_OUTPUT_DIR, 'phase4_metrics.csv')

    if os.path.exists(p3_file):
        p3 = pd.read_csv(p3_file)
        for _, r in p3.iterrows():
            bm = ''
            if 'ROC-AUC' in r['Metric']: bm = '≥ 0.85'
            elif 'Brier' in r['Metric']: bm = '≤ 0.10'
            elif 'F1' in r['Metric']: bm = '≥ 0.80'
            table_data.append(['Phase 3', r['Metric'], r['Value'], bm])

    if os.path.exists(p4_file):
        p4 = pd.read_csv(p4_file)
        for _, r in p4.iterrows():
            bm = ''
            if 'Catch' in r['Metric']: bm = '≥ 0.95'
            elif 'FP' in r['Metric']: bm = '0.30-0.60'
            table_data.append(['Phase 4', r['Metric'], r['Value'], bm])

    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for j in range(len(headers)):
            table[0, j].set_facecolor(COLORS['dark'])
            table[0, j].set_text_props(color='white', fontweight='bold')

        ax.set_title('AlertIQ — Benchmark Comparison', fontsize=16,
                     fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, '14_benchmark_summary.png'), bbox_inches='tight')
    plt.close()

    print(f"\n  ✅ Benchmark summary saved to {EVAL_OUTPUT_DIR}/14_benchmark_summary.png")


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AlertIQ — Comprehensive EDA & Evaluation Suite")
    print("=" * 70)

    ensure_dirs()

    # Phase 1: Raw data EDA
    raw_df = phase1_eda()

    # Phase 2: Alert generation EDA
    phase2_eda(raw_df)

    # Phase 3: Suppression model evaluation
    phase3_evaluation()

    # Phase 4: RL agent evaluation
    phase4_evaluation()

    # Phase 5: Drift detection evaluation
    phase5_evaluation()

    # Benchmark summary
    generate_benchmark_summary()

    print(SEP)
    print("  ALL EVALUATIONS COMPLETE")
    print(f"  EDA charts:        {os.path.abspath(EDA_OUTPUT_DIR)}")
    print(f"  Evaluation charts: {os.path.abspath(EVAL_OUTPUT_DIR)}")
    print(SEP)
