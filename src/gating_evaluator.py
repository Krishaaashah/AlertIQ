import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_thresholds(eval_df: pd.DataFrame, thresholds=[0.80, 0.90, 0.95, 0.97, 0.99], output_dir='.'):
    """
    Evaluate the performance of the alert suppression model across different confidence thresholds.
    eval_df MUST contain:
        - isFraud (actual label: 1 for Fraud, 0 for False Positive)
        - prob_benign (predicted probability of being a False Positive / Benign)
    """
    print("\n" + "="*50)
    print("GATING EVALUATOR: THRESHOLD SENSITIVITY ANALYSIS")
    print("="*50)

    total_alerts = len(eval_df)
    total_fraud = eval_df['isFraud'].sum()
    total_fp = total_alerts - total_fraud
    
    print(f"Test Set Summary: {total_alerts} Total Alerts | {total_fraud} True Frauds | {total_fp} False Positives")
    
    # Calculate Analyst Baseline Recall if simulation was performed
    if 'analyst_decision' in eval_df.columns:
        analyst_caught = eval_df[(eval_df['isFraud'] == 1) & (eval_df['analyst_decision'] == 'escalate')]
        baseline_recall = len(analyst_caught) / total_fraud if total_fraud > 0 else 0
        print(f"Analyst Baseline Fraud Recall: {baseline_recall:.4f}")
    else:
        baseline_recall = 1.0

    results = []
    
    for thresh in thresholds:
        # Evaluate suppression logic
        is_suppressed = eval_df['prob_benign'] >= thresh
        
        # Alerts routed to the analyst (not suppressed)
        routed_to_analyst = eval_df[~is_suppressed]
        
        # Measurement 1: Fraud Recall (What % of frauds were NOT suppressed?)
        frauds_survived_suppression = routed_to_analyst['isFraud'].sum()
        fraud_recall = frauds_survived_suppression / total_fraud if total_fraud > 0 else 0
        
        # Measurement 2: False Positive Reduction (What % of FPs were suppressed?)
        suppressed_fps = eval_df[is_suppressed]['isFraud'] == 0
        fp_reduction = suppressed_fps.sum() / total_fp if total_fp > 0 else 0
        
        # Measurement 3: Workload Savings (What % of TOTAL alerts were suppressed?)
        num_suppressed = is_suppressed.sum()
        workload_savings = num_suppressed / total_alerts if total_alerts > 0 else 0
        
        results.append({
            'Threshold': thresh,
            'Fraud Recall': fraud_recall,
            'FP Reduction': fp_reduction,
            'Workload Savings': workload_savings
        })
        
    results_df = pd.DataFrame(results)
    
    print("\n--- Threshold Comparison Report ---")
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("-----------------------------------")
    
    # Save the sensitivity analysis results
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'threshold_report.csv')
    results_df.to_csv(report_path, index=False)
    
    # Generate charts
    plot_evaluation_charts(results_df, baseline_recall, output_dir)
    
    return results_df

def plot_evaluation_charts(results_df: pd.DataFrame, baseline_recall: float, output_dir: str):
    """Generate and save evaluation charts visualizing tradeoffs."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(results_df['Threshold'], results_df['Fraud Recall'], marker='o', label='Fraud Recall', color='#d62728', linewidth=2)
    plt.plot(results_df['Threshold'], results_df['FP Reduction'], marker='s', label='False Positive Reduction', color='#1f77b4', linewidth=2)
    plt.plot(results_df['Threshold'], results_df['Workload Savings'], marker='^', label='Overall Workload Savings', color='#2ca02c', linewidth=2)
    
    # Add a baseline recall reference line
    plt.axhline(y=baseline_recall, color='k', linestyle='--', label='Baseline Analyst Recall', alpha=0.6)
    
    plt.title('Alert Suppression Layer: Performance vs. Confidence Threshold', fontsize=14)
    plt.xlabel('Suppression Threshold (prob_benign >= X)', fontsize=12)
    plt.ylabel('Metric Rate (0.0 to 1.0)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')
    
    chart_path = os.path.join(output_dir, 'evaluation_charts.png')
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nModel evaluation charts saved to: {chart_path}")
    print(f"Threshold report data saved to:   {os.path.join(output_dir, 'threshold_report.csv')}")

if __name__ == "__main__":
    pass
