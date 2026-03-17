def print_results(results_df, baseline_recall):
    
    print("\n===== CONFIDENCE GATING EVALUATION =====\n")
    print("Baseline Fraud Recall:", round(baseline_recall,5))
    print("\nThreshold Results:\n")
    print(results_df)
    
    print("\nInterpretation:")
    print("- Fraud recall must remain >= baseline")
    print("- False positive reduction should increase with threshold")
