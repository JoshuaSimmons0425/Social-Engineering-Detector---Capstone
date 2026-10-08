import os
import json
import re
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def load_file_content(path: str) -> str:
    """Safely loads file content as a lowercase string."""
    if not os.path.exists(path):
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_risk_level_to_binary(text: str) -> int:
    """
    Parses the SLM text using Risk Management Theory mapping rules:
    Low Risk -> 0 (Benign)
    Medium / High / Critical -> 1 (Malicious)
    """
    if not text:
        return 0
        
    # Isolate the final conclusion block (last 500 characters) to ensure context accuracy
    conclusion_block = text[-500:]
    if len(text) < 500:
        conclusion_block = text

    # Search for explicit risk keywords prioritising the highest severity found
    if any(kw in conclusion_block for kw in ["Critical", "High", "Medium"]):
        return 1
    if "Low" in conclusion_block:
        return 0
        
    # Fallback backup: Scan whole text if conclusion block is ambiguous
    if any(kw in text for kw in ["Critical", "High", "Medium"]):
        return 1
        
    return 0  # Default fallback to safe/benign if no risk tags fire

def main():
    # Path to your already saved, external stratified dataset
    dataset_path = "data/splits/stratified_data.csv"
    base_results_dir = "results/risk_assessments"
    output_dir = "stat_test"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Stratified dataset missing at: {dataset_path}")
        return
        
    # 1. Read your existing saved file directly (No reconstruction/sampling loops)
    stratified_df = pd.read_csv(dataset_path)
    
    # 2. Extract and map your retained native labels to binary codes
    # Safely strips and lowercases to avoid leading/trailing whitespace mismatches
    ground_truth_labels = [
        0 if str(label).strip().lower() == 'benign' else 1 
        for label in stratified_df['Label'].tolist()
    ]

    # Verify that your dataset exactly matches your 50 processed folder targets
    if len(ground_truth_labels) != 50:
        print(f"Warning: Your saved dataset has {len(ground_truth_labels)} rows, "
              f"but you have 50 experiment folders. Processing the first 50 match slots.")

    y_true = []
    y_no_evidence = []
    y_baseline_xai = []
    y_bert_xai = []

    # 3. Pull and cross-check predictions across your 50 experiments
    for idx in range(min(50, len(ground_truth_labels))):
        exp_folder = f"experiment_{idx}"
        exp_path = os.path.join(base_results_dir, exp_folder)
        
        no_file = os.path.join(exp_path, "no_evidence", "no_evidence_reply.md")
        base_file = os.path.join(exp_path, "baseline_evidence", "baseline_evidence_reply.md")
        bert_file = os.path.join(exp_path, "bert_evidence", "bert_evidence_reply.md")
        
        if not all(os.path.exists(f) for f in [no_file, base_file, bert_file]):
            print(f"Skipping index {idx}: Incomplete experiment folder setup.")
            continue
            
        y_true.append(ground_truth_labels[idx])
        y_no_evidence.append(parse_risk_level_to_binary(load_file_content(no_file)))
        y_baseline_xai.append(parse_risk_level_to_binary(load_file_content(base_file)))
        y_bert_xai.append(parse_risk_level_to_binary(load_file_content(bert_file)))

    if not y_true:
        print("Error: No valid experiment files could be loaded.")
        return

    # 4. Calculate Performance Matrices
    tiers = {
        "Control (No Evidence)": y_no_evidence,
        "Tier 1 (Logistic Regression xAI)": y_baseline_xai,
        "Tier 2 (Contextual BERT xAI)": y_bert_xai
    }
    
    summary_data = []
    for tier_name, y_pred in tiers.items():
        acc = accuracy_score(y_true, y_pred)
        
        # Build confusion matrix variables safely
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # False Positive Rate (Benign emails misclassified as Medium/High/Critical risk)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        # False Negative Rate (Active threats missed entirely / misclassified as Low risk)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        summary_data.append({
            "Experimental Tier": tier_name,
            "Accuracy (Ground Truth)": f"{acc:.2%}",
            "False Positive Rate (FPR)": f"{fpr:.2%}",
            "False Negative Rate (FNR)": f"{fnr:.2%}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "risk_ground_truth_alignment.csv"), index=False)
    
    print("\n=========================================================================")
    print("         RISK-THEORY MAPPED GROUND TRUTH LABEL ALIGNMENT MATRIX          ")
    print("=========================================================================\n")
    print(summary_df.to_markdown(index=False))
    print("\n=========================================================================")

if __name__ == "__main__":
    main()
