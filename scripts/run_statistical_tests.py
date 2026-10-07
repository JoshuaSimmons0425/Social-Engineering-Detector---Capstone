import os
import json
import re
import pandas as pd
import scipy.stats as stats

def load_json_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    eval_dir = "results/evaluations"
    output_dir = "stat_test"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory '{eval_dir}' not found.")
        return

    # 1. Gather and naturally sort all report targets
    report_files = [
        f for f in os.listdir(eval_dir) 
        if f.startswith("experiment_") and f.endswith("_local_eval_report.json")
    ]
    report_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    if not report_files:
        print("No evaluation reports found.")
        return

    print(f"Aggregating data across {len(report_files)} experimental observations...")

    # Lists to stack the overall average metric vector for each experiment instance
    no_evidence_samples = []
    baseline_evidence_samples = []
    bert_evidence_samples = []
    
    # 2. Extract metrics and compile composite averages row by row
    for file_name in report_files:
        data = load_json_file(os.path.join(eval_dir, file_name))
        
        # Accumulators for this specific experiment instance
        scores_no = []
        scores_base = []
        scores_bert = []
        
        for metric_name, configs in data.items():
            if "no_evidence" in configs and "score" in configs["no_evidence"]:
                scores_no.append(configs["no_evidence"]["score"])
            if "baseline_evidence" in configs and "score" in configs["baseline_evidence"]:
                scores_base.append(configs["baseline_evidence"]["score"])
            if "bert_evidence" in configs and "score" in configs["bert_evidence"]:
                scores_bert.append(configs["bert_evidence"]["score"])
                
        # Calculate the overall mean across all 5 evaluation criteria
        if scores_no and scores_base and scores_bert:
            no_evidence_samples.append(sum(scores_no) / len(scores_no))
            baseline_evidence_samples.append(sum(scores_base) / len(scores_base))
            bert_evidence_samples.append(sum(scores_bert) / len(scores_bert))

    # Construct clean tabular snapshot for tracking rows
    experiments_df = pd.DataFrame({
        "Experiment_ID": [f"Experiment_{i}" for i in range(len(no_evidence_samples))],
        "Control_No_Evidence": no_evidence_samples,
        "Tier1_Baseline_Evidence": baseline_evidence_samples,
        "Tier2_BERT_Evidence": bert_evidence_samples
    })
    experiments_df.to_csv(os.path.join(output_dir, "aggregated_experiments_scores.csv"), index=False)

    # 3. Execute Omnibus Test: Friedman's ANOVA by Ranks
    # Null Hypothesis (H0): The distributions of overall scores are equal across all tiers.
    friedman_stat, friedman_p = stats.friedmanchisquare(
        no_evidence_samples, 
        baseline_evidence_samples, 
        bert_evidence_samples
    )

    # Define standard significance bar parameters
    initial_alpha = 0.05
    num_comparisons = 3
    bonferroni_alpha = initial_alpha / num_comparisons  # Adjusted Alpha threshold = 0.0167

    text_output = []
    text_output.append("=========================================================================")
    text_output.append("            NON-PARAMETRIC WITHIN-SUBJECTS STATISTICAL REPORT            ")
    text_output.append("=========================================================================\n")
    text_output.append(f"Number of Independent Experiment Blocks (N): {len(no_evidence_samples)}")
    text_output.append(f"Initial Alpha Level (α): {initial_alpha}")
    text_output.append(f"Bonferroni Adjusted Alpha Threshold (α_adj): {bonferroni_alpha:.4f}\n")
    text_output.append("-------------------------------------------------------------------------")
    text_output.append("STAGE 1: GLOBAL OMNIBUS TEST (Friedman's ANOVA by Ranks)")
    text_output.append("-------------------------------------------------------------------------")
    text_output.append(f"Friedman Chi-Square Statistic: {friedman_stat:.4f}")
    text_output.append(f"Asymptotic P-Value:            {friedman_p:.4e}")
    
    omnibus_significant = friedman_p < initial_alpha
    text_output.append(f"Omnibus Statistical Effect?   {'[SUCCESS] Significant Difference Detected' if omnibus_significant else '[FAIL] No Significant Effect Found'}\n")

    # 4. Stage 2: Execute Post-Hoc Pairwise Matched Wilcoxon Signed-Rank Tests
    text_output.append("-------------------------------------------------------------------------")
    text_output.append("STAGE 2: POST-HOC PAIRWISE COMPARISONS (Wilcoxon Signed-Rank Tests)")
    text_output.append("-------------------------------------------------------------------------")
    
    if omnibus_significant:
        matchups = [
            ("Control (No Evidence)", "Tier 1 Treatment (Baseline Heuristics)", no_evidence_samples, baseline_evidence_samples),
            ("Control (No Evidence)", "Tier 2 Treatment (BERT xAI)", no_evidence_samples, bert_evidence_samples),
            ("Tier 1 Treatment (Baseline Heuristics)", "Tier 2 Treatment (BERT xAI)", baseline_evidence_samples, bert_evidence_samples)
        ]
        
        for name_a, name_b, data_a, data_b in matchups:
            # Execute Wilcoxon two-tailed test routine
            w_stat, p_val = stats.wilcoxon(data_a, data_b)
            is_pairwise_sig = p_val < bonferroni_alpha
            
            mean_a = sum(data_a) / len(data_a)
            mean_b = sum(data_b) / len(data_b)
            
            text_output.append(f"Matchup: {name_a} vs {name_b}")
            text_output.append(f"  - Mean Performance:  {mean_a:.2f} vs {mean_b:.2f}")
            text_output.append(f"  - Wilcoxon W-Stat:   {w_stat:.1f}")
            text_output.append(f"  - Pairwise P-Value:  {p_val:.4e}")
            text_output.append(f"  - Significant?       {'YES (P < α_adj)' if is_pairwise_sig else 'NO (P >= α_adj)'}\n")
    else:
        text_output.append("Post-Hoc tests bypassed. Omnibus test did not reject the null hypothesis.")

    # Print report summary back out to shell terminal interface
    report_string = "\n".join(text_output)
    print(report_string)

    # 5. Document logs directly to disk text records
    with open(os.path.join(output_dir, "statistical_test_results.txt"), "w", encoding='utf-8') as f:
        f.write(report_string)
    print(f"Comprehensive statistical text reports successfully saved inside the '{output_dir}/' folder.")

if __name__ == "__main__":
    main()