import os
import json
import re
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_json_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_box_plot(df: pd.DataFrame, output_dir: str):
    """Generates a publication-quality box plot showing performance shifts and means."""
    # 1. Melt the DataFrame from wide to long format for Seaborn compatibility
    plot_df = df.melt(
        id_vars=["Experiment_ID"], 
        value_vars=["Control_No_Evidence", "Tier1_Baseline_Evidence", "Tier2_BERT_Evidence"],
        var_name="Experimental_Tier", 
        value_name="Overall_Average_Score"
    )
    
    # Clean up labels for presentation presentation
    tier_mapping = {
        "Control_No_Evidence": "Control\n(No Evidence)",
        "Tier1_Baseline_Evidence": "Tier 1 Treatment\n(Baseline Heuristics)",
        "Tier2_BERT_Evidence": "Tier 2 Treatment\n(BERT xAI)"
    }
    plot_df["Experimental_Tier"] = plot_df["Experimental_Tier"].map(tier_mapping)

    # 2. Configure academic figure styles via Seaborn
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.1)
    
    plt.figure(figsize=(9, 6.5))
    
    # 3. Plot the Box and Whisker elements with muted scholarly palette
    # showmeans=True injects a distinct indicator calculating mathematical expectations
    ax = sns.boxplot(
        x="Experimental_Tier", 
        y="Overall_Average_Score", 
        data=plot_df,
        palette="muted",
        width=0.45,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8}
    )
    
    # 4. Lay down jittered raw data points over boxes to display group size density (N=20)
    sns.stripplot(
        x="Experimental_Tier", 
        y="Overall_Average_Score", 
        data=plot_df,
        color="black",
        size=5,
        alpha=0.4,
        jitter=0.15
    )

    # 5. Fine-tune axis ranges and styling elements
    plt.title("Distribution of Overall Quality Scores Across Experimental Framework Tiers", pad=20, weight='bold')
    plt.xlabel("Evaluation Assessment Configurations", labelpad=15, weight='semibold')
    plt.ylabel("Overall Mean Score (Scale 1 - 5)", labelpad=15, weight='semibold')
    
    # Force Y-Axis boundaries to tightly frame your 1-5 evaluation limits
    plt.ylim(1.0, 5.2)
    plt.yticks([1.0, 2.0, 3.0, 4.0, 5.0])

    # Add a subtitle anchor note describing visual elements
    plt.figtext(
        0.15, 0.005, 
        "Note: Boxes bound the IQR (middle 50%); horizontal bars indicate medians; white diamonds (◆) represent means.", 
        alpha=0.75, style='italic', fontsize=10
    )

    plt.tight_layout()
    
    # 6. Save the figure as a print-ready vector graphic asset
    plot_save_path = os.path.join(output_dir, "evaluation_distribution_plot.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Publication-ready distribution figure successfully rendered to: {plot_save_path}")

def main():
    eval_dir = "results/evaluations"
    output_dir = "stat_test"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory '{eval_dir}' not found.")
        return

    # Gather and naturally sort all report targets
    report_files = [
        f for f in os.listdir(eval_dir) 
        if f.startswith("experiment_") and f.endswith("_local_eval_report.json")
    ]
    report_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    if not report_files:
        print("No evaluation reports found.")
        return

    print(f"Aggregating data across {len(report_files)} experimental observations...")

    no_evidence_samples = []
    baseline_evidence_samples = []
    bert_evidence_samples = []
    
    # Extract metrics and compile composite averages row by row
    for file_name in report_files:
        data = load_json_file(os.path.join(eval_dir, file_name))
        
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
                
        if scores_no and scores_base and scores_bert:
            no_evidence_samples.append(sum(scores_no) / len(scores_no))
            baseline_evidence_samples.append(sum(scores_base) / len(scores_base))
            bert_evidence_samples.append(sum(scores_bert) / len(scores_bert))

    # Construct tracking frame row
    experiments_df = pd.DataFrame({
        "Experiment_ID": [f"Experiment_{i}" for i in range(len(no_evidence_samples))],
        "Control_No_Evidence": no_evidence_samples,
        "Tier1_Baseline_Evidence": baseline_evidence_samples,
        "Tier2_BERT_Evidence": bert_evidence_samples
    })
    experiments_df.to_csv(os.path.join(output_dir, "aggregated_experiments_scores.csv"), index=False)

    # Trigger distribution chart compilation step
    generate_box_plot(experiments_df, output_dir)

    # Execute Omnibus Test: Friedman's ANOVA by Ranks
    friedman_stat, friedman_p = stats.friedmanchisquare(
        no_evidence_samples, 
        baseline_evidence_samples, 
        bert_evidence_samples
    )

    initial_alpha = 0.05
    num_comparisons = 3
    bonferroni_alpha = initial_alpha / num_comparisons 

    text_output = []
    text_output.append("\n=========================================================================")
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

    report_string = "\n".join(text_output)
    print(report_string)

    with open(os.path.join(output_dir, "statistical_test_results.txt"), "w", encoding='utf-8') as f:
        f.write(report_string)
    print(f"Comprehensive reports and data logs saved inside the '{output_dir}/' folder.")

if __name__ == "__main__":
    main()