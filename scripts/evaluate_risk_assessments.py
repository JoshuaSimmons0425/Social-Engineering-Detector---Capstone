import os
import json
import re
import sys
# Import the new Gemini judge class instead
from src.LLM_judgment.gemini_flash_judge import GeminiEvalJudge

def load_file_content(path: str) -> str:
    """Safely loads file content as string."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required experiment file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    # Enforce API Key verification before initiating execution
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is missing.")
        print("Please export it via terminal: export GEMINI_API_KEY='your_key'")
        sys.exit(1)

    # 1. Base folders
    base_results_dir = "results/risk_assessments"
    base_xai_dir = "xAI_outputs"
    evaluation_output_dir = "results/evaluations"
    os.makedirs(evaluation_output_dir, exist_ok=True)

    if not os.path.exists(base_results_dir):
        print(f"Error: Risk assessments directory '{base_results_dir}' not found.")
        sys.exit(1)

    # 2. Locate and naturally sort all experiment directories
    experiment_folders = [
        d for d in os.listdir(base_results_dir)
        if os.path.isdir(os.path.join(base_results_dir, d)) and d.startswith("experiment_")
    ]
    experiment_folders.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    if not experiment_folders:
        print("No experiment folders found inside results/risk_assessments.")
        sys.exit(0)

    # 3. Define metrics configuration mapping to custom rubric markdown criteria files
    metrics = {
        "relevance": {
            "task_description": "Evaluating the focus and specificity of a social engineering risk assessment.",
            "criterion": "Focusing on characteristics that actually matter over generic security commentary.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/relevance.md",
            "pairwise_file": "prompts/LLM-Judge/pairwise_rubrics/relevance.md"
        },
        "justification_soundness": {
            "task_description": "Evaluating the explanation quality and analytical faithfulness of a risk assessment.",
            "criterion": "Logical reasoning chains that strongly support the final risk conclusions.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/justification.md",
            "pairwise_file": "prompts/LLM-Judge/pairwise_rubrics/justification.md"
        },
        "contextual_awareness": {
            "task_description": "Evaluating a small language model's ability to synthesize nuanced contextual data.",
            "criterion": "Understanding complex social engineering situations and the interaction of subtle cues.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/contextual_awareness.md",
            "pairwise_file": "prompts/LLM-Judge/pairwise_rubrics/contextual_awareness.md"
        },
        "accuracy": {
            "task_description": "Evaluating the objective validity of a security risk decision.",
            "criterion": "Appropriateness and accuracy of the final risk judgment.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/accuracy.md",
            "pairwise_file": "prompts/LLM-Judge/pairwise_rubrics/accuracy.md"
        },
        "guidance_appropriateness": {
            "task_description": "Evaluating the real-world utility of security recommendations.",
            "criterion": "Practicality, safety, and operational sense of the recommended mitigations.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/guidance.md",
            "pairwise_file": "prompts/LLM-Judge/pairwise_rubrics/guidance.md"
        }
    }

    # 4. Instantiate the Gemini Judge class cloud context
    judge = GeminiEvalJudge()

    # 5. Master loop across all discovered experiments
    for exp_folder in experiment_folders:
        print(f"\n=======================================================")
        print(f"RUNNING GEMINI 2.5 FLASH EVALUATION FOR: {exp_folder.upper()}")
        print(f"=======================================================")
        
        exp_results_path = os.path.join(base_results_dir, exp_folder)
        exp_xai_path = os.path.join(base_xai_dir, exp_folder)

        no_evidence_file = os.path.join(exp_results_path, "no_evidence", "no_evidence_reply.md")
        baseline_evidence_file = os.path.join(exp_results_path, "baseline_evidence", "baseline_evidence_reply.md")
        bert_evidence_file = os.path.join(exp_results_path, "bert_evidence", "bert_evidence_reply.md")
        input_text_file = os.path.join(exp_xai_path, "input_text", "cleaned_input.txt")

        # Verify execution files exist for this experiment block
        required_files = [no_evidence_file, baseline_evidence_file, bert_evidence_file, input_text_file]
        if not all(os.path.exists(f) for f in required_files):
            print(f"Skipping {exp_folder}: Missing target assessment outputs or source files.")
            continue

        original_message = load_file_content(input_text_file)
        
        model_outputs = {
            "no_evidence": load_file_content(no_evidence_file),
            "baseline_evidence": load_file_content(baseline_evidence_file),
            "bert_evidence": load_file_content(bert_evidence_file)
        }

        # Contextual ground-truth reference sequence text
        full_instruction = (
            f"Original Input Message (Reference Context):\n\"\"\"\n{original_message}\n\"\"\"\n\n"
            f"Task Instruction:\nEvaluate the security risks present inside the target text above, "
            f"validating indicators of deception or persuasion techniques."
        )

        experiment_report = {}

        # Loop through every metric configuration to build this experiment's evaluation matrix
        for metric_name, cfg in metrics.items():
            print(f"\n  Evaluating Metric: {metric_name.upper()}")
            
            try:
                direct_rubric_text = load_file_content(cfg["direct_file"])
                pairwise_rubric_text = load_file_content(cfg["pairwise_file"])
            except FileNotFoundError as e:
                print(f"    Skipping metric {metric_name}: {e}")
                continue

            rubrics_pack = {
                "task_description": cfg["task_description"],
                "criterion": cfg["criterion"],
                "direct_rubric": direct_rubric_text,
                "pairwise_rubric": pairwise_rubric_text
            }

            experiment_report[metric_name] = {"direct": {}, "pairwise": {}}

            # --- Phase A: Run Direct Appraisals ---
            for model_name, output_content in model_outputs.items():
                res = judge.evaluate_direct(full_instruction, output_content, rubrics_pack)
                experiment_report[metric_name]["direct"][model_name] = res
                print(f"    Direct Score [{model_name}]: {res['parsed_score']}/5")

            # --- Phase B: Run Pairwise Combos ---
            combos = [
                ("no_evidence", "baseline_evidence"),
                ("no_evidence", "bert_evidence"),
                ("baseline_evidence", "bert_evidence")
            ]
            
            for model_a, model_b in combos:
                res = judge.evaluate_pairwise(full_instruction, model_outputs[model_a], model_outputs[model_b], rubrics_pack)
                
                winner_label = res["parsed_decision"]
                resolved_winner = model_a if winner_label == "A" else (model_b if winner_label == "B" else winner_label)
                
                experiment_report[metric_name]["pairwise"][f"{model_a}_vs_{model_b}"] = {
                    "winner": resolved_winner,
                    "feedback": res["feedback"]
                }
                print(f"    Pairwise Win [{model_a} vs {model_b}]: {resolved_winner}")

        # 6. Export an individual JSON matrix for this unique experiment tracking row
        output_report_path = os.path.join(evaluation_output_dir, f"{exp_folder}_eval_report.json")
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_report, f, indent=4, ensure_ascii=False)
            
        print(f"\nSaved clean JSON metrics report for {exp_folder} -> {output_report_path}")

    print("\nAll experiment evaluations successfully compiled via Gemini Flash.")

if __name__ == "__main__":
    main()
