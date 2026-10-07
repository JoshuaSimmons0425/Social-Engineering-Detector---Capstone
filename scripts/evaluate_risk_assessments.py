import os
import json
from src.LLM_judgment.prometheous2_judge import SocialEngineeringEvalJudge

def load_file_content(path: str) -> str:
    """Safely loads file content as string."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required experiment file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    # 1. Load inputs and outputs
    system_prompt = load_file_content("inputs/system_prompt.txt")
    task_input = load_file_content("inputs/task_input.txt")
    full_instruction = f"System Prompt: {system_prompt}\nUser Task: {task_input}"

    model_outputs = {
        "baseline_slm": load_file_content("outputs/baseline_slm_output.txt"),
        "xai_augmented_slm_1": load_file_content("outputs/xai_augmented_slm_1_output.txt"),
        "xai_augmented_slm_2": load_file_content("outputs/xai_augmented_slm_2_output.txt")
    }

    # 2. Define the metrics map pointing directly to the specific markdown files
    metrics = {
        "relevance": {
            "task_description": "Evaluating the focus and specificity of a social engineering risk assessment.",
            "criterion": "Focusing on characteristics that actually matter over generic security commentary.",
            "direct_file": "rubrics/1_relevance_direct.md",
            "pairwise_file": "rubrics/1_relevance_pairwise.md"
        },
        "justification_soundness": {
            "task_description": "Evaluating the explanation quality and analytical faithfulness of a risk assessment.",
            "criterion": "Logical reasoning chains that strongly support the final risk conclusions.",
            "direct_file": "rubrics/2_justification_direct.md",
            "pairwise_file": "rubrics/2_justification_pairwise.md"
        },
        "contextual_awareness": {
            "task_description": "Evaluating a small language model's ability to synthesize nuanced contextual data.",
            "criterion": "Understanding complex social engineering situations and the interaction of subtle cues.",
            "direct_file": "rubrics/3_context_direct.md",
            "pairwise_file": "rubrics/3_context_pairwise.md"
        },
        "accuracy": {
            "task_description": "Evaluating the objective validity of a security risk decision.",
            "criterion": "Appropriateness and accuracy of the final risk judgment.",
            "direct_file": "rubrics/4_accuracy_direct.md",
            "pairwise_file": "rubrics/4_accuracy_pairwise.md"
        },
        "guidance_appropriateness": {
            "task_description": "Evaluating the real-world utility of security recommendations.",
            "criterion": "Practicality, safety, and operational sense of the recommended mitigations.",
            "direct_file": "rubrics/5_guidance_direct.md",
            "pairwise_file": "rubrics/5_guidance_pairwise.md"
        }
    }

    # 3. Instantiate the corrected Judge class
    judge = SocialEngineeringEvalJudge()
    master_report = {}

    # 4. Loop through every metric configuration to build the report matrix
    for metric_name, cfg in metrics.items():
        print(f"\n======== EVALUATING METRIC: {metric_name.upper()} ========")
        
        # Load the custom markdown criteria files directly
        direct_rubric_text = load_file_content(cfg["direct_file"])
        pairwise_rubric_text = load_file_content(cfg["pairwise_file"])

        rubrics_pack = {
            "task_description": cfg["task_description"],
            "criterion": cfg["criterion"],
            "direct_rubric": direct_rubric_text,
            "pairwise_rubric": pairwise_rubric_text
        }

        master_report[metric_name] = {"direct": {}, "pairwise": {}}

        # --- Phase A: Run Direct Appraisals ---
        print(f"Running Direct evaluations for {metric_name}...")
        for model_name, output_content in model_outputs.items():
            res = judge.evaluate_direct(full_instruction, output_content, rubrics_pack)
            master_report[metric_name]["direct"][model_name] = res
            print(f"  [{model_name}] Score: {res['parsed_score']}")

        # --- Phase B: Run Pairwise Combos ---
        print(f"Running Pairwise matchups for {metric_name}...")
        combos = [
            ("baseline_slm", "xai_augmented_slm_1"),
            ("baseline_slm", "xai_augmented_slm_2"),
            ("xai_augmented_slm_1", "xai_augmented_slm_2")
        ]
        
        for model_a, model_b in combos:
            res = judge.evaluate_pairwise(full_instruction, model_outputs[model_a], model_outputs[model_b], rubrics_pack)
            
            # Resolve the actual text name of the model that won the pairing
            winner_label = res["parsed_decision"]
            resolved_winner = model_a if winner_label == "A" else (model_b if winner_label == "B" else winner_label)
            
            master_report[metric_name]["pairwise"][f"{model_a}_vs_{model_b}"] = {
                "winner": resolved_winner,
                "feedback": res["feedback"]
            }
            print(f"  [{model_a} vs {model_b}] Winner: {resolved_winner}")

    # 5. Export multi-dimensional JSON matrix
    output_report_path = "multi_metric_evaluation_report.json"
    with open(output_report_path, 'w', encoding='utf-8') as f:
        json.dump(master_report, f, indent=4, ensure_ascii=False)
        
    print(f"\nEvaluation loop completed successfully. Matrix logged in: {output_report_path}")

if __name__ == "__main__":
    main()
