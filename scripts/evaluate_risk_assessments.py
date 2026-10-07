import os
import json
import re
import sys
# Import DeepEval core infrastructure
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

from src.LLM_judgment.ollamajudge import LocalOllamaJudge

# 1. Create a custom wrapper class so DeepEval can talk to local Ollama

def load_file_content(path: str) -> str:
    """Safely loads file content as string."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required experiment file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():

    # 2. Base folders
    base_results_dir = "results/risk_assessments"
    base_xai_dir = "xAI_outputs"
    evaluation_output_dir = "results/evaluations"
    os.makedirs(evaluation_output_dir, exist_ok=True)

    if not os.path.exists(base_results_dir):
        print(f"Error: Risk assessments directory '{base_results_dir}' not found.")
        sys.exit(1)

    # 3. Locate and naturally sort all experiment directories
    experiment_folders = [
        d for d in os.listdir(base_results_dir)
        if os.path.isdir(os.path.join(base_results_dir, d)) and d.startswith("experiment_")
    ]
    experiment_folders.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    if not experiment_folders:
        print("No experiment folders found inside results/risk_assessments.")
        sys.exit(0)

    # 4. Define metrics configurations mapped to explicit, step-by-step auditing pathways
    metrics_config = {
        "relevance": {
            "evaluation_steps": [
                "Step 1: Check the original input message to understand its true intent (whether benign or malicious).",
                "Step 2: Read the candidate risk assessment output carefully.",
                "Step 3: Evaluate whether the assessment focuses directly on characteristics and triggers native to this specific message text.",
                "Step 4: Penalize the score if the assessment relies on copy-pasted, generic security commentary or platitudes that do not directly apply to this message content."
            ],
            "direct_file": "prompts/LLM-Judge/direct_rubrics/relevance.md"
        },
        "justification_soundness": {
            "evaluation_steps": [
                "Step 1: Parse the original message context to determine the grounding baseline.",
                "Step 2: Examine the logical reasoning chains presented inside the candidate risk assessment.",
                "Step 3: Verify if the leaps from language observation to final risk determination are mathematically or logically sound, rather than speculative.",
                "Step 4: Confirm that the final risk decision is fully supported by concrete textual evidence referenced directly from the target message."
            ],
            "direct_file": "prompts/LLM-Judge/direct_rubrics/justification.md"
        },
        "contextual_awareness": {
            "evaluation_steps": [
                "Step 1: Classify the original reference message text as either malicious, suspicious, or completely benign.",
                "Step 2: Check if the candidate risk assessment evaluates the message as an integrated whole rather than isolating single keywords.",
                "Step 3: Determine if the assessment accurately balances explicit textual statements against implicit psychological triggers.",
                "Step 4: Check if the model correctly maintains a benign baseline configuration when evaluating clean corporate/personal communications, actively avoiding false-positive paranoia."
            ],
            "direct_file": "prompts/LLM-Judge/direct_rubrics/contextual_awareness.md"
        },
        "accuracy": {
            "evaluation_steps": [
                "Step 1: Analyze the original ground-truth input message to establish the objectively correct security risk level.",
                "Step 2: Isolate the core threat verdict and classified risk parameters within the candidate assessment.",
                "Step 3: Rate the objective accuracy and safety profiles of the assessment's final security judgment.",
                "Step 4: Strictly penalize critical analytical errors, such as misclassifying highly dangerous vectors as safe, or flagging standard benign text as high-risk malicious attacks."
            ],
            "direct_file": "prompts/LLM-Judge/direct_rubrics/accuracy.md"
        },
        "guidance_appropriateness": {
            "evaluation_steps": [
                "Step 1: Evaluate the operational environment and context implied by the original message context.",
                "Step 2: Review all actionable mitigations, instructions, or recommendations supplied by the candidate assessment.",
                "Step 3: Assess whether these action items are practical, protective, and contextually appropriate for the target scenario.",
                "Step 4: Penalize defensive suggestions if they introduce unnecessary operational friction for a benign email, or if they lack the defensive depth required to block an active security exploit."
            ],
            "direct_file": "prompts/LLM-Judge/direct_rubrics/guidance.md"
        }
    }

    # 5. Instantiate our custom Local Ollama model judge
    local_judge = LocalOllamaJudge(model_name="llama3.1:8b")

    # 6. Master loop across all discovered experiments
    for exp_folder in experiment_folders:
        print(f"\n=======================================================")
        print(f"DEEPEVAL LOCAL G-EVAL RUNNING FOR: {exp_folder.upper()}")
        print(f"=======================================================")
        
        exp_results_path = os.path.join(base_results_dir, exp_folder)
        exp_xai_path = os.path.join(base_xai_dir, exp_folder)

        input_text_file = os.path.join(exp_xai_path, "input_text", "cleaned_input.txt")
        no_evidence_file = os.path.join(exp_results_path, "no_evidence", "no_evidence_reply.md")
        baseline_evidence_file = os.path.join(exp_results_path, "baseline_evidence", "baseline_evidence_reply.md")
        bert_evidence_file = os.path.join(exp_results_path, "bert_evidence", "bert_evidence_reply.md")

        if not all(os.path.exists(f) for f in [input_text_file, no_evidence_file, baseline_evidence_file, bert_evidence_file]):
            print(f"Skipping {exp_folder}: Incomplete records.")
            continue

        original_message = load_file_content(input_text_file)
        model_outputs = {
            "no_evidence": load_file_content(no_evidence_file),
            "baseline_evidence": load_file_content(baseline_evidence_file),
            "bert_evidence": load_file_content(bert_evidence_file)
        }

        experiment_report = {}

        for metric_name, cfg in metrics_config.items():
            print(f"  Evaluating Metric: {metric_name.upper()}...")
            detailed_rubric = load_file_content(cfg["direct_file"])

            # 7. Instantiate the dynamic DeepEval G-Eval Metric object with locked execution steps
            # DeepEval will now skip generation steps and inject these arrays directly into the evaluation prompt template
            geval_metric = GEval(
                name=metric_name,
                evaluation_steps=cfg["evaluation_steps"],
                criteria=f"Detailed Scale Guidelines:\n{detailed_rubric}",
                evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                model=local_judge,
                threshold=0.5
            )

            experiment_report[metric_name] = {}

            # 8. Build and execute test cases for each risk evaluation variant
            for model_name, output_content in model_outputs.items():
                test_case = LLMTestCase(
                    input=f"Analyze risks for this message: {original_message}",
                    actual_output=output_content
                )

                # Execute local evaluation call
                geval_metric.measure(test_case)

                # Convert DeepEval's native 0.0-1.0 float back to your legacy 1-5 scale mapping
                scaled_score = round(geval_metric.score * 5, 2)
                reasoning = geval_metric.reason

                experiment_report[metric_name][model_name] = {
                    "score": scaled_score,
                    "reasoning": reasoning
                }
                print(f"    [{model_name}] Score: {scaled_score}/5")

        # Export report matrix for this experiment run
        output_report_path = os.path.join(evaluation_output_dir, f"{exp_folder}_local_eval_report.json")
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_report, f, indent=4, ensure_ascii=False)
            
        print(f"Saved local JSON metrics report for {exp_folder} -> {output_report_path}")

if __name__ == "__main__":
    main()