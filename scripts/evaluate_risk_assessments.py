import os
import json
import re
import sys
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
# Import DeepEval core infrastructure
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.llms import Ollama

from src.LLM_judgment.ollamajudge import LocalOllamaJudge

# 1. Create a custom wrapper class so DeepEval can talk to local Ollama

def load_file_content(path: str) -> str:
    """Safely loads file content as string."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required experiment file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    load_dotenv()

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

    # 4. Define metrics configurations mapped to custom rubric text instructions
    metrics_config = {
        "relevance": {
            "criteria": "Focusing on characteristics that actually matter over generic security commentary.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/relevance.md"
        },
        "justification_soundness": {
            "criteria": "Logical reasoning chains that strongly support the final risk conclusions.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/justification.md"
        },
        "contextual_awareness": {
            "criteria": "Understanding complex social engineering situations and the interaction of subtle cues.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/contextual_awareness.md"
        },
        "accuracy": {
            "criteria": "Appropriateness and accuracy of the final risk judgment.",
            "direct_file": "prompts/LLM-Judge/direct_rubrics/accuracy.md"
        },
        "guidance_appropriateness": {
            "criteria": "Practicality, safety, and operational sense of the recommended mitigations.",
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

            # 7. Instantiate the dynamic DeepEval G-Eval Metric object pointing to local judge
            geval_metric = GEval(
                name=metric_name,
                criteria=f"{cfg['criteria']}\nDetailed Scale Guidelines:\n{detailed_rubric}",
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
