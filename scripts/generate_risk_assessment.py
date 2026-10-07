import os
import sys
import re
from src.risk_assessment.ai_risk_assessor import AIRiskAssessor

def main():
    api_key = 'lm-studio' 
    base_url = 'http://localhost:1234/v1'

    user_prompt_path = 'prompts/explainer/explainer_prompt.md'
    system_prompt_path = 'prompts/explainer/system_prompt.md'
    model_name = "google/gemma-2-9b"
    
    # Base directories
    base_xai_dir = "xAI_outputs"
    base_results_dir = "results/risk_assessments"

    if not os.path.exists(base_xai_dir):
        print(f"Error: Base directory '{base_xai_dir}' not found.")
        sys.exit(1)

    # Find and sort all experiment directories numerically (experiment_0, experiment_1, etc.)
    experiment_folders = [
        d for d in os.listdir(base_xai_dir) 
        if os.path.isdir(os.path.join(base_xai_dir, d)) and d.startswith("experiment_")
    ]
    # Sort naturally by the trailing integer ID
    experiment_folders.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    if not experiment_folders:
        print("No experiment folders found inside xAI_outputs.")
        sys.exit(0)

    print(f"Found {len(experiment_folders)} experiments to assess.\n")

    # Iterate through each experiment sequentially
    for exp_folder in experiment_folders:
        print(f"=== Processing {exp_folder} ===")
        exp_path = os.path.join(base_xai_dir, exp_folder)

        # Build paths to the saved evidence fragments for this specific experiment
        input_text_path = os.path.join(exp_path, "input_text", "cleaned_input.txt")
        baseline_evidence_path = os.path.join(exp_path, "baseline", "explanations.txt")
        bert_evidence_path = os.path.join(exp_path, "bert", "explanations.txt")

        # Verify necessary files exist before executing the LLM calls
        if not all(os.path.exists(p) for p in [input_text_path, baseline_evidence_path, bert_evidence_path]):
            print(f"Skipping {exp_folder}: One or more required explanation source files are missing.")
            continue

        # Read the respective text files
        with open(input_text_path, 'r', encoding='utf-8') as f:
            input_text = f.read()
        with open(baseline_evidence_path, 'r', encoding='utf-8') as f:
            baseline_evidence = f.read()
        with open(bert_evidence_path, 'r', encoding='utf-8') as f:
            bert_evidence = f.read()

        # Map out structures
        structures = {
            "no_evidence": "no evidence",
            "baseline_evidence": baseline_evidence,
            "bert_evidence": bert_evidence
        }
        
        # Run the assessment loop across all evaluation tiers
        for structure in structures:
            print(f"  Running structure: {structure}...")
            
            assessor = AIRiskAssessor(
                model_id=model_name,
                input_text=input_text,
                api_key=api_key,
                base_url=base_url,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                temperature=0.3,
                evidence=structures[structure]
            )

            response = assessor.call_chat_model()
            
            # Map clean output targets to mirror the specific experiment folder structures
            output_dir = os.path.join(base_results_dir, exp_folder, structure)
            os.makedirs(output_dir, exist_ok=True)
            
            output_file_path = os.path.join(output_dir, f"{structure}_reply.md")
            assessor.save_reply_to_file(output_file_path)
            
        print(f"Finished assessments for {exp_folder}.\n")

    print("All risk assessments completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()