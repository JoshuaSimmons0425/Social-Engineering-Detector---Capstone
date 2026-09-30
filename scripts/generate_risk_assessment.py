import sys
from src.risk_assessment.ai_risk_assessor import AIRiskAssessor

def main():

    api_key = 'lm-studio' # Not an actual API key, but a placeholder in order to call LMStudio properly.
    base_url = 'http://localhost:1234/v1'

    user_prompt_path = 'prompts/explainer/explainer_prompt.md'
    system_prompt_path = 'prompts/explainer/system_prompt.md'

    baseline_evidence_path = "xAI_outputs/baseline/explanations.txt"
    with open(baseline_evidence_path, 'r') as f:
        baseline_evidence = f.read()

    bert_evidence_path = "xAI_outputs/bert/explanations.txt"

    with open(bert_evidence_path, 'r') as f:
        bert_evidence = f.read()

    input_text_path = "xAI_outputs/input_text/cleaned_input.txt"
    with open(input_text_path, 'r') as f:
        input_text = f.read()

    model_name = "google/gemma-2-9b"

    no_evidence = "no evidence"
    
    structures = {"no_evidence": no_evidence,
                  "baseline_evidence": baseline_evidence,
                  "bert_evidence": bert_evidence
    }
    
    for structure in structures:
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
        print(response)

        assessor.save_reply_to_file(f"results/risk_assessments/{structure}/{structure}_reply.md")

    sys.exit(0)

if __name__ == "__main__":
    main()