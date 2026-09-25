import sys
from src.explaination.ai_risk_assessor import AIRiskAssessor

def main():

    api_key = 'lm-studio' # Not an actual API key, but a placeholder in order to call LMStudio properly.
    base_url = 'http://localhost:1234/v1'

    user_prompt_path = 'prompts/explainer/explainer_prompt.md'
    system_prompt_path = 'prompts/explainer/system_prompt.md'

    bert_evidence_path = "xAI_outputs/bert/explanations.txt"

    with open(bert_evidence_path, 'r') as f:
        bert_evidence = f.read()

    model_name = "llama-3.2-3b-instruct"

    input_text = """Buck up, your troubles caused by small dimension will soon be over!
Become a lover no woman will be able to resist!
http://whitedone.com/


come. Even as Nazi tanks were rolling down the streets, the dreamersphilosopher or a journalist. He was still not sure.I do the same."""


    assessor = AIRiskAssessor(
        model_id=model_name,
        input_text=input_text,
        api_key=api_key,
        base_url=base_url,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        temperature=0.4,
        evidence=bert_evidence
    )

    response = assessor.call_chat_model()
    print(response)

    sys.exit(0)

if __name__ == "__main__":
    main()