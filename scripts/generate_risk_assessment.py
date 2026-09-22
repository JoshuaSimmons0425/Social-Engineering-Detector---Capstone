from src.explaination.ai_risk_assessor import AIRiskAssessor

def main():

    api_key = 'lm-studio' # Not an actual API key, but a placeholder in order to call LMStudio properly.
    base_url = 'http://localhost:1234/v1'

    user_prompt_path = 'prompts/explainer/explainer_prompt.md'
    system_prompt_path = 'prompts/explainer/system_prompt.md'

    model_name = "qwen2.5-1.5b-instruct"

    input_text = "Notice, your account requires verification. To restore access, reset your password and enter your username, 2FA verification code, and recovery key.\n\nAccount Services,\nMorgan Martinez"

    assessor = AIRiskAssessor(
        model_id=model_name,
        input_text=input_text,
        api_key=api_key,
        base_url=base_url,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        temperature=0.01
    )

    response = assessor.call_chat_model()
    print(response)

if __name__ == "__main__":
    main()