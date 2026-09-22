import os
from envs.nanogpt_env.Lib import re
import numpy as np  
from openai import OpenAI

class AIRiskAssessor:
    def __init__(self, model_id, input_text, api_key, base_url, user_prompt_path, system_prompt_path, temperature = 0.0, evidence='no evidence'):
        self.model_id = model_id
        self.input_text = input_text
        self.api_key = api_key
        self.base_url = base_url
        self.user_prompt = user_prompt_path
        self.system_prompt = system_prompt_path
        self.temperature = temperature
        self.evidence = evidence
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def load_prompts(self):
        with open(self.user_prompt, "r", encoding="utf-8") as prompt:
            self.user_prompt_content = prompt.read()

        with open(self.system_prompt, "r", encoding="utf-8") as system:
            self.system_prompt_content = system.read()

    def inject_evidence_and_input(self):
        if hasattr(self, "user_prompt_content"):
            self.user_prompt_content = self.user_prompt_content.replace("{evidence}", self.evidence)
        if hasattr(self, "user_prompt_content"):
            self.user_prompt_content = self.user_prompt_content.replace("{message}", self.input_text)

    def process_prompts(self):
        self.load_prompts()
        self.inject_evidence_and_input()

    def get_processed_prompts(self):
        self.process_prompts()
        return self.user_prompt_content, self.system_prompt_content

    def call_chat_model(self):
        user_prompt_content, system_prompt_content = self.get_processed_prompts()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_prompt_content}
            ],
            max_tokens=600,
            temperature=self.temperature
        )
        reply = response.choices[0].message.content.strip()

        if "<think>" in reply:
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

        return reply
